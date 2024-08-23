import torch

from transformers.pytorch_utils import Conv1D
from datasets import load_dataset
from lm_eval.utils import make_table
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

import gc, json
from modules import qmgr, qbase
import modules as ms

import os
os.environ["http_proxy"] = "http://1.yanglab.icu:7890" 
os.environ["https_proxy"] = "http://1.yanglab.icu:7890"

token = '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'
eval_tasks = None
model = None
tokenizer = None
g_model_name = None
org_sd = None

log = None

def print_(str):
    print(str)
    log.write(str + '\n')
    log.flush()

def print_task(task):
    str = json.dumps({module.__class__.__name__: {k: v for k, v in module.config.items() if k != 'device'} for module in task})
    print_(f'config = {str}')

TEST = False

# LlamaSdpaAttention

state_dict = {}

from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaDecoderLayer, logger, apply_rotary_pos_emb, repeat_kv
from typing import Optional, Tuple
from functools import partial

save_state = {}
last_idx = -1

def save_attn(attn_output: torch.Tensor, name: str, idx: int):
    global save_state, last_idx
    if idx < last_idx:
        torch.save(save_state, 'attn.pth')
        exit(0)
    save_state[name] = attn_output.cpu()
    last_idx = idx
    
    return attn_output

def do_quantize(attn_output: torch.Tensor, modules, cacheKey: str, name: str):
    key = f'{cacheKey}_{name}'
    
    q = qmgr(modules, print_)
    
    # for module in modules:
    #    if not isinstance(module, qbase):
    #        q.force_init.append(module)
    
    q.state = state_dict.get(key, None)
    
    state = q.process(attn_output.contiguous().to(ms.g_device))
    
    state_dict[key] = q.state
    
    val = state['tensor'].to('cuda:0').to(torch.float16)
    
    qmse = (attn_output - val) ** 2
    
    print_(f'{name} qmse: {qmse.max()}, {qmse.mean()}')

    attn_output = val
    
    del state
    gc.collect()
    
    return attn_output

def hook_llama(layer: LlamaSdpaAttention, processor):
    
    self = layer
    
    def forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        assert output_attentions == False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
        # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        ## add
        
        attn_output = processor(attn_output)
        
        
        ## add end
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
    
    layer.forward = forward

def init(model_name, log_name):
    global eval_tasks, model, tokenizer, g_model_name, log, org_sd
    
    # model_name = 'bigscience/bloomz-7b1'
    eval_tasks = "piqa,arc_challenge,boolq"
    kwargs = {"torch_dtype": torch.float16, 'token': token}
    tokenizer_kwargs = {
        "use_fast": True,
        "revision": "main",
        "use_auth_token": None,
        'token': token
    }

    if TEST:
        import os
        eval_tasks = 'wikitext'
        #model_name = r'C:\Users\Administrator\.cache\huggingface\hub\models--facebook--opt-125m\snapshots\3d2b5f275bdf882b8775f902e1bfdb790e2cfc32'
        model_name = 'facebook/opt-125m'
        #os.environ["http_proxy"] = "http://127.0.0.1:1080" 
        #os.environ["https_proxy"] = "http://127.0.0.1:1080"

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        ms.g_device = torch.device('cuda:0')
        
        from transformers.models.opt.modeling_opt import OPTForCausalLM
        from transformers import AutoTokenizer
        print("Loading model...")
        model = OPTForCausalLM.from_pretrained(model_name, **kwargs).cuda('cuda:0')
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    # load model and tokenizer
    elif model_name == "bigscience/bloomz-7b1":
        model_name = '/home/cc/huggingface/hub/models--bigscience--bloomz-7b1/snapshots/2f4c4f3ebcf171dbbe2bae989ea2d2f3d3486a97/'
        from transformers.models.bloom.modeling_bloom import BloomForCausalLM
        from transformers.models.bloom.tokenization_bloom_fast import BloomTokenizerFast
        print("Loading model...")
        model = BloomForCausalLM.from_pretrained(model_name, **kwargs).cuda('cuda:0')
        print("Loading tokenizer...")
        tokenizer = BloomTokenizerFast.from_pretrained(model_name, **tokenizer_kwargs)
    elif model_name == "gpt2-xl":
        model_name = '/home/cc/huggingface/hub/models--gpt2-xl/snapshots/33cdb5c0db5423c1879b1b9f16c352988e8754a8/'
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
        from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
        print("Loading model...")
        model = GPT2LMHeadModel.from_pretrained(model_name, **kwargs).cuda('cuda:0')
        print("Loading tokenizer...")
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name, **tokenizer_kwargs)
    elif model_name == "facebook/opt-6.7b" or model_name == "facebook/opt-125m":
        if model_name == 'facebook/opt-125m':
            model_name = '/home/cc/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6/'
        else:
            model_name = '/home/cc/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0/'
        from transformers.models.opt.modeling_opt import OPTForCausalLM
        from transformers import AutoTokenizer
        print("Loading model...")
        model = OPTForCausalLM.from_pretrained(model_name, **kwargs).cuda('cuda:0')
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    elif model_name == 'meta-llama/Meta-Llama-3-8B':
        model_name = '/home/cc/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6/'
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        from transformers import AutoTokenizer
        print("Loading model...")
        model = LlamaForCausalLM.from_pretrained(model_name, **kwargs).cuda('cuda:0')
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        # tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
      
        # prepare llamda model
        
          
          
          
      
    g_model_name = model_name
    log = open(log_name, 'a')
    org_sd = {k: v.cpu() for k, v in model.state_dict().items()}

def main(modules, cacheKey):

    print_task(modules)
    
    
    for idx, layer in enumerate(model.base_model.layers):
        layer: LlamaDecoderLayer
        assert type(layer) is LlamaDecoderLayer
        
        name = f'attention_{idx}'
        # hook_llama(layer.self_attn, partial(do_quantize, modules=modules, name=name, idx=idx))
        hook_llama(layer.self_attn, partial(save_attn, name=name, idx=idx))
    
    for name, m in model.named_modules():
        if 'lm_head' in name:
            continue
        if isinstance(m, torch.nn.Linear) or isinstance(m, Conv1D):
            
            key = f'{cacheKey}_{name}'
            
            q = qmgr(modules, print_)
            
            q.state = state_dict.get(key, None)
            
            state = q.process(m.weight.data.to(ms.g_device))
            
            state_dict[key] = q.state
            
            val = state['tensor'].to('cuda:0').to(torch.float16)
            
            qmse = (m.weight.data - val) ** 2
            
            print_(f'{name} qmse: {qmse.max()}, {qmse.mean()}')

            m.weight.data = val
            
            del state
            gc.collect()
    
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer)
    # evaluation function
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=eval_tasks.split(","),
        batch_size=1,
        num_fewshot=0
    )

    print_(make_table(results))
    
    model.load_state_dict(org_sd)