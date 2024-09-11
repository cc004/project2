import os
os.environ["http_proxy"] = "http://1.yanglab.icu:7890" 
os.environ["https_proxy"] = "http://1.yanglab.icu:7890"
token = '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'
model_name = '/home/cc/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6/'
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import torch
from transformers.pytorch_utils import Conv1D
kwargs = {"torch_dtype": torch.float16, 'token': token}
print("Loading model...")
device = 'cuda:0'
model: torch.nn.Module = LlamaForCausalLM.from_pretrained(model_name, **kwargs).cuda(device)

N = 65536
half_N = N // 2

all_stat = torch.zeros(N, dtype=torch.int32, device=device)

for name, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) or isinstance(m, Conv1D):
        scale = m.weight.data.abs().max().float()
        scale_0 = half_N / scale
        q = (m.weight.data.float() * scale_0).int() + half_N
        q = torch.clamp(q, 0, N - 1)
        stat = q.view(-1).bincount(minlength=N)
        all_stat += stat

all_stat = all_stat.cpu().numpy()

import numpy as np

np.save('llama3_stat.npy', all_stat)