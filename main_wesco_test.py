import os
os.environ["http_proxy"] = "http://127.0.0.1:1080" 
os.environ["https_proxy"] = "http://127.0.0.1:1080"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from hash import *
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D
from datasets import load_dataset
from lm_eval import evaluator
from lm_evaluate_adaptor import LMEvalAdaptor

from modules import *

import gc

log = open('log.txt', 'a')

def print_(str):
    print(str)
    log.write(str + '\n')


def main(modules):
    # model configs
    model_name = 'gpt2-xl'
    model_name = 'facebook/opt-125m'
    # model_name = 'bigscience/bloomz-7b1'
    tasks = "wikitext"
    kwargs = {"torch_dtype": torch.float16}

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    for name, m in model.named_modules():
        if 'lm_head' in name:
            continue
        if isinstance(m, torch.nn.Linear) or isinstance(m, Conv1D):
            
            state = qmgr(modules, print).process(m.weight.data)
            
            val = state['tensor'].to(torch.float16)
            
            qmse = (m.weight.data - val) ** 2
            
            print(f'{name} qmse: {qmse.max()}, {qmse.mean()}')

            m.weight.data = val
            
            del state
            gc.collect()
        

    lm_eval_model = LMEvalAdaptor(model_name, model, tokenizer)
    # evaluation function
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=tasks.split(","),
        batch_size=1,
        no_cache=True,
        num_fewshot=0,
    )

    print_(evaluator.make_table(results))


if __name__ == "__main__":
    main([
        rquan(ratio=0.002, encoding_1='int4'),
        hashq(),
    ])
