import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

import runner
from runner import init, main
from modules import *

ratio_best = 0.0015
ratio_best_int4 = 0.002

if __name__ == "__main__":
    # model configs
    # model_name = 'gpt2-xl'
    model_name = 'meta-llama/Meta-Llama-3-8B'

    runner.eval_tasks = "piqa"
    init(model_name, 'llama_4bit.txt')

    tasks = []

    for weight in [0, 0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1, 10, 100]:
        tasks.append(
            ([sigmaq(max_error_weight=weight)], f'w{weight}')
        )
    
    for task, key in tasks:
        main(task, key)
