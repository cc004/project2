import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

from runner import init, main
from modules import *

ratio_best = 0.0015
ratio_best_int4 = 0.002

if __name__ == "__main__":
    # model configs
    # model_name = 'gpt2-xl'
    model_name = 'meta-llama/Meta-Llama-3-8B'

    init(model_name, 'wesco_col128.txt')

    tasks = []

    errs = [0, 1e-6, 3e-6, 1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 1e-2]

    # fig 3 (20 points)

    for err in errs:
        tasks.append([
            int4rquan(),
            hashq(),
            adderror(error_percent=err),
            wesco_v2_col(col=128)
        ])
    
    for _ in range(10):
        for task in tasks:
            main(task, 'one')
