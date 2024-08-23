import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from runner import init, main
from modules import *

ratio_best = 0.0015
ratio_best_int4 = 0.002

if __name__ == "__main__":
    # model configs
    # model_name = 'gpt2-xl'
    model_name = 'facebook/opt-6.7b'

    init(model_name, 'fig4.2.2_test1e-2.txt')

    tasks = []

    errs = [1e-2]

    # fig 3 (20 points)

    for err in errs:
        tasks.append([
            int4rquan(),
            hashq(),
            adderror(error_percent=err),
            fullerr(correct_error_low=2, fame_count=1)
        ])
    
    for _ in range(10):
        for task in tasks:
            main(task, 'one')
