import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from runner import init, main
from modules import *

ratio_best = 0.0015
ratio_best_int4 = 0.002

if __name__ == "__main__":
    # model configs
    # model_name = 'gpt2-xl'
    model_name = 'facebook/opt-6.7b'

    init(model_name, 'fig4.txt')

    tasks = []

    err = 1

    tasks.append([
        rquan(ratio=1, encoding_1='int8'),
        hashq()
    ])
    
    for task in tasks:
        main(task, 'one')
