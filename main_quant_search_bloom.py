import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from runner import init, main
from modules import *

if __name__ == "__main__":
    # model configs
    # model_name = 'gpt2-xl'
    model_name = 'bigscience/bloomz-7b1'

    init(model_name, 'quant_search_bloom2.txt')

    tasks = []

    ratio = 0.001
    while ratio < 0.01:
        tasks.append([
            rquan(ratio=ratio, encoding_1='int4', encoding_2='int2'),
            hashq()
        ])
        tasks.append([
            rquan(ratio=ratio, encoding_1='flint', encoding_2='int2'),
            hashq()
        ])
        ratio += 0.0002
    
    for task in tasks:
        main(task, 'one')
