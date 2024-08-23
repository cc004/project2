import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'

from runner import init, main
from modules import *

if __name__ == "__main__":
    # model configs
    # model_name = 'gpt2-xl'
    model_name = 'meta-llama/Meta-Llama-3-8B'

    init(model_name, 'quant_search.txt')

    tasks = []

    ratio = 0.001
    while ratio < 0.005:
        tasks.append([
            rquan(ratio=ratio, encoding_1='int4'),
            hashq()
        ])
        ratio += 0.0002
    
    for task in tasks:
        main(task, 'one')
