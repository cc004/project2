import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'

from runner import init, main
from modules import *

if __name__ == "__main__":
    # model configs
    # model_name = 'gpt2-xl'
    model_name = 'facebook/opt-6.7b'

    init(model_name, 'quant_search_opt2.txt')

    tasks = []

    ratio = 0.001
    while ratio < 0.02:
        tasks.append(([
            rquan(ratio=ratio, encoding_1='int8', encoding_2='int3'),
            hashq()
        ], f'ratio{ratio}'))
        ratio += 0.001
    
    for task, key in tasks:
        main(task, key)
