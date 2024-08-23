import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '2,6'

from runner import init, main
from modules import *
from hash import *

if __name__ == "__main__":
    # model configs
    # model_name = 'gpt2-xl'
    model_name = 'facebook/opt-6.7b'

    init(model_name, 'quant_search2.txt')

    tasks = []

    ratio = 0.0002
    while ratio < 0.005:
        tasks.append([
            rquan(ratio=ratio, encoding_1='int4'),
            hashq(hash=MinCI16384_flower)
        ])
        ratio += 0.0002
    
    for task in tasks:
        main(task, 'one')
