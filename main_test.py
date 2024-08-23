import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'

from runner import init, main
from modules import *

if __name__ == "__main__":
    model_name = 'meta-llama/Meta-Llama-3-8B'

    init(model_name, 'main_test.txt')

    main([], 'main')
