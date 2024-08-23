import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '2,1'

from runner import init, main
from modules import *

if __name__ == "__main__":
    
    # model configs
    # model_name = 'gpt2-xl'
    # model_name = 'facebook/opt-6.7b'
    model_name = 'facebook/llama-7b'
    
    init(model_name, 'fig1_new2_test.txt')

    tasks = []

    # fig 1 (15 points)
    for ratio in [1e-3]:
      '''
      tasks.append(([
          rquan(ratio=ratio, encoding_1='flint'),
      ], f'ratio:{ratio}'))
      tasks.append(([
          rquan(ratio=ratio, encoding_1='int4'),
      ], f'ratio:{ratio}'))'''
      for percent in [1e-3]:
          if percent > ratio: continue
          tasks.append(([
              rquan(ratio=ratio, encoding_1='flint'),
              adderrorq(error_percent=percent, error_on='outlier'),
          ], f'ratio:{ratio}'))
          tasks.append(([
              rquan(ratio=ratio, encoding_1='int4'),
              adderrorq(error_percent=percent, error_on='outlier'),
          ], f'ratio:{ratio}'))
    for _ in range(20):
      for task, key in tasks:
          main(task, key)
