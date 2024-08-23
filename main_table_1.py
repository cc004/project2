import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from runner import init, main
from modules import *

ratio_best = 0.0015
ratio_best_int4 = 0.002

if __name__ == "__main__":
    # model configs
    model_name = 'gpt2-xl'
    # model_name = 'facebook/opt-6.7b'

    init(model_name, 'fig4.txt')

    tasks = []

    errs = [0, 1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]

    # fig 3 (20 points)

    tasks.append(([
    ] , 'rbase'))

    tasks.append(([
        rquan(ratio=0)
    ], 'rquan0'))

    tasks.append(([
        rquan(ratio=1, encoding_1='flint')
    ], 'rquan1'))

    tasks.append(([
        olivewise(sigmaq(encoding_1='flint'), encoding_1='flint')
    ], 'olivewise'))

    tasks.append(([
        rquan(ratio=ratio_best_int4, encoding_1='int4'),
        hashq()
    ], 'rint4'))

    tasks.append(([
        rquan(ratio=ratio_best, encoding_1='flint'),
        hashq()
    ], 'rflint'))

    tasks.append(([
        sigmaq(encoding_1='int4'),
        hashq()
    ], 'rsint4'))

    tasks.append(([
        sigmaq(encoding_1='flint'),
        hashq()
    ], 'rflint'))

    tasks.append(([
        rquan(ratio=ratio_best_int4, encoding_1='int8'),
        hashq()
    ] , 'rint8'))

    for task, key in tasks:
        main(task, key)
