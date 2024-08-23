from dataset_100 import testloader
from model_100 import reload_model, evaluatea, fusenet
import math
from modules import *

model = reload_model()
model = fusenet(model)

evaluatea(model, testloader)

for ratio in [
    #1e-8, 1e-4, 3e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2, 3e-2, 1e-1
    #7e-3, 8e-3, 9e-3, 10e-3, 11e-3, 12e-3, 13e-3, 14e-3
    3e-3
]:
    model = reload_model()
    model = fusenet(model)
    
    qmask_all = []
    
    s = 0; n = 0; s0 = 0; n0 = 0
    col = 128
    def handler(state):
        global s, n, s0, n0
        qmask = state['qmask'].flatten()
        tensor = state['tensor'].flatten()
        
        num = qmask.numel()
        pad = math.ceil(num / col) * col - num
        
        qmask = torch.cat([qmask, torch.zeros(pad, dtype=qmask.dtype, device=qmask.device)])
        tensor = torch.cat([tensor, torch.zeros(pad, dtype=tensor.dtype, device=tensor.device)])
        
        qmask = qmask.view(-1, col)
        tensor = tensor.view(-1, col)
        
        s += qmask.shape[0]
        s0 += qmask.numel()
        
        n += (qmask.sum(dim=1) > 0).sum().item()
        t = qmask.sum(dim=1)
        
        outliner = qmask * tensor
        outliner_max = outliner.abs().max(dim=1).values
        scale = tensor.abs().max()
        
       # n += ((t == 1) * (outliner_max < 0.5 * scale)).sum().item()
       # n0 += qmask.sum().item()
        
    w = wrapper([rquan(ratio=ratio, encoding_1='int4'), olive()])
    # w = wrapper([sigmaq(), rowreduce(), hashq()])
    w.process(model, handler)
    del w
    
    x = 1 - (1 - ratio) ** col
    
    print(f'percent: {n / s * 100}% ~ {x * 100}% all {n0 / s0 * 100}%')
    
    print(f'ratio: {ratio}')
    evaluatea(model, testloader) 