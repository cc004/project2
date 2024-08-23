
def readdata(file):
    state = -1
    with open(file) as fp:
        for line in fp.readlines():
            if 'piqa' in line:
                state = 0
                res = []
            if state >= 0:
                state += 1
                if state == 2 or state == 4 or state == 5:
                    res.append(float(line.split('|')[4]))
                elif state == 6:
                    state = -1
                    yield res


fp = open('fig.csv', 'w')

def f(data, idxs, label, funcs = None):
    if not funcs: funcs = [lambda x:x, lambda x:x, lambda x:x]
    trans = [data[idx] for idx in idxs]
    fp.write(f'{label}-piqa,{",".join(str(-funcs[0](x[0])) for x in trans)}\n')
    fp.write(f'{label}-arc_challenge,{",".join(str(-funcs[1](x[1])) for x in trans)}\n')
    fp.write(f'{label}-boolq,{",".join(str(-funcs[2](x[2])) for x in trans)}\n')
    fp.write(f'{label}-avg,{",".join(str(-(funcs[0](x[0]) + funcs[1](x[1]) + funcs[2](x[2])) / 3) for x in trans)}\n')

data = list(readdata('fig1.txt'))
fp.write('fig1,1e-8,3e-7,1e-7,3e-6,1e-6\n')
f(data, [0+2*i for i in range(5)], 'flint')
f(data, [1+2*i for i in range(5)], 'int4')

data = list(readdata('fig2.txt'))

base = data[0]

fp.write('fig2,1e-4,1e-3,2e-3,3e-3,5e-3,1e-2\n')
fs = [lambda x: base[0] - x, lambda x: base[1] - x, lambda x: base[2] - x]
f(data, [1+2*i for i in range(6)], 'outlier', fs)
f(data, [2+2*i for i in range(6)], 'non-outlier', fs)

data = list(readdata('fig2.1.txt'))

fp.write('fig21,2e-4,3e-4,5e-4\n')
f(data, [0+2*i for i in range(3)], 'outlier', fs)
f(data, [1+2*i for i in range(3)], 'non-outlier', fs)

try:
    data = list(readdata('fig3.txt'))

    fp.write('fig3,1e-4,1e-3,2e-3,3e-3,5e-3,1e-2\n')
    f(data, [3+4*i for i in range(6)], 'all', fs)
    f(data, [2+4*i for i in range(6)], 'all-1', fs)
    f(data, [1+4*i for i in range(6)], 'all-2', fs)
    f(data, [0+4*i for i in range(6)], 'all-3', fs)
except:
    pass

data = list(readdata('fig4.txt'))


fp.write('table,int4,ant,olive,best-int4,best-flint,3s-int4,3s-flint\n')
f(data, [0,1,2,3,4,5,6], 'table') # 47=int8

try:
    data = list(readdata('fig4.1.txt'))

    fp.write('fig4,0,1e-4,3e-4,1e-3,3e-3,1e-2\n')
    base=baset=data[0]
    f(data, [0+4*i for i in range(1,6)], 'none,1', fs)
    base=data[1]
    f(data, [1+4*i for i in range(1,6)], 'full,1', fs)
    base=data[2]
    f(data, [2+4*i for i in range(1,6)], 'zero,1', fs)
    base=data[3]
    f(data, [3+4*i for i in range(1,6)], 'w32,1', fs)
except:
    pass

try:
    data = list(readdata('fig6.txt'))
    base=baset
    fp.write('fig6,1e-4,1e-3,2e-3,3e-3,5e-3,1e-2\n')
    f(data, [1+3*i for i in range(6)], 'full', fs)
    f(data, [2+3*i for i in range(6)], 'full-1', fs)
    f(data, [3+3*i for i in range(6)], 'full-2', fs)
except:
    pass

fp.close()

'''
    # fig 6
    
    tasks.append([
        rquan(ratio=ratio_best_int4, encoding_1='int4'),
        hashq()
    ])
    for err in simplified_errs:
        for bit_low in [3, 2, 1]:
            tasks.append([
                rquan(ratio=ratio_best_int4, encoding_1='int4'),
                hashq(),
                adderror(error_percent=err),
                fullerr(correct_error_low=bit_low)
            ])

'''

        
