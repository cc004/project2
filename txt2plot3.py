
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


fp = open('fig4.csv', 'w')

def f(data, idxs, label, funcs = None):
    if not funcs: funcs = [lambda x:x, lambda x:x, lambda x:x]
    trans = [data[idx] for idx in idxs]
    fp.write(f'{label}-piqa,{",".join(str(-funcs[0](x[0])) for x in trans)}\n')
    fp.write(f'{label}-arc_challenge,{",".join(str(-funcs[1](x[1])) for x in trans)}\n')
    fp.write(f'{label}-boolq,{",".join(str(-funcs[2](x[2])) for x in trans)}\n')
    fp.write(f'{label}-avg,{",".join(str(-(funcs[0](x[0]) + funcs[1](x[1]) + funcs[2](x[2])) / 3) for x in trans)}\n')

data = list(readdata('fig4.2.1.txt'))
data += list(readdata('fig4.2.2.txt'))
data += list(readdata('fig4.2.3.txt'))
data += list(readdata('fig4.2.4.txt'))
fp.write('fig1,1e-4,3e-4,1e-3,3e-3,1e-2\n')
base = [0.7225, 0.2978, 0.6165]
fs = [lambda x: base[0] - x, lambda x: base[1] - x, lambda x: base[2] - x]

f(data, [0+i for i in range(5)], 'none', fs)
f(data, [5+i for i in range(5)], 'full', fs)
f(data, [10+i for i in range(5)], 'reset', fs)
f(data, [15+i for i in range(5)], 'w32', fs)

fp.close()
