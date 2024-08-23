
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


fp = open('fig_tab.csv', 'w')

def f(data, idxs, label, funcs = None):
    if not funcs: funcs = [lambda x:x, lambda x:x, lambda x:x]
    trans = [data[idx] for idx in idxs]
    fp.write(f'{label}-piqa,{",".join(str(funcs[0](x[0])) for x in trans)}\n')
    fp.write(f'{label}-arc_challenge,{",".join(str(funcs[1](x[1])) for x in trans)}\n')
    fp.write(f'{label}-boolq,{",".join(str(funcs[2](x[2])) for x in trans)}\n')
    fp.write(f'{label}-avg,{",".join(str((funcs[0](x[0]) + funcs[1](x[1]) + funcs[2](x[2])) / 3) for x in trans)}\n')

data = list(readdata('fig_tab1.txt'))

ff = [
    lambda x: 100*x,
    lambda x: 100*x,
    lambda x: 100*x,
]
fp.write('table,int4,ant,olive,best-int4,best-flint,3s-int4,3s-flint,int8\n')
f(data, [0,1,2,3,4,5,6,7], 'table', ff) # 47=int8

data = list(readdata('fig_tab2.txt'))

fp.write('table,int4,ant,olive,best-int4,best-flint,3s-int4,3s-flint,int8\n')
f(data, [0,1,2,3,4,5,6,7], 'table', ff) # 47=int8

data = list(readdata('fig_main_table.txt'))

fp.write('table,int4,ant,olive,best-int4,best-flint,3s-int4,3s-flint,int8\n')
f(data, [0,1,2,3,4,5,6,7], 'table', ff) # 47=int8

fp.close()