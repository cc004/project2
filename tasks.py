from modules import *

tasks = []

# fig 1 (15 points)

for ratio in [1e-8, 1e-7, 1e-6]:
    for percent in [1e-8, 3e-7, 1e-7, 3e-6, 1e-6]:
        if percent > ratio: continue
        tasks.append([
            rquan(ratio=ratio, encoding_1='flint'),
            adderrorq(error_percent=percent, error_on='outlier'),
        ])

errs = [0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]

simplified_errs = [1e-6, 1e-5, 1e-4, 1e-3, 3e-3]

# fig 2 (33 points)

err_ons = ['outlier', 'non-outlier']

tasks.append([
    rquan(ratio=0.005, encoding_1='int4'),
])
tasks.append([
    olivewise(quan(encoding_1='flint')),
])
tasks.append([
    rquan(ratio=1, encoding_1='flint'),
])

for err in simplified_errs:
    for err_on in err_ons:
        tasks.append([
            rquan(ratio=0.005, encoding_1='int4'),
            adderrorq(error_percent=err, error_on=err_on),
        ])
        tasks.append([
            olivewise(quan(encoding_1='flint')),
            adderrorq(error_percent=err, error_on=err_on),
        ])
        tasks.append([
            rquan(ratio=1, encoding_1='flint'),
            adderrorq(error_percent=err, error_on=err_on),
        ])
    
# fig 3 (20 points)

for err in simplified_errs:
    for bit_high in [4, 3, 2, 1]:
        tasks.append([
            rquan(ratio=0.005, encoding_1='int4'),
            adderror(error_percent=err, error_high=bit_high, err_low=0),
        ])

# find ratio best

'''
for ratio in [
    1e-3, 2e-3, 3e-3, 4e-3, 5e-3,
    6e-3, 7e-3, 8e-3, 9e-3, 1e-2
]:
    tasks.append([
        rquan(ratio=ratio, encoding_1='int4'),
        hashq()
    ])

'''

ratio_best = 0.003 # TO BE DETERMINED

# table (1 point)

tasks.append([
    rquan(ratio=ratio_best, encoding_1='int4'),
    hashq()
])

# fig 4 ?

for err in errs:
    tasks.append([
        rquan(ratio=ratio_best, encoding_1='int4'),
        hashq(),
        adderror(error_percent=err),
        hasherr()
    ])
    tasks.append([
        rquan(ratio=ratio_best, encoding_1='int4'),
        hashq(),
        adderror(error_percent=err),
        wesco()
    ])
    
'''
# fig 5

for err in errs:
    tasks.append([
        rquan(ratio=ratio_best, encoding_1='int4'),
        hashq(),
        adderror(error_percent=err),
        hasherr()
    ])
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        tasks.append([
            rquan(ratio=ratio_best, encoding_1='int4'),
            hashq(),
            adderror(error_percent=err),
            hasherr(),
            rowreduce_new(reduce_threshold=threshold)
        ])
'''

# fig 6 (25 points)

for err in simplified_errs:
    for bit_low in [3, 2, 1, 0]:
        tasks.append([
            rquan(ratio=ratio_best, encoding_1='int4'),
            hashq(),
            adderror(error_percent=err),
            fullerr(correct_error_low=bit_low)
        ])
    tasks.append([
        rquan(ratio=ratio_best, encoding_1='int4'),
        hashq(),
        adderror(error_percent=err),
        weightzeroerr()
    ])
    tasks.append([
        rquan(ratio=ratio_best, encoding_1='int4'),
        hashq(),
        adderror(error_percent=err),
        weightzeroerr_v2()
    ])


print(f'task count = {len(tasks)}')