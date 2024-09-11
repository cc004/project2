import numpy as np

filenames = ['opt_stat.npy', 'llama3_stat.npy', 'gpt_stat.npy', 'llama3_stat_col3.npy']

for f in filenames:
    print(f)
    stat = np.load(f)
    N = len(stat)

    def center(M):
        Mnum = N // M
        left = N // 2 - Mnum // 2
        right = N // 2 + Mnum // 2
        percent = stat[left:right].sum() / stat.sum()
        print(f"Percent of center 1/{M}: {percent:.2%}")

    center(256)
    center(128)
    center(64)
    center(32)
    center(16)
    center(8)
    center(4)