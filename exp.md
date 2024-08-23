## figure 1

错误全在最大的percent(1e-6, 1e-5, 1e-4) 错误数 vs accuracy

## figure 2.1

{
    encoding_1=int4 -> ours
    ratio=0.005
}

1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3 (logscale)

错误出现在(all, non-outlier, outlier) 错误数/总参数量 vs accuracy

## figure 2.2

{
    encoding_1=flint -> olive
    ratio=0.005
}
{
    encoding_1=flint -> ant
    ratio=1
}
1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3 (logscale)

错误出现在(all, non-outlier, outlier) 错误数/总参数量 vs accuracy-drop

## figure 3

ratio=0.005
1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3 (logscale)

错误出现在(all, all-1, a-2, a-3) 错误数/总参数量 vs accuracy

## table

ratio=? => best accuracy
encoding_1=4bit

## figure 4

ratio=best
encoding_1=4bit
error=1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3 (logscale)

## figure 5

ratio=best
encoding_1=4bit
error=1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3 (logscale)

error vs outlier和错误总行数 相对于纯随机?

1. normal
2. 剪threshold=?

## figure 6

ratio=best
1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3 (logscale)

只纠正前n个bit/weightzero 错误比例 vs accuracy
只纠正前n个bit/weightzero 错误比例 vs 存储的overhead

bitzero: 非符号位->0， 符号位=0

n=1, 2, 3, 4

