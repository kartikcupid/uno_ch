# Results - Cahn-Hilliard UNO

Report material only. Every file here is regenerated from the rollout
caches in `Work/rollout_cache/` by `python make_submission.py`; no model is
loaded and no trajectory is re-solved. One folder per run, named
`train_t<END>[_lap][_<size>]_pred_t<T>`: a model trained on data to t = END,
rolled out to t = T. `_lap` marks the Laplacian-output model (the head predicts
M and the increment is its 5-point Laplacian); runs without it are the base
model (direct increment) and are kept for comparison. `<size>` (`m24`, `w48`,
`w44m22`) marks a model with other than the default 20 Fourier modes / width 40 and
`s<N>` (`w32s1`) a model trained with training seed N instead of 0 (a seed
replicate of the same architecture); an untagged run has the default size
and seed 0.

## Runs

| run | model | parameters | trained to | predicted to | seeds | diverged | critical R2(t_end) | folder |
|---|---|---|---|---|---|---|---|---|
| train_t0300_pred_t10000 | base | 17.90M | t = 300 | t = 10000 | 50 | 0 | R2(10000) = 0.6229 +- 0.2201 | `train_t0300_pred_t10000/` |
| train_t0300_lap_pred_t10000 | Laplacian output | 17.90M | t = 300 | t = 10000 | 50 | 0 | R2(10000) = 0.7702 +- 0.1235 | `train_t0300_lap_pred_t10000/` |
| train_t0500_pred_t10000 | base | 17.90M | t = 500 | t = 10000 | 50 | 0 | R2(10000) = 0.1759 +- 0.2574 | `train_t0500_pred_t10000/` |
| train_t0500_lap_pred_t10000 | Laplacian output | 17.90M | t = 500 | t = 10000 | 50 | 0 | R2(10000) = 0.8279 +- 0.1185 | `train_t0500_lap_pred_t10000/` |
| train_t1000_pred_t10000 | base | 17.90M | t = 1000 | t = 10000 | 50 | 0 | R2(10000) = 0.0100 +- 0.2564 | `train_t1000_pred_t10000/` |
| train_t1000_lap_pred_t10000 | Laplacian output | 17.90M | t = 1000 | t = 10000 | 50 | 0 | R2(10000) = 0.9033 +- 0.0879 | `train_t1000_lap_pred_t10000/` |
| train_t1000_lap_m24_pred_t10000 | Laplacian output, modes 24 | 25.11M | t = 1000 | t = 10000 | 50 | 0 | R2(10000) = 0.8953 +- 0.1031 | `train_t1000_lap_m24_pred_t10000/` |
| train_t1000_lap_w24_pred_t10000 | Laplacian output, width 24, modes 20 | 6.47M | t = 1000 | t = 10000 | 50 | 0 | R2(10000) = 0.9319 +- 0.1213 | `train_t1000_lap_w24_pred_t10000/` |
| train_t1000_lap_w24m8_pred_t10000 | Laplacian output, width 24, modes 8 | 1.68M | t = 1000 | t = 10000 | 50 | 0 | R2(10000) = 0.9341 +- 0.1100 | `train_t1000_lap_w24m8_pred_t10000/` |
| train_t1000_lap_w32_pred_t10000 | Laplacian output, width 32, modes 20 | 11.47M | t = 1000 | t = 10000 | 50 | 0 | R2(10000) = 0.9327 +- 0.1313 | `train_t1000_lap_w32_pred_t10000/` |
| train_t1000_lap_w48_pred_t10000 | Laplacian output, width 48, modes 20 | 25.76M | t = 1000 | t = 10000 | 50 | 0 | R2(10000) = 0.8559 +- 0.0960 | `train_t1000_lap_w48_pred_t10000/` |

R2(t_end) is the mean over seeds of each seed's R2 against the ground
truth at the end of that run's rollout, +- one standard deviation
across seeds. A seed whose rollout diverged (non-finite or |psi|
beyond the limit) is counted in the `diverged` column and left out of
every R2 mean; the survival statistics count it as failed. Runs with
different horizons are not comparable at their end points;
`comparison.png` puts the critical curves of the default-size, seed-0
runs (every training horizon, base and Laplacian output) on one log-t
axis instead. `comparison_t<END>.png` does the same for the model-size
and training-seed variants of one END (Laplacian output); written:
`comparison_t1000.png`. In the per-run tables, max and min are the
extreme seeds' R2(t_end); the best seed is chosen by trajectory-mean
R2, so it need not be the one with the max.

## train_t0300_pred_t10000

Trained to t = 300 (base, modes 20, 17.9M), rolled out from t = 100 to t = 10000; checkpoint `ccecc0f076d7`.

| case | R2(t = 10000) | median | max | min | trajectory-mean R2 | best seed (mean R2) | diverged |
|---|---|---|---|---|---|---|---|
| psi0m0.40 | 0.2073 +- 0.2815 | 0.2080 | 0.7529 | -0.7526 | 0.5065 +- 0.1476 | 54464 (0.7638) | 0 of 50 |
| psi0m0.20 | 0.1841 +- 0.3190 | 0.2086 | 0.7820 | -0.5405 | 0.4973 +- 0.1879 | 61803 (0.7949) | 0 of 50 |
| psi0p0.00 | 0.6229 +- 0.2201 | 0.7130 | 0.9068 | 0.0343 | 0.7122 +- 0.1194 | 36963 (0.8941) | 0 of 50 |
| psi0p0.20 | 0.2094 +- 0.2837 | 0.2723 | 0.6391 | -0.9175 | 0.5147 +- 0.1666 | 92280 (0.7269) | 0 of 50 |
| psi0p0.40 | 0.1127 +- 0.2972 | 0.0954 | 0.7713 | -0.7554 | 0.4902 +- 0.1388 | 99999 (0.8093) | 0 of 50 |
| l160 | 0.5719 +- 0.1739 | 0.6292 | 0.8534 | 0.0798 | 0.6979 +- 0.0991 | 22911 (0.8257) | 0 of 50 |
| l192 | 0.4997 +- 0.1775 | 0.5341 | 0.7830 | 0.0866 | 0.6511 +- 0.0984 | 51170 (0.8102) | 0 of 50 |
| l256 | 0.4714 +- 0.0932 | 0.4813 | 0.6730 | 0.2010 | 0.6371 +- 0.0515 | 57721 (0.7498) | 0 of 50 |

## train_t0300_lap_pred_t10000

Trained to t = 300 (Laplacian output, modes 20, 17.9M), rolled out from t = 100 to t = 10000; checkpoint `70e255ec2c21`.

| case | R2(t = 10000) | median | max | min | trajectory-mean R2 | best seed (mean R2) | diverged |
|---|---|---|---|---|---|---|---|
| psi0m0.40 | 0.4894 +- 0.2313 | 0.5357 | 0.7882 | -0.1897 | 0.7086 +- 0.1350 | 48417 (0.8846) | 0 of 50 |
| psi0m0.20 | 0.5198 +- 0.2037 | 0.5561 | 0.8232 | 0.0847 | 0.7541 +- 0.1119 | 99999 (0.9095) | 0 of 50 |
| psi0p0.00 | 0.7702 +- 0.1235 | 0.8053 | 0.9348 | 0.4475 | 0.8819 +- 0.0732 | 85507 (0.9666) | 0 of 50 |
| psi0p0.20 | 0.5142 +- 0.2169 | 0.5677 | 0.8667 | -0.0268 | 0.7507 +- 0.1166 | 33052 (0.9143) | 0 of 50 |
| psi0p0.40 | 0.5312 +- 0.2289 | 0.5839 | 0.9200 | -0.0500 | 0.7456 +- 0.1129 | 35064 (0.9074) | 0 of 50 |
| l160 | 0.8155 +- 0.1277 | 0.8531 | 0.9472 | 0.3428 | 0.9044 +- 0.0649 | 34617 (0.9627) | 0 of 50 |
| l192 | 0.7646 +- 0.1314 | 0.8125 | 0.8953 | 0.2761 | 0.8838 +- 0.0710 | 28096 (0.9526) | 0 of 50 |
| l256 | 0.8076 +- 0.0518 | 0.8089 | 0.8911 | 0.6325 | 0.9039 +- 0.0286 | 31458 (0.9431) | 0 of 50 |

## train_t0500_pred_t10000

Trained to t = 500 (base, modes 20, 17.9M), rolled out from t = 100 to t = 10000; checkpoint `0b3842c641fb`.

| case | R2(t = 10000) | median | max | min | trajectory-mean R2 | best seed (mean R2) | diverged |
|---|---|---|---|---|---|---|---|
| psi0m0.40 | -0.1522 +- 0.3170 | -0.1451 | 0.4477 | -1.1446 | 0.4488 +- 0.1496 | 31458 (0.7740) | 0 of 50 |
| psi0m0.20 | -0.4114 +- 0.2659 | -0.4010 | 0.2571 | -0.9802 | 0.2979 +- 0.1687 | 68620 (0.6938) | 0 of 50 |
| psi0p0.00 | 0.1759 +- 0.2574 | 0.1775 | 0.6619 | -0.5315 | 0.4995 +- 0.1426 | 48417 (0.7931) | 0 of 50 |
| psi0p0.20 | -0.3787 +- 0.2474 | -0.3746 | 0.2286 | -0.9106 | 0.2752 +- 0.1733 | 52066 (0.6732) | 0 of 50 |
| psi0p0.40 | -0.4941 +- 0.3616 | -0.5309 | 0.1845 | -1.1687 | 0.3033 +- 0.1758 | 99999 (0.6381) | 0 of 50 |
| l160 | 0.1371 +- 0.1798 | 0.1670 | 0.4741 | -0.2500 | 0.4808 +- 0.1112 | 57721 (0.6952) | 0 of 50 |
| l192 | 0.1063 +- 0.1874 | 0.1018 | 0.4907 | -0.2586 | 0.4551 +- 0.1083 | 84331 (0.6386) | 0 of 50 |
| l256 | 0.0629 +- 0.1336 | 0.0655 | 0.3024 | -0.2919 | 0.4282 +- 0.0755 | 98895 (0.5751) | 0 of 50 |

## train_t0500_lap_pred_t10000

Trained to t = 500 (Laplacian output, modes 20, 17.9M), rolled out from t = 100 to t = 10000; checkpoint `da30a2ce3495`.

| case | R2(t = 10000) | median | max | min | trajectory-mean R2 | best seed (mean R2) | diverged |
|---|---|---|---|---|---|---|---|
| psi0m0.40 | 0.6588 +- 0.1970 | 0.6922 | 0.9304 | 0.0554 | 0.8049 +- 0.1193 | 98895 (0.9435) | 0 of 50 |
| psi0m0.20 | 0.6797 +- 0.1978 | 0.7414 | 0.9272 | 0.0613 | 0.8234 +- 0.1322 | 99999 (0.9686) | 0 of 50 |
| psi0p0.00 | 0.8279 +- 0.1185 | 0.8644 | 0.9475 | 0.4247 | 0.9384 +- 0.0636 | 48417 (0.9840) | 0 of 50 |
| psi0p0.20 | 0.6691 +- 0.2580 | 0.7591 | 0.9402 | -0.0811 | 0.8388 +- 0.1402 | 92280 (0.9741) | 0 of 50 |
| psi0p0.40 | 0.6882 +- 0.2065 | 0.7825 | 0.9297 | 0.1665 | 0.8341 +- 0.1023 | 35064 (0.9647) | 0 of 50 |
| l160 | 0.8204 +- 0.0902 | 0.8425 | 0.9426 | 0.4991 | 0.9316 +- 0.0502 | 34617 (0.9784) | 0 of 50 |
| l192 | 0.7640 +- 0.1156 | 0.7990 | 0.8819 | 0.2946 | 0.9144 +- 0.0653 | 76214 (0.9671) | 0 of 50 |
| l256 | 0.7898 +- 0.0674 | 0.8012 | 0.8913 | 0.5794 | 0.9198 +- 0.0392 | 31415 (0.9658) | 0 of 50 |

## train_t1000_pred_t10000

Trained to t = 1000 (base, modes 20, 17.9M), rolled out from t = 100 to t = 10000; checkpoint `d25e45559e9e`.

| case | R2(t = 10000) | median | max | min | trajectory-mean R2 | best seed (mean R2) | diverged |
|---|---|---|---|---|---|---|---|
| psi0m0.40 | -0.5367 +- 0.2839 | -0.5170 | 0.0683 | -1.0953 | 0.2886 +- 0.1584 | 31458 (0.6464) | 0 of 50 |
| psi0m0.20 | -0.5926 +- 0.2428 | -0.6040 | 0.1290 | -1.0901 | 0.1742 +- 0.1757 | 27924 (0.6641) | 0 of 50 |
| psi0p0.00 | 0.0100 +- 0.2564 | 0.0305 | 0.5334 | -0.6721 | 0.4471 +- 0.1390 | 48417 (0.7538) | 0 of 50 |
| psi0p0.20 | -0.5147 +- 0.2186 | -0.5161 | 0.0384 | -0.9058 | 0.2364 +- 0.1551 | 29935 (0.6206) | 0 of 50 |
| psi0p0.40 | -0.6543 +- 0.2716 | -0.7158 | -0.1825 | -1.1772 | 0.2323 +- 0.1558 | 28096 (0.4979) | 0 of 50 |
| l160 | -0.0034 +- 0.1788 | 0.0096 | 0.3536 | -0.4865 | 0.4386 +- 0.1172 | 84331 (0.6479) | 0 of 50 |
| l192 | -0.0363 +- 0.1794 | -0.0440 | 0.3291 | -0.4957 | 0.4052 +- 0.1069 | 43125 (0.5839) | 0 of 50 |
| l256 | -0.0867 +- 0.1445 | -0.0704 | 0.1673 | -0.4807 | 0.3760 +- 0.0802 | 29935 (0.5524) | 0 of 50 |

## train_t1000_lap_pred_t10000

Trained to t = 1000 (Laplacian output, modes 20, 17.9M), rolled out from t = 100 to t = 10000; checkpoint `7a8a86a3a5ff`.

| case | R2(t = 10000) | median | max | min | trajectory-mean R2 | best seed (mean R2) | diverged |
|---|---|---|---|---|---|---|---|
| psi0m0.40 | 0.7629 +- 0.2075 | 0.8352 | 0.9603 | -0.1286 | 0.8683 +- 0.1041 | 58713 (0.9720) | 0 of 50 |
| psi0m0.20 | 0.7949 +- 0.1896 | 0.8688 | 0.9794 | 0.0952 | 0.8902 +- 0.1018 | 70389 (0.9819) | 0 of 50 |
| psi0p0.00 | 0.9033 +- 0.0879 | 0.9294 | 0.9813 | 0.5275 | 0.9613 +- 0.0458 | 48417 (0.9904) | 0 of 50 |
| psi0p0.20 | 0.7063 +- 0.2106 | 0.7801 | 0.9633 | 0.2056 | 0.8501 +- 0.1169 | 70389 (0.9813) | 0 of 50 |
| psi0p0.40 | 0.7911 +- 0.1559 | 0.8583 | 0.9806 | 0.4242 | 0.8784 +- 0.0801 | 99999 (0.9717) | 0 of 50 |
| l160 | 0.8209 +- 0.1185 | 0.8540 | 0.9378 | 0.3426 | 0.9297 +- 0.0629 | 34617 (0.9770) | 0 of 50 |
| l192 | 0.8329 +- 0.1018 | 0.8732 | 0.9316 | 0.4682 | 0.9305 +- 0.0627 | 34617 (0.9792) | 0 of 50 |
| l256 | 0.8325 +- 0.0653 | 0.8475 | 0.9195 | 0.6520 | 0.9323 +- 0.0383 | 88080 (0.9725) | 0 of 50 |

## train_t1000_lap_m24_pred_t10000

Trained to t = 1000 (Laplacian output, modes 24, 25.1M), rolled out from t = 100 to t = 10000; checkpoint `dbec368d5f28`.

| case | R2(t = 10000) | median | max | min | trajectory-mean R2 | best seed (mean R2) | diverged |
|---|---|---|---|---|---|---|---|
| psi0m0.40 | 0.7655 +- 0.1789 | 0.8227 | 0.9582 | 0.2425 | 0.8537 +- 0.1031 | 98895 (0.9617) | 0 of 50 |
| psi0m0.20 | 0.7536 +- 0.2082 | 0.8311 | 0.9824 | 0.0816 | 0.8556 +- 0.1142 | 86420 (0.9719) | 0 of 50 |
| psi0p0.00 | 0.8953 +- 0.1031 | 0.9253 | 0.9700 | 0.4121 | 0.9434 +- 0.0610 | 58713 (0.9763) | 0 of 50 |
| psi0p0.20 | 0.7190 +- 0.2186 | 0.7778 | 0.9598 | 0.0681 | 0.8480 +- 0.1199 | 70389 (0.9745) | 0 of 50 |
| psi0p0.40 | 0.7681 +- 0.1799 | 0.8259 | 0.9620 | 0.2499 | 0.8655 +- 0.0871 | 43125 (0.9581) | 0 of 50 |
| l160 | 0.8333 +- 0.1000 | 0.8615 | 0.9523 | 0.4666 | 0.9218 +- 0.0636 | 94245 (0.9764) | 0 of 50 |
| l192 | 0.8222 +- 0.1146 | 0.8687 | 0.9234 | 0.3596 | 0.9178 +- 0.0687 | 57721 (0.9709) | 0 of 50 |
| l256 | 0.8425 +- 0.0575 | 0.8482 | 0.9161 | 0.6599 | 0.9264 +- 0.0355 | 67517 (0.9663) | 0 of 50 |

## train_t1000_lap_w24_pred_t10000

Trained to t = 1000 (Laplacian output, width 24, modes 20, 6.5M), rolled out from t = 100 to t = 10000; checkpoint `599ca19cc20b`.

| case | R2(t = 10000) | median | max | min | trajectory-mean R2 | best seed (mean R2) | diverged |
|---|---|---|---|---|---|---|---|
| psi0m0.40 | 0.7335 +- 0.2039 | 0.7486 | 0.9743 | 0.0797 | 0.8636 +- 0.1135 | 48417 (0.9761) | 0 of 50 |
| psi0m0.20 | 0.7633 +- 0.1809 | 0.8155 | 0.9828 | 0.2820 | 0.8823 +- 0.1066 | 33052 (0.9780) | 0 of 50 |
| psi0p0.00 | 0.9319 +- 0.1213 | 0.9750 | 0.9934 | 0.4301 | 0.9642 +- 0.0730 | 57721 (0.9935) | 0 of 50 |
| psi0p0.20 | 0.7104 +- 0.2437 | 0.7666 | 0.9761 | -0.0195 | 0.8407 +- 0.1402 | 85507 (0.9760) | 0 of 50 |
| psi0p0.40 | 0.7542 +- 0.1911 | 0.8261 | 0.9710 | 0.1345 | 0.8735 +- 0.0871 | 88971 (0.9630) | 0 of 50 |
| l160 | 0.9089 +- 0.1059 | 0.9592 | 0.9882 | 0.5226 | 0.9530 +- 0.0595 | 92280 (0.9943) | 0 of 50 |
| l192 | 0.8956 +- 0.1168 | 0.9524 | 0.9901 | 0.5096 | 0.9450 +- 0.0627 | 67517 (0.9942) | 0 of 50 |
| l256 | 0.9279 +- 0.0564 | 0.9528 | 0.9845 | 0.7272 | 0.9624 +- 0.0303 | 98895 (0.9907) | 0 of 50 |

## train_t1000_lap_w24m8_pred_t10000

Trained to t = 1000 (Laplacian output, width 24, modes 8, 1.7M), rolled out from t = 100 to t = 10000; checkpoint `d188df84478c`.

| case | R2(t = 10000) | median | max | min | trajectory-mean R2 | best seed (mean R2) | diverged |
|---|---|---|---|---|---|---|---|
| psi0m0.40 | 0.7678 +- 0.1668 | 0.8223 | 0.9669 | 0.1532 | 0.8766 +- 0.0912 | 77086 (0.9630) | 0 of 50 |
| psi0m0.20 | 0.7516 +- 0.2248 | 0.8043 | 0.9728 | -0.0126 | 0.8677 +- 0.1389 | 39748 (0.9827) | 0 of 50 |
| psi0p0.00 | 0.9341 +- 0.1100 | 0.9811 | 0.9905 | 0.4858 | 0.9625 +- 0.0700 | 36963 (0.9947) | 0 of 50 |
| psi0p0.20 | 0.7220 +- 0.2411 | 0.8125 | 0.9799 | -0.1778 | 0.8570 +- 0.1386 | 60079 (0.9817) | 0 of 50 |
| psi0p0.40 | 0.7794 +- 0.1770 | 0.8258 | 0.9879 | 0.2921 | 0.8829 +- 0.0862 | 33052 (0.9784) | 0 of 50 |
| l160 | 0.8681 +- 0.1388 | 0.9381 | 0.9807 | 0.2849 | 0.9275 +- 0.0820 | 34617 (0.9889) | 0 of 50 |
| l192 | 0.8851 +- 0.1018 | 0.9407 | 0.9837 | 0.6448 | 0.9385 +- 0.0582 | 31458 (0.9919) | 0 of 50 |
| l256 | 0.8877 +- 0.0762 | 0.9082 | 0.9698 | 0.5728 | 0.9368 +- 0.0462 | 29935 (0.9883) | 0 of 50 |

## train_t1000_lap_w32_pred_t10000

Trained to t = 1000 (Laplacian output, width 32, modes 20, 11.5M), rolled out from t = 100 to t = 10000; checkpoint `c230b5d44fe3`.

| case | R2(t = 10000) | median | max | min | trajectory-mean R2 | best seed (mean R2) | diverged |
|---|---|---|---|---|---|---|---|
| psi0m0.40 | 0.8049 +- 0.1537 | 0.8582 | 0.9687 | 0.2460 | 0.8871 +- 0.0899 | 99999 (0.9740) | 0 of 50 |
| psi0m0.20 | 0.7239 +- 0.2260 | 0.7686 | 0.9732 | -0.1615 | 0.8475 +- 0.1427 | 33052 (0.9774) | 0 of 50 |
| psi0p0.00 | 0.9327 +- 0.1313 | 0.9864 | 0.9936 | 0.4646 | 0.9628 +- 0.0756 | 58713 (0.9953) | 0 of 50 |
| psi0p0.20 | 0.7473 +- 0.2280 | 0.8492 | 0.9710 | 0.0495 | 0.8678 +- 0.1289 | 33052 (0.9776) | 0 of 50 |
| psi0p0.40 | 0.7645 +- 0.1876 | 0.7949 | 0.9828 | 0.0790 | 0.8734 +- 0.0926 | 42192 (0.9733) | 0 of 50 |
| l160 | 0.8923 +- 0.1216 | 0.9605 | 0.9847 | 0.4823 | 0.9467 +- 0.0676 | 34617 (0.9936) | 0 of 50 |
| l192 | 0.9124 +- 0.0961 | 0.9663 | 0.9873 | 0.6007 | 0.9532 +- 0.0673 | 29935 (0.9937) | 0 of 50 |
| l256 | 0.9184 +- 0.0580 | 0.9345 | 0.9814 | 0.7600 | 0.9588 +- 0.0318 | 66260 (0.9924) | 0 of 50 |

## train_t1000_lap_w48_pred_t10000

Trained to t = 1000 (Laplacian output, width 48, modes 20, 25.8M), rolled out from t = 100 to t = 10000; checkpoint `a45df0036465`.

| case | R2(t = 10000) | median | max | min | trajectory-mean R2 | best seed (mean R2) | diverged |
|---|---|---|---|---|---|---|---|
| psi0m0.40 | 0.8075 +- 0.1866 | 0.8651 | 0.9805 | -0.0166 | 0.9000 +- 0.0997 | 58713 (0.9818) | 0 of 50 |
| psi0m0.20 | 0.6923 +- 0.2303 | 0.7464 | 0.9631 | -0.0304 | 0.8461 +- 0.1444 | 63866 (0.9783) | 0 of 50 |
| psi0p0.00 | 0.8559 +- 0.0960 | 0.8793 | 0.9853 | 0.5210 | 0.9582 +- 0.0495 | 32393 (0.9947) | 0 of 50 |
| psi0p0.20 | 0.7163 +- 0.2034 | 0.7650 | 0.9747 | 0.0446 | 0.8751 +- 0.1151 | 54321 (0.9742) | 0 of 50 |
| psi0p0.40 | 0.7481 +- 0.1932 | 0.8024 | 0.9745 | 0.1699 | 0.8805 +- 0.0980 | 33052 (0.9756) | 0 of 50 |
| l160 | 0.7758 +- 0.0891 | 0.7978 | 0.9133 | 0.5474 | 0.9136 +- 0.0455 | 34617 (0.9673) | 0 of 50 |
| l192 | 0.7700 +- 0.0806 | 0.7943 | 0.9042 | 0.5037 | 0.9215 +- 0.0510 | 66260 (0.9718) | 0 of 50 |
| l256 | 0.7783 +- 0.0624 | 0.7896 | 0.8611 | 0.5987 | 0.9221 +- 0.0326 | 39748 (0.9559) | 0 of 50 |

## What each run folder holds

Cases: `psi0m0.40` ... `psi0p0.40` are the five compositions on the
128 x 128 test box; `l160`, `l192`, `l256` are the critical quench on
larger boxes with the critical quench's own stride schedule. In the
box-size plots `l128` is the critical composition `psi0p0.00`.

Best seed (one seed per case, the highest trajectory-mean R2, i.e.
the mean of R2 over every rollout time):

- `snapshots_<case>.png` - ground truth, prediction and absolute error
  at the main snapshot times (100, 500, 1000, 2000 for a new run); the
  title names the best, median and worst seed with their mean R2
- `snapshots_long_<case>.png` - the same rows at the long snapshots,
  every 2000 up to t_end (2000, 4000, ..., 10000); absent for older
  caches that did not store them

Seed averages (every test seed):

- `r2_sweep.png` / `.pdf` - mean R2(t) +- 1 sd, (a) by composition,
  (b) by box size; dashed: threshold, dotted: training horizon
- `phase_ordering_kinetics_<case>.png` - Puri ch. 1: (a) zero-crossing
  L(t) +- 1 sd with fitted exponents, (b) S(k,t), (c) scaling collapse,
  (d) C(r,t) vs r/L, (e) free energy, (f) <psi>(t) and the A/B
  composition
- `composition_<case>.png` - seed-mean <psi>(t), truth vs prediction
  +- 1 sd, with the A / B percent axis and the A / B split at
  t_start and t_end; `composition_all.png` - c_A %(t) of the five
  compositions on one axis; `composition.csv` - the same numbers at the
  snapshot times
- `domain_growth_<case>.png` - Puri's half-maximum L(t) +- 1 sd against
  the Lifshitz-Slyozov t^(1/3) law
- `psi_distribution_<case>.png` - P(psi) at t_end, all seeds pooled

Plus `report.txt` (model card, A/B composition and its conservation,
solve time, summary table, low-k increment response) and `manifest.json` (every number the
figures show, the seeds, schedules and the best / median / worst
seed ids). Growth exponents are fitted over [t_end/4, t_end] and
over the training window [t_start, train_t_end].

`train_t0300_pred_t02000/earlier_run_2026-09-08/` holds graphs kept from the
Sep 8 run (10 seeds, single-seed physics panels, box curves on the
+0.4 schedule); see its NOTE.md. The drawing code never touches it.

## Finite-size regime

Time axes are shaded where the seed-mean zero-crossing length exceeds
l/4, i.e. fewer than 4 domains across the box. From there on the
box, not the physics, limits coarsening: l = 128 enters it around
t ~ 3400. Nothing is dropped; the larger boxes show the trend. The
manifest records the first such time per case.

## Regenerate

    python make_submission.py                  redraw every complete run
    python make_submission.py --run LABEL      redraw one run
    python uno_predict.py --end N              tune, roll out, diagnose, draw
    python make_submission.py --rollout --end N   refill a rollout cache
    (add --variant base for the old architecture; default is lap)

## Where everything else lives

- `Work/rollout_cache/<run>/` - per-seed rollout cache the figures are
  drawn from (`<case>/seed<N>.npz` + `meta.json`)
- `Work/predict/<run>/` - `results.json` and `predict.log`
- `Work/models/train_t<END>[_lap][_<size>]/` - checkpoints, training log and memmaps
- `Work/gt_cache/` - the ground-truth cache shared by every run
- `Work/data/` - datasets written by `ch_gpu.py`
- `Archive/` - earlier Results/ and Submission/ folders, intact
