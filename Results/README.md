# Results - Cahn-Hilliard UNO

Report material only. Every file here is regenerated from the rollout
caches in `Work/rollout_cache/` by `python make_submission.py`; no model is
loaded and no trajectory is re-solved. One folder per run, named
`train_t<END>[_lap]_pred_t<T>`: a model trained on data to t = END,
rolled out to t = T. `_lap` marks the Laplacian-output model (the head predicts
M and the increment is its 5-point Laplacian); runs without it are the base
model (direct increment) and are kept for comparison.

## Runs

| run | model | trained to | predicted to | seeds | critical R2(t_end) | folder |
|---|---|---|---|---|---|---|
| train_t0300_pred_t10000 | base | t = 300 | t = 10000 | 50 | R2(10000) = 0.6229 +- 0.2201 | `train_t0300_pred_t10000/` |
| train_t0300_lap_pred_t10000 | Laplacian output | t = 300 | t = 10000 | 50 | R2(10000) = 0.7702 +- 0.1235 | `train_t0300_lap_pred_t10000/` |
| train_t0500_pred_t10000 | base | t = 500 | t = 10000 | 50 | R2(10000) = 0.1759 +- 0.2574 | `train_t0500_pred_t10000/` |
| train_t0500_lap_pred_t10000 | Laplacian output | t = 500 | t = 10000 | 50 | R2(10000) = 0.8279 +- 0.1185 | `train_t0500_lap_pred_t10000/` |
| train_t1000_pred_t10000 | base | t = 1000 | t = 10000 | 50 | R2(10000) = 0.0100 +- 0.2564 | `train_t1000_pred_t10000/` |
| train_t1000_lap_pred_t10000 | Laplacian output | t = 1000 | t = 10000 | 50 | R2(10000) = 0.9033 +- 0.0879 | `train_t1000_lap_pred_t10000/` |

R2(t_end) is the mean over seeds of each seed's R2 against the ground
truth at the end of that run's rollout, +- one standard deviation
across seeds. Runs with different horizons are not comparable at
their end points; `comparison.png` puts the critical curves of every
run on one log-t axis instead. In the per-run tables, max and min
are the extreme seeds' R2(t_end); the best seed is chosen by
trajectory-mean R2, so it need not be the one with the max.

## train_t0300_pred_t10000

Trained to t = 300 (base), rolled out from t = 100 to t = 10000; checkpoint `ccecc0f076d7`.

| case | R2(t = 10000) | median | max | min | trajectory-mean R2 | best seed (mean R2) |
|---|---|---|---|---|---|---|
| psi0m0.40 | 0.2073 +- 0.2815 | 0.2080 | 0.7529 | -0.7526 | 0.5065 +- 0.1476 | 54464 (0.7638) |
| psi0m0.20 | 0.1841 +- 0.3190 | 0.2086 | 0.7820 | -0.5405 | 0.4973 +- 0.1879 | 61803 (0.7949) |
| psi0p0.00 | 0.6229 +- 0.2201 | 0.7130 | 0.9068 | 0.0343 | 0.7122 +- 0.1194 | 36963 (0.8941) |
| psi0p0.20 | 0.2094 +- 0.2837 | 0.2723 | 0.6391 | -0.9175 | 0.5147 +- 0.1666 | 92280 (0.7269) |
| psi0p0.40 | 0.1127 +- 0.2972 | 0.0954 | 0.7713 | -0.7554 | 0.4902 +- 0.1388 | 99999 (0.8093) |
| l160 | 0.5719 +- 0.1739 | 0.6292 | 0.8534 | 0.0798 | 0.6979 +- 0.0991 | 22911 (0.8257) |
| l192 | 0.4997 +- 0.1775 | 0.5341 | 0.7830 | 0.0866 | 0.6511 +- 0.0984 | 51170 (0.8102) |
| l256 | 0.4714 +- 0.0932 | 0.4813 | 0.6730 | 0.2010 | 0.6371 +- 0.0515 | 57721 (0.7498) |

## train_t0300_lap_pred_t10000

Trained to t = 300 (Laplacian output), rolled out from t = 100 to t = 10000; checkpoint `70e255ec2c21`.

| case | R2(t = 10000) | median | max | min | trajectory-mean R2 | best seed (mean R2) |
|---|---|---|---|---|---|---|
| psi0m0.40 | 0.4894 +- 0.2313 | 0.5357 | 0.7882 | -0.1897 | 0.7086 +- 0.1350 | 48417 (0.8846) |
| psi0m0.20 | 0.5198 +- 0.2037 | 0.5561 | 0.8232 | 0.0847 | 0.7541 +- 0.1119 | 99999 (0.9095) |
| psi0p0.00 | 0.7702 +- 0.1235 | 0.8053 | 0.9348 | 0.4475 | 0.8819 +- 0.0732 | 85507 (0.9666) |
| psi0p0.20 | 0.5142 +- 0.2169 | 0.5677 | 0.8667 | -0.0268 | 0.7507 +- 0.1166 | 33052 (0.9143) |
| psi0p0.40 | 0.5312 +- 0.2289 | 0.5839 | 0.9200 | -0.0500 | 0.7456 +- 0.1129 | 35064 (0.9074) |
| l160 | 0.8155 +- 0.1277 | 0.8531 | 0.9472 | 0.3428 | 0.9044 +- 0.0649 | 34617 (0.9627) |
| l192 | 0.7646 +- 0.1314 | 0.8125 | 0.8953 | 0.2761 | 0.8838 +- 0.0710 | 28096 (0.9526) |
| l256 | 0.8076 +- 0.0518 | 0.8089 | 0.8911 | 0.6325 | 0.9039 +- 0.0286 | 31458 (0.9431) |

## train_t0500_pred_t10000

Trained to t = 500 (base), rolled out from t = 100 to t = 10000; checkpoint `0b3842c641fb`.

| case | R2(t = 10000) | median | max | min | trajectory-mean R2 | best seed (mean R2) |
|---|---|---|---|---|---|---|
| psi0m0.40 | -0.1522 +- 0.3170 | -0.1451 | 0.4477 | -1.1446 | 0.4488 +- 0.1496 | 31458 (0.7740) |
| psi0m0.20 | -0.4114 +- 0.2659 | -0.4010 | 0.2571 | -0.9802 | 0.2979 +- 0.1687 | 68620 (0.6938) |
| psi0p0.00 | 0.1759 +- 0.2574 | 0.1775 | 0.6619 | -0.5315 | 0.4995 +- 0.1426 | 48417 (0.7931) |
| psi0p0.20 | -0.3787 +- 0.2474 | -0.3746 | 0.2286 | -0.9106 | 0.2752 +- 0.1733 | 52066 (0.6732) |
| psi0p0.40 | -0.4941 +- 0.3616 | -0.5309 | 0.1845 | -1.1687 | 0.3033 +- 0.1758 | 99999 (0.6381) |
| l160 | 0.1371 +- 0.1798 | 0.1670 | 0.4741 | -0.2500 | 0.4808 +- 0.1112 | 57721 (0.6952) |
| l192 | 0.1063 +- 0.1874 | 0.1018 | 0.4907 | -0.2586 | 0.4551 +- 0.1083 | 84331 (0.6386) |
| l256 | 0.0629 +- 0.1336 | 0.0655 | 0.3024 | -0.2919 | 0.4282 +- 0.0755 | 98895 (0.5751) |

## train_t0500_lap_pred_t10000

Trained to t = 500 (Laplacian output), rolled out from t = 100 to t = 10000; checkpoint `da30a2ce3495`.

| case | R2(t = 10000) | median | max | min | trajectory-mean R2 | best seed (mean R2) |
|---|---|---|---|---|---|---|
| psi0m0.40 | 0.6588 +- 0.1970 | 0.6922 | 0.9304 | 0.0554 | 0.8049 +- 0.1193 | 98895 (0.9435) |
| psi0m0.20 | 0.6797 +- 0.1978 | 0.7414 | 0.9272 | 0.0613 | 0.8234 +- 0.1322 | 99999 (0.9686) |
| psi0p0.00 | 0.8279 +- 0.1185 | 0.8644 | 0.9475 | 0.4247 | 0.9384 +- 0.0636 | 48417 (0.9840) |
| psi0p0.20 | 0.6691 +- 0.2580 | 0.7591 | 0.9402 | -0.0811 | 0.8388 +- 0.1402 | 92280 (0.9741) |
| psi0p0.40 | 0.6882 +- 0.2065 | 0.7825 | 0.9297 | 0.1665 | 0.8341 +- 0.1023 | 35064 (0.9647) |
| l160 | 0.8204 +- 0.0902 | 0.8425 | 0.9426 | 0.4991 | 0.9316 +- 0.0502 | 34617 (0.9784) |
| l192 | 0.7640 +- 0.1156 | 0.7990 | 0.8819 | 0.2946 | 0.9144 +- 0.0653 | 76214 (0.9671) |
| l256 | 0.7898 +- 0.0674 | 0.8012 | 0.8913 | 0.5794 | 0.9198 +- 0.0392 | 31415 (0.9658) |

## train_t1000_pred_t10000

Trained to t = 1000 (base), rolled out from t = 100 to t = 10000; checkpoint `d25e45559e9e`.

| case | R2(t = 10000) | median | max | min | trajectory-mean R2 | best seed (mean R2) |
|---|---|---|---|---|---|---|
| psi0m0.40 | -0.5367 +- 0.2839 | -0.5170 | 0.0683 | -1.0953 | 0.2886 +- 0.1584 | 31458 (0.6464) |
| psi0m0.20 | -0.5926 +- 0.2428 | -0.6040 | 0.1290 | -1.0901 | 0.1742 +- 0.1757 | 27924 (0.6641) |
| psi0p0.00 | 0.0100 +- 0.2564 | 0.0305 | 0.5334 | -0.6721 | 0.4471 +- 0.1390 | 48417 (0.7538) |
| psi0p0.20 | -0.5147 +- 0.2186 | -0.5161 | 0.0384 | -0.9058 | 0.2364 +- 0.1551 | 29935 (0.6206) |
| psi0p0.40 | -0.6543 +- 0.2716 | -0.7158 | -0.1825 | -1.1772 | 0.2323 +- 0.1558 | 28096 (0.4979) |
| l160 | -0.0034 +- 0.1788 | 0.0096 | 0.3536 | -0.4865 | 0.4386 +- 0.1172 | 84331 (0.6479) |
| l192 | -0.0363 +- 0.1794 | -0.0440 | 0.3291 | -0.4957 | 0.4052 +- 0.1069 | 43125 (0.5839) |
| l256 | -0.0867 +- 0.1445 | -0.0704 | 0.1673 | -0.4807 | 0.3760 +- 0.0802 | 29935 (0.5524) |

## train_t1000_lap_pred_t10000

Trained to t = 1000 (Laplacian output), rolled out from t = 100 to t = 10000; checkpoint `7a8a86a3a5ff`.

| case | R2(t = 10000) | median | max | min | trajectory-mean R2 | best seed (mean R2) |
|---|---|---|---|---|---|---|
| psi0m0.40 | 0.7629 +- 0.2075 | 0.8352 | 0.9603 | -0.1286 | 0.8683 +- 0.1041 | 58713 (0.9720) |
| psi0m0.20 | 0.7949 +- 0.1896 | 0.8688 | 0.9794 | 0.0952 | 0.8902 +- 0.1018 | 70389 (0.9819) |
| psi0p0.00 | 0.9033 +- 0.0879 | 0.9294 | 0.9813 | 0.5275 | 0.9613 +- 0.0458 | 48417 (0.9904) |
| psi0p0.20 | 0.7063 +- 0.2106 | 0.7801 | 0.9633 | 0.2056 | 0.8501 +- 0.1169 | 70389 (0.9813) |
| psi0p0.40 | 0.7911 +- 0.1559 | 0.8583 | 0.9806 | 0.4242 | 0.8784 +- 0.0801 | 99999 (0.9717) |
| l160 | 0.8209 +- 0.1185 | 0.8540 | 0.9378 | 0.3426 | 0.9297 +- 0.0629 | 34617 (0.9770) |
| l192 | 0.8329 +- 0.1018 | 0.8732 | 0.9316 | 0.4682 | 0.9305 +- 0.0627 | 34617 (0.9792) |
| l256 | 0.8325 +- 0.0653 | 0.8475 | 0.9195 | 0.6520 | 0.9323 +- 0.0383 | 88080 (0.9725) |

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
- `Work/models/train_t<END>[_lap]/` - checkpoints, training log and memmaps
- `Work/gt_cache/` - the ground-truth cache shared by every run
- `Work/data/` - datasets written by `ch_gpu.py`
- `Archive/` - earlier Results/ and Submission/ folders, intact
