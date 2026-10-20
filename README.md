# UNO for the Cahn–Hilliard equation — trained on l = 64 to END ≤ 1000, autoregressive rollout to t = 10000 on l = 128–256

Neural-operator surrogate for the dimensionless Cahn–Hilliard–Cook equation

$$\partial_t\psi = \nabla^2\left(\psi^3 - \psi - \nabla^2\psi\right)$$

(Puri, *Kinetics of Phase Transitions*, Eq. 1.182 — exactly what `ch.py` integrates).

Protocol, per the project brief: train on the dataset `ch.py` produces (64×64 boxes,
data up to t = END), then predict **unseen** trajectories on a **128×128** lattice,
starting at **t = 100** and rolling autoregressively in real time, scored by **R²**
plus snapshot comparison. The brief's horizon is t = 2000; the current runs go five
times further, to **t = 10000**, and also on the larger boxes l = 160, 192 and 256 at
the same lattice spacing. The models are trained to END = 300, 500 or 1000 and have a
Laplacian-output head (*Current status*, next). The rollout runs the full horizon; the
**0.80** threshold is reported (the time R² first crosses it, per seed and for the
mean) rather than used to truncate.

Sections 1 and 2 describe the method and are current. Sections 3 to 5 report the
September run (base model, END = 300, rollout to t = 2000), the ablation and the
diagnostics around it, and are marked *Historical* wherever its numbers are
superseded. Section 6 is the reference for the
files, labels, options and job scripts. The numbers of 2026-10-07 to 2026-10-18 are
collected in `Work/analysis/findings_2026-10-15.md` (its closing addenda hold the
2026-10-16 re-predictions, the w12 m12 run and the w24 m8 seed replicates, and the
2026-10-18 removal of runs from the drawn results and the new snapshot colours).

---

## Current status (2026-10-18)

Unless stated otherwise, every number in this section is from a model trained on
l = 64 boxes to t = END with the Laplacian output, rolled out from t = 100 to
**t = 10000** on unseen boxes and scored by R² against the exact solver over 50 test
seeds per case. The 8 cases are ψ₀ = −0.4, −0.2, +0.2, +0.4 and the critical quench
ψ₀ = 0, all at l = 128, and the critical quench at l = 160, 192 and 256. *Pooled*
means the mean over all 8 cases × 50 seeds (400 rollouts), *off-critical* the four
l = 128 compositions, *boxes* the three larger boxes. The test seeds are disjoint from
the 10 validation / tuning seeds and from the l = 64 training-data seeds (checked in
the rollout caches, the `meta.json` files and `ch_gpu.py plan()`).

**What changed on 2026-10-18.** The w24 m8 seed replicates are in (below): over three
seeds it is stable and matches the default at long times. Seven runs are no longer
drawn, and their numbers stay in the tables below: w24 m12, w16, w12, w8, w12 m12 and
the two w24 m8 seed replicates (seeds 1 and 2). Their `Results/` folders and their
rollout caches, predictions and models were moved, not deleted, to
`Archive/removed_runs_2026-10-18/` (4.1 GB; deleting that folder frees the space), and
their job logs stay in the project root. `Results/` holds 11 runs: END = 300, 500 and
1000, base head and Laplacian output, plus the END = 1000 sizes default, w48, m24, w32,
w24 and w24 m8 (seed 0). The snapshot figures now use the colours of the presentation
deck (§6, *Snapshot colours*).

### The Laplacian output fixed the long-END runs

Training to END = 1000 with the original head failed in two stacked ways. The first
model picked epoch 12 by a late-heavy checkpoint score (a damped, under-trained model,
validation survival about 416–424); it was retired and archived. Separately, every base
model moved the lowest wavenumbers 2.4–3.5× too far per step when it went from l = 64 to
l = 128 (the box has modes below the training box's fundamental). Retrained with
survival-based selection (epoch 16, validation survival 1852), the base head still
scores R²(10000) = 0.010 on the critical quench. The head now predicts M and the
increment is its 5-point Laplacian, the stencil the solver uses, so mass is conserved
exactly for any Δt and the lowest wavenumbers are suppressed by k². Every run in the
table, base head included (END = 300, 500 and 1000), chooses its checkpoint by
validation survival (R² ≥ 0.8, ties broken on in-range R²), so the table isolates the
head. Critical-quench R²(10000), 50 seeds:

| END | Laplacian output | base head |
|---|---|---|
| 300 | 0.770 | 0.623 |
| 500 | 0.828 | 0.176 |
| 1000 | 0.903 | 0.010 |

The base head's R²(10000) fell as END grew; the Laplacian output's rose with END in
these three single-seed runs. This is the critical quench only, and the step from 0.770
to 0.828 (+0.058) is inside the 0.04–0.08 training-run SD (a pooled-R² figure, see
below); the margin over the base head is much larger. The idea is not new: Fantasia et
al. 2025 (arXiv:2507.21760) use the Laplacian of a scalar output field (their Eq. 11),
Lanzoni et al. 2024 (MLST) output a flux through a fixed divergence layer, in 3-D with
a CRNN, and Yadav et al. 2026 (arXiv:2606.26128) work in the same setting
(`Work/analysis/literature_laplacian_output_2026-10-08.md`). Details: §6,
*Laplacian output*; the failure analysis is in
`Work/analysis/t1000_failure_2026-10-06/`.

### Training-seed noise, and why size rankings need replicates

Two seed replicates were trained (seed 1 against the seed-0 run of the same size).
Pooled R²(10000):

| model | seed 0 | seed 1 | difference |
|---|---|---|---|
| default (17.9M) | 0.806 | 0.664 | −0.141 |
| w32 (11.5M) | 0.837 | 0.778 | −0.059 |

Seed 1 was worse in 8 of 8 cases in both pairs. The training-run SD of pooled
R²(10000) is about 0.04–0.08, against about 0.008 for the test-seed SE of one run's
pooled mean. The noise appears only after t ≈ 4000: at t ≤ 2000 the runs agree within
0.01–0.03, except where an under-trained checkpoint was picked. The paired ± of two runs
covers the 50 test seeds only, so it says nothing about which *training* was luckier, and
a single-seed difference of a few hundredths in R²(10000) cannot rank two sizes: w32
against the default, averaged over 2 seeds each, is +0.07 ± 0.08 (p ≈ 0.3–0.4), so no
size effect is detectable, and the earlier "smaller is better" result (+0.031 ± 0.007
against default seed 0) was within seed noise. Validation catches gross failures but
cannot resolve differences of 0.03–0.05. The three w24 m8 seeds (below) give a spread of
0.060 in pooled R²(10000), inside that range. The confidence interval, the mechanism,
the confounds (GPU node, nearly blind checkpoint selection, tuner grid edge) and the
confirmation runs: §6, *Training seed*.

### Model size: how small can the model be?

The advisor's question, answered with eleven sizes trained to END = 1000 (once each,
except w24 m8, which has three training seeds) and scored the same way at t = 10000.
`python3 Work/analysis/size_sweep.py` writes
`Work/analysis/size_sweep/size_sweep.{txt,csv,png}`. Five rows of the table (w16,
w24 m12, w12, w8 and w12 m12) are numbers only since 2026-10-18: those runs were removed
from the drawn results, and their folders are in `Archive/removed_runs_2026-10-18/`. The
full 13-run table of 2026-10-16 (the eleven sizes plus the two w24 m8 replicates) is
frozen in `Work/analysis/size_sweep/size_sweep_all_runs_2026-10-16.{txt,csv,png}`; the
files `size_sweep.*` are still identical to it, and the next run of the tool writes the
remaining runs only.

| run | params | R²(2000) | R²(5000) | R²(10000) | off-crit @10k | crit @10k | boxes @10k | verdict |
|---|---|---|---|---|---|---|---|---|
| w48 | 25.76M | 0.964 | 0.908 | 0.768 | 0.741 | 0.856 | 0.775 | good |
| m24 (w40) | 25.11M | 0.942 | 0.884 | 0.800 | 0.752 | 0.895 | 0.833 | epoch-24 pick |
| default w40 m20 | 17.90M | 0.960 | 0.904 | 0.806 | 0.764 | 0.903 | 0.829 | reference |
| w32 | 11.47M | 0.958 | 0.907 | 0.837 | 0.760 | 0.933 | 0.908 | good |
| w24 | 6.47M | 0.961 | 0.907 | 0.828 | 0.740 | 0.932 | 0.911 | good |
| w16 | 2.89M | 0.948 | 0.862 | 0.713 | 0.620 | 0.796 | 0.808 | borderline |
| w24 m12 | 2.86M | 0.950 | 0.863 | 0.684 | 0.654 | 0.802 | — | unstable, 147 seeds diverged |
| **w24 m8** | **1.68M** | **0.957** | **0.901** | **0.824** | **0.755** | **0.934** | **0.880** | **good** |
| w12 | 1.63M | 0.937 | 0.845 | 0.705 | 0.573 | 0.849 | 0.835 | worse, late overshoot |
| w8 | 0.74M | 0.772 | 0.555 | 0.291 | 0.064 | 0.544 | 0.499 | fails, 25 diverged |
| w12 m12 | 0.73M | 0.900 | 0.692 | 0.115 | 0.109 | 0.315 | — | fails, 236 diverged |

All columns are pooled R² unless named (*off-crit* and *crit* are the l = 128 groups,
*boxes* the three larger boxes). The verdict column is the reading of the sweep; the
tool prints automatic verdicts from fixed tolerances, and for m24, w16 and w12 those read
"fails early" because their R²(2000) is below the early line. The w24 m12, w8 and w12 m12
rows cover only the seeds that reached each time (the diverged seeds are counted in the
verdict column and left out of the means), so their R²(5000) and R²(10000) are survivor
means. w24 m12 and w8 were re-predicted on 2026-10-16 with the per-seed divergence
handling, so no seed of any of the three is cut short by a chunk-mate any more (the
sweep tool's `n_collat` is 0 for all three). The w24 m12 *boxes* cell is empty because
only 3 of its 150 big-box seeds reach t = 10000, and none of w12 m12's 150 do. The
w24 m8 row is seed 0; its replicates are below.

**w24 m8 (modes per level 8, 4, 2, 2) is as good as the default within noise.** For
seed 0, the paired differences (w24m8 − default) are −0.003 ± 0.002 at t = 2000,
−0.003 ± 0.005 at t = 5000 and +0.019 ± 0.008 at t = 10000 (test-seed errors only, see
the seed noise above). It has the best validation survival of any run (9491; its
in-range R² of 0.985 is 0.001 below w48's), no rollout diverges or overshoots
(max |ψ| ≤ 1.09), and it is 10.7× smaller than the default. Two more training seeds were
run on 2026-10-16 (jobs `utp_547746` on gpu2 and `utp_547747` on gpu1; their folders are
now in the archive, see above). Numbers from `size_sweep.txt` of that date, max |ψ| from
the snapshot caches:

| seed | node | epoch picked | val in-range R² | R²(2000) | R²(5000) | R²(10000) | off / crit / boxes @10k | diverged | max \|ψ\| |
|---|---|---|---|---|---|---|---|---|---|
| 0 | gpu2 | 38 | 0.985 | 0.957 | 0.901 | 0.824 | 0.755 / 0.934 / 0.880 | 0 | 1.09 |
| 1 | gpu2 | 28 | 0.975 | 0.944 | 0.880 | 0.751 | 0.704 / 0.856 / 0.779 | 0 | 1.08 |
| 2 | gpu1 | 22 | 0.942 | 0.910 | 0.837 | 0.706 | 0.650 / 0.811 / 0.746 | 0 | 1.06 |
| mean | | | | 0.937 | 0.873 | 0.761 (SD 0.060) | | 0 of 1200 | |

The default's two seeds give R²(2000) of 0.960 and 0.957 and R²(10000) of 0.806 and
0.664 (mean 0.735). Against the pass criteria (§6, *Training seed*): **stability passes
on all three seeds** (no divergence, no snapshot above |ψ| = 1.1); **late accuracy
passes** (every seed has R²(10000) ≥ 0.664, and the mean of 0.761 is in line with the
default's 0.735); **early accuracy (R²(2000) ≥ 0.95) passes only for seed 0** (seed 1,
0.944, is just below the line; seed 2, 0.910, fails), and the mean early gap to the
default is about −0.02. Seed 2's early shortfall coincides with an under-trained
checkpoint pick: its epoch-22 checkpoint had a validation in-range R² of 0.942, whereas
its epoch-60 EMA weights (still in `uno_ch_state.pt`) reached 0.979, and survival-based
selection preferred epoch 22 because two hard validation seeds survived longer there.
That is the same pattern as m24 and w32 seed 1 (both picked at epoch 24), so part of the
early gap is checkpoint selection, not capacity. The advisor paused selection changes on
2026-10-08, so the evidence should be raised with him. **Over three seeds, w24 m8
(1.68M, 10.7× smaller) is stable and matches the default at long times; at t ≤ 2000 it
is about 0.02 lower on average, and that gap depends on which epoch the selection
picks.** Width 24 (6.47M) remains the no-loss option.

**Cutting modes looked much cheaper than cutting width (provisional).** At about 1.7M
parameters w24m8 beats w12 by +0.163 ± 0.012 in R²(10000) on the l = 128 cases (a
test-seed error only, from two single-seed runs, against a training-run SD of
0.04–0.08), and at modes 20 the width degrades monotonically below 24. Quality is not
monotone in size, though: w16 (2.89M, 0.713) and w24 m12 (2.86M, 0.684 over its
survivors, with 147 of 150 big-box seeds diverged) are larger than w24m8 and worse. The
advice to keep the width at 24 or more and cut modes is still provisional: the +0.163 is
w24m8 seed 0 against one w12 run, and in pooled R²(10000) the three w24m8 seeds (0.824,
0.751, 0.706) reach down to w12's 0.705, so it should not be read as a margin of that
size for every training seed. w16, w12 and the other small widths have one seed each.

What goes wrong in the small models (numbers only: those runs are removed from the drawn
results, see above):

- Off-critical cases fail first. In the critical case and the big boxes the small nets
  under-coarsen (w16's late growth exponent at the critical quench is 0.288 against
  0.328 for the truth).
- w12 overshoots the amplitude late: 138 of 400 test rollouts exceed max |ψ| = 1.1,
  while the truth stays at or below 1.043. w8 degrades everywhere and overshoots
  systematically; after re-prediction 25 of its 400 rollouts diverge (13 at ψ₀ = −0.4,
  1 at −0.2, 2 at +0.2, 3 at l = 192, 6 at l = 256), and its pooled R² is
  0.772 / 0.555 / 0.291 at t = 2000 / 5000 / 10000 over the survivors.
- w12 m12 (0.73M, trained and predicted 2026-10-16 on gpu2; epoch 58, validation
  survival 5188, in-range R² 0.954) fails badly: 236 of 400 rollouts diverge, including
  all 150 big-box seeds and 45 of the 50 critical seeds, and its R² is 0.900 at t = 2000
  but 0.115 at t = 10000 over the survivors.
- w24 m12 has an abrupt high-k runaway in its big boxes. With per-seed divergence
  handling, 147 of its 150 big-box seeds diverge sooner or later: 47 of 50 at l = 160
  (t ≈ 5100–9700), 50 of 50 at l = 192 (t ≈ 4100–7700) and 50 of 50 at l = 256
  (t ≈ 3300–5000). The l = 128 cases have 0 diverged and give a pooled R²(10000) of
  0.684 over the survivors (critical quench 0.802). Before the divergence fix 51 seeds
  were counted as diverged and the other 99 big-box seeds as cut short with a chunk-mate;
  most of those 99 blow up later anyway, so the old truncation only hid how widespread
  the failure is. This checkpoint is unstable in boxes larger than 128. Whether that is
  systematic for modes 12 or specific to this one training run would need the seed
  replicate (w24 m12 seed 1), which is optional now and not run.
- More epochs would not help: the small nets plateau by epoch about 40 and are already
  behind at epoch 24.

**The sweep is complete.** With w12 m12 the table has eleven runs, six of them at or
below 2.9M parameters (w16, w24 m12, w24 m8, w12, w8 and w12 m12). Nothing under 1M
parameters works (w8, w12 m12). Below 2.9M only w24 m8 (1.68M) is both accurate and
stable. Stability is not monotone in size: w24 m12 (2.86M) is unstable in the big boxes
while the smaller w24 m8 is not, so stability in boxes larger than 128 is a pass
criterion of its own, to be checked next to R². All of this is on one training seed per
size, except w24 m8, which has three.

The w24 m8 confirmation seeds are done (above). A w24 m12 seed 1 (to see whether its
instability is systematic) and a w24 seed 1 were planned and are optional now (§6,
*Training seed*). Options and measured costs: §6, *Model size*.

### Time cost

`python3 Work/analysis/timing_table.py` writes `Work/analysis/timing/timing.txt`,
`timing_runs.csv` and `timing_cases.csv` from the job logs. Both HPC nodes are A100s and
are assigned randomly, so compare times only within one node. The tables and ranges below
include the runs removed on 2026-10-18 (w12, w16, w8, ...) and come from the all-runs output of
2026-10-16, frozen as `timing_all_runs_2026-10-16.txt`, `timing_runs_all_runs_2026-10-16.csv`
and `timing_cases_all_runs_2026-10-16.csv` in the same folder; a re-run of the tool now
covers only the remaining runs.

**Training** (60 epochs × 200 steps). On gpu1 every size from 0.74M to 25M trains in
26–32 min, at 136–153 ms/step at unroll 4: a smaller model does not shorten training
(17.9M to 6.5M saves 9 %). On gpu2 the time grows with size, and even w12 on gpu2 is
slower than any size on gpu1, so the node matters more than the size:

| gpu2 training | w12 (1.63M) | w16 (2.89M) | w32 (11.47M) | default (17.90M) | w48 (25.76M) |
|---|---|---|---|---|---|
| minutes | 36 | 42 | 66 | 93 | 115 |

**Prediction** (tuning, 400 test rollouts, diagnostics, drawing) takes 25–33 min on
gpu1 at any size:

| part | time |
|---|---|
| step-size tuning (estimated) | ~5 min |
| 400 test rollouts | 4–5 min for the intact Laplacian runs (2.5–5.0 over all gpu1 runs) |
| other (derived) | 18–24 min |

The tuning time is an estimate (the log does not time it; `timing.txt` computes it from
the model calls of the 20 candidates), and "other" is the prediction total minus tuning
minus rollouts. The shortest rollout times on gpu1 are the base-head run (3.5 min) and
the first-pass predictions of w24m12 (2.5) and w8 (3.8), which the old chunk handling
had truncated (their 2026-10-16 re-predictions are not comparable). "Other" is
ground-truth loading, per-seed metrics and cache writes, the diagnostics (including a
batched exact-solver predictability run) and drawing (about 2 min).

**Per-case rollout cost**, default model on gpu1, seeds batched (10 per chunk at
l = 128):

| case group | ms per trajectory-step | tuning |
|---|---|---|
| off-critical | 1.20 | ~248 s for the 4 compositions |
| critical | 1.20 | ~62 s |
| big boxes | 3.91 | none (they reuse the critical schedule) |

One trajectory at a time an A100 step costs about 9 ms for every size: the GPU is
latency-bound.

**Speedup** (estimated) against the explicit dt = 0.01 solver is 72–99× like-for-like
and 207–869× batched across the END = 1000 Laplacian-output runs, depending on the step
count and the node. The solver time (about 210–250 s per trajectory) is extrapolated
from a 2000-step probe, not timed inside the job (the "est speedup" column of Table 4 in
`timing.txt`, whose note says not to quote it from a single run). It does not track
model size, and the headline is unstable between runs (see the speed paragraph in §3).

**A small model saves time on a CPU or the Mac**, not on the A100. Per step at
l = 128, batch 1:

| model | CPU (ms/step) | Apple GPU / MPS (ms/step) |
|---|---|---|
| w40 m20 | 146 | 49 |
| w32 | 98 | 31 |
| w24 | 68 | 23 |
| w24 m8 | 34 | 16 |
| w16 | 36 | 15 |
| w12 | 18 | 13 |
| w8 | 10 | 14 |

w24m8 is 4.3× faster on the CPU and 3× faster on MPS than the default; on MPS every
model under about 2M parameters costs the same per step.

**Job overhead.** The last step of `utp.sh` / `up.sh` used to redraw every Results
folder, about 1.2 min per folder, i.e. 8–20 min per job. It is now
`make_submission.py --overview`, which takes well under a minute (§6, *Files*).

### Rollout divergence handling (found and fixed 2026-10-15)

Seeds are rolled out in chunks, and one non-finite seed used to stop its whole chunk:
w24m12 had 99 further seeds cut short this way (86 of them still at R² ≥ 0.8 when cut)
and w8 78. Huge but finite explosions were not caught at all and gave R² of −10⁴⁴ to
−10⁴⁸ (w24m12). A seed now
diverges when its field is non-finite or max |ψ| > 10 (healthy runs peak at 1.55). Only
that seed stops: it is saved up to its own divergence, counted, and left out of the R²
means. Tuning and training validation score it R² = −1 from its divergence on, the
checkpoint-selection rule is unchanged, and results are bit-identical to before when
nothing diverges. The count is shown in the summary, `Results/README.md`, `report.txt`
and the legends. The two affected runs were re-predicted on 2026-10-16 (no retraining;
jobs `up_547734` for w24m12 and `up_547735` for w8). The cut seeds of w24m12 turned out
to blow up later anyway (147 of its 150 big-box seeds diverge in the end), so the old
chunk truncation had hidden how widespread its failure is; w8 has 25 diverged seeds of
400, and neither run has a seed lost to a chunk-mate any more. How it works and how old
caches are handled: §6, *Diverged rollouts* and *Caches from before the divergence fix*.
Code backups are in `Archive/code_2026-10-15_before_divergence_fix/`. Both runs were
removed from the drawn results on 2026-10-18 (their caches are in
`Archive/removed_runs_2026-10-18/`); the counts above stay as numbers.

### Open

- Done: the w24 m8 confirmation seeds (seeds 1 and 2, 2026-10-16) and the re-predictions
  of w24 m12 and w8. Optional, not run: w24 m12 seed 1 (is its big-box instability
  systematic for modes 12?), w24 seed 1, a third default seed and w20 m8 (1.17M).
- HPC sync (to do): copy `make_submission.py` to the HPC folder, and move the seven
  removed runs' folders out of `Results/`, `Work/rollout_cache/`, `Work/predict/` and
  `Work/models/` there too; otherwise the `--overview` step at the end of each job lists
  them again (§6, *Running*, step 7).
- The early-time gap of w24 m8: seed 2's epoch-22 pick (validation in-range R² 0.942,
  against 0.979 for its epoch-60 weights) is evidence to raise with the advisor, who
  paused selection changes on 2026-10-08.
- Checkpoint selection is saturated (paused by the advisor), the step-size tuner's
  alpha sits at the bottom of its grid in 22 of 35 picks, and the speedup headline is
  unstable. The full list is in `Work/analysis/findings_2026-10-15.md`, section 9 and the
  addenda.

---

## 1. The problem that had to be solved first

Training happens on a 64×64 box; testing happens on a 128×128 box **with the same
lattice spacing dx = 1**. That is not super-resolution — it is *domain extension*.
The physics is identical; the box is simply twice as wide.

This breaks a textbook FNO/UNO. Spectral weights are indexed by mode number `m`, but
mode `m` means wavenumber `k = 2πm/N`. Change `N` and the same learned weight silently
denotes a different physical wavenumber: a filter trained to act at k = 1.57 ends up
acting at k = 0.785. The operator that gets evaluated at l = 128 is not the operator
that was trained.

**Fix.** Spectral weights are stored on a *physical* wavenumber grid and resampled
whenever the box changes size (`SpectralConv2d` in `uno_train.py`). For N = 2·N_ref
the resampling is an exact linear interpolation at half-integer mode indices, so the
128-box operator *is* the 64-box operator. Everything else in the network is chosen
to be box-size agnostic as well: local convolutions, average pooling, nearest
upsampling, no normalisation layers (which would couple to statistics over a box whose
size we are deliberately changing), and FFTs with `norm='forward'`.

### The test that proves it

Tiling a 64-box state 2×2 produces a legitimate 128-box state, and the exact solver
satisfies `solver(tile(ψ)) == tile(solver(ψ))`. Any correct operator must too. This is
a structural property — it needs no training — and `uno_predict.py` checks it on
every run (`CONFIG["do_transfer"]`), alongside one-step R² at l = 128 far outside
the training window.

Two real bugs were found this way, both of which pass every conventional unit test:

1. **Wrong wavenumber scale below the first level.** The scale factor was derived from
   the *current tensor* size, but the U-net's pooling coarsens `dx` without shrinking
   the domain — the box stays the same physical width at every depth. Every level
   below the first was resampling by the wrong factor.

2. **An ill-posed inverse real FFT.** The C2R transform over-determines the `m₂ = 0`
   column: a real output demands `out[-m₁,0] = conj(out[m₁,0])`, which unconstrained
   learned weights do not satisfy. cuFFT and MKL resolve the inconsistency
   *differently* — CPU gave the right answer and CUDA was off by 10%, while every
   intermediate tensor still agreed to 1e-7. Projecting onto the Hermitian part fixes
   it, and is exactly the statement that the convolution kernel is real-valued.
   Most public FNO implementations carry this latent bug.

---

## 2. What else drives the result

**Variable time step — the single biggest lever.** `ch.py` stores frames 0.1 apart, and
`y[i] == X[i+10]` exactly, so `train_y` is a shifted copy of `train_X` carrying no new
information. Dropping it halves the data *and* frees the time step: any multiple of 0.1
can be used, not just the 1.0 implied by `n_ahead=100`. Rolling t = 100 → 2000 at
Δt = 1 costs **1900** compounding steps. Since coarsening slows as t^(1/3), a stride
Δt ∝ t keeps the change per step constant and reaches t = 2000 in **~50**. The model is
conditioned on Δt (FiLM), so one model serves every stride, and the schedule becomes an
inference-time knob tuned on held-out data rather than a baked-in choice. (The
counts are for the September horizon of t = 2000. The current rollouts go to t = 10000,
and the tuned schedules of the default Laplacian model take 222–320 model calls,
`Work/analysis/timing/timing.txt`.)

**Mass conservation by construction** (`CONFIG["conserve_mass"]`, default `True`).
Cahn–Hilliard is Model B: the right-hand side is a divergence, so on a periodic box
⟨ψ⟩ is a constant of motion. The network predicts an increment whose spatial mean is
subtracted exactly, making conservation a property of the architecture rather than a
penalty term. Measured drift is ~1e-9 per step and ~1e-8 over the whole ~50-step
rollout. Turning it off is an ablation, not an option: with a random head, drift goes
from 3e-9 to 4e-3 per step and **accumulates linearly** to 0.22 over 50 steps — enough
to move ⟨ψ⟩ from a critical quench to an off-critical one, changing the morphology
itself. The flag is recorded in the checkpoint's `arch`, so `uno_predict.py` rebuilds
the model exactly as trained. (The drift numbers are those of the September base
model over its ~50-step rollout to t = 2000. With the Laplacian output the increment
is the Laplacian of a field and has zero mean, so conservation is exact for any Δt
and the projection only removes float rounding; §6, *Laplacian output*.)

**Rollout-aware training.** A model trained only on one-step targets has never seen its
own output distribution — which is why one-step-trained models look excellent and then
diverge after a few autoregressive steps. Stage 2 unrolls the model and backpropagates
through the last `grad_steps` transitions (pushforward), so a 10-step unroll costs
the memory of `grad_steps` — 4 on an A100, which puts real gradient through more of
the drift the model has to correct.

**Physics-exact augmentation.** The equation is equivariant under the lattice symmetry
group D4 and odd under ψ → −ψ (which maps the off = +0.4 ensemble onto off = −0.4).
Both are exact, so they are free data.

**Metric-aligned loss.** R² = 1 − MSE/Var, so the primary term is per-sample MSE/Var.
A second term normalises by the *increment* instead, which stops small-Δt samples from
being ignored, and a gradient term protects the interfaces.

---

## 3. Results

*Historical, September 2026: the first run (base model, END = 300, 10 test seeds,
rollout t = 100 → 2000, job 399765). Its numbers are kept because the tables below
and the later sections refer to them, but they are no longer the current result: with
50 seeds and a rollout to t = 10000 the critical-quench R² of the base END = 300 model
is 0.623 and that of its Laplacian-output counterpart 0.770, and the Laplacian END = 1000
model reaches 0.903 (see *Current status*).*

NVIDIA A100-SXM4-40GB, **fp32**, `train_off=None`, width 40 / modes 20,
17,901,252 parameters, 60 epochs with unroll 4 from epoch 12, no early stopping
(`patience = 0`). Best checkpoint at **epoch 18**, validation rollout 0.8934.
Training 1.49 h, prediction 6.4 min with the ground truth already cached. Ten unseen
test seeds per quench, rollout t = 100 → 2000 at l = 128, stride tuned per quench on
ten validation seeds. Dataset generated by `ch_gpu.py`; job 399765, 2026-09-07.

| ψ₀ | α | dt_max | steps | mean of all R² | R²(t = 2000) | R² ≥ 0.80 held to | never below |
|---|---|---|---|---|---|---|---|
| −0.4 | 0.10 | 40 | 57 | 0.8874 ± 0.0407 | 0.744 ± 0.098 | 1577 ± 433 | 3/10 |
| −0.2 | 0.20 | 45 | 47 | 0.8972 ± 0.0444 | 0.785 ± 0.096 | 1714 ± 305 | 4/10 |
| **0.0** | 0.20 | 50 | 44 | 0.8987 ± 0.0487 | 0.758 ± 0.125 | 1741 ± 334 | 3/10 |
| +0.2 | 0.50 | 50 | 42 | 0.8925 ± 0.0370 | 0.783 ± 0.065 | 1784 ± 351 | 5/10 |
| +0.4 | 0.10 | 45 | 54 | **0.9054 ± 0.0209** | 0.767 ± 0.060 | 1710 ± 271 | 3/10 |

Mean over the five quenches **0.8962**. Best ψ₀ = +0.4, worst ψ₀ = −0.4.

The persistence baseline falls below 0.80 at t ≈ 130 and reaches zero by t ≈ 370.

### Box size does not matter — the central result

Same model, critical quench, ten seeds, boxes it never saw:

| box | 128 (test) | 160 | 192 | 256 |
|---|---|---|---|---|
| mean of all R² | 0.900 | **0.911** | 0.902 | 0.909 |
| R²(t = 2000) | 0.739 | **0.764** | 0.741 | 0.761 |

L = 256 is **4× the l = 64 training box and 16× its area**, and scores above l = 128.
Tile invariance 1.4e-7 in fp32. The physical-wavenumber spectral weights do what they
were built to do: the operator is genuinely domain-transferable, not fitted to one
grid. (This is the t = 2000 result of the September model. At t = 10000 the box
transfer still holds: for the default END = 1000 Laplacian model the three larger
boxes score a mean R²(10000) of 0.829, against 0.903 for the critical quench at
l = 128, which is itself in its finite-size regime after t ≈ 3400, see §6.)

### Physics

| quantity at t = 2000 | ψ₀ = 0 truth | UNO | ψ₀ = −0.2 truth | UNO |
|---|---|---|---|---|
| growth exponent n | 0.3082 | **0.3107** | 0.2562 | **0.2554** |
| domain scale L | 26.13 | **26.38** | 21.59 | 21.01 |
| free energy monotone | yes | **yes** | yes | **yes** |
| order-parameter drift | — | **2.2e-8** | — | 4.2e-8 |

The growth exponents above are **one seed** (`figure_physics` fits the first test
seed) on the zero-crossing length. Averaged over 50 seeds
(`Archive/Submission_2026-09-23/B_train_t300_eval_50seed/manifest.json`, formerly
`Submission/`) the same length gives 0.324
truth against 0.344 UNO at the critical quench, **+6.0 %**; Puri's half-maximum
length gives 0.301 against 0.318, +5.9 %. At l = 160 / 192 / 256 the excess is
+9.0 to +9.3 %. L(t) tracks the truth within ~2 % to t = 1000 and then
over-coarsens.

Per-step coarsening bias −10.6 % inside the training L range, −14.2 % outside (§5a).
Mixture composition is held to **0.00000 percentage points** on every quench: the
model ends the rollout separating exactly the A/B mixture it was given.

**Speed (September model):** 67× like-for-like against the explicit dt = 0.01 solver
(both unbatched), 203× when the surrogate rolls ten trajectories at once. The current
figures are in the update below.

This figure is **not stable between runs.** Three measurements of the same quantity
on the same A100 gave 96× / 56× / 67× like-for-like (690× / 170× / 203× batched);
all three are still in `Archive/Results_2026-09-08/uno_ch_predict.log` (formerly
`Results/`), which is append-mode. Two
things move it: the per-quench stride tuning selects a different number of
autoregressive steps each run (51–66 here), and per-step wall time varies by ~33 %
between nodes (7.8 vs 10.4 ms). Quote what the current run prints, not the best
figure ever seen. Both numbers are against the explicit dt = 0.01 scheme, not a
semi-implicit spectral solver, which would need far fewer steps.

Update 2026-10-15: the 400-rollout Laplacian-output runs (t = 100 → 10000) give an
estimated 72–99× like-for-like and 207–869× batched, depending on the step count and the
node. The solver time is extrapolated from a 2000-step probe (Table 4 of `timing.txt`,
column "est speedup", which says not to quote it from a single run). The speedup does
not track model size, and the headline is still unstable (*Current status*, *Time
cost*).

### Five runs, and what they settle

*Historical: five base-model runs (END = 300, rollout to t = 2000, 10 test seeds), all
before the Laplacian output. Kept as the record of what was ruled out.*

| run | unroll 4 from | val best | test mean | bias in/out | h | artifacts |
|---|---|---|---|---|---|---|
| fp32, 60 ep, 5 val seeds | ep 20 | 0.9034 | **0.9035** | **−8.5 / −7.7** | 0.76 | ✗ |
| fp64, 200 ep, 10 val | ep 71 | 0.8562 | 0.8883 | −12.1 / −12.8 | 2.78 | ✗ |
| fp64, 40 ep, 10 val | ep 12 | **0.8927** | 0.8888 | −11.1 / −13.3 | **0.40** | ✗ |
| fp32, 60 ep, 10 val | ep 12 | 0.8891 | 0.8906 | −11.4 / −17.1 | 0.75 | ✗ |
| fp32, 60 ep, `ch_gpu` data | ep 12 | 0.8934 | 0.8962 | −10.6 / −14.2 | 1.49 | ✓ §3 |

**Only the last row still has artifacts on disk.** The first four are transcribed from
cluster logs that were deleted during a reorganisation; their numbers cannot be
re-derived here, and §3 above reports the last row. Treat rows 1–4 as recorded
history, not as reproducible results.

**Precision does not matter.** The last two runs share stage timing, validation seeds
and selection rule and differ only in precision. They peak at the **same epoch (18)**
with the **same val1** (0.00107 vs 0.00109) and land **0.0017 apart** against a
standard error of 0.012. fp64 costs ~1.8× the wall time and buys nothing measurable.

**What the original run had is not precision.** It remains the best at 0.9035, and the
only differences left are the curriculum timing (unroll 4 from epoch 20 rather than
12) and the 5-seed selector. The coarsening bias — a teacher-forced systematic
measurement, far less noisy than a 10-seed R² mean — separates by *timing*, not by
precision or seed count: −8.5 % for the late transition against −11 to −17 % for both
early ones.

---

## 3e. The `conserve_mass` ablation

*Historical, 2026-08-25 (base model, END = 300, rollout to t = 2000, so the R² values
are not comparable with the current t = 10000 tables). The conclusion stands: R² alone
does not detect a broken conservation law, so the constraint is argued from the
physics panels. With the Laplacian output the constraint is also built into the output
form (§6, *Laplacian output*).*

A controlled pair from 2026-08-25: fp32, 60 epochs, identical stage timing, identical
10 validation seeds, identical selection rule. Its ON column is the fourth run in the
table above, **not** the §3 run — so its per-quench numbers differ slightly from §3's.
The pair is what matters here, and both halves of it were measured together. The only
difference is the projection (the `if self.conserve_mass:` branch in the model's
forward pass in [uno_train.py](uno_train.py)).

| ψ₀ | mean R², ON | mean R², OFF | Δ | growth-exponent error OFF | free energy monotone OFF |
|---|---|---|---|---|---|
| −0.4 | 0.8662 | 0.8971 | +0.031 | 2.8 % | **no** |
| −0.2 | 0.9007 | **0.9399** | +0.039 | 0.9 % | yes |
| 0.0 | 0.8920 | 0.9194 | +0.027 | 1.0 % | yes |
| +0.2 | 0.8848 | 0.9049 | +0.020 | 4.8 % | yes |
| +0.4 | **0.9092** | 0.8772 | −0.032 | **13.4 %** | **no** |
| **mean** | 0.8906 | **0.9077** | **+0.017** | | **3/5** |

**Removing the constraint improves R².** That is not a mistake, and it is the
interesting part.

### Why the metric goes the wrong way

Without the projection the model gains one extra degree of freedom per step: it can
shift ⟨ψ⟩ freely. R² = 1 − SS_res/SS_tot penalises any mean offset between prediction
and truth, so a model allowed to move the mean can absorb accumulated error into it
and mechanically shrink the residual. It buys the metric by spending the conservation
law.

### What it spends

| | ON | OFF |
|---|---|---|
| order-parameter drift \|⟨ψ⟩(t) − ψ₀\| at t = 2000 | **~1e-8** | **~1e-1** |
| free energy monotone | 5/5 quenches | **3/5** |
| growth exponent, ψ₀ = +0.4 | 0.3066 vs 0.3082 (0.5 %) | 0.2682 vs 0.3097 (**13.4 %**) |
| domain scale L, ψ₀ = +0.4 | 24.27 vs 24.38 | **21.42** vs 24.38 |
| seeds holding R² ≥ 0.80, ψ₀ = +0.4 | 3/10 | **0/10** |

Seven orders of magnitude of drift. On the ψ₀ = +0.4 quench a drift of ~0.1 is **a
quarter of the quench depth** — by t = 2000 the surrogate is no longer simulating the
system it was given. The growth exponent is 13 % wrong, the domain scale 12 % too
small, and the free energy goes *up* in places, which the Cahn–Hilliard dynamics
forbid.

### The point for the presentation

**R² alone cannot detect a broken conservation law — it rewards breaking it.** The
unconstrained model looks better by the headline metric on four of five quenches
while silently changing the composition it is evolving. Only the physics panels
(order-parameter drift, free-energy monotonicity, growth exponent) reveal it. That is
the argument for the hard constraint, and it is stronger than a plain accuracy loss
would have been.

**Timing caveat:** the OFF run reports 1.50 h against 0.47 h (452 vs 142 ms/step).
That is the node, not the change — it landed on `gpu2` with a cpu-bind mask in the
`.err`, peak memory was identical at 5.85 GB, and one mean-reduction per step cannot
cost 3×. Do not attribute it to the constraint. (The 2026-10 size sweep confirms the
node effect: the same 17.9M architecture trains in 31 min on gpu1 and 93 min on gpu2,
*Current status*, *Time cost*.)

---

## 3a. One run covers every quench

`eval_offs` defaults to `(-0.4, -0.2, 0.0, 0.2, 0.4)`, so a single
`python uno_predict.py` produces the full rollout (now t = 100 → 10000, the
`PAIRS[END]` horizon; originally t = 2000) for **every**
composition `ch.py` supplies — no editing and resubmitting per quench. The three larger
boxes (`l160`, `l192`, `l256`, critical quench) are rolled out in the same run, so a run
has 8 cases. Each composition
gets its own tuned stride schedule, its own case in the rollout cache
(`psi0m0.40` … `psi0p0.40`), its own figures in `Results/<label>/` and its own
block in `Work/predict/<label>/results.json`, followed by one table across every
case (the example below is from the older 10-seed, t = 2000 version of that table):

```
[summary] all quenches, 10 test seeds, l = 128, t = 100 -> 2000
    psi0  alpha  dt_max  steps            mean R2           final R2     R2 >= thr to t n never
    -0.4   0.50    50.0     42    0.8753 +-0.0421    0.7536 +-0.0709        1610 +-457    1/10
    ...
  best psi0 = ..., worst psi0 = ...
```

The stride schedule is tuned **per quench**, because the optimum genuinely
differs by morphology — the earlier separate runs selected α = 0.35 for the
critical quench and α = 0.50 for the droplet one. Set `do_tune=False` to skip
tuning and use `alpha`/`dt_max` verbatim.

Diagnostics that need a single reference trajectory (transfer, predictability,
drift) use the critical quench when it is in `eval_offs`, otherwise the first
entry.

**First-run cost**, then cached: ~30 min of solver time for the new test and
validation ground truth across the four uncached quenches, plus the sweep-figure
boxes — roughly **50 min end to end**, and ~10 min on every run after that. The
cache grows by ~2.5 GB. (Historical: these are the September figures, to t = 2000 with
10 seeds. Ground truth to t = 10000 for 50 seeds is ~112 GB, see "Disk on the cluster"
in §6, and a prediction takes 25–33 min on gpu1 once it exists, see "Time cost" in
Current status.)

---

## 3b. Paper-style sweep figure

`make_submission.py` writes `Results/<label>/r2_sweep.png` and `.pdf` — a
two-panel REVTeX-style figure: R²(t) by **mixture composition** in panel (a) and by
**box size** in panel (b) (128 = the critical quench), each curve the mean over
seeds with a ±1 sd band, every case on its own schedule, the threshold dashed and
the training horizon dotted. Computer
Modern serif, inward ticks on all four sides, minor ticks, frameless in-panel
legends, shared x axis — styled for a physics journal, but it is our own
figure and reproduces no published one.

| knob | default | note |
|---|---|---|
| `eval_offs` | `(-0.4, -0.2, 0.0, 0.2, 0.4)` | every composition `ch.py` supplies, so panel (a) reports in-distribution performance across the whole training set |
| `paper_sizes` | `(160, 192, 256)` | 2.5×, 3× and 4× the l = 64 training box (up to 16× the area) |
| `paper_size_off` | `0.0` | composition used for panel (b) |
| `paper_seeds` | the 50 test seeds | three seeds showed a ψ → −ψ gap that ten seeds do not support |
| `rollout_cells` | `10·128²` | rollout batch budget — ten seeds at a time at l = 128, 2 at L = 256 |
| `paper_save_every` | `0.0` | `0` means reuse `save_every`; raise it to shrink the L = 256 cache |

Both panels autoscale y to the data — a fixed 0.78–1.00 window would clip our
curves — with the lower limit at min(0, data) minus a margin, so a negative R² at
long horizons stays visible.

Panel (a) doubles as a **sign-symmetry check**. CH is invariant under ψ → −ψ and
training augments with it, but nothing in the architecture enforces it, so the
ψ₀ = ±0.2 and ψ₀ = ±0.4 curves should coincide to within seed noise. Any visible
gap between a pair measures residual learned asymmetry.

Cost: panel (a) is **free** — it is drawn from the same rollout cache as every other
figure. Panel (b)'s ground truth is part of the prewarm: 50 seeds at each of
L = 160/192/256, ~12 GB of cache to t = 2000 (to t = 10000 the whole ground-truth
cache is ~112 GB, see "Disk on the cluster" in §6). The solver is launch-bound at
these sizes, so L = 256 costs about the same per trajectory as L = 128. Raise
`paper_save_every` to shrink the disk without changing solve time.

---

## 4. There is no intrinsic predictability limit here

It is tempting to explain a decaying rollout R² as "error accumulation is unavoidable
in a nonlinear PDE." For this problem that is measurably false, and it is worth
knowing before blaming the model.

`uno_predict.py` (`CONFIG["do_predictability"]`) perturbs the ground truth at
t = 100 by a mass-preserving amount and integrates with the **exact** `ch.py`
dynamics to t = 2000:

| relative perturbation | R²(t=500) | R²(t=1000) | R²(t=2000) |
|---|---|---|---|
| 1e-4 | 1.00000 | 1.00000 | 1.00000 |
| 1e-3 | 1.00000 | 1.00000 | 1.00000 |
| 1e-2 | 0.99993 | 0.99997 | 0.99998 |

Even a 1 % perturbation returns R² = 0.99998 at t = 2000. Cahn–Hilliard coarsening is
strongly **contracting** — the tanh interface profile is an attractor, so unbiased
error is damped rather than amplified. (Contrast a chaotic system such as
Navier–Stokes at high Re, where this table would decay to zero.)

The table is the exact solver's, so it does not depend on the model. The diagnostic
now runs to t = 10000 with every prediction (`results.json["predictability"]`): in
`Work/predict/train_t1000_lap_pred_t10000/results.json` a 1 % perturbation still returns
R² = 0.99999 at t = 10000, and 1e-3 and 1e-4 return 1.00000.

Two consequences:

* Any rollout error is **systematic model bias**, not noise and not chaos. That is
  exactly what unrolled ("pushforward") training removes, and it is why one-step
  accuracy and rollout stability were observed pulling in opposite directions:
  one-step R² at Δt = 40 plateaued at 0.9996 while the t = 2000 rollout *degraded*.
* R² > 0.80 at the horizon is a fair target, so model selection should be done on the
  real objective. Training therefore evaluates a full 128-box rollout on held-out
  trajectories at every checkpoint, rather than on a short 64-box proxy that
  saturates at 0.999 and stops discriminating. (In September this rollout went to
  t = 2000 on one trajectory. It now goes to t = PAIRS[END] = 10000 on 10 validation
  seeds, and the checkpoint is chosen by survival, §6, *What `uno_train.py` does*.)

---

## 5. The binding constraint is the length of the training trajectories

*Historical, September 2026 (base model, END = 300, rollout to t = 2000). This
analysis motivated the longer training trajectories, and the END = 500 and 1000 models
followed. Its recommendation held only after the Laplacian output: with the base head
the critical-quench R²(10000) fell as END grew (0.623, 0.176, 0.010 at END = 300, 500,
1000), with the Laplacian output it rose (0.770, 0.828, 0.903), so the head mattered
more than END (see *Current status*). The numbers in (a) and (b) below are those of the
END = 300 base checkpoint.*

`ch.py` sets `end = 300`, and that single parameter turns out to govern how far the
rollout can go. Two independent measurements say so.

**(a) The per-step coarsening rate is mis-calibrated in a way that tracks L.**
`uno_predict.py` (`CONFIG["do_drift"]`) takes one model step from the *true* state at
each time — so nothing accumulates — and compares the change in domain scale L against
the truth:

| t | L | per-step bias in ΔL |
|---|---|---|
| 100–300 (**in** training range) | 13.95 → 18.76 | **−11.1 … −9.7 %** |
| 400–800 | 20.24 → 25.19 | −10.4 … −9.8 % |
| 1100–1900 | 27.74 → 33.14 | **−12.4 … −21.9 %** |

Mean **−10.6 %** inside the training L range, **−14.2 %** outside it.

The sign never changes: the model **under**-predicts the coarsening rate everywhere,
by a roughly constant ~10 % while L stays inside the range the training data covers,
then progressively worse once L leaves it — reaching −21.9 % by t = 1900. The
coarsening rate is calibrated for the domain sizes it was trained on and degrades
monotonically beyond them, which is the signature of an extrapolation error rather
than a modelling error.

(The autoregressive rollout ends at L = 26.38 against a true 26.13 on the critical
quench. That is not the same measurement: this table is teacher-forced and uses the
`moment` estimator, while the rollout figure uses the quantised zero-crossing one, so
the two are not directly comparable in sign or magnitude.)

(Note this is checkpoint-specific. A weaker, one-step-trained model showed a uniformly
positive and growing bias — it over-coarsened everywhere. Rollout training removed the
over-coarsening inside the training range and converted the extrapolation error into
the L-dependent drift above.)

**(b) Rollout training is capped by the available runway.**
A training chain of n steps at stride Δt needs n·Δt of trajectory, and only ~265 usable
time units exist. So n·Δt ≤ 265, while the test rollout must cover **1900**. The two
demands of rollout training pull against each other:

* long chains remove drift best, but force Δt ≤ 26;
* deployment wants Δt ≈ 40 so the rollout takes ~50 steps rather than ~76.

Measured directly: 10-step chains (Δt ≤ 26) scored R²(2000) = 0.51 against 0.63 for a
configuration reaching deployment strides. The compromise used here is **6-step chains
at Δt ≤ 44**, the largest n·Δt product that still covers the deployment stride.

**Recommendation.** Raising the dataset's `END` (`ch_gpu.py`, 300 as supplied) is the
highest-value change available. It shrinks the L-extrapolation and lifts the n·Δt
ceiling, at a few minutes of solver time. Nothing else here is close in leverage —
the intrinsic dynamics impose no limit at all (§4). The next dataset was to extend it
to **t = 1000 and t = 2000** (`ch_gpu.py --end`). The END = 500 and 1000 models were
trained and every model is evaluated to t = 10000 (`train_t0500_pred_t10000`,
`train_t1000_pred_t10000` and the `_lap_` runs, §6); END = 2000 is supported but has
not been trained.

A 64² box stays usable well past t = 500: solving the exact dynamics on 64², 128²
and 256² boxes to t = 2000, the 64-box's real-space domain length tracks the
256-box to within 3.4 %. (A −20 % gap appears in the `moment` estimator, but that is
the estimator running out of k-resolution on the coarse 64-box grid, not physics.)

---

## 6. Files

| file | role |
|---|---|
| `ch_gpu.py` | dataset generator: `ch.py`'s dynamics, seeds and frame selection, batched in float64 on the GPU. `--end N` writes `Work/data/test_t<N>.npz` plus a `.json` sidecar (device, dtype, torch version, shapes, timing). It refuses to overwrite an existing file. `--verify` checks it against a numpy transcription of `ch.py`'s loop |
| `uno_train.py` | **training**: CH solver, ground-truth cache, UNO, epoch loop with unroll curriculum, survival-based checkpoint selection. `--end N [--variant lap or base] [--modes M] [--width W] [--seed N] [--npz PATH] [--smoke] [--bench]`. The default `lap` variant predicts M and applies the 5-point Laplacian (see *Laplacian output* below). It also holds the path helpers and the low-k gain (`lowk_gains`) the other scripts import |
| `uno_predict.py` | **prediction**: ground-truth prewarm, per-quench stride tuning, 50-seed rollouts into the rollout cache, diagnostics (transfer, predictability, drift, low-k gain, speedup), `results.json`, then the drawing of this run. `--end N [--variant lap or base] [--modes M] [--width W] [--seed N] [--t-end T] [--gt-only] [--smoke] [--ckpt PATH]` |
| `make_submission.py` | **drawing**: `Results/` from the rollout caches alone, with no model and no torch. With no option it redraws every complete run folder, then `Results/README.md`, `comparison.png` and `comparison_t<END>.png`. `--overview` writes only `Results/README.md` and the comparison images and redraws no run folder (it cannot be combined with `--run` or `--rollout`). `--run LABEL` redraws one run. `--rollout` refills a cache first. The snapshot palette is set by four `DRAW` keys (*Snapshot colours*) |
| `migrate_layout.py` | one-off move of the pre-2026-09-29 layout into the one below. Dry run by default; `--apply` does the moves |
| `check_code.py` | static checks: compile, ≤ 88 columns, Python-3.11 f-string rules, no comments in `uno_train.py` / `uno_predict.py` |
| `gt.sh` `ch_gpu.sh` `ut.sh` `up.sh` `utp.sh` | Slurm jobs: ground truth only; dataset; train; predict + draw; train + predict + draw. `END`, `T_END`, `VARIANT` (`lap` default, `base` for the old architecture), `MODES` (default 20), `WIDTH` (default 40) and `SEED` (default 0) go in through `--export`, and the usage is at the top of each script. Every job prints `python --version`. `utp.sh` and `up.sh` **end with `make_submission.py --overview`**: the run's own `Results/<label>/` folder is drawn by `uno_predict.py`, and the last step only refreshes `Results/README.md` and the comparison images, which takes well under a minute (it used to redraw every run folder, about 1.2 min each) |
| `sp.sh` | Slurm job (`test` partition, 1 CPU): the **full redraw**, `python make_submission.py` over every complete run. Needed only after the drawing code changes or to refresh every run folder |
| `Work/analysis/size_sweep.py` | `python3 Work/analysis/size_sweep.py [--end 1000] [--variant lap] [--tol-early 0.01] [--tol-late 0.08]`. Finds every `train_t<END>[_lap]<tag>_pred_t10000` rollout cache (tag empty, `w<W>`, `m<M>`, `w<W>m<M>`, optionally with `s<N>`) and prints per run the parameter count, training metadata (epoch, validation survival, in-range R²), the pooled R² (8 cases × all seeds) at t = 500, 1000, 2000, 5000 and 10000, the R²(10000) of the off-critical, critical and big-box groups, the excess late growth exponent and the domain-size ratio pred / truth at t = 10000, and a verdict (*good*, *early only*, *fails early*, or *diverged* for a run with a diverged or chunk-truncated rollout, which is kept out of the reference). The default model is the reference. Writes `size_sweep.{txt,csv,png}` into `Work/analysis/size_sweep/` (another `--end` or `--variant` adds a `_<variant>_t<END>` suffix, e.g. `size_sweep_base_t1000.*`). Late verdicts rest on one training seed per configuration (three for w24 m8) and are provisional. Since 2026-10-18 it covers only the runs still in `Work/rollout_cache/`; the 13-run output of 2026-10-16 is frozen as `size_sweep_all_runs_2026-10-16.{txt,csv,png}` |
| `Work/analysis/timing_table.py` | `python3 Work/analysis/timing_table.py [--logs "utp_*.out up_*.out"] [--end 1000]`. Reads the job logs in the project root and `Work/predict/<label>/results.json` and prints four tables (per run, per case, per case group, speed diagnostic) with the node, training minutes, ms/step per unroll stage, peak GPU memory, the prediction total split into tuning, rollouts and other, and the ms per trajectory-step of every case. Tuning is not timed in the logs, so `tuning~` is an estimate from the model calls of the 20 candidates. Writes `timing.txt`, `timing_runs.csv` and `timing_cases.csv` into `Work/analysis/timing/`. Compare times only within one node. Since 2026-10-18 it covers only the runs still in `Work/predict/`; the all-runs output of 2026-10-16 is frozen as `timing_all_runs_2026-10-16.txt`, `timing_runs_all_runs_2026-10-16.csv` and `timing_cases_all_runs_2026-10-16.csv` |
| `Work/analysis/paired_compare.py` | `python3 Work/analysis/paired_compare.py <label_A> <label_B>`. Seed-by-seed comparison of two rollout caches, B minus A (positive = B better), per case: R² at 500, 1000, 2000, 4000, 6000, 8000 and 10000, the log-time mean R², the interpolated survival (R² ≥ 0.8), the zero-crossing domain-size ratio at t_end and the win count, then the pooled R²(t_end). **A seed pair is skipped, and counted, when either rollout diverged or stopped before t_end**, so a short rollout is never scored as if it had survived |
| `Work/analysis/findings_2026-10-15.md` | the numbers of 2026-10-07 to 2026-10-18 in one place (Laplacian output, model capacity, seed noise, size sweep, time cost, divergence handling, HPC notes, next steps, the 2026-10-16 addenda with the w24m12 and w8 re-predictions, the w12 m12 run and the w24 m8 seed replicates, and the 2026-10-18 addendum with the runs removed from the drawn results, the snapshot colours and the HPC sync); the source of *Current status* above |
| `Work/analysis/literature_laplacian_output_2026-10-08.md`, `Work/analysis/t1000_failure_2026-10-06/` | the prior-art check for the Laplacian output, and the analysis of the END = 1000 failure (`analysis.json` and its figures) |
| `hpc_storage.py`, `hpc_storage.sh` | used / available storage and quota on the HPC for `$HOME`, scratch and the working directory (`sbatch hpc_storage.sh` shows what a compute node sees) |

Every setting lives in a `CONFIG` dict at the top of each Python file, so a run is
reproducible from those files alone. `uno_predict.py` imports the model and the
solver from `uno_train.py`. The numpy-only physics helpers live in
`make_submission.py`, so drawing never imports torch.

### Runs, labels and where things live

A run is a model trained on data up to t = END and rolled out to t = T. Its label is
`train_t<END:04d>[_lap][_<size>]_pred_t<T:05d>`, for example `train_t0300_pred_t10000` (base
model), `train_t0300_lap_pred_t10000` (Laplacian-output model) or
`train_t1000_lap_m24_pred_t10000` (25.11M parameters, see *Model size*) or
`train_t1000_lap_w24m8_pred_t10000` (1.68M). **Every
model is predicted to T = 10000**: `PAIRS` in `uno_train.py` maps 300, 500, 1000
and 2000 to 10000, and `default_t_end` returns 10000 for any other END. Training
validates to the same horizon (`val_t1 = 0` means PAIRS[END]), so every END
validates against ground truth to 10000. The September 2000-long runs
(`train_t0300_pred_t02000`, `train_t0500_pred_t02000`) are no longer in the working
tree: only `_pred_t10000` runs of the same checkpoints are kept, and the September
t = 2000 material is in `Archive/Results_2026-09-08/` and `Archive/Submission_2026-09-23/`.
The runs in `Results/` now are END = 300, 500 and 1000, each with the base head and the
default-size Laplacian output (6), plus five END = 1000 size variants (m24, w48, w32,
w24, w24m8; *Model size*). Seven other runs (w24m12, w16, w12, w8, w12m12 and the w24m8
seed replicates s1 and s2) were removed on 2026-10-18: their `Results/`,
`Work/rollout_cache/`, `Work/predict/` and `Work/models/` folders are in
`Archive/removed_runs_2026-10-18/` (4.1 GB), and their numbers are in *Current status*.

**Snapshots.** `CONFIG["snapshots"] = (100, 500, 1000, 2000)` in `uno_predict.py`
is the fixed *main* set (those ≤ T), and every multiple of
`long_snap_every = 2000` in (t_start, T] is a *long* snapshot, so a T = 10000 run
stores 100, 500, 1000, 2000, 4000, 6000, 8000, 10000 in each seed file. All of
them are must-hit times of the rollout schedule and of every tuning schedule.
Tuning is scored at (T/4, T/2, 3T/4, T). `make_submission.py` draws
`snapshots_<case>.png` at the main times (the layout is unchanged) and the new
`snapshots_long_<case>.png` at the long times, with the same truth / UNO / |error|
rows, the same best-seed rule and "every 2000" in the title. An older cache that
stored fewer than two long times (the t = 300 → 2000 and the epoch-12 t = 1000 →
10000 caches) skips the long figure with a log line, and one with fewer than two main
times draws the main figure at whatever snapshots it holds.

**Snapshot colours.** The snapshot figures use the palette of the presentation deck
("Solving the Cahn-Hilliard Equation with a U-shaped Neural Operator"), so the PNGs in
`Results/` can be dropped into the slides without regenerating them. In
`snapshots_<case>.png` and `snapshots_long_<case>.png`, ψ = −1, 0, +1 run from blue
`#1F4E8C` through ivory `#FFFEFB` to orange `#D9480F`, the |error| row runs from
`#FAF7F0` (0) to `#C2410C` (1), the figure background is `#FFFEFB` and the text is
`#14213D`; the panel frames are removed. The four colours live in the `DRAW` dict at the
top of `make_submission.py` (`snap_field`, `snap_error`, `snap_bg`, `snap_text`): change
those four keys to change the palette, then redraw (`python3 make_submission.py`, or
`--run LABEL` for one run). Only the snapshot figures use these keys. All 11 run folders
in `Results/` were redrawn on the Mac on 2026-10-18, and the previous
`make_submission.py` is in `Archive/code_2026-10-18_before_snapshot_colours/`. On the
HPC, `uno_predict.py` draws a run's folder with `make_submission.py`, so copy the new
file there too, or new runs come out in the previous colours (*Running*, step 7).

```
Results/                        report material only; make_submission.py regenerates it
                                (11 run folders since 2026-10-18)
  README.md                     index of every run (generated, never edit by hand)
  comparison.png                critical-quench mean R²(t) over the 50 test seeds (log t) of
                                the 6 default-size, seed-0 runs only: base head and
                                Laplacian output at END = 300, 500, 1000
  comparison_t<END>.png         the same curve for the size and training-seed variants of
                                one END (Laplacian output): one colour per size, the
                                default in black, one line style per seed
                                (comparison_t1000.png exists: 6 runs, default, w48, m24,
                                w32, w24, w24m8, all seed 0)
  project_*.png/.pdf,           the block diagrams (make_diagram.py)
  uno_architecture.png/.pdf
  <label>/                      report.txt, manifest.json, composition.csv, composition_all.png,
                                r2_sweep.png/.pdf and per case snapshots_ and
                                snapshots_long_ (BEST seed), phase_ordering_kinetics_,
                                domain_growth_, psi_distribution_, composition_
                                (seed AVERAGES)
Work/                           heavy and machine-side, never part of the report
  analysis/                     size_sweep.py, timing_table.py, paired_compare.py, their
                                outputs (size_sweep/, timing/), findings_2026-10-15.md,
                                the literature check and the END = 1000 failure analysis
  data/test_t<END>.npz, .json   datasets (ch_gpu.py)
  models/train_t<END>/          uno_ch_best.pt, uno_ch_best_prev.pt, uno_ch_state.pt,
                                uno_ch_state_stale.pt, uno_ch_train.log,
                                uno_ch_train_config.json, cache/{train,val}_X_f32.npy
  gt_cache/                     ONE ground-truth cache, shared by every run and by the
                                training validation
  predict/<label>/              results.json, predict.log (rollout.log for --rollout)
  predict/gt_t<T>/predict.log   log of a --gt-only run
  rollout_cache/<label>/        <case>/seed<N>.npz + meta.json; every figure is drawn
                                from these
  smoke/                        everything a --smoke run writes (its own models,
                                gt_cache, rollout_cache, predict and Results/)
Archive/                        Results_2026-09-08/, Submission_2026-09-23/ (moved there
                                intact by migrate_layout.py), dated code backups
                                (code_<date>_before_<change>/), dated document backups
                                (docs_<date>_before_<change>/, latest
                                docs_2026-10-18_before_runs_removed/),
                                removed_runs_2026-10-18/ (Results/ and Work/{models,
                                predict,rollout_cache}/ of seven runs, 4.1 GB; deleting
                                it frees the space),
                                superseded/<label>_<old fingerprint>_<time>/ (created
                                when a run is superseded, see below)
```

**Superseded runs.** A new checkpoint trained for the same END has a different
fingerprint from the one a label's rollout cache was built from, and mixing two
models in one cache would corrupt every figure. The rollout engine therefore
moves (never deletes) `Work/rollout_cache/<label>`, `Work/predict/<label>` and
`Results/<label>` into `Archive/superseded/<label>_<old fingerprint>_<YYYYmmdd-HHMMSS>/`
(`rollout_cache/`, `predict/`, `Results/`), logs each move and carries on with a
fresh cache. Retraining END = 1000 with the new selection did this to the
epoch-12 `train_t1000_pred_t10000`; the base-head run of that label now in `Results/`
is the retraining (survival-selected, epoch 16, `Work/predict/train_t1000_pred_t10000/results.json`).
A `--smoke` run archives under
`Work/smoke/Archive/superseded/`. `--schedule-from` keeps its own fingerprint check.
The old model files are rotated by the trainer (`uno_ch_best_prev.pt`,
`uno_ch_state_stale.pt`).

**Diverged rollouts.** Seeds are rolled out in chunks (10 at l = 128, 6 / 4 / 2 at
l = 160 / 192 / 256). The model acts on every sample separately (spectral convolution,
convolutions, FiLM and the spatial means are all per sample), so one seed's blow-up cannot
change another's. A seed is *diverged* when its field goes non-finite or its
max |ψ| exceeds `diverge_abs = 10` (`CONFIG` of `uno_predict.py` and `uno_train.py`;
physical |ψ| stays below about 1, the largest legitimate transient seen is 1.55).
`uno_predict.rollout` freezes such a seed at its last good state, stores NaN for it from
then on, logs `WARNING: k of n rollout(s) diverged (non-finite or |psi| > 10) at t = ...;
the others continue` and keeps rolling the rest; it stops only when every seed has
diverged (`none left, stopping`). The seed's file in the rollout cache is cut just before
its divergence and carries `diverged_at` (NaN for a seed that never diverged), so
`cached_ok` takes it as final. `load_case` leaves diverged seeds out of every stacked
array: every R² mean, best / median / worst seed, growth exponent and figure covers the
seeds that reached t_end. The count is shown wherever a case is reported: `n_diverged`
(and `n_truncated_legacy`) in the `results.json` summary, the `n div` column of the
`[summary]` table and of `report.txt`, `manifest.json` (per case `n_diverged`,
`diverged_seeds`, `diverged_at`), the `diverged` column of `Results/README.md`, and the
comparison legends (` (3 of 50 diverged, excluded)`). Survival counts a diverged seed as
failed at its first crossing of the threshold, or at its last good time if it diverged
first. A case whose seeds all diverged is skipped with a printed warning (it still counts
in the `diverged` column; if it is the critical case, the run's row and its comparison
curve show another composition, labelled `critical case: all N seeds diverged`). Stride tuning
scores a diverged validation seed R² = -1 from its divergence time on (never its last
value held flat; `div` column of the tuning table), and the validation rollout in
`uno_train.py` scores it the same way with its own survival time; the checkpoint-selection
rule is unchanged.

**Caches from before the divergence fix.** Until 2026-10-15 a non-finite state truncated
the whole chunk, so chunk-mates lost their trajectory too (`w24m12`: 150 short
files, `w8`: 94; `Work/analysis/size_sweep.py` estimated 51 + 99 and 16 + 78 as diverged
+ collateral). Those short files carry no `diverged_at`, so `cached_ok` rejects them and the
next prediction rolls out only them again; complete old files stay cached. For a cache
that has not been re-predicted, `make_submission.py` cuts such a case to its shortest seed
(printed warning) and `Results/README.md` reports "older truncated seed files
(re-predict)". Both affected runs were re-predicted on 2026-10-16, no retraining
(`up_547734.out`, `up_547735.out`): `w24m12` now has 147 diverged and `w8` 25, both with 0
collateral. The commands were:

```bash
sbatch --export=ALL,END=1000,WIDTH=24,MODES=12 up.sh
sbatch --export=ALL,END=1000,WIDTH=8 up.sh
```

**Composition.** Every run folder gets `composition_<case>.png` (seed-mean ⟨ψ⟩(t),
truth against UNO, ±1 sd, with the A / B percent axis and the A / B split at t_start
and t_end in the title), `composition_all.png` (c_A %(t) of the five compositions on
one axis) and `composition.csv` (case, psi0, t, mean ψ and A / B % for truth and
UNO, and the UNO drift, at every snapshot time). The convention is ψ = c_A − c_B
with c_A + c_B = 1, so c_A = (1 + ψ)/2 and c_B = (1 − ψ)/2: ψ = 0 is 50 / 50 and
ψ = +0.05 is 52.5 % A / 47.5 % B. `report.txt` section 2 prints the same A % / B %
table for every case. Panel (f) of `phase_ordering_kinetics_<case>.png` is unchanged.


Cases: `psi0m0.40` … `psi0p0.40` are the five compositions on the 128 box, and
`l160`, `l192`, `l256` are the critical quench on larger boxes. The boxes use the
critical quench's own tuned schedule, so `l128` in the box plots is exactly
`psi0p0.00`. The best seed is the one with the highest trajectory-mean R² (the mean
over every rollout time), and the same rule is used everywhere.

**Finite-size caveat at the long horizons.** After t ≈ 3400 the zero-crossing domain
length on the l = 128 test box exceeds l/4, so fewer than 4 domains fit across it.
From then on the box, not the physics, limits coarsening, and the l = 128 R² at
T = 10000 describes that regime. Nothing is dropped. Every time axis shades
it as "finite-size regime (< 4 domains)", and the manifest records the first such
time for each case. The 160 / 192 / 256 boxes, with all 50 seeds at every horizon,
show the trend.

A checkpoint records its training horizon (`train_t_end`, `train_shape`,
`val_shape`). The migrated t = 300 checkpoint predates that and falls back to the
memmap in its own model folder (2990 frames, so t = 300). Seeds already in a
rollout cache are skipped when the checkpoint fingerprint (sha256[:12]), label and
schedule match, so an interrupted prediction resumes where it stopped. A cache
with a different checkpoint under the same label is moved into
`Archive/superseded/` (nothing is deleted) and rebuilt; a different horizon is refused.

Ground-truth files are `gt_l<l>_off<off>_seed<s>_t<t0>-<t1>_ev<ev>.npz`. They are
uncompressed zips written frame by frame, and they are read through a memmap. New
files also record the device, the solver dtype and dt, and the torch version. A
request is served by any longer file of the same (l, off, seed, t0, ev), because
the first frames of a longer solve are bit-identical. A solve to 10000 therefore
also covers every 2000 request (and a 20000 solve, if one is ever made, every 10000
request). Inside a Slurm job (`SLURM_JOB_ID` set)
ground truth must run on CUDA (`gt_require_cuda = "auto"`) and the job stops if
there is none. Elsewhere it falls back to the CPU and logs that it does.

### Laplacian output

`CONFIG["laplacian_output"] = True` (the default) changes what the network's head
means. Cahn–Hilliard is ∂ψ/∂t = ∇²μ, and the solver's whole update over any Δt is
ψ(t+Δt) − ψ(t) = ∇²₅ₚₜ M with M = Σ dt·μₙ over the solver steps, because the
discrete 5-point periodic Laplacian is linear. So a network that predicts M and then
applies the *same* 5-point Laplacian (`lap5` in `uno_train.py`, the `roll`
stencil of `_features`) represents every true update exactly: it is a change of
variables, not an approximation, and inside the training box it should do as well as
the base head. Three things come with it for free:

- **Low-k suppression.** The increment of Fourier mode k carries the factor
  −(4 sin²(kₓ/2) + 4 sin²(k_y/2)) ≈ −k². The 8 longest modes of the l = 128 box
  (|n| = 1 and (1,1), k = 0.049–0.069) lie below the l = 64 training box's
  fundamental 2π/64 = 0.098, so they are damped by (k/k_min,train)² = 0.25 for the
  |n| = 1 shell and 0.5 for the (1,1) shell, without tuning.
- **Mass conservation.** The Laplacian of anything has zero mean, so ⟨ψ⟩ is exact
  (the `conserve_mass` subtraction stays and only removes float rounding).
- **A smoother target.** M is a smoother field than Δψ, which the U-shaped
  architecture should handle well.

Why: `Work/analysis/t1000_failure_2026-10-06/analysis.json` found that every base
model (END 300/500/1000) steps those 8 modes 2.4–3.5× too far (they were never
trained on) and collapses at l = 128 after t ≈ 2000. Damping them by 0.4 in a
post-hoc test lifted R²(5000) from −0.15 to 0.84; the k² factor gives a damping of
that size from the physics. The new low-k diagnostic (`diag_lowk` in `uno_predict.py`,
section 6 of `report.txt`, `results.json["lowk"]`) measures it: the teacher-forced
gain g(shell) = Σ Re(conj ΔP · ΔT) / Σ |ΔT|² per wavenumber shell ("below" = the 8
modes, "first" = |n| ∈ [2, 2.9], "mid" = [3, 8], "high" = rest) at t0 ∈ (500, 1000,
2000, 5000), Δt = 48. A well-trained model has g ≈ 1 in every shell. Measured on the
base END = 500 model (`Work/models/train_t0500`, 2 seeds, critical quench, l = 128):
below 2.54 / 2.35 / 2.28 at t0 = 500 / 1000 / 2000, first 0.89–0.91, mid 0.77, high
0.76–0.80. Training logs the lowest-shell and first-shell gain in every validation
line (`lowk g <below>/<first>@<t0>`); it is not used for checkpoint selection. The
findings file quotes the same defect as 2.3–3.5×.

**Result (2026-10-15).** The Laplacian models do not have the defect. For the END = 300,
500 and 1000 runs (`results.json["lowk"]` of `Work/predict/train_t0300_lap_pred_t10000`,
`train_t0500_lap_pred_t10000`, `train_t1000_lap_pred_t10000`) the "below" shell has
g = 0.73–0.82 at t0 = 500, 1000 and 2000 and 0.84–0.93 at t0 = 5000, against 2.28–2.54
for the base END = 500 model above. Critical-quench R²(10000) is 0.770, 0.828 and 0.903
for END = 300, 500 and 1000 with the Laplacian output, against 0.623, 0.176 and 0.010
with the base head (*Current status*). All six runs choose their checkpoint by validation
survival (§6, *What `uno_train.py` does*), so this comparison isolates the head. The
survival-based selection was a separate change: the late-heavy checkpoint score had
picked epoch 12 for the first, retired base END = 1000 model.

**Prior art.** The idea is not new (`Work/analysis/literature_laplacian_output_2026-10-08.md`,
searched 2026-10-08; several entries were checked at abstract level only). Fantasia et
al. 2025 (arXiv:2507.21760) have the network output a scalar field and add its Laplacian
to the state, for Cahn–Hilliard in 2-D, and Lanzoni et al. 2024 (*Mach. Learn.: Sci.
Technol.* 5, 045017) output a flux through a fixed divergence layer; both use
convolutional recurrent networks rather than a spectral U-shaped operator. Yadav et al.
2026 (arXiv:2606.26128) work in the same setting as this project (2-D Cahn–Hilliard,
trained on 64×64, applied to L = 160, 192, 256) but conserve the mean by a projection, so
they have no k² suppression of the unseen modes. The papers checked do not discuss the k²
suppression of modes that were never seen in training, which is the point of the diagnosis here.

**Risk.** ∇² amplifies grid-scale wiggles of the head output by up to 8×. If the
network's output is noisy, interfaces can come out rough early in training, and the
learning rate or an output scale may need to change. Watch interface sharpness in
`snapshots_*` and the first epochs' `val1`. Outcome: the END = 300, 500 and 1000 runs
all train to usable models (R²(10000) above), so the risk did not stop training.

**Variants.** `--variant lap` (default) and `--variant base` on `uno_train.py`,
`uno_predict.py` and `make_submission.py --rollout` (and `VARIANT=lap|base` in
`ut.sh` / `up.sh` / `utp.sh`) pick the architecture. The lap model lives in
`Work/models/train_t<END>_lap/` and its run label is
`train_t<END>_lap_pred_t10000`; the base runs keep their old labels and paths
(`train_t<END>`, `train_t<END>_pred_t10000`) and stay in `Results/` for comparison
(`README.md` and `comparison.png` list them as separate, labelled rows and
curves). `uno_predict.py` refuses a checkpoint whose `arch.laplacian_output` does
not match `--variant`; an old checkpoint without the key builds the base model
unchanged, bit for bit. Retraining the lap model costs about 0.5 h per END on an A100
(31 min measured on gpu1; the same architecture took 93 min on gpu2, see *Time cost* in
*Current status*).

Job order (ground truth already exists, so no `gt.sh` dependency). The first line made
the Laplacian END = 300, 500 and 1000 runs; the END = 2000 lines have not been run:

```bash
for E in 300 500 1000; do sbatch --export=ALL,END=$E utp.sh; done   # VARIANT=lap
d=$(sbatch --parsable --export=ALL,END=2000 ch_gpu.sh)             # if not built yet
sbatch --dependency=afterok:$d --export=ALL,END=2000 utp.sh
sbatch sp.sh                                                        # full redraw; utp.sh and up.sh already end with --overview
```

### Model size (Fourier modes and width)

`uno_train.py`, `uno_predict.py` and `make_submission.py --rollout` take
`--modes M` and `--width W` (defaults = `CONFIG["modes"] = 20` and
`CONFIG["width"] = 40`, which stay as they are, so every existing label, folder
and checkpoint is unchanged). Precedence is CLI > `--smoke` overrides > CONFIG, so
`--smoke --modes 10` is a smoke model with 10 modes. A non-default size adds a
tag after the variant: `m24`, `w48`, `w44m22` (`run_tag` in `uno_train.py`, which
also adds the training-seed tag below).

| model | modes per level | parameters | peak GPU memory in training |
|---|---|---|---|
| default (width 40, modes 20) | 20, 10, 5, 2 | 17.90M | 8.52 GB |
| `--modes 24` | 24, 12, 6, 3 | **25.11M** | 8.70 GB |
| `--width 48` (modes 20) | 20, 10, 5, 2 | 25.76M | 11.91 GB |
| `--width 24 --modes 8` | 8, 4, 2, 2 | 1.68M | 6.44 GB |

The parameter count grows roughly as width² × modes², because the spectral weights
dominate. The peak memory is the measured value in
`Work/analysis/timing/timing.txt`. The other sizes of the sweep (widths 8, 12, 16, 24
and 32 at modes 20, width 24 with modes 12 and width 12 with modes 12) are in the
results table under *Current status* (w8, w12, w16, w24 m12 and w12 m12 as numbers only
since 2026-10-18).

Labels and folders: `Work/models/train_t1000_lap_m24/`, run
`train_t1000_lap_m24_pred_t10000`, its rollout cache, predict folder and
`Results/` folder carry the same label, so the 25.11M run sits **side by side**
with `train_t1000_lap_pred_t10000`: nothing is overwritten, and the superseded
archiving (same label, different checkpoint) never treats one as the other
because the labels differ. `uno_predict.py` checks `arch.modes` and `arch.width`
of the checkpoint against `--modes` / `--width` and stops with a message
otherwise. Modes 24 fits every grid: level mode counts 24, 12, 6, 3 are within
the l = 64 limits (31, 15, 7, 3) and the l = 128 .. 256 boxes need no more.
`Results/README.md` lists a parameter-count column. `comparison.png` shows only the
default-size, seed-0 runs; the size variants of one END are drawn in
`comparison_t<END>.png` (`comparison_t1000.png`: default, w48, m24, w32, w24 and w24m8),
one colour per size with the default in black. The size colours are distinct hues, not
shades of one colour (`SIZE_COLORS` in `make_submission.py`: blue, orange, green, red,
purple, then brown, pink, cyan, olive, by parameter count).

**Measured cost** (60 epochs × 200 steps, `Work/analysis/timing/timing.txt`). The early
estimate of 1.2–1.4× the compute per step for the 25M models turned out too high on the
A100: on gpu1 the 25.11M model takes 153 ms/step at unroll 4 against 151 for the 17.90M
default, and every size from 0.74M to 25M trains in 26–32 min, so a smaller model does
not shorten training there. On gpu2 the training time grows with size (36 min for w12 to 115 min for
w48), but the node effect is larger than the size effect. Peak memory is in the table.
Where a small model does save time is inference on a CPU or the Mac (*Time cost* in
*Current status*).

**Caution, recorded when the option was added (2026-10-12):** the 17.9M model already
overfits mildly (validation survival peaks at epoch 26–28 of 60), and the data is 4
trajectories per quench, so 25M was an experiment, not an upgrade. The sweep fits that:
no larger model scored better, and the default's selected epoch is 28
(`size_sweep.txt`).

**Result.** Bigger is not better. The 25M models are not better than the default (pooled
R²(10000): m24 0.800, w48 0.768, default 0.806; m24 was picked at the under-trained epoch
24). The smallest model as good as the default is w24 m8 (1.68M; pooled R²(10000) 0.824
for seed 0 and 0.761 averaged over three seeds, stable on all three and about 0.02 below
the default at t ≤ 2000), but quality is not monotone in size: w16 (2.89M, 0.713) and w24 m12
(2.86M, 0.684 over its survivors, 147 of its 150 big-box seeds diverged) are larger than
w24 m8 and worse. No model under 1M parameters works (w8 0.291 with 25 of 400 rollouts
diverged, w12 m12 0.115 with 236 of 400), and below 2.9M only w24 m8 is both accurate and
stable; stability in boxes larger than 128 is a pass criterion of its own. Cutting modes
looked much cheaper than cutting width (w24 m8 beats w12 at about the same size by
+0.163 ± 0.012 in R²(10000) on the l = 128 cases, a test-seed error only, comparing
w24 m8 seed 0 with the single w12 run, against a training-run SD of 0.04–0.08), so the
provisional advice is
to keep the width at 24 or more and cut modes. Apart from w24 m8, which has three
training seeds, the verdicts rest on one seed per size (*Training seed* below). The
commands:

```bash
sbatch --export=ALL,END=1000,MODES=24 utp.sh              # 25.11M, train, predict to 10000, overview
sbatch --export=ALL,END=1000,WIDTH=24,MODES=8 utp.sh      # w24 m8, 1.68M
python3 Work/analysis/size_sweep.py                       # the sweep table
```

Compare two runs at matched times with `Work/analysis/paired_compare.py` (below) or in
`comparison_t1000.png`; `report.txt` section 1 prints the width and modes. `ut.sh`,
`up.sh` and `utp.sh` take `MODES` (default 20), `WIDTH` (default 40) and `SEED`
(default 0).

### Training seed

`uno_train.py`, `uno_predict.py` and `make_submission.py --rollout` take
`--seed N` (default `CONFIG["seed"] = 0`; precedence CLI > `--smoke` > CONFIG).
The seed sets the weight initialisation (torch and numpy global seeds), the
training and validation pair draws (the two `PairSampler`s) and the augmentation
noise generator. It does not touch the 50 test seeds, the 10 validation/tune
seeds or the benchmark. A non-zero seed adds `s<N>` after the size tag (`s1`,
`w32s1`, `w44m22s2`; `run_tag`), so `train_t1000_lap_w32s1/` and
`train_t1000_lap_w32s1_pred_t10000` sit next to the seed-0 run, whose names do not
change. The seed is stored in the checkpoint (`config["seed"]`) and in the resume
stamp, and `uno_predict.py` stops if the checkpoint was trained with another seed
than `--seed` (an old checkpoint without it counts as seed 0).
The streams are seeded from `1000 * seed` plus a per-rank offset (seed 0 keeps the
old values), so replicate seeds do not overlap across ranks; a negative seed is
rejected.

Why: the paired +- SE of two runs only covers the 50 test seeds, i.e. the noise of
evaluating one trained model, not of training it. Seed replicates measure the run-to-run
noise. Same seed is not bit-reproducible on CUDA (non-deterministic kernels), so a
"replicate" with the same seed still differs slightly.

```bash
sbatch --export=ALL,END=1000,WIDTH=32,SEED=1 utp.sh   # ut.sh / up.sh take SEED too
python3 Work/analysis/paired_compare.py train_t1000_lap_w32_pred_t10000 \
    train_t1000_lap_w32s1_pred_t10000
```

`paired_compare.py` skips, and counts, every seed pair in which either rollout diverged
or stopped before t_end (*Files*). The seed-1 runs were deleted on the HPC after they
were scored (see below), so the second label of this example no longer has a cache. The
command works for any two complete runs. The size variants and seed replicates of one END
are drawn in `comparison_t<END>.png` (one colour per size, the default in black, one line
style per seed), not in `comparison.png`, and `Results/README.md` names the seed in the
model column (`seed 1`). At present `comparison_t1000.png` has six runs, all seed 0, so
only one line style is in use; the w24 m8 replicates (seeds 1 and 2) are in the archive.

**Result (2026-10-14).** Two replicates were trained, the default size and w32, each with
training seed 1 against the seed-0 run of the same size (the pooled R²(10000) table is
in *Current status*: 0.806 against 0.664 for the default, 0.837 against 0.778 for w32).
Seed 1 was worse in 8 of 8 cases in both pairs. Their models, caches and `Results/`
folders were later deleted on the HPC to save space, so their numbers survive only in
`utp_547101.out` (default seed 1), `utp_547087.out` (w32 seed 1) and
`Work/analysis/findings_2026-10-15.md`.

- **How large the noise is.** The training-run SD of pooled R²(10000) is about
  0.04–0.08: 0.076 from the two pairs (df = 2, 95 % CI 0.04–0.48) and 0.041 from the
  same-node w32 pair alone. Six of the seven larger runs lie within SD 0.027; default
  seed 1 is the outlier. The noise appears only after t ≈ 4000. At
  t ≤ 2000 the runs agree within 0.01–0.03, except where an under-trained checkpoint
  was picked. The test-seed SE of one run's pooled mean is about 0.008, so the paired
  ± of a comparison is far smaller than the training-run spread.
- **Size effect.** w32 against the default, averaged over 2 seeds each, is
  +0.07 ± 0.08 (p ≈ 0.3–0.4): no size effect is detectable. The earlier "smaller is
  better" result (+0.031 ± 0.007 against default seed 0) was within seed noise. This is why a
  size ranking from single-seed runs needs replicates, which is why w24m8 got two more
  seeds (below).
- **Mechanism (descriptive, not a proven cause).** The bad late runs over-coarsen. In the
  worst four the late growth exponent is too high by +0.025 to +0.054, and the
  domain-size ratio L_pred / L_true at t = 10000 is 1.04–1.07.
- **Validation catches gross failures.** Among the seven larger runs (w24 and up,
  6.5M–25.8M parameters, the seed-1 replicates included) default seed 1 had the lowest
  validation survival (8304; the smaller w24m12, w12 and w8 have 7854, 7059 and 4845)
  and the lowest tuner ψ₀ = 0 validation R²(10000) (0.78, against ≥ 0.89 for the other
  runs), and validation picked the better member of both replicate pairs. It cannot
  resolve differences of 0.03–0.05.

**Confounds.**

- **GPU node.** Both nodes are A100s and are assigned randomly. gpu1 trains at
  136–153 ms/step for every size (unroll 4), while gpu2 is about 3× slower for the 17.9M model.
  Compare times only within one node.
- **Checkpoint selection is nearly blind.** In most runs 8 of the 10 validation seeds
  reach t = 10000, so two seeds decide the pick. m24 and w32 seed 1 were picked at epoch
  24, under-trained, and have the lowest R² at t ≤ 2000 among the seven larger runs
  (m24: 0.942 at t = 2000; the smaller w12 and w8 are lower still, 0.937 and 0.772).
  The advisor paused changes to the selection (2026-10-08).
- **Step-size tuner.** Alpha sits at the bottom of the grid in 22 of 35 picks.

**Confirmation seeds for w24m8 (done 2026-10-16).** The two re-predictions (w24m12 and
w8) were done first; then the two w24m8 replicates were trained with seeds 1 and 2 (jobs
`utp_547746` and `utp_547747`):

```bash
for c in WIDTH=24,MODES=8,SEED=1 WIDTH=24,MODES=8,SEED=2; do
  sbatch --export=ALL,END=1000,$c utp.sh
done
```

Pass criteria and outcome for each w24m8 seed (the table and the discussion are under
*Model size* in *Current status*):

- R²(2000) ≥ 0.95: seed 0 passes (0.957), seed 1 is just below (0.944), seed 2 fails
  (0.910), which coincides with its under-trained epoch-22 pick;
- no divergence and no snapshot with max |ψ| > 1.1: all three seeds pass (0 of 1200
  rollouts diverged, max |ψ| ≤ 1.09);
- R²(10000) not below the default's lower seed (≥ 0.664 pooled; the default's two seeds
  gave 0.806 and 0.664): all three pass (0.824, 0.751, 0.706).

Not run, and optional now: w24 m12 seed 1 (`WIDTH=24,MODES=12,SEED=1`: does the big-box
instability belong to modes 12 or to the first training run?), w24 seed 1
(`WIDTH=24,SEED=1`), a third default seed (the seed-1 caches are gone) and w20 m8
(1.17M). Do not pin jobs to a node: nodes are assigned randomly (*Running from scratch
storage*).

### Running

On the cluster, in this order (usage is also at the top of each `.sh`). Steps (1) to (3)
are how the current set of runs was made and are done; (4) END = 2000 has not been run.
The size and seed variants are single `utp.sh` calls (*Model size*, *Training seed*).

```bash
python migrate_layout.py                  # once per copy: prints the plan ...
python migrate_layout.py --apply          # ... then moves (never deletes, never overwrites)

g=$(sbatch --parsable gt.sh)              # (1) ground truth to T_END=10000, ~12 h budget

# (2) retrain END = 1000 (VARIANT=lap by default; add VARIANT=base for the old
#     architecture, which re-archives the epoch-12 run train_t1000_pred_t10000).
#     The lap run has its own label train_t1000_lap_pred_t10000, so nothing
#     existing is superseded.
sbatch --dependency=afterok:$g --export=ALL,END=1000 utp.sh           #     -> T = 10000

# (3) existing base END = 300 and END = 500 checkpoints, label pred_t10000
sbatch --dependency=afterok:$g --export=ALL,END=300,VARIANT=base up.sh   # -> T = 10000
sbatch --dependency=afterok:$g --export=ALL,END=500,VARIANT=base up.sh   # -> T = 10000

# (4) END = 2000 (dataset needs ~17 GB of host RAM; ch_gpu.sh asks for 32G)
d=$(sbatch --parsable --export=ALL,END=2000 ch_gpu.sh)
sbatch --dependency=afterok:$d:$g --export=ALL,END=2000 utp.sh        #     -> T = 10000

# (5) full redraw of every run folder and the index (utp.sh and up.sh already end with
#     make_submission.py --overview, so this is only needed after changing the drawing code)
sbatch sp.sh                              # or: python make_submission.py
```

`utp.sh` is `ut.sh` and `up.sh` in one job; `ut.sh` and `up.sh` separately are
`sbatch --dependency=afterok:$d --export=ALL,END=1000 ut.sh` followed by
`--dependency=afterok:$t:$g --export=ALL,END=1000 up.sh`. The ordering rules
behind the chains above:

- **`gt.sh` goes first and needs no model**, so it runs alongside training and
  dataset jobs. One job with the default T_END = 10000 writes every ground-truth
  file that every training validation and every prediction needs.
- **A prediction must start after `gt.sh` has finished.** Otherwise it solves the
  files that are still missing itself, inside its own time limit.
- **Training waits for `gt.sh` too.** Its validation ground truth (10 seeds, l = 128,
  critical, to 10000) is exactly the set of files `gt.sh` writes. Two jobs solving
  the same file at once write separate partials
  (`<name>.<host>-<pid>.partial.npz`) and the last rename installs identical bytes,
  so an overlap is safe but solves those seeds twice. A `*.partial.npz` left by a
  killed job is incomplete and can be removed.
- **If `gt.sh` hits its time limit**, resubmit it, then point the jobs that were
  waiting on it at the new id (`scontrol update job=<id>
  dependency=afterok:<new_g>`): an `afterok` on a job that timed out is never
  satisfied, so those jobs would sit in the queue with `DependencyNeverSatisfied`.
- `ch_gpu.sh` exits 1 when the dataset already exists, which cancels an `afterok`
  chain. Leave it out when the file is already there.
- Training resumes from `Work/models/train_t<END>/uno_ch_state.pt`, and prediction
  resumes from the rollout cache, so a job that hits its wall clock can simply be
  resubmitted. Longer runs can raise `--time` on the `sbatch` line. Do not rerun
  END = 300: see the note in `ut.sh`.

(6) **Copy the results back** to the Mac and refresh the index:

```bash
for r in train_t1000_lap_pred_t10000 train_t1000_lap_w24m8_pred_t10000; do   # the labels you need
  rsync -a hpc:<project>/Results/$r Results/
  rsync -a hpc:<project>/Work/rollout_cache/$r Work/rollout_cache/   # what the figures are drawn from
  rsync -a hpc:<project>/Work/predict/$r Work/predict/               # results.json (report section 5)
done
python3 make_submission.py --overview     # Results/README.md + comparison images over every run
```

`--overview` redraws no run folder, which is what the copied `Results/<label>/` folders
already are. Plain `python3 make_submission.py` redraws them all as well.

(7) **Keep the HPC in step with the Mac** (2026-10-18, to do). Copy `make_submission.py`
to the HPC folder (it holds the snapshot palette, *Snapshot colours*), and move the seven
removed runs' folders there as well: `train_t1000_lap_{w24m12, w16, w12, w8, w12m12,
w24m8s1, w24m8s2}_pred_t10000` in `Results/`, `Work/rollout_cache/` and `Work/predict/`,
and `train_t1000_lap_<tag>/` in `Work/models/`. Otherwise the `--overview` step at the
end of each job lists them again.

Disk on the cluster: ~112 GB of ground truth at T = 10000 (save_every = 4, all 50
seeds at every box size), plus ~68 GB of datasets and memmaps (the t1000 and t2000
`.npz` files are 16.4 and 32.8 GB). A dataset can be deleted once its memmaps exist,
because training reuses memmaps whose frame count matches END. Once the 10000 files
exist, the old 2000-long files (21.5 GB) are redundant: prefix reuse serves the
2000 requests from the longer files.

**Running from scratch storage.** The home quota on the HPC is 200 GB, of which 184 GB
were used on 2026-10-15. Scratch has 1 TB (`/scratch/data/m25ph1006/chc_uno_1`). The
job scripts use relative paths (`source env/bin/activate`, `python uno_train.py`; the
Python files work relative to their own folder, so `Work/` and `Results/` follow the
copy), so the project runs from scratch unchanged, provided `env/` is **linked** from
the home copy rather than copied (`ln -s <home copy>/env env`). Check the scratch purge
policy before relying on it for anything that is expensive to rebuild.
`sbatch hpc_storage.sh` (or
`python3 hpc_storage.py`) reports the used and available space and the quota of each
location.

**Nodes are assigned randomly.** The GPU a job lands on is not under your control, and
it matters: the same 17.9M architecture trains in 31 min on gpu1 and 93 min on gpu2
(*Time cost* in *Current status*). Do not pin jobs to a node, which only delays them.
Compare times only within one node (`timing_table.py` prints it), and do not read a
node effect as a model effect.

Directly, without Slurm:

```bash
python ch_gpu.py --end 1000                        # -> Work/data/test_t1000.npz
python uno_train.py --end 1000                     # -> Work/models/train_t1000/
python uno_predict.py --gt-only --t-end 10000      # ground truth only, no model
python uno_predict.py --end 1000                   # prewarm, tune, roll out, diagnose, draw
python make_submission.py                          # redraw every complete run, no model
python make_submission.py --overview               # only Results/README.md + the comparison images
python make_submission.py --run train_t0300_pred_t10000
python make_submission.py --rollout --end 1000 --schedule-from auto
                                                   # refill one rollout cache, then draw it
```

**Regenerating `Results/`** needs only `Work/rollout_cache/`. `python
make_submission.py` redraws every run whose cache is complete (every planned case
rolled out, every seed file present, and `meta.json` marked `complete` once the last
case is done, so a run still tuning or killed mid-way is listed as not complete) into
`Results/<label>/`, then `Results/README.md`, `comparison.png` and
`comparison_t<END>.png`. `--overview` skips the run folders and writes only those last
files, which is why `utp.sh` and `up.sh` end with it. It overwrites only
the file names it writes and deletes nothing. `--schedule-from auto` reads
`Work/predict/<label>/results.json`, and only if its checkpoint fingerprint matches.
On the Mac (`python3`; MPS for the model, CPU for the float64 solver), `python3
uno_train.py --smoke` and `python3 uno_predict.py --smoke` are ~1-minute end-to-end
checks. They read the real dataset but write only under `Work/smoke/`, and before
migration they need `--npz test.npz` on the training call.

Key switches, all in `CONFIG` (or `PRECISION`) at the top of each file:

| where | knob | effect |
|---|---|---|
| train | `train_end` | END when `--end` is not given (300) |
| train | `PRECISION` | `"fp32"` / `"tf32"` / `"fp64"` — dtype for the whole network |
| train | `train_off` | quenches to train on; `None` = all five, `(0.0,)` = critical only |
| train | `laplacian_output` | `True` (default): the head predicts M and the increment is `lap5(M)` (*Laplacian output*); `--variant base` sets `False` |
| train | `width`, `modes`, `seed` | channel width (40), Fourier modes at the first level (20) and training seed (0); `--width`, `--modes` and `--seed` override them (*Model size*, *Training seed*) |
| train | `conserve_mass` | exact Model B projection; `False` is the ablation |
| train | `epochs`, `steps_per_epoch`, `unroll_stages` | length and unroll curriculum |
| train | `val_seeds`, `val_t1` | held-out trajectories used for checkpoint selection (10); `val_t1 = 0` validates to PAIRS[END] = 10000 |
| train | `select`, `select_min_delta` | `"survival"`: best checkpoint = longest mean interpolated survival time; epochs within `select_min_delta` (1.0 time unit) of it are separated by the in-range accuracy |
| train / predict | `diverge_abs` | a rollout seed diverges when non-finite or max \|ψ\| exceeds it (10); see *Diverged rollouts*. A cached seed file does not record the limit: after changing it, move the affected `Work/rollout_cache/<run>/` aside before predicting again |
| predict | `snapshots`, `long_snap_every` | main snapshot set (100, 500, 1000, 2000) and the spacing of the long set (2000: 2000, 4000, ..., 10000) |
| train / predict | `gt_require_cuda` | `"auto"` = CUDA required for ground truth inside a Slurm job |
| predict | `train_end`, `t_end` | END and T when not given on the command line; `t_end = 0` means PAIRS[END] = 10000 |
| predict | `eval_offs` | which quenches to report; defaults to all five, evaluated in one run |
| predict | `test_seeds`, `paper_seeds` | unseen seeds to evaluate (50 each, all ≥ 20000; the first 10 are the original set) |
| predict | `device` | `cuda`, else `mps`, else `cpu`; on MPS the float64 solver runs on the CPU |

`python uno_train.py --bench` times fp32/tf32/fp64 on the GPU you actually have.

### Seed budget

Three separate seed sets, with different limits:

* **training** — hard cap of **20**, fixed by `ch.py`'s `train_nens = 25` (80/20
  split). Raising it means regenerating `test.npz`, which costs ~196 MB per
  trajectory and breaks comparability with the other students' models. Past ~40
  trajectories `np.random.randint(0, 10000, N)` also starts repeating seeds
  (27 % chance of a collision at N = 80), so switch to `np.random.choice(...,
  replace=False)` if you do.
* **validation** (`val_seeds`, 10) and **test** (`test_seeds`, 50) — no practical
  cap on the GPU. Per test seed the ground-truth cache costs 31 MB at l = 128 and
  49 / 70 / 125 MB at l = 160 / 192 / 256, so fifty seeds across the five
  compositions and three transfer boxes is ~20 GB of cache to t = 2000 (~112 GB
  to 10000, ~225 GB to 20000). Solved in batches it takes minutes on an A100 at
  t = 2000 and roughly three hours on an M1 CPU, where float64 is memory-bandwidth
  bound and MPS cannot help.

Test seeds must be ≥ 20000 so they cannot collide with `ch.py`'s training
`[0,10000)` or validation `[10000,20000)` draws; this is asserted at load. The
tune seed is asserted not to be a test seed. Validation and test ground truth
share one cache, so the tune-seed trajectory solved during training is reused by
`uno_predict.py`.

`--smoke` on either gives a ~1-minute end-to-end check (useful on a login node
before submitting); it writes only under `Work/smoke/`. Multi-GPU is `torchrun
--nproc_per_node=N uno_train.py --end N`; training auto-resumes from
`Work/models/train_t<END>/uno_ch_state.pt`, so a job that hits a wall clock limit
can simply be resubmitted.

### What `uno_train.py` does

One epoch loop with an **unroll curriculum**: it starts on the one-step map and
lengthens the unroll (1 → 2 → 4 → 8) at fixed fractions of the run, dropping the
learning rate at each stage. Early epochs learn the flow map; later ones train the
model on its own output distribution, which is what stops autoregressive drift.

Checkpoints are selected on a **real 128-box rollout to t = PAIRS[END] = 10000**
(the horizon every model is evaluated on) across the `val_seeds` trajectories — not
on one-step accuracy, which actively fights rollout stability (§4), and not on a
single trajectory, which is noisy.

**Selection by survival, not by mean R².** The first rule scored an epoch by the
unclipped mean R² over every rollout time in [val_t0, val_t1]. With val_t1 = 10000
about 80 % of those rows are late-time rows where every model has long since
decayed, so that mean rewards a damped, smooth model: for END = 1000 it picked the
under-trained epoch 12 (survival 424) while epoch 60 survives ~1800
(`Work/analysis/t1000_failure_2026-10-06/`). The rule is now, for every END:

1. `eval_rollout` returns, per validation seed, the survival time with **linear
   interpolation** of the first crossing of `val_threshold` (0.80) between the two
   rollout times around it (a seed that never drops below it counts t[-1], a
   non-finite rollout fails at that step), and the **in-range accuracy** A, the
   mean over seeds and rollout times t ≤ min(2·END, val_t1) of R² clipped to [0, 1].
2. The best epoch has the highest mean interpolated survival. A later epoch replaces
   it when its survival exceeds the best by more than `select_min_delta` (1.0 time
   unit), or is within `select_min_delta` of it and has the higher A.
3. The log line per validation is `rollout <old mean R2> | in-range R2 <A> (t <=
   2·END) | survives t=<mean> [<per seed>]` then `<- best` or `(no gain k/cap)`;
   the old mean is for reference only. `[done]` reports the selected epoch with its
   survival and A, and `best.pt` carries `metrics = dict(select="survival", survives,
   per_seed, r2_in_range, in_range_t1, rollout, epoch, val1)`, which `report.txt`
   shows in the model card.
4. `select` and `select_min_delta` are part of the resume stamp, so a state trained
   under the old rule starts fresh and the old `best.pt` is kept as
   `uno_ch_best_prev.pt`.

Training stops early when the selection has not improved for `patience`
validations; `patience = 0` disables stopping entirely, and the no-gain counter is
still logged so the epoch it would have fired at can be read off the log afterwards.

**Not fixed by this change.** The l = 128 late-time collapse (the box-transfer
defect in the 8 longest-wavelength modes, `Work/analysis/t1000_failure_2026-10-06/`)
is a property of the architecture, not of the checkpoint choice. Survival-based
selection stops picking a damped model, but it does not make the l = 128 box
survive to 10000. *(Update 2026-10-15: the Laplacian output, §6 *Laplacian output*,
removed the defect. With the base head the survival-selected END = 1000 model still
scores a critical-quench R²(10000) of 0.010, with the Laplacian output 0.903.)*

**Selection is saturated.** In most runs 8 of the 10 validation seeds reach
t = 10000, so two seeds decide the pick, and the tuner's alpha sits at the bottom of
its grid in 22 of 35 picks. The advisor paused changes to the selection on 2026-10-08;
see the confounds under *Training seed*.

**The counter has no stage awareness.** It is reset only by an improving
validation, and the stop test carries no stage term, so early stopping *can* fire
before the curriculum reaches its final unroll. With the shipped schedule that
never happens — in the September run the score improved at every validation up to the
peak at epoch 18, so `patience = 4` would first fire at epoch 26, eight epochs past the
peak. But a
schedule that delays the final stage (for example unroll 4 from epoch 20 rather
than 12) leaves enough room for a plateau in the earlier stages to end the run
before the final stage begins. Use `patience = 0` for such a run, or widen
`patience`, and check the log.

### Which quenches are trained on

`CONFIG["train_off"] = (0.0,)` trains on the **critical quench only** — `ch.py`
hands out `off_values` in contiguous blocks, so this selects training
trajectories 8–11 and validation trajectory 2. Set it to `None` for all five, or
to any subset. The ψ → −ψ augmentation is an exact symmetry of CHC but maps
off → −off, so it is enabled only when the selected set is closed under negation;
otherwise it would quietly train on quenches you asked to exclude.

### A100 notes

* The dataset lives in **GPU memory** (`data_on_gpu`) — 196 MB for the critical
  quench — so batch assembly is a device-side gather and the host leaves the inner
  loop entirely.
* At batch 32 an A100 measured 40 ms/step, which is launch-latency bound rather
  than compute bound. Defaults are therefore batch 128 (phase 1) / 64 (phase 2),
  `width=40`, `modes=20`, and `grad_steps=4` — the 40 GB affords real gradient
  through more of the unroll. Run `python uno_train.py --bench` to re-measure on
  your own card before changing them.
* `modes` 16 → 20 raises the spectral cutoff from k = 1.57 to 1.96. The interface
  is ~1.4 lattice units wide, so that extra band is interface curvature — which is
  what sets the coarsening-rate bias in §5.
* `PRECISION` (fp32/tf32) is a hard switch at the top of `uno_train.py`; `--bench`
  reports both on the hardware you actually have.
* Measured in the 2026-10 size sweep: on gpu1 every size from 0.74M to 25M parameters
  trains at 136–153 ms/step at unroll 4, so shrinking the model does not make training
  cheaper there (*Time cost* in *Current status*).

---

## 7. Notes on the setup

* **Ground truth is verified against `ch.py` itself.** The GPU solver reproduces the
  cached training data bit-exactly. Doing so surfaced that `ch.py` writes `data[i]`
  *after* stepping, so stored frame `i` is t = 0.1·i + **0.01**, not 0.1·i. Frame
  *differences* — the only thing the training Δt depends on — are unaffected.
* **Test seeds are genuinely unseen.** `ch.py` draws training seeds from [0, 10000) and
  validation seeds from [10000, 20000). Test seeds are ≥ 20000, asserted at load time.
  Re-checked on 2026-10-15 in the rollout caches, the `meta.json` files and
  `ch_gpu.py plan()`: the 50 test seeds are disjoint from the 10 validation / tuning
  seeds and from the l = 64 training-data seeds.
* **Schedule tuning is held out.** The stride schedule is selected on the ten
  `tune_seeds` — the same trajectories training used for validation — scored as
  the mean R² on a **fixed** time grid (T/4, T/2, 3T/4, T: t = 500, 1000, 1500,
  2000 for T = 2000) so schedules of
  different length stay comparable. A disjointness assert fails the run if any tune
  seed appears in `test_seeds`. The tuner warns when the winner sits on the edge of
  the search grid, which it does: longer strides keep winning because they compound
  the per-step bias fewer times, so the grid must not be widened past the `dt_max`
  the model was trained on. (In the 2026-10 runs alpha is at the bottom of the grid,
  alpha = 0.1, in 22 of 35 picks; this is an open issue, not a fix.)
* **Python version.** Both files are syntax-verified on **3.9, 3.10, 3.11, 3.12
  and 3.14**, so they run on whatever the cluster module provides. This matters:
  a multi-line expression inside an f-string is legal from 3.12 but a SyntaxError
  before it, and a laptop on 3.14 will happily compile code the cluster rejects.
  `python3 check_code.py` compiles every file and flags the 3.12-only f-string
  forms, so version skew is caught before the queue wait; every job prints
  `python --version`. The dataset itself is
  version-independent — NumPy's legacy `RandomState` is stable, and the solver
  reproduces the cached `test.npz` frames to 0.0.
* `test.npz` — now `Work/data/test_t<END>.npz` — is the *training* set despite the
  name (that is what `ch.py`'s `fname` is set to); the l = 128 test trajectories
  are generated separately.
* **Multi-GPU** was validated with 2 ranks: the sampled unroll length is drawn from
  an RNG seeded identically on every rank, because DDP deadlocks if ranks disagree
  on how many forward/backward calls to make. Verified that the unroll sequence
  matches across ranks and that weights stay bit-identical after gradient sync.
* The earlier multi-file `legacy/` version is not part of this copy of the project.
  `Archive/code_2026-09-29_before_next_phase/` holds the code and docs as they were
  before the current layout.
