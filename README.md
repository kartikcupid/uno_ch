# UNO for the Cahn–Hilliard equation — autoregressive rollout to t = 2000

Neural-operator surrogate for the dimensionless Cahn–Hilliard–Cook equation

$$\partial_t\psi = \nabla^2\left(\psi^3 - \psi - \nabla^2\psi\right)$$

(Puri, *Kinetics of Phase Transitions*, Eq. 1.182 — exactly what `ch.py` integrates).

Protocol, per the project brief: train on the dataset `ch.py` produces, then predict
**unseen** trajectories on a **128×128** lattice, starting at **t = 100** and rolling
autoregressively to **t = 2000** in real time, scored by **R²** plus snapshot
comparison. The rollout runs the full horizon; the **0.80** threshold is reported
(the time R² first crosses it, per seed and for the mean) rather than used to
truncate.

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
inference-time knob tuned on held-out data rather than a baked-in choice.

**Mass conservation by construction** (`CONFIG["conserve_mass"]`, default `True`).
Cahn–Hilliard is Model B: the right-hand side is a divergence, so on a periodic box
⟨ψ⟩ is a constant of motion. The network predicts an increment whose spatial mean is
subtracted exactly, making conservation a property of the architecture rather than a
penalty term. Measured drift is ~1e-9 per step and ~1e-8 over the whole ~50-step
rollout. Turning it off is an ablation, not an option: with a random head, drift goes
from 3e-9 to 4e-3 per step and **accumulates linearly** to 0.22 over 50 steps — enough
to move ⟨ψ⟩ from a critical quench to an off-critical one, changing the morphology
itself. The flag is recorded in the checkpoint's `arch`, so `uno_predict.py` rebuilds
the model exactly as trained.

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
grid.

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

**Speed:** 67× like-for-like against the explicit dt = 0.01 solver (both unbatched),
203× when the surrogate rolls ten trajectories at once. Quote the first.

This figure is **not stable between runs.** Three measurements of the same quantity
on the same A100 gave 96× / 56× / 67× like-for-like (690× / 170× / 203× batched);
all three are still in `Archive/Results_2026-09-08/uno_ch_predict.log` (formerly
`Results/`), which is append-mode. Two
things move it: the per-quench stride tuning selects a different number of
autoregressive steps each run (51–66 here), and per-step wall time varies by ~33 %
between nodes (7.8 vs 10.4 ms). Quote what the current run prints, not the best
figure ever seen. Both numbers are against the explicit dt = 0.01 scheme, not a
semi-implicit spectral solver, which would need far fewer steps.

### Five runs, and what they settle

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

A controlled pair from 2026-08-25: fp32, 60 epochs, identical stage timing, identical
10 validation seeds, identical selection rule. Its ON column is the fourth run in the
table above, **not** the §3 run — so its per-quench numbers differ slightly from §3's.
The pair is what matters here, and both halves of it were measured together. The only difference is the projection at
[uno_train.py:527](uno_train.py:527).

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
cost 3×. Do not attribute it to the constraint.

---

## 3a. One run covers every quench

`eval_offs` defaults to `(-0.4, -0.2, 0.0, 0.2, 0.4)`, so a single
`python uno_predict.py` produces the full t = 100 → 2000 rollout for **every**
composition `ch.py` supplies — no editing and resubmitting per quench. Each one
gets its own tuned stride schedule, its own case in the rollout cache
(`psi0m0.40` … `psi0p0.40`), its own figures in `Results/<label>/` and its own
block in `Work/predict/<label>/results.json`, followed by one table across every
case (the example below is from the older 10-seed version of that table):

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
cache grows by ~2.5 GB.

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
L = 160/192/256, ~12 GB of cache to t = 2000. The solver is launch-bound at these
sizes, so L = 256 costs about the same per trajectory as L = 128. Raise
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

Two consequences:

* Any rollout error is **systematic model bias**, not noise and not chaos. That is
  exactly what unrolled ("pushforward") training removes, and it is why one-step
  accuracy and rollout stability were observed pulling in opposite directions:
  one-step R² at Δt = 40 plateaued at 0.9996 while the t = 2000 rollout *degraded*.
* R² > 0.80 at t = 2000 is a fair target, so model selection should be done on the
  real objective. Training therefore evaluates a full 128-box rollout to t = 2000 on
  a held-out trajectory at every checkpoint, rather than on a short 64-box proxy that
  saturates at 0.999 and stops discriminating.

---

## 5. The binding constraint is the length of the training trajectories

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
the intrinsic dynamics impose no limit at all (§3). The next dataset extends it to
**t = 1000 and t = 2000** (`ch_gpu.py --end`), evaluated to t = 10000 as
runs `train_t1000_pred_t10000` and `train_t2000_pred_t10000` beside
`train_t0300_pred_t02000` and `train_t0300_pred_t10000` (§6).

A 64² box stays usable well past t = 500: solving the exact dynamics on 64², 128²
and 256² boxes to t = 2000, the 64-box's real-space domain length tracks the
256-box to within 3.4 %. (A −20 % gap appears in the `moment` estimator, but that is
the estimator running out of k-resolution on the coarse 64-box grid, not physics.)

---

## 6. Files

| file | role |
|---|---|
| `ch_gpu.py` | dataset generator: `ch.py`'s dynamics, seeds and frame selection, batched in float64 on the GPU. `--end N` writes `Work/data/test_t<N>.npz` plus a `.json` sidecar (device, dtype, torch version, shapes, timing). It refuses to overwrite an existing file. `--verify` checks it against a numpy transcription of `ch.py`'s loop |
| `uno_train.py` | **training**: CH solver, ground-truth cache, UNO, epoch loop with unroll curriculum, survival-based checkpoint selection. `--end N [--variant lap or base] [--npz PATH] [--smoke] [--bench]`. The default `lap` variant predicts M and applies the 5-point Laplacian (see *Laplacian output* below). It also holds the path helpers and the low-k gain (`lowk_gains`) the other scripts import |
| `uno_predict.py` | **prediction**: ground-truth prewarm, per-quench stride tuning, 50-seed rollouts into the rollout cache, diagnostics (transfer, predictability, drift, low-k gain, speedup), `results.json`, then the drawing of this run. `--end N [--variant lap or base] [--t-end T] [--gt-only] [--smoke] [--ckpt PATH]` |
| `make_submission.py` | **drawing**: `Results/` from the rollout caches alone, with no model and no torch. `--rollout` refills a cache first |
| `migrate_layout.py` | one-off move of the pre-2026-09-29 layout into the one below. Dry run by default; `--apply` does the moves |
| `check_code.py` | static checks: compile, ≤ 88 columns, Python-3.11 f-string rules, no comments in `uno_train.py` / `uno_predict.py` |
| `gt.sh` `ch_gpu.sh` `ut.sh` `up.sh` `utp.sh` | Slurm jobs: ground truth only; dataset; train; predict + draw; train + predict + draw. `END`, `T_END` and `VARIANT` (`lap` default, `base` for the old architecture) go in through `--export`, and the usage is at the top of each script. Every job prints `python --version` |

Every setting lives in a `CONFIG` dict at the top of each Python file, so a run is
reproducible from those files alone. `uno_predict.py` imports the model and the
solver from `uno_train.py`. The numpy-only physics helpers live in
`make_submission.py`, so drawing never imports torch.

### Runs, labels and where things live

A run is a model trained on data up to t = END and rolled out to t = T. Its label is
`train_t<END:04d>[_lap]_pred_t<T:05d>`, for example `train_t0300_pred_t10000` (base
model) or `train_t0300_lap_pred_t10000` (Laplacian-output model). **Every
model is predicted to T = 10000**: `PAIRS` in `uno_train.py` maps 300, 500, 1000
and 2000 to 10000, and `default_t_end` returns 10000 for any other END. Training
validates to the same horizon (`val_t1 = 0` means PAIRS[END]), so every END
validates against ground truth to 10000. The existing 2000-long runs
(`train_t0300_pred_t02000`, `train_t0500_pred_t02000`) stay as they are and are
joined by `_pred_t10000` runs of the same checkpoints.

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

```
Results/                        report material only; make_submission.py regenerates it
  README.md, comparison.png     index of every run; critical mean R²(t) of every run
                                (log t, drawn once there are two or more runs)
  <label>/                      report.txt, manifest.json, composition.csv, composition_all.png,
                                r2_sweep.png/.pdf and per case snapshots_ and
                                snapshots_long_ (BEST seed), phase_ordering_kinetics_,
                                domain_growth_, psi_distribution_, composition_
                                (seed AVERAGES)
  train_t0300_pred_t02000/earlier_run_2026-09-08/
                                the old kinetics and R² sweep graphs + NOTE.md; the
                                drawing code never touches this folder
Work/                           heavy and machine-side, never part of the report
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
                                intact by migrate_layout.py), code backups,
                                superseded/<label>_<old fingerprint>_<time>/ (see below)
```

**Superseded runs.** A new checkpoint trained for the same END has a different
fingerprint from the one a label's rollout cache was built from, and mixing two
models in one cache would corrupt every figure. The rollout engine therefore
moves (never deletes) `Work/rollout_cache/<label>`, `Work/predict/<label>` and
`Results/<label>` into `Archive/superseded/<label>_<old fingerprint>_<YYYYmmdd-HHMMSS>/`
(`rollout_cache/`, `predict/`, `Results/`), logs each move and carries on with a
fresh cache. Retraining END = 1000 with the new selection does this to the
epoch-12 `train_t1000_pred_t10000`. A `--smoke` run archives under
`Work/smoke/Archive/superseded/`. `--schedule-from` keeps its own fingerprint check.
The old model files are rotated by the trainer (`uno_ch_best_prev.pt`,
`uno_ch_state_stale.pt`).

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
line (`lowk g <below>/<first>@<t0>`); it is not used for checkpoint selection.

**Risk.** ∇² amplifies grid-scale wiggles of the head output by up to 8×. If the
network's output is noisy, interfaces can come out rough early in training, and the
learning rate or an output scale may need to change. Watch interface sharpness in
`snapshots_*` and the first epochs' `val1`. This is the main unknown and needs the
END = 500 run to judge.

**Variants.** `--variant lap` (default) and `--variant base` on `uno_train.py`,
`uno_predict.py` and `make_submission.py --rollout` (and `VARIANT=lap|base` in
`ut.sh` / `up.sh` / `utp.sh`) pick the architecture. The lap model lives in
`Work/models/train_t<END>_lap/` and its run label is
`train_t<END>_lap_pred_t10000`; the base runs keep their old labels and paths
(`train_t<END>`, `train_t<END>_pred_t10000`) and stay in `Results/` for comparison
(`README.md` and `comparison.png` list them as separate, labelled rows and
curves). `uno_predict.py` refuses a checkpoint whose `arch.laplacian_output` does
not match `--variant`; an old checkpoint without the key builds the base model
unchanged, bit for bit. Retraining the lap model costs about 0.5 h per END on an A100.

Job order (ground truth already exists, so no `gt.sh` dependency):

```bash
for E in 300 500 1000; do sbatch --export=ALL,END=$E utp.sh; done   # VARIANT=lap
d=$(sbatch --parsable --export=ALL,END=2000 ch_gpu.sh)             # if not built yet
sbatch --dependency=afterok:$d --export=ALL,END=2000 utp.sh
sbatch sp.sh                                                        # index + comparison.png
```

### Running

On the cluster, in this order (usage is also at the top of each `.sh`):

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

# (5) redraw everything and the index
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

(4) **Copy the results back** to the Mac and redraw the index:

```bash
for r in train_t0300_pred_t10000 train_t0500_pred_t10000 train_t1000_pred_t10000 train_t2000_pred_t10000; do
  rsync -a hpc:<project>/Results/$r Results/
  rsync -a hpc:<project>/Work/rollout_cache/$r Work/rollout_cache/   # what the figures are drawn from
  rsync -a hpc:<project>/Work/predict/$r Work/predict/               # results.json (report section 5)
done
python3 make_submission.py                # README.md + comparison.png over every run
```

Disk on the cluster: ~112 GB of ground truth at T = 10000 (save_every = 4, all 50
seeds at every box size), plus ~68 GB of datasets and memmaps (the t1000 and t2000
`.npz` files are 16.4 and 32.8 GB). A dataset can be deleted once its memmaps exist,
because training reuses memmaps whose frame count matches END. Once the 10000 files
exist, the old 2000-long files (21.5 GB) are redundant: prefix reuse serves the
2000 requests from the longer files.

Directly, without Slurm:

```bash
python ch_gpu.py --end 1000                        # -> Work/data/test_t1000.npz
python uno_train.py --end 1000                     # -> Work/models/train_t1000/
python uno_predict.py --gt-only --t-end 10000      # ground truth only, no model
python uno_predict.py --end 1000                   # prewarm, tune, roll out, diagnose, draw
python make_submission.py                          # redraw every complete run, no model
python make_submission.py --run train_t0300_pred_t10000
python make_submission.py --rollout --end 1000 --schedule-from auto
                                                   # refill one rollout cache, then draw it
```

**Regenerating `Results/`** needs only `Work/rollout_cache/`. `python
make_submission.py` redraws every run whose cache is complete (every planned case
rolled out, every seed file present, and `meta.json` marked `complete` once the last
case is done, so a run still tuning or killed mid-way is listed as not complete) into
`Results/<label>/`, then `Results/README.md` and `comparison.png`. It overwrites only
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
| train | `conserve_mass` | exact Model B projection; `False` is the ablation |
| train | `epochs`, `steps_per_epoch`, `unroll_stages` | length and unroll curriculum |
| train | `val_seeds`, `val_t1` | held-out trajectories used for checkpoint selection (10); `val_t1 = 0` validates to PAIRS[END] = 10000 |
| train | `select`, `select_min_delta` | `"survival"`: best checkpoint = longest mean interpolated survival time; epochs within `select_min_delta` (1.0 time unit) of it are separated by the in-range accuracy |
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
survive to 10000.

**The counter has no stage awareness.** It is reset only by an improving
validation, and the stop test carries no stage term, so early stopping *can* fire
before the curriculum reaches its final unroll. With the shipped schedule that
never happens — the score improves at every validation up to the peak at epoch 18,
so `patience = 4` would first fire at epoch 26, eight epochs past the peak. But a
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

---

## 7. Notes on the setup

* **Ground truth is verified against `ch.py` itself.** The GPU solver reproduces the
  cached training data bit-exactly. Doing so surfaced that `ch.py` writes `data[i]`
  *after* stepping, so stored frame `i` is t = 0.1·i + **0.01**, not 0.1·i. Frame
  *differences* — the only thing the training Δt depends on — are unaffected.
* **Test seeds are genuinely unseen.** `ch.py` draws training seeds from [0, 10000) and
  validation seeds from [10000, 20000). Test seeds are ≥ 20000, asserted at load time.
* **Schedule tuning is held out.** The stride schedule is selected on the ten
  `tune_seeds` — the same trajectories training used for validation — scored as
  the mean R² on a **fixed** time grid (T/4, T/2, 3T/4, T: t = 500, 1000, 1500,
  2000 for T = 2000) so schedules of
  different length stay comparable. A disjointness assert fails the run if any tune
  seed appears in `test_seeds`. The tuner warns when the winner sits on the edge of
  the search grid, which it does: longer strides keep winning because they compound
  the per-step bias fewer times, so the grid must not be widened past the `dt_max`
  the model was trained on.
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
