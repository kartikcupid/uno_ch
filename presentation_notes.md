# Presentation notes — "What work am I doing?"

For the UNO / Cahn–Hilliard project. A few versions at different lengths, plus slide bullets — pick what fits your time slot, or blend them. Updated 2026-10-18; every number comes from `Work/analysis/findings_2026-10-15.md`, including its addenda of 2026-10-16 (re-predictions, the width-12 12-mode run, the 1.68M-model seed replicates) and 2026-10-18 (runs removed from the drawn results; see "Numbers to quote" at the end). The earlier version of these notes (target t = 2000, "R² above 0.80 for most of the 1900-step rollout", "~96× faster") is in `Archive/docs_2026-10-15_before_findings_update/`; the version before the 2026-10-16 addendum is in `Archive/docs_2026-10-16_before_addendum/`, and the version before the 2026-10-18 update is in `Archive/docs_2026-10-18_before_runs_removed/`.

## One sentence

I'm training a neural network to act as a fast stand-in for a physics simulation of phase separation — trained only on small 64-by-64 boxes up to time 1000 — and testing whether it still gets the physics right on boxes up to four times wider and for ten times longer than it ever saw.

## ~30 seconds

My project is to build a fast machine-learning surrogate for the Cahn–Hilliard equation — the PDE that describes how a mixture separates into two phases over time, like oil separating from water. I train a neural operator on small 64-by-64 simulations, only up to time 1000, then let it roll forward on its own, each prediction feeding the next, from time 100 all the way to time 10000 — on 128-by-128 boxes and on boxes up to 256-by-256 that it has never seen. What made the long horizon work was changing the output: the network predicts a field, and the update is that field's Laplacian, the same stencil the solver uses, so the total composition is conserved exactly. I borrowed that idea from earlier papers, I didn't invent it. With it, the even-mix case scores R² of 0.90 at time 10000 against the exact solver, where the old output gives 0.01 when trained the same way. And it runs an estimated 70 to 100 times faster than the explicit solver, like for like.

## ~90 seconds (more technical — good for the advisor)

Concretely, I'm training a neural operator — a U-Net combined with a Fourier-based architecture, so it's called a UNO — on Cahn–Hilliard simulations on a 64-by-64 grid, only up to t = 1000. The test is on boxes it has never seen: 128-by-128, plus 160, 192 and 256 per side, where 256 is sixteen times the training area. Starting from t = 100, it rolls forward autoregressively to t = 10000, scored by R² against the exact solver over 50 unseen test seeds in each of eight cases: four off-critical compositions, the critical quench (mean composition zero), and the three big boxes.

That's not just upscaling. Making the box bigger at the same resolution changes which physical wavenumber each part of the network is responsible for, and the standard architecture gets that silently wrong. I fixed that, along with an ill-posed inverse FFT that the CPU and GPU resolved differently, and I train with rollout-aware unrolling so the model learns to correct its own compounding errors.

The newest piece is the output. Training out to t = 1000 first failed in two stacked ways. The checkpoint score was late-heavy and picked a damped, under-trained epoch-12 model, which I retired. And with the old output, going from 64 to 128 the lowest wavenumbers moved 2.4 to 3.5 times too far per step: even after I fixed the checkpoint selection, the old output scores R² of 0.01 on the critical quench at t = 10000. The fix is that the network predicts a field M and the increment is its 5-point Laplacian, the stencil the solver uses. That makes mass conservation exact for any time step, and it suppresses the lowest wavenumbers by k². To be clear, this output form is not new: Fantasia et al. 2025 use the Laplacian of a scalar field in a convolutional recurrent network, Lanzoni et al. 2024 output a flux through a fixed divergence layer, in 3-D, and Yadav et al. 2026 work in the same setting. [Note to self: the first author of that last paper is Vijay Yadav, the same name as the advisor. Check whether it is his own paper, and say so if it is.] Critical-quench R² at t = 10000 goes 0.770, 0.828, 0.903 for training to t = 300, 500, 1000, against 0.623, 0.176, 0.010 for the old output. All six pick their checkpoint by validation survival, so the comparison is the output form alone.

Across all eight cases the default model's pooled R² is 0.96 at t = 2000, 0.90 at t = 5000 and 0.81 at t = 10000; the big boxes alone are 0.83 at t = 10000. On speed: against the explicit dt = 0.01 solver it is an estimated 72 to 99 times faster one trajectory at a time, and hundreds of times (207 to 869) when seeds are batched; the solver time is extrapolated from a 2000-step probe. That figure moves from run to run, so I quote the range, not a single number.

## Model size — the advisor's question

The question was how small the model can be. I trained eleven sizes, once each, to t = 1000 with the Laplacian output and scored all of them the same way at t = 10000 (the table shows the main ones), and then trained two more seeds of the 1.68M model. The sweep is complete. The runs of width 16, width 12, width 8, width 24 with 12 modes and width 12 with 12 modes, and the two extra seeds of the 1.68M model, are shown as numbers only: their figures were removed from the results on 2026-10-18.

| Model | Parameters | R²(2000) | R²(10000) | Verdict |
|---|---|---|---|---|
| width 48 | 25.76M | 0.964 | 0.768 | good |
| default (width 40, 20 modes) | 17.90M | 0.960 | 0.806 | reference |
| width 32 | 11.47M | 0.958 | 0.837 | good |
| width 24 | 6.47M | 0.961 | 0.828 | good |
| width 16 | 2.89M | 0.948 | 0.713 | borderline |
| width 24, 12 modes | 2.86M | 0.950 | 0.684 | unstable, 147 seeds diverged |
| **width 24, 8 modes (seed 0)** | **1.68M** | **0.957** | **0.824** | **good** |
| width 24, 8 modes (mean of 3 seeds) | 1.68M | 0.937 | 0.761 | stable on all three |
| width 12 | 1.63M | 0.937 | 0.705 | worse, late overshoot |
| width 8 | 0.74M | 0.772 | 0.291 | fails, 25 diverged |
| width 12, 12 modes | 0.73M | 0.900 | 0.115 | fails, 236 diverged |

The three rows with diverged seeds (width 24 with 12 modes, width 8, width 12 with 12 modes) are means over the seeds that did not diverge; the diverged ones are counted in the verdict column and left out.

A 1.68M-parameter model (width 24, 8 Fourier modes in the first level) matches the 17.9M default at long times, and it is 10.7 times smaller. That now holds over three training seeds. For seed 0 the paired differences against the default are −0.003 ± 0.002 at t = 2000, −0.003 ± 0.005 at t = 5000 and +0.019 ± 0.008 at t = 10000; it also had the best validation survival of any run (9491; its in-range R² of 0.985 is just below the width-48 model's 0.986). Over the three seeds R²(10000) is 0.824, 0.751 and 0.706 (mean 0.761, SD 0.060), against 0.806 and 0.664 for the default's two seeds (mean 0.735); R²(2000) is 0.957, 0.944 and 0.910 (mean 0.937) against 0.960 and 0.957. None of the 1200 rollouts diverged and the largest |ψ| is 1.09. The early-time caveat: at t ≤ 2000 it is about 0.02 lower than the default on average, and only seed 0 reaches R² of 0.95. Seed 2's checkpoint was picked at epoch 22 (validation in-range R² 0.942, against 0.979 for its epoch-60 weights), the same under-trained pattern as the 25M 24-mode model and width-32 seed 1, so part of the early gap is checkpoint selection, not capacity. The advisor paused changes to the selection, so that evidence is one to raise with him.

Cut Fourier modes, not width (still provisional: the width-12 model has one seed). At about 1.7M parameters the 8-mode model (seed 0) beats the width-12 model by +0.163 ± 0.012 in R²(10000) on the l = 128 cases, but the three 8-mode seeds, pooled over all cases, reach down to the width-12 model's 0.705, so I would not quote that margin for every training seed. At 20 modes quality drops steadily once width goes below 24. Too-small models fail by overshoot and blow-up: width 12 overshoots late (138 of 400 test rollouts exceed max |ψ| 1.1, the truth peaks at 1.043), and width 8 degrades everywhere, with 25 of its 400 rollouts diverging. The width-12, 12-mode model (0.73M) fails badly: 236 of 400 rollouts diverge, including all the big-box ones and 45 of the 50 critical ones, and its R² is 0.900 at t = 2000 but 0.115 at t = 10000 over the survivors. More epochs would not rescue the small ones; they plateau by about epoch 40.

The width-24, 12-mode model (2.86M) is the surprise. After I re-predicted it with the per-seed divergence handling, 147 of its 150 big-box rollouts blow up sooner or later (47 of 50 at l = 160, 50 of 50 at l = 192 and at l = 256), while none of its l = 128 rollouts diverged (pooled R² at t = 10000 of 0.684 over the survivors, critical quench 0.802). The earlier count of 51 was an undercount: the old code cut whole chunks at the first blow-up, and the seeds it had cut short mostly blow up later anyway, so the truncation had hidden how widespread the failure is. This checkpoint is unstable in boxes larger than 128. Whether that is systematic for 12 modes or specific to this one training run I cannot say; it would need a seed replicate, which I have not run and which is optional now.

So the sweep says: nothing under 1M parameters works, and below 2.9M only the 1.68M model (width 24, 8 modes) is both accurate and stable. Stability in boxes larger than 128 has to be checked as a pass criterion of its own, and it is not monotone in size: the 2.86M model is unstable where the smaller 1.68M one is not.

Smaller does not save HPC training time: on the faster node every size from 0.74M to 25M trains in 26–32 minutes, and shrinking 17.9M to 6.5M saves 9%; which node the job lands on matters more than the size. Where it pays is inference on a laptop: per step at l = 128 the 1.68M model takes 34 ms on a CPU against 146 ms for the default (4.3 times faster), and 16 against 49 ms on the Apple GPU (3 times).

The caveat: every size except the 1.68M model rests on one training seed. Retraining the same model moves late-time R² by about 0.04–0.08 (0.060 across the three 1.68M seeds), so +0.019 is "as good as", not "better". The width-24, 12-mode and width-8 rows were re-predicted on 2026-10-16 after the rollout-divergence bug fix, so those rows are final for the one seed they have. The confirmation seeds for the 1.68M model are done. A second seed of the width-24, 12-mode model, to see whether its big-box instability is systematic, and a second seed of width 24 were planned and are optional now.

## Honest caveats

- **Seed noise.** Retraining the same model changes pooled R²(10000) by a standard deviation of about 0.04–0.08. Default seed 0 scores 0.806 and seed 1 scores 0.664; width 32 goes 0.837 to 0.778. Seed 1 was worse in all 8 cases in both pairs. The test seeds are not the problem (standard error about 0.008). Up to t = 2000 runs agree within 0.01–0.03 (except where an under-trained checkpoint was picked); the noise appears after t ≈ 4000.
- **Single seeds.** Most configurations have one training run; the 1.68M model has three. An earlier "smaller is better" result (+0.031 ± 0.007) was within seed noise, and averaged over two seeds each, width 32 against the default is +0.07 ± 0.08, so no size effect is detectable either way.
- **Selection.** Validation catches gross failures but cannot resolve differences of 0.03–0.05. Eight of the ten validation seeds reach t = 10000 in most runs, so two seeds decide the checkpoint. The advisor paused changes to selection on 2026-10-08. Seed 2 of the 1.68M model (epoch 22, in-range R² 0.942 against 0.979 at epoch 60) is a fresh example of the pick being under-trained.
- **Speedup depends on the comparison.** It is an estimate against the explicit dt = 0.01 solver (about 210–250 s per trajectory, extrapolated from a 2000-step probe): 72–99 times one trajectory at a time, 207–869 times batched, depending on step count and node. It does not track model size and is unstable between runs, so I do not quote it from a single run.
- **The main idea is borrowed.** The Laplacian output is from Fantasia et al. 2025 (Lanzoni et al. 2024 have a related flux-divergence output). I did not find the k² explanation of why the old output fails at long horizons in the papers I checked, but several of those checks were at abstract level only, so I would not claim it as new.

## Slide bullets

The snapshot PNGs in `Results/<run>/` (`snapshots_<case>.png` and `snapshots_long_<case>.png`) now use the deck's colours and can be dropped into the slides directly.

- Goal: fast neural-network surrogate for the Cahn–Hilliard (phase-separation) equation
- Train on 64×64 simulations up to t = 1000; test on unseen boxes from 128×128 up to 256×256, autoregressive rollout t = 100 → 10000
- Core challenge: domain extension ≠ super-resolution — same physics, bigger box, which breaks standard spectral neural operators
- Key fix: Laplacian output (borrowed from Fantasia 2025; related flux-divergence output in Lanzoni 2024) — exact mass conservation, low wavenumbers damped by k²; critical-quench R²(10000) 0.90 vs 0.01 for the old output
- Pooled R² 0.96 / 0.90 / 0.81 at t = 2000 / 5000 / 10000 over 8 cases × 50 unseen seeds
- Model size: a 1.7M model matches the 17.9M one at long times over three training seeds (about 0.02 lower at t ≤ 2000) — cut Fourier modes, not width; nothing under 1M works, and below 2.9M only that model is both accurate and stable in the big boxes
- Speed: an estimated 70–100× faster than the explicit solver like for like, hundreds of times batched
- Caveat: training-seed spread of 0.04–0.08 in late-time R²; the 1.7M model has three seeds, the other sizes one

## If you only have one line to open with

"I'm testing whether a neural network trained on small, short physics simulations can still get the physics right on much bigger boxes and ten times longer than it has ever seen — and along the way, finding out which small design choices quietly break this."

## Numbers to quote

All from `Work/analysis/findings_2026-10-15.md` (the size-sweep rows from its 2026-10-16 addenda where noted) unless another file is named; "default" is width 40, 20 modes, 17.90M parameters, training seed 0. The runs removed from the drawn results on 2026-10-18 (widths 16, 12 and 8, width 24 with 12 modes, width 12 with 12 modes, and the 1.68M seeds 1 and 2) are numbers only; their full tables are frozen in `Work/analysis/size_sweep/size_sweep_all_runs_2026-10-16.txt` and `Work/analysis/timing/timing_all_runs_2026-10-16.txt`, and the figures and caches are in `Archive/removed_runs_2026-10-18/`.

| Number | What it is | Where from |
|---|---|---|
| 0.903 vs 0.010 | critical-quench R²(10000), Laplacian vs old output, trained to t = 1000 | §1 |
| 0.770 / 0.828 / 0.903 vs 0.623 / 0.176 / 0.010 | the same, trained to t = 300 / 500 / 1000 | §1 |
| 2.4–3.5× | how many times too far the lowest wavenumbers moved per step with the old output, l = 64 to l = 128 | `Work/analysis/t1000_failure_2026-10-06/analysis.json` (findings §1 says 2.3–3.5×) |
| 0.960 / 0.904 / 0.806 | default, pooled R² at t = 2000 / 5000 / 10000 | §4, `Work/analysis/size_sweep/size_sweep.txt` |
| 0.764 / 0.903 / 0.829 | default, R²(10000): off-critical / critical / big boxes | §4, `size_sweep.txt` |
| 1.68M, 0.824, 10.7× smaller | width 24, 8 modes, seed 0: parameters, pooled R²(10000), size vs default | §4, `size_sweep.txt` |
| 0.957 / 0.944 / 0.910; 0.824 / 0.751 / 0.706 | width 24, 8 modes, seeds 0 / 1 / 2: pooled R²(2000); pooled R²(10000) (mean 0.937 and 0.761, SD 0.060) | Addendum 2026-10-16 (later), `size_sweep_all_runs_2026-10-16.txt` |
| 0.960 / 0.957; 0.806 / 0.664 | default, seeds 0 / 1: pooled R²(2000); pooled R²(10000) (mean 0.735) | §3, addendum 2026-10-16 (later) |
| 0 of 1200, 1.09 | 1.68M model, three seeds: rollouts that diverge; largest max \|ψ\| | Addendum 2026-10-16 (later) |
| epoch 22, 0.942 vs 0.979 | 1.68M model seed 2: epoch picked, validation in-range R² of that checkpoint vs the epoch-60 EMA weights | Addendum 2026-10-16 (later) |
| +0.163 ± 0.012 | 8-mode (seed 0) vs width-12 model at about 1.7M parameters, R²(10000), l = 128 cases | §4 |
| 147 of 150 | width 24, 12 modes (2.86M): big-box rollouts that diverge (47 / 50 / 50 at l = 160 / 192 / 256); 0 of the l = 128 rollouts | Addendum 2026-10-16, `size_sweep_all_runs_2026-10-16.txt` |
| 25 of 400, 236 of 400 | rollouts that diverge: width 8 (0.74M), width 12 with 12 modes (0.73M); R²(10000) 0.291 and 0.115 over the survivors | Addendum 2026-10-16, `size_sweep_all_runs_2026-10-16.txt` |
| 0.04–0.08 | training-seed SD of pooled R²(10000); test-seed SE about 0.008 | §3 |
| 0.806 vs 0.664, 0.837 vs 0.778 | pooled R²(10000), seed 0 vs seed 1: default, width 32 | §3 |
| 26–32 min, 9% | HPC training time at any size on gpu1 (gpu2 takes 36–115 min); saved by 17.9M → 6.5M | §5, `Work/analysis/timing/timing_all_runs_2026-10-16.txt` |
| 34 vs 146 ms, 16 vs 49 ms | per step on CPU / Apple GPU: 1.68M model vs default (4.3× / 3×) | §5 |
| 72–99× / 207–869× | estimated speedup vs explicit dt = 0.01 solver: one trajectory / batched (solver time extrapolated from a 2000-step probe) | §5, `timing_all_runs_2026-10-16.txt` Table 4 |
