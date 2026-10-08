# Presentation notes — "What work am I doing?"

For the UNO / Cahn–Hilliard project. A few versions at different lengths, plus slide bullets — pick what fits your time slot, or blend them.

## One sentence

I'm training a neural network to act as a fast stand-in for a physics simulation of phase separation, and testing whether it still works correctly when the simulation domain is scaled up far beyond anything it was trained on.

## ~30 seconds

My project is to build a fast machine-learning surrogate for the Cahn–Hilliard equation — the PDE that describes how a mixture separates into two phases over time, like oil separating from water. Instead of running the expensive numerical solver at every time step, I train a neural operator to learn the underlying physics directly from simulation data, so once it's trained it can predict how a new mixture evolves nearly a hundred times faster than solving the equation directly. The specific challenge my advisor set is generalization: the model is trained only on small, 64-by-64 grids, but it's tested on grids twice as wide that it has never seen, and it has to correctly predict almost 2000 time steps into the future — so the real work is making the model generalize in both space and time, not just memorize its training data.

## ~90 seconds (more technical — good for the advisor)

Concretely, I'm training a neural operator — a U-Net combined with a Fourier-based architecture, so it's called a UNO — on Cahn–Hilliard simulations on a 64-by-64 grid. The real test is on a 128-by-128 grid the model has never seen, starting from partway through the simulation at t = 100 and rolling forward autoregressively — each prediction feeding into the next — all the way out to t = 2000, scored by R² and by comparing snapshots directly.

That's not just upscaling. Making the box bigger while keeping the same resolution changes which physical wavenumber each part of the network is responsible for, and the standard version of this architecture gets that silently wrong. I tracked that down and fixed it, along with a second bug where the GPU and CPU disagreed on a subtle FFT calculation. I also built the equation's exact conservation law — the fact that the total composition can't change — directly into the network architecture rather than leaving it to training, and trained the model to correct its own compounding errors over long rollouts instead of only matching single time steps.

The result: R² stays above 0.80 for most of that 1900-step rollout — well past where a trivial baseline collapses to zero, around t = 370 — and, the main result, the same model keeps working even on grids sixteen times larger in area than anything it was trained on, all while running roughly a hundred times faster than the conventional solver.

## Slide bullets

- Goal: fast neural-network surrogate for the Cahn–Hilliard (phase-separation) equation
- Train on 64×64 lattice simulations; test on unseen 128×128, autoregressive rollout t = 100 → 2000
- Core challenge: domain extension ≠ super-resolution — same physics, bigger box, which breaks standard spectral neural operators
- Physics built into the architecture: exact mass conservation, rollout-aware training
- Headline result: generalizes to 256×256 (16× the training area), ~96× faster than the direct solver

## If you only have one line to open with

"I'm testing whether a neural network trained on small physics simulations can still get the physics right at a much larger scale it's never seen — and along the way, fixing bugs that quietly break this in most implementations."
