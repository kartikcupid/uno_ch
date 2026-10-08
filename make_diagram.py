#!/usr/bin/env python
"""Block diagram of the UNO / Cahn-Hilliard project.

    python3 make_diagram.py

Writes, into Results/:
    project_pipeline.pdf/.png      data -> training -> selection -> prediction
                                   -> metrics -> report
    uno_architecture.pdf/.png      the network, one UNO block, the spectral
                                   convolution
    project_block_diagram.pdf/.png both on one page

Layer widths, Fourier modes per level, the parameter count and the run
settings are read from uno_train / uno_predict, so the figure follows the code.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Results")

PAL = {
    "data": ("#dceaf7", "#2b6cb0"),
    "train": ("#e2f3df", "#2f7d4f"),
    "model": ("#ebe3f7", "#5b3fa8"),
    "eval": ("#fde9d4", "#b4541b"),
    "out": ("#ececec", "#3f4a5a"),
    "cond": ("#fff4cf", "#9a6a12"),
    "key": ("#ffe1e1", "#b42323"),
}
INK = "#2d3748"


def facts():
    """Numbers the diagram prints, taken from the code."""
    sys.path.insert(0, HERE)
    import uno_predict as up
    import uno_train as ut
    c = ut.CONFIG
    net = ut.UNO(width=c["width"], modes=c["modes"], n_ref=64,
                 levels=c["levels"], cond_dim=c["cond_dim"],
                 conserve_mass=c["conserve_mass"],
                 laplacian_output=c["laplacian_output"])
    widths = [int(c["width"] * 1.5 ** i) for i in range(c["levels"] + 1)]
    p = up.CONFIG
    return dict(
        cfg=c, pcfg=p, widths=widths, modes=list(net.mode_list),
        n_par=sum(q.numel() for q in net.parameters()),
        t_end=float(next(iter(ut.PAIRS.values()))),
        n_test=len(p["test_seeds"]), n_tune=len(p["tune_seeds"]),
        sizes=list(p["paper_sizes"]), offs=list(p["eval_offs"]),
        alphas=list(p["tune_alphas"]), dtmaxs=list(p["tune_dt_maxs"]),
        snaps=list(p["snapshots"]), every=float(p["long_snap_every"]),
        save_every=float(p["save_every"]))


def pretty(t):
    """Typographic symbols for plain (non-mathtext) labels."""
    if not t or "$" in t:
        return t
    for a, b in ((" -> ", " \u2192 "), ("->", "\u2192"), (">=", "\u2265"),
                 ("<=", "\u2264"), ("!=", "\u2260"), ("1x1", "1\u00d71"),
                 ("3x3", "3\u00d73"), (" x ", " \u00d7 "), ("R^2", "R\u00b2"),
                 ("sigma", "\u03c3"), ("P(psi)", "P(\u03c8)"),
                 ("ln dt", "ln \u0394t"), ("dt log", "\u0394t log"),
                 ("t0 ", "t\u2080 "), (" x2", " \u00d72")):
        t = t.replace(a, b)
    return t


def box(ax, x, y, w, h, title, lines=(), kind="model", tag=None, fs=8.2,
        tfs=9.4, center=False):
    face, edge = PAL[kind]
    title, lines = pretty(title), [pretty(t) for t in lines]
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.3,rounding_size=1.0",
                                fc=face, ec=edge, lw=1.3, zorder=2))
    ax.text(x + w / 2, y + h - 1.0, title, ha="center", va="top",
            fontsize=tfs, fontweight="bold", color=edge, zorder=3)
    if tag:
        ax.text(x + w - 0.6, y + 0.5, tag, ha="right", va="bottom",
                fontsize=6.6, family="monospace", color=edge, zorder=3)
    if lines:
        body = "\n".join(lines)
        if center:
            ax.text(x + w / 2, y + h - 3.6, body, ha="center", va="top",
                    fontsize=fs, linespacing=1.32, color=INK, zorder=3)
        else:
            ax.text(x + 1.0, y + h - 3.9, body, ha="left", va="top",
                    fontsize=fs, linespacing=1.32, color=INK, zorder=3)


def arrow(ax, p, q, color=INK, lw=1.3, ls="-", rad=0.0, style="-|>",
          text=None, tx=None, fs=7.2, tcolor=None, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11,
                                 lw=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.5, shrinkB=1.5, zorder=4))
    if text:
        text = pretty(text)
        x, y = tx if tx else ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 1.0)
        ax.text(x, y, text, ha=ha, va="bottom", fontsize=fs,
                color=tcolor or color, zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))


def path(ax, pts, color=INK, lw=1.3, ls="-", text=None, tx=None, fs=7.2):
    xs, ys = zip(*pts[:-1])
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=4,
            solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls)
    if text:
        ax.text(*tx, pretty(text), ha="center", va="bottom", fontsize=fs, color=color,
                zorder=5, bbox=dict(fc="white", ec="none", pad=0.6,
                                    alpha=0.9))


def band(ax, y0, y1, label, color):
    ax.add_patch(FancyBboxPatch((0.3, y0), 199.4, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=1.5",
                                fc=color, ec="none", alpha=0.35, zorder=0))
    ax.text(1.5, y1 - 0.8, pretty(label), ha="left", va="top", fontsize=9.6,
            fontweight="bold", color="#4a5568", zorder=1)


def canvas(ax, h):
    ax.set_xlim(0, 200)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_pipeline(ax, f):
    """Panel A: the whole project as data flow."""
    c, p = f["cfg"], f["pcfg"]
    canvas(ax, 83)
    t_end = f"{f['t_end']:g}"
    offs = ", ".join(f"{o:g}" for o in f["offs"])
    ax.text(100, 82.5, "UNO surrogate for the Cahn-Hilliard equation: "
            "project pipeline", ha="center", va="top", fontsize=13,
            fontweight="bold", color=INK)
    band(ax, 47, 78.5, "TRAINING   (64 x 64 periodic box, uno_train.py)",
         "#d9ecd6")
    band(ax, 1.5, 42, "", "#f6e3cf")
    ax.text(198.5, 41.2, pretty("PREDICTION AND EVALUATION   (128 x 128, "
            "plus " + " / ".join(str(s) for s in f["sizes"]) + ")"),
            ha="right", va="top", fontsize=9.6, fontweight="bold",
            color="#4a5568", zorder=1)

    w, h1, y1 = 36.5, 25, 49.5
    xs = [2.5, 42.0, 81.5, 121.0, 160.5]
    box(ax, xs[0], y1, w, h1, "1  Ground-truth solver", [
        r"$\partial_t\psi=\nabla^2\mu,\;\;\mu=\psi^3-\psi-\nabla^2\psi$",
        r"$\psi=c_A-c_B$  (binary A/B mixture)",
        "explicit Euler, solver dt = "
        f"{c['solver_dt']:g}, float64",
        "on CUDA (A100); 5-point Laplacian",
        "periodic 64 x 64 lattice, dx = 1",
        r"5 quenches $\bar\psi_0$ = {" + offs + "}",
        "25 trajectories: 20 train + 5 val",
        "t = 0 -> END = 300 / 500 / 1000 / 2000",
    ], "data", "ch_gpu.py")
    box(ax, xs[1], y1, w, h1, "2  Training dataset", [
        "frames every 0.1 (GAP = 10 solver steps)",
        "train_X, train_y: (20, 10 END - 10, 64, 64)",
        "val_X, val_y: (5, 10 END - 10, 64, 64)",
        "float64 on disk (shared format, ch.py)",
        "-> float32 memmaps (build_cache)",
        "y = X shifted by dt = 1 (N_AHEAD = 100 steps)",
        "",
        "Work/data/test_tEND.npz + .json",
        "(device, dtype, shapes recorded)",
    ], "data", "Work/data")
    box(ax, xs[2], y1, w, h1, "3  Training pairs", [
        r"chains $\psi(t_0),\psi(t_0+\Delta t),\ldots,"
        r"\psi(t_0+n\Delta t)$",
        f"dt log-uniform in [{c['dt_min']:g}, {c['dt_max']:g}],"
        f"  t0 >= {c['t_min']:g}",
        "unroll n = 1 -> 2 -> 4  (curriculum,",
        "   from epochs 1 / 5 / 12 of 60)",
        "augment: D4 rotations and flips,",
        r"   $\psi\to-\psi$ (symmetric quench set)",
        f"input noise sigma in [0, {c['noise']:g}]",
        f"batch {c['batch']}, sampled on the GPU",
    ], "train", "PairSampler")
    box(ax, xs[3], y1, w, h1, "4  Training the UNO", [
        f"{f['n_par'] / 1e6:.1f} M parameters (panel B)",
        r"loss $=(1-R^2)$"
        f" + {c['w_inc']:g} incr. + {c['w_grad']:g} grad.",
        f"backprop through the last {c['grad_steps']} unroll steps",
        f"AdamW, lr {c['lr']:g}, cosine to {c['lr_min']:g}",
        f"grad clip {c['grad_clip']:g}, EMA {c['ema_decay']:g},"
        f" wd {c['weight_decay']:g}",
        f"{c['epochs']} epochs x {c['steps_per_epoch']} steps,"
        " ~0.5-1.5 h on A100",
        "fp32; resume-safe state.pt",
        "",
        "Work/models/train_tEND_lap/",
    ], "train", "uno_train.py")
    box(ax, xs[4], y1, w, h1, "5  Validation and selection", [
        f"{len(c['val_seeds'])} held-out seeds, 128 x 128,"
        r" $\bar\psi_0=0$",
        f"rollout t = {c['val_t0']:g} -> {t_end} vs float64 truth",
        "every 2 epochs",
        "keep the epoch whose R^2 stays >= "
        f"{c['val_threshold']:g}",
        "   for longest (interpolated survival)",
        "tie-break: mean R^2 over t <= 2 END",
        "logged: low-k step gain (ideal ~ 1)",
        "",
        "-> uno_ch_best.pt",
    ], "eval", "eval_rollout")

    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w, y1 + h1 / 2), (b, y1 + h1 / 2))

    h2, y2 = 25, 9.5
    box(ax, xs[0], y2, w, h2, "6  Test ground truth", [
        "same CH solver, float64 on CUDA",
        f"128 x 128: {f['n_test']} test + {f['n_tune']} tune seeds",
        "   x 5 quenches",
        "box sizes " + " / ".join(str(s) for s in f["sizes"])
        + r" ($\bar\psi_0=0$,",
        f"   {f['n_test']} seeds)",
        f"t = {p['t_start']:g} -> {t_end}, frames every "
        f"{f['save_every']:g}",
        "streamed to disk, device recorded",
        "",
        "Work/gt_cache  (shared by all models)",
    ], "data", "ground_truth_batch")
    box(ax, xs[1], y2, w, h2, "7  Stride tuner", [
        r"$\Delta t(t)=\mathrm{clip}(\alpha\,t,\;1,\;\Delta t_{max})$",
        r"$\alpha$ in {" + ", ".join(f"{a:g}" for a in f["alphas"])
        + "}",
        r"$\Delta t_{max}$ in {"
        + ", ".join(f"{d:g}" for d in f["dtmaxs"]) + "}",
        f"chosen per quench on the {f['n_tune']} tune seeds",
        "(never on test seeds)",
        "boxes use the critical schedule",
    ], "eval", "tune()")
    snaps = ", ".join(f"{s:g}" for s in f["snaps"])
    box(ax, xs[2], y2, w, h2, "8  Autoregressive rollout", [
        r"$\hat\psi(t+\Delta t)=\mathrm{UNO}(\hat\psi(t),\,\Delta t)$",
        f"seeded with the truth at t = {p['t_start']:g}, then",
        "   the model never sees the truth again",
        f"~210 steps to t = {t_end}",
        f"snapshot times: {snaps},",
        f"   then every {f['every']:g} to {t_end}",
        "batched seeds, fp32, CUDA / MPS",
        "",
        "50 seeds x 8 cases per model",
    ], "model", "rollout_run")
    box(ax, xs[3], y2, w, h2, "9  Metrics and diagnostics", [
        r"$R^2(t)$ vs truth; survival ($R^2\geq0.8$)",
        "domain size L(t): Puri half-max, zero-crossing",
        "S(k, t), C(r, t), free energy, P(psi)",
        r"composition $\langle\psi\rangle(t)$:  "
        r"$c_A=(1+\langle\psi\rangle)/2$",
        "low-k step gain, coarsening drift,",
        "   box-size transfer, speedup vs solver",
        "",
        "Work/rollout_cache/<label>/seed*.npz",
        "Work/predict/<label>/results.json",
    ], "eval", "uno_predict.py")
    box(ax, xs[4], y2, w, h2, "10  Report  (Results/<label>/)", [
        "best seed: snapshots (t <= 2000),",
        f"   snapshots_long (every {f['every']:g})",
        "seed averages (50 seeds):",
        "   r2_sweep, domain_growth + LS t^(1/3),",
        "   psi_distribution, phase_ordering_kinetics,",
        "   composition (+ composition.csv)",
        "report.txt, manifest.json",
        "Results/README.md, comparison.png",
        "label: train_tEND[_lap]_pred_t10000",
    ], "out", "make_submission.py")

    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w, y2 + h2 / 2), (b, y2 + h2 / 2))
    cx = [x + w / 2 for x in xs]
    path(ax, [(cx[4], y1), (cx[4], 44.5), (cx[2], 44.5), (cx[2], y2 + h2)],
         color=PAL["model"][1], lw=1.8, text="uno_ch_best.pt",
         tx=((cx[2] + cx[4]) / 2, 44.7))
    arrow(ax, (cx[0], y1), (cx[0], y2 + h2), color=PAL["data"][1], ls="--",
          text="same solver", tx=(cx[0] + 7, 43.6))
    path(ax, [(cx[0] - 8, y2), (cx[0] - 8, 5.0), (cx[3], 5.0),
              (cx[3], y2)], color=PAL["data"][1], lw=1.1,
         text=r"truth frames for every metric", tx=(cx[2] + 19, 2.4))
    arrow(ax, (cx[2], 5.0), (cx[2], y2), color=PAL["data"][1], lw=1.1)
    ax.text(cx[2] + 1.5, 5.6, r"$\psi(t=100)$", fontsize=7.2,
            color=PAL["data"][1], ha="left", va="bottom")


def uno_block(ax, x, y, w, h, title, ch, modes, size, kind="model"):
    box(ax, x, y, w, h, title, [f"{ch} ch, {modes} modes", size], kind,
        fs=7.8, tfs=8.6, center=True)


def node(ax, x, y, sym, r=1.6, color=INK):
    ax.add_patch(Circle((x, y), r, fc="white", ec=color, lw=1.2, zorder=5))
    ax.text(x, y, sym, ha="center", va="center", fontsize=9, color=color,
            zorder=6, fontweight="bold")


def draw_arch(ax, f):
    """Panel B: the network, one UNO block, the spectral convolution."""
    c = f["cfg"]
    W, M = f["widths"], f["modes"]
    canvas(ax, 118)
    ax.text(100, 117, f"UNO architecture  ({f['n_par'] / 1e6:.1f} M "
            "parameters; widths " + "/".join(map(str, W))
            + "; Fourier modes " + "/".join(map(str, M))
            + " per level)", ha="center", va="top", fontsize=13,
            fontweight="bold", color=INK)
    ax.text(100, 113.0, r"one call maps $\psi(t)\to\psi(t+\Delta t)$ for "
            r"any $\Delta t\in[1,50]$ and any box size $l$ (trained at "
            r"$l=64$, applied at $l=128\ldots256$)", ha="center", va="top",
            fontsize=8.8, color="#4a5568")

    box(ax, 2, 84, 22, 9, r"$\psi(t)$", [r"$1\times l\times l$"], "data",
        fs=8, tfs=10, center=True)
    box(ax, 2, 68, 22, 12, "Input features", [r"$\psi,\ \psi^3,\ "
        r"\nabla^2\psi,\ \langle\psi\rangle$", "4 channels"], "data",
        fs=7.8, tfs=8.6, center=True)
    box(ax, 2, 56, 22, 8.5, "Lift", [f"1x1 conv  4 -> {W[0]}"], "model",
        fs=7.8, tfs=8.6, center=True)
    arrow(ax, (13, 84), (13, 80))
    arrow(ax, (13, 68), (13, 64.5))

    box(ax, 30, 98, 12, 8, r"$\Delta t$", ["step size"], "cond", fs=7.4,
        tfs=9.6, center=True)
    box(ax, 46, 98, 40, 8, "Fourier embedding of ln dt", [
        f"{c['cond_dim'] // 2} sin + {c['cond_dim'] // 2} cos + ln dt = "
        f"{2 * (c['cond_dim'] // 2) + 1}"], "cond", fs=7.4, tfs=8.4,
        center=True)
    box(ax, 90, 98, 34, 8, "MLP (GELU)", [
        f"{2 * (c['cond_dim'] // 2) + 1} -> {c['cond_dim']} -> "
        f"{c['cond_dim']}"], "cond", fs=7.4, tfs=8.4, center=True)
    arrow(ax, (42, 102), (46, 102))
    arrow(ax, (86, 102), (90, 102))
    bus_y = 93.5
    cond_c = PAL["cond"][1]
    ax.plot([107, 107], [98, bus_y], color=cond_c, lw=1.4, zorder=4)
    ax.plot([37, 166], [bus_y, bus_y], color=cond_c, lw=1.4, zorder=4)
    ax.text(127, bus_y + 0.6, r"conditioning $c\in\mathbb{R}^{64}$ "
            r"$\to$ FiLM ($\gamma,\beta$) in every UNO block", fontsize=7.6,
            color=cond_c, ha="left", va="bottom")

    bw, bh = 19, 12
    lv = [76, 57, 38, 19]
    ex = [30, 51, 72]
    dx = [135, 114, 93]
    sizes = [r"$l\times l$", r"$l/2$", r"$l/4$", r"$l/8$"]
    for i in range(3):
        uno_block(ax, ex[i], lv[i], bw, bh, "UNO block", W[i], M[i],
                  sizes[i])
        uno_block(ax, dx[i] + 21, lv[i], bw, bh, "UNO block",
                  f"{2 * W[i]} -> {W[i]}", M[i], sizes[i])
    xm = 93
    uno_block(ax, xm, lv[3], bw + 2, bh, "Bottleneck", W[3], M[3],
              sizes[3])
    dpos = [dx[i] + 21 for i in range(3)]

    path(ax, [(24, 60.2), (27, 60.2), (27, lv[0] + bh / 2),
              (ex[0], lv[0] + bh / 2)])
    for i in range(3):
        a = (ex[i] + bw / 2, lv[i])
        b = ((ex[i + 1] if i < 2 else xm) + bw / 2, lv[i + 1] + bh)
        arrow(ax, a, b, text=f"1x1 {W[i]}->{W[i + 1]}\navg-pool /2",
              tx=((a[0] + b[0]) / 2 - 7.5, (a[1] + b[1]) / 2 - 2.6), fs=6.6)
        top = (dpos[i] + bw / 2, lv[i])
        bot = ((dpos[i + 1] if i < 2 else xm + 2) + bw / 2, lv[i + 1] + bh)
        arrow(ax, bot, top, text=f"1x1 {W[i + 1]}->{W[i]}\nnearest x2, 3x3",
              tx=((top[0] + bot[0]) / 2 + 7.5, (top[1] + bot[1]) / 2 - 2.6),
              fs=6.6)
        y = lv[i] + bh / 2
        arrow(ax, (ex[i] + bw, y), (dpos[i], y), color="#718096", ls="--",
              text="skip (concatenate)", tx=((ex[i] + bw + dpos[i]) / 2,
                                            y + 0.3), fs=6.8)
    for xx, yy in ([(e + bw / 2, lv[i] + bh) for i, e in enumerate(ex)]
                   + [(d + bw / 2, lv[i] + bh) for i, d in enumerate(dpos)]
                   + [(xm + bw / 2 + 1, lv[3] + bh)]):
        ax.plot([xx, xx], [bus_y, yy + 0.3], color=cond_c, lw=0.9, ls=":",
                zorder=3)

    ox, ow = 178, 19
    box(ax, ox, 74, ow, 14, "Head", ["1x1 -> GELU -> 1x1", "(zero init)",
                                     r"$\to m(x)$"], "model", fs=7.4,
        tfs=8.6, center=True)
    arrow(ax, (dpos[0] + bw, lv[0] + bh / 2), (ox, 81))
    box(ax, ox, 60, ow, 10, r"$\times$ gain $g(\Delta t)$",
        [r"$e^{\,a+b\ln\Delta t}$ (learned)"], "cond", fs=7.4, tfs=8.4,
        center=True)
    box(ax, ox, 38, ow, 18, "Laplacian output", [
        r"$\Delta\psi=\nabla_5^2\,(g\,m)$", "exact Model-B form",
        r"low-k steps $\propto k^2$", "mass conserved"], "key", fs=7.4,
        tfs=8.6, center=True)
    box(ax, ox, 26, ow, 8, r"$-\ \langle\Delta\psi\rangle$",
        ["(rounding only)"], "model", fs=7.2, tfs=8.4, center=True)
    for a, b in ((74, 70), (60, 56), (38, 34)):
        arrow(ax, (ox + ow / 2, a), (ox + ow / 2, b))
    node(ax, ox + ow / 2, 20.5, "+")
    arrow(ax, (ox + ow / 2, 26), (ox + ow / 2, 22.2))
    box(ax, ox, 7, ow, 9, r"$\psi(t+\Delta t)$", [r"$1\times l\times l$"],
        "data", fs=8, tfs=10, center=True)
    arrow(ax, (ox + ow / 2, 18.9), (ox + ow / 2, 16))
    path(ax, [(2.5, 88.5), (0.9, 88.5), (0.9, 108.2), (199.2, 108.2),
              (199.2, 20.5), (ox + ow / 2 + 1.7, 20.5)],
         color=PAL["data"][1], lw=1.2,
         text=r"residual: $\psi(t)$ is added back", tx=(182, 108.4))

    ax.add_patch(FancyBboxPatch((1.5, 1.0), 87, 28.5,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                fc="#f8f8fb", ec="#a0aec0", lw=1.0, ls="--",
                                zorder=1))
    ax.text(3, 28.8, "Inside every UNO block", fontsize=9, fontweight="bold",
            color=INK, va="top")
    ax.text(4, 15.5, r"$x$", fontsize=10, ha="center", va="center")
    for y, t1, t2 in ((20, "Spectral conv", "(see right)"),
                      (12, "3x3 conv", "circular"),
                      (4, "1x1 conv", "pointwise")):
        box(ax, 9, y, 15, 6, t1, [t2], "model", fs=6.6, tfs=7.2,
            center=True)
        arrow(ax, (5.5, 15.5), (9, y + 3.0), lw=1.0)
        arrow(ax, (24, y + 3.0), (28.6, 15.5), lw=1.0)
    node(ax, 30.2, 15.5, "+", r=1.5)
    box(ax, 34, 12, 11, 7, "FiLM", [r"$(1+\gamma)h+\beta$"], "cond", fs=6.4,
        tfs=7.6, center=True)
    ax.plot([39.5, 39.5], [19, 25.5], color=cond_c, lw=0.9, ls=":")
    ax.text(40, 25.6, "c", fontsize=7.4, color=cond_c, va="bottom")
    box(ax, 48, 12, 8, 7, "GELU", [], "model", tfs=7.4)
    box(ax, 59, 12, 10, 7, "1x1", ["mix"], "model", fs=6.6, tfs=7.4,
        center=True)
    node(ax, 73, 15.5, "+", r=1.5)
    box(ax, 77, 12, 8.5, 7, "GELU", [], "model", tfs=7.4)
    for a, b in ((31.7, 34), (45, 48), (56, 59), (69, 71.5), (74.5, 77)):
        arrow(ax, (a, 15.5), (b, 15.5), lw=1.0)
    path(ax, [(4, 14), (4, 2.6), (73, 2.6), (73, 14)], color="#718096",
         lw=1.0, ls="--", text="skip (1x1 when channels change)",
         tx=(40, 2.8), fs=6.6)

    ax.add_patch(FancyBboxPatch((92, 1.0), 81, 15.5,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                fc="#f8f8fb", ec="#a0aec0", lw=1.0, ls="--",
                                zorder=1))
    ax.text(93.5, 15.6, "Spectral convolution on physical wavenumbers",
            fontsize=9, fontweight="bold", color=INK, va="top")
    ax.text(93.5, 12.3, pretty("\n".join([
        "rFFT2 of every channel; keep the lowest modes |n_x| <= f, "
        "0 <= n_y <= f,",
        "   with f = modes x s and s = l / 64, then multiply by a learned "
        "complex",
        "   kernel W(k) (in x out per mode) and transform back (irFFT2)",
        "W lives on the 64-box wavenumbers and is bilinearly resampled "
        "for s != 1,",
        "   so the same weights run on 64, 128, 160, 192 and 256 boxes",
    ])), fontsize=7.2, color=INK, va="top", linespacing=1.3)


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote Results/{stem}.pdf and .png")


def main():
    f = facts()
    fig, ax = plt.subplots(figsize=(20, 8.3))
    draw_pipeline(ax, f)
    save(fig, "project_pipeline")
    fig, ax = plt.subplots(figsize=(20, 11.8))
    draw_arch(ax, f)
    save(fig, "uno_architecture")
    fig, axes = plt.subplots(2, 1, figsize=(20, 20.3),
                             gridspec_kw=dict(height_ratios=[83, 118],
                                              hspace=0.02))
    draw_pipeline(axes[0], f)
    draw_arch(axes[1], f)
    for ax, lab in zip(axes, "AB"):
        ax.text(-0.5, ax.get_ylim()[1], lab, fontsize=18, fontweight="bold",
                va="top", ha="left", color=INK)
    save(fig, "project_block_diagram")


if __name__ == "__main__":
    main()
