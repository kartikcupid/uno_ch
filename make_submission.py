"""Drawing stage of the UNO / Cahn-Hilliard project: Results/ from the caches.

    python make_submission.py                  redraw every complete run in
                                               Work/rollout_cache/ into
                                               Results/<label>/, then write
                                               Results/README.md and
                                               Results/comparison.png
    python make_submission.py --run LABEL      redraw one run, e.g.
                                               train_t0300_pred_t10000 or
                                               train_t0300_lap_pred_t10000
    python make_submission.py --rollout --end 1000 [--t-end T] [--seeds K]
                              [--schedule-from auto|PATH] [--ckpt P]
                              [--variant lap|base]
                                               (re)fill that run's rollout
                                               cache with uno_predict's
                                               rollout engine, then draw it
    python make_submission.py --smoke          the same for Work/smoke/
    python make_submission.py --work W --results R
                                               draw from W/rollout_cache/
                                               into R/ (other roots)

The draw loads no model, solves no trajectory and imports no torch; only
--rollout does. A run is complete when every case of its meta.json holds
every seed that meta.json lists.

Figures per run folder, and whether each is one seed or a seed average:

    snapshots_<case>.png                BEST SEED: the seed with the highest
                                        trajectory-mean R^2 (mean of R^2
                                        over every rollout time), per case,
                                        at the main snapshot times
    snapshots_long_<case>.png           BEST SEED, same rule and layout, at
                                        the long snapshots (every 2000);
                                        skipped when a cache has < 2 of them
    phase_ordering_kinetics_<case>.png  AVERAGE over seeds (Puri, ch. 1)
    domain_growth_<case>.png            AVERAGE: half-maximum L(t)
    psi_distribution_<case>.png         AVERAGE: P(psi) at t_end, pooled
    composition_<case>.png              AVERAGE: <psi>(t) and A / B % +- 1 sd
    composition_all.png                 AVERAGE: c_A %(t) of every composition
    r2_sweep.png / .pdf                 AVERAGE: mean R^2(t) +- 1 sd

composition.csv (seed-mean <psi> and A / B % at the snapshot times of every
case), report.txt and manifest.json complete the folder.  Only these names
are ever written into a run folder; nothing there is deleted
(Results/train_t0300_pred_t02000/ keeps its earlier_run_2026-09-08/
subfolder untouched).

This module also holds the numpy-only physics helpers (domain_length,
structure_factor, pair_correlation, free_energy_density, psi_to_a, a_to_psi,
r2_score, first_below) and the cache helpers, so the draw needs no torch;
uno_predict imports them from here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import textwrap
import warnings
from datetime import datetime
from typing import Any

import numpy as np

WORK = "Work"
RESULTS = "Results"
ARCHIVE = "Archive"
SMOKE_ROOT = os.path.join(WORK, "smoke")
SMOKE_RESULTS = os.path.join(SMOKE_ROOT, RESULTS)
TRAIN_KEYS = ("train_end", "npz", "epochs", "steps_per_epoch", "batch", "lr",
              "lr_min", "unroll_stages", "stage_lr_mult", "grad_steps",
              "weight_decay", "grad_clip", "ema_decay", "noise", "w_inc",
              "w_grad", "seed", "train_off", "t_min", "dt_min", "dt_max",
              "val_seeds", "val_t1", "val_every", "patience",
              "select", "select_min_delta", "conserve_mass",
              "laplacian_output")
EARLIER = "earlier_run_2026-09-08"
LAP_NAME = "Laplacian output"

DRAW: dict[str, Any] = dict(
    model_name   = "UNO",
    pdf_bins     = 44,
    pdf_range    = (-1.1, 1.1),
    truth_color  = "#bf00bf",
    pred_color   = "#1a55b0",
    kin_truth    = "#1b7f5f",
    kin_pred     = "#7b3fa0",
    markevery    = 0.045,
    min_domains  = 4,
    sweep_dpi    = 300,
)

FIG_KINDS = ("phase_ordering_kinetics", "domain_growth", "psi_distribution",
             "snapshots", "snapshots_long", "composition")
MAIN_SNAPS = (100.0, 500.0, 1000.0, 2000.0)
LONG_EVERY = 2000.0
CSV_COLS = ("case", "psi0", "t", "psi_mean_truth", "psi_mean_pred",
            "A_pct_truth", "B_pct_truth", "A_pct_pred", "B_pct_pred",
            "drift_pred", "n_seeds")
SERIES = ("r2", "L_half_truth", "L_half_pred", "L_zero_truth", "L_zero_pred",
          "mean_psi_truth", "mean_psi_pred")

PAPER_C = ("#1f2fd0", "#e11ec4", "#137a2e", "#c07000", "#00868b")
PAPER_M = ("o", "s", "^", "D", "v")
PAPER_RC: Any = {
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.labelsize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "xtick.minor.size": 2.8,
    "ytick.minor.size": 2.8,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "legend.frameon": False,
    "legend.handlelength": 1.9,
    "legend.labelspacing": 0.35,
    "savefig.bbox": "tight",
}
RUN_COLORS = ("#1b4f9c", "#c2188b", "#3aa655", "#c07000", "#00868b",
              "#7b3fa0")


def free_energy_density(psi):
    gx = np.roll(psi, -1, -2) - psi
    gy = np.roll(psi, -1, -1) - psi
    return (-0.5 * psi ** 2 + 0.25 * psi ** 4
            + 0.5 * (gx ** 2 + gy ** 2)).mean(axis=(-2, -1))


def structure_factor(psi):
    """Angle-averaged S(k) of the fluctuation psi - <psi>."""
    psi = np.asarray(psi, dtype=np.float64)
    n = psi.shape[-1]
    fk = np.fft.fft2(psi - psi.mean(axis=(-2, -1), keepdims=True), norm="ortho")
    power = (fk.real ** 2 + fk.imag ** 2).mean(axis=tuple(range(psi.ndim - 2)))
    kx = 2.0 * np.pi * np.fft.fftfreq(n)
    kmag = np.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2)
    dk, nb = 2.0 * np.pi / n, n // 2
    idx = np.clip((kmag / dk).astype(int), 0, nb - 1)
    cnt = np.bincount(idx.ravel(), minlength=nb)
    s = np.bincount(idx.ravel(), weights=power.ravel(), minlength=nb)
    ok = cnt > 0
    return ((np.arange(nb) + 0.5) * dk)[ok], (s[ok] / cnt[ok])


def psi_to_a(psi):
    """Percent of material A in the binary mixture: c_A = (1 + psi)/2."""
    return 50.0 * (1.0 + psi)


def a_to_psi(pct):
    """Inverse of psi_to_a, for the composition axis."""
    return pct / 50.0 - 1.0


def domain_length(psi, method="zero"):
    """Domain scale L(t).  'zero' is quantised; use 'moment' for rates.

    'half' is Puri's definition (ch. 1, Fig. 1.7): the distance over which
    the correlation function C(r, t) falls to half its maximum, linearly
    interpolated between lattice shells.
    """
    psi = np.asarray(psi, dtype=np.float64)
    if method == "half":
        r, c = pair_correlation(psi)
        below = np.nonzero(c < 0.5)[0]
        if below.size == 0 or below[0] == 0:
            return float("nan")
        j = int(below[0])
        return float(r[j - 1] + (c[j - 1] - 0.5) * (r[j] - r[j - 1])
                     / (c[j - 1] - c[j]))
    if method == "zero":
        s = np.sign(psi - psi.mean(axis=(-2, -1), keepdims=True))
        cx = (s != np.roll(s, -1, -2)).mean(axis=(-2, -1))
        cy = (s != np.roll(s, -1, -1)).mean(axis=(-2, -1))
        return 1.0 / np.maximum(0.5 * (cx + cy), 1e-12)
    k, S = structure_factor(psi)
    if method == "moment":
        return 2.0 * np.pi * S.sum() / max((k * S).sum(), 1e-12)
    if method == "peak":
        return 2.0 * np.pi / k[int(np.argmax(S))]
    raise ValueError(method)


def pair_correlation(psi, rmax=None):
    psi = np.asarray(psi, dtype=np.float64)
    n = psi.shape[-1]
    fk = np.fft.fft2(psi - psi.mean(axis=(-2, -1), keepdims=True))
    corr = np.fft.ifft2(fk * np.conj(fk)).real / (n * n)
    corr = corr.mean(axis=tuple(range(psi.ndim - 2)))
    corr = corr / corr.flat[0]
    ax = np.minimum(np.arange(n), n - np.arange(n))
    r = np.sqrt(ax[:, None] ** 2 + ax[None, :] ** 2)
    rmax = rmax or n // 2
    idx = np.clip(np.rint(r).astype(int), 0, rmax)
    cnt = np.bincount(idx.ravel(), minlength=rmax + 1)
    s = np.bincount(idx.ravel(), weights=corr.ravel(), minlength=rmax + 1)
    ok = cnt > 0
    return np.arange(rmax + 1)[ok], s[ok] / cnt[ok]


def r2_score(pred, true):
    pred = np.asarray(pred, np.float64)
    true = np.asarray(true, np.float64)
    return float(1.0 - ((pred - true) ** 2).sum()
                 / max(((true - true.mean()) ** 2).sum(), 1e-30))


def first_below(times, r2, threshold):
    bad = np.nonzero(np.asarray(r2) < threshold)[0]
    return float(times[bad[0]]) if bad.size else None


def fingerprint(path):
    """First 12 hex digits of the checkpoint's sha256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()[:12]


def comp_case(off):
    return f"psi0{'m' if off < 0 else 'p'}{abs(off):.2f}"


def savez_compressed_atomic(path, **arrays):
    tmp = path[:-4] + ".partial.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def dump_json(path, obj):
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def seed_file(cdir, case, seed):
    return os.path.join(cdir, case, f"seed{seed}.npz")


def rollout_root(work=WORK):
    return os.path.join(work, "rollout_cache")


def parse_label(label):
    """(END, variant, t_end) of train_t<END>[_lap]_pred_t<T>, else None.

    variant is 'lap' or 'base'; END and t_end are ints.
    """
    m = re.fullmatch(r"train_t(\d+)(_lap)?_pred_t(\d+)", label)
    if not m:
        return None
    return int(m.group(1)), ("lap" if m.group(2) else "base"), int(m.group(3))


def run_variant(label, meta=None):
    """'lap' or 'base': from the label, else from the cached arch."""
    p = parse_label(label)
    if p:
        return p[1]
    arch = (meta or {}).get("arch") or {}
    return "lap" if arch.get("laplacian_output") else "base"


def variant_text(variant):
    return f"{LAP_NAME}" if variant == "lap" else "base"


def train_dir_name(end, variant):
    return f"train_t{int(round(float(end))):04d}" + (
        "_lap" if variant == "lap" else "")


def results_json(label, work=WORK):
    return os.path.join(work, "predict", label, "results.json")


def meta_seeds(meta):
    return [int(s) for s in (meta.get("seeds") or meta.get("seed_order") or [])]


def case_seeds(meta, case):
    """Seeds of one case: its own list when meta records one."""
    own = (meta.get("cases") or {}).get(case, {}).get("seeds")
    return [int(s) for s in own] if own else meta_seeds(meta)


def case_schedule(meta, c):
    """alpha / dt_min / dt_max / source of one case, new or old meta."""
    s = dict((meta.get("schedule") or {}).get(c.get("schedule", ""), {}))
    for k in ("alpha", "dt_min", "dt_max", "source"):
        if k in c:
            s[k] = c[k]
    return s


def run_status(cdir, meta):
    """None when the cache is complete, else why it is not.

    A cache written by the current rollout engine records the cases it plans
    and a complete flag set only once the last case is rolled out; older
    caches (t300) predate both and are judged by their seed files alone.
    """
    cases = meta.get("cases") or {}
    if not cases:
        return "meta.json lists no cases"
    left = [c for c in meta.get("planned_cases") or [] if c not in cases]
    if left:
        return (f"{len(left)} planned case(s) not started yet (first "
                f"{left[0]})")
    if meta.get("complete") is False:
        return "the rollout has not finished (meta.json complete = false)"
    need = [(case, sd) for case in cases for sd in case_seeds(meta, case)]
    if not need:
        return "meta.json lists no seeds"
    miss = [(c, s) for c, s in need if not os.path.exists(seed_file(cdir, c, s))]
    if miss:
        return (f"{len(miss)} of {len(need)} seed files missing (first "
                f"{miss[0][0]}/seed{miss[0][1]}.npz)")
    return None


def discover(work=WORK):
    """Every rollout cache under <work>/rollout_cache/ that has a meta.json."""
    root = rollout_root(work)
    out: list[dict[str, Any]] = []
    if not os.path.isdir(root):
        return out
    def order(name):
        p = parse_label(name)
        return (0, p[0], p[1] == "lap", p[2], name) if p else (1, 0, 0, 0, name)

    for name in sorted(os.listdir(root), key=order):
        cdir = os.path.join(root, name)
        mpath = os.path.join(cdir, "meta.json")
        if not os.path.isfile(mpath):
            continue
        meta = read_json(mpath)
        out.append(dict(label=name, cdir=cdir, meta=meta,
                        why=run_status(cdir, meta)))
    return out


def load_case(cdir, case, seeds, meta, snaps=True):
    """Stack the cached per-seed arrays of one case.

    Seeds are cut to a common length if a rollout ever stopped early (a
    non-finite state); full_len keeps each seed's own length.  Files that
    predate fe_*, t_end and train_t_end are read too: t_end is then the
    last rollout time and train_t_end comes from meta.
    """
    rows = []
    for sd in seeds:
        with np.load(seed_file(cdir, case, sd)) as z:
            keep = [k for k in z.files
                    if snaps or k not in ("snap_truth", "snap_pred")]
            rows.append({k: z[k] for k in keep})
    full = np.array([len(r["times"]) for r in rows])
    n = int(full.min())
    if full.max() != n:
        print(f"[draw] WARNING {case}: rollouts of different length "
              f"({n}..{full.max()} points); cut to {n}", flush=True)
    d: dict[str, Any] = dict(case=case, seeds=[int(s) for s in seeds],
                             full_len=full,
                             times=np.asarray(rows[0]["times"][:n], float))
    for k in SERIES:
        d[k] = np.stack([np.asarray(r[k][:n], dtype=np.float64) for r in rows])
    if all("fe_truth" in r for r in rows):
        for k in ("fe_truth", "fe_pred"):
            d[k] = np.stack([np.asarray(r[k][:n], np.float64) for r in rows])
    d["snap_times"] = np.asarray(rows[0]["snap_times"], dtype=np.float64)
    if snaps:
        d["snap_truth"] = np.stack([r["snap_truth"] for r in rows])
        d["snap_pred"] = np.stack([r["snap_pred"] for r in rows])
    r0 = rows[0]
    d["l"], d["off"] = int(r0["l"]), float(r0["off"])
    d["alpha"], d["dt_max"] = float(r0["alpha"]), float(r0["dt_max"])
    d["t_end"] = (float(r0["t_end"]) if "t_end" in r0
                  else float(max(r["times"][-1] for r in rows)))
    tt = float(r0["train_t_end"]) if "train_t_end" in r0 else 0.0
    d["train_t_end"] = tt or float(meta.get("train_t_end") or 0.0) or None
    return d


def sd_of(x, axis=None):
    x = np.asarray(x, dtype=np.float64)
    n = x.size if axis is None else x.shape[axis]
    if n < 2:
        return 0.0 if axis is None else np.zeros(np.delete(x.shape, axis))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanstd(x, axis=axis, ddof=1)
    return float(out) if axis is None else out


def nanmean0(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.asarray(x, dtype=np.float64), axis=0)


def find_index(times, x):
    """Index of the time exactly equal to x (to 1e-6), else None."""
    times = np.asarray(times, dtype=np.float64)
    if times.size == 0:
        return None
    k = int(np.argmin(np.abs(times - x)))
    return k if abs(times[k] - x) < 1e-6 else None


def pick_seeds(d):
    """One best-seed rule everywhere: the highest trajectory-mean R^2.

    A seed whose rollout stopped early cannot be the best.  Median and worst
    are picked by the same trajectory mean.
    """
    traj = d["r2"].mean(axis=1)
    full = d["full_len"] == d["full_len"].max()
    best = int(np.argmax(np.where(full, traj, -np.inf)))
    order = np.argsort(traj, kind="stable")
    return dict(best=best, median=int(order[(len(order) - 1) // 2]),
                worst=int(order[0])), traj


def case_stats(d, thr):
    """R^2 at t_end, trajectory mean, survival and the picked seeds."""
    t, r2 = d["times"], d["r2"]
    fin = r2[:, -1]
    picks, traj = pick_seeds(d)
    cross = [first_below(t, r2[k], thr) for k in range(r2.shape[0])]
    held = [float(t[-1]) if c is None else c for c in cross]
    return dict(
        n_seeds=len(d["seeds"]), steps=int(len(t) - 1), t_end=float(t[-1]),
        r2_at_end=dict(mean=float(fin.mean()), sd=sd_of(fin),
                       median=float(np.median(fin)), best=float(fin.max()),
                       worst=float(fin.min())),
        r2_trajectory_mean=dict(mean=float(traj.mean()), sd=sd_of(traj)),
        survival=dict(threshold=float(thr), mean=float(np.mean(held)),
                      sd=sd_of(held),
                      n_survive_to_end=int(sum(c is None for c in cross)),
                      mean_curve_crosses=first_below(t, r2.mean(axis=0), thr)),
        seeds={k: dict(seed=int(d["seeds"][i]),
                       r2_trajectory_mean=float(traj[i]),
                       r2_at_end=float(fin[i])) for k, i in picks.items()})


def horizons(meta, d=None):
    """(t_start, t_end, train_t_end or None) of a run."""
    t0, t1 = float(meta["t_start"]), float(meta["t_end"])
    tt = (d or {}).get("train_t_end") or meta.get("train_t_end")
    return t0, t1, (float(tt) if tt else None)


def fit_windows(meta, d=None):
    """Late window [t_end/4, t_end] and the training window."""
    t0, t1, tt = horizons(meta, d)
    late = (max(t0, 0.25 * t1), t1)
    train = (t0, tt) if tt and tt > t0 else None
    return late, train


def growth_fit(t, L, lo=-np.inf, hi=np.inf):
    t = np.asarray(t, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    ok = np.isfinite(L) & (L > 0) & (t >= lo - 1e-9) & (t <= hi + 1e-9)
    if ok.sum() < 2:
        return None
    return float(np.polyfit(np.log(t[ok]), np.log(L[ok]), 1)[0])


def growth_record(d, meta, kind):
    t = d["times"]
    lt, lp = nanmean0(d[f"L_{kind}_truth"]), nanmean0(d[f"L_{kind}_pred"])
    late, train = fit_windows(meta, d)
    out: dict[str, Any] = dict(
        late_window=list(late), n_truth=growth_fit(t, lt, *late),
        n_pred=growth_fit(t, lp, *late), L_start_truth=float(lt[0]),
        L_end_truth=float(lt[-1]), L_end_pred=float(lp[-1]))
    if train:
        out.update(train_window=list(train),
                   n_truth_train=growth_fit(t, lt, *train),
                   n_pred_train=growth_fit(t, lp, *train))
    return out


def finite_size(d):
    """Spans of t where the mean zero-crossing L exceeds l / min_domains."""
    t = d["times"]
    over = nanmean0(d["L_zero_truth"]) > d["l"] / DRAW["min_domains"]
    spans, i = [], 0
    while i < len(t):
        if over[i]:
            j = i
            while j + 1 < len(t) and over[j + 1]:
                j += 1
            spans.append((float(t[i]), float(t[min(j + 1, len(t) - 1)])))
            i = j + 1
        else:
            i += 1
    first = float(t[int(np.argmax(over))]) if over.any() else None
    return spans, first


def composition(d):
    tru, pre = d["mean_psi_truth"], d["mean_psi_pred"]
    psi0 = tru[:, :1]
    t = d["times"]
    slopes = [float(np.polyfit(t, pre[k], 1)[0]) for k in range(len(pre))]
    return dict(
        A_percent_start_truth=psi_to_a(float(tru[:, 0].mean())),
        A_percent_start_pred=psi_to_a(float(pre[:, 0].mean())),
        A_percent_end_truth=psi_to_a(float(tru[:, -1].mean())),
        A_percent_end_pred=psi_to_a(float(pre[:, -1].mean())),
        max_abs_psi_drift_truth=float(np.abs(tru - psi0).max()),
        max_abs_psi_drift_pred=float(np.abs(pre - psi0).max()),
        psi_drift_rate_pred=float(np.mean(slopes)))


def ab_pct(psi):
    """(A %, B %) of a mixture with mean order parameter psi."""
    a = psi_to_a(psi)
    return a, 100.0 - a


def composition_series(d, meta):
    """Seed-mean <psi> and A / B % at the snapshot times of one case.

    The times are every snapshot the seed files hold plus t_start, kept when
    they lie on the rollout time grid; one dict per time.
    """
    t0 = float(meta["t_start"])
    want = sorted({float(x) for x in d["snap_times"]} | {t0})
    mt, mp = nanmean0(d["mean_psi_truth"]), nanmean0(d["mean_psi_pred"])
    psi0 = float(mt[0])
    rows = []
    for x in want:
        k = find_index(d["times"], x)
        if k is None:
            continue
        at, bt = ab_pct(float(mt[k]))
        ap, bp = ab_pct(float(mp[k]))
        rows.append(dict(t=x, psi0=psi0, psi_mean_truth=float(mt[k]),
                         psi_mean_pred=float(mp[k]), A_pct_truth=at,
                         B_pct_truth=bt, A_pct_pred=ap, B_pct_pred=bp,
                         drift_pred=float(mp[k]) - psi0,
                         n_seeds=len(d["seeds"])))
    return rows


def write_composition_csv(path, cases):
    """composition.csv: one row per (case, t) of every composition_series."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for case, c in cases.items():
            for r in c["composition_series"]:
                w.writerow([case, format(r["psi0"], ".6f"), format(r["t"], "g"),
                            *[format(r[k], ".6f") for k in CSV_COLS[3:10]],
                            r["n_seeds"]])


def fmt(x, spec=".3f"):
    return "n/a" if x is None else format(x, spec)


def r2_limits(values):
    """Lower limit min(0, data) with a margin: negative R^2 stays visible."""
    lo = min(0.0, min(float(np.nanmin(v)) for v in values))
    return lo - 0.05 * (1.02 - lo), 1.02


def case_title(d, meta):
    off, l = d["off"], d["l"]
    if l != meta.get("base_l", 128):
        kind = f"box transfer to l = {l}, critical quench"
    elif abs(off) < 1e-9:
        kind = "critical quench"
    else:
        kind = "off-critical quench"
    return kind + rf", $\bar\psi_0 = {off:+.1f}$"


def mark_time(ax, meta, d, spans):
    """Finite-size shading and the training-horizon line on a time axis."""
    t0, t1, tt = horizons(meta, d)
    for i, (a, b) in enumerate(spans):
        ax.axvspan(a, b, color="0.55", alpha=0.16, lw=0,
                   label=(f"finite-size regime (< {DRAW['min_domains']} "
                          f"domains)") if i == 0 else None)
    if tt and t0 < tt < t1:
        ax.axvline(tt, color="0.25", ls=":", lw=1.4,
                   label=f"training horizon $t$ = {tt:g}")


def snapshot_sets(d, meta):
    """(main, long, every): the snapshot times of a case a figure draws.

    The sets the run recorded (meta main_snapshots / long_snapshots) are used
    when present, else MAIN_SNAPS and the multiples of LONG_EVERY, always cut
    to the times the seed files hold.  A cache holding fewer than two main
    times (an old horizon) draws the main figure at all its snapshot times.
    """
    have = [float(x) for x in d["snap_times"]]
    t0 = float(meta["t_start"])
    want_m = meta.get("main_snapshots") or MAIN_SNAPS
    every = float(meta.get("long_snap_every") or LONG_EVERY)
    want_l = meta.get("long_snapshots")
    if want_l is None:
        want_l = [x for x in have if x > t0 + 1e-9
                  and abs(x / every - round(x / every)) < 1e-6]
    main = [x for x in have if any(abs(x - w) < 1e-6 for w in want_m)]
    long = [x for x in have if any(abs(x - w) < 1e-6 for w in want_l)]
    if len(main) < 2:
        main = have
    return main, long, every


def fig_snapshots(plt, d, meta, st, path, times=None, every=None):
    """Truth, prediction and error for the best seed at the snapshot times.

    times picks the columns (default every snapshot of the case); every marks
    the long-horizon figure in the title.
    """
    name = DRAW["model_name"]
    t0, t1, _ = horizons(meta, d)
    pk = st["seeds"]
    b = d["seeds"].index(pk["best"]["seed"])
    cols = [j for j, x in enumerate(d["snap_times"])
            if times is None or any(abs(x - w) < 1e-6 for w in times)]
    n_s = len(cols)
    fig = plt.figure(figsize=(3.4 * n_s + 1.3, 10.9))
    gs = fig.add_gridspec(3, n_s + 1, width_ratios=[1] * n_s + [0.05],
                          wspace=0.06, hspace=0.10, top=0.82)
    rows = ("ground truth", f"{name} prediction", "absolute error")
    im_f = im_e = None
    shown = []
    for c, j in enumerate(cols):
        ts = d["snap_times"][j]
        tru, pre = d["snap_truth"][b, j], d["snap_pred"][b, j]
        k = find_index(d["times"], ts)
        r2 = float(d["r2"][b, k]) if k is not None else None
        shown.append(dict(t=float(ts), r2=r2))
        for r in range(3):
            ax = fig.add_subplot(gs[r, c])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                im_f = ax.imshow(tru, cmap="RdBu_r", vmin=-1, vmax=1)
                ax.set_title(f"$t$ = {ts:g}" + "\n"
                             + f"$R^2$ = {fmt(r2)}", fontsize=12)
            elif r == 1:
                im_f = ax.imshow(pre, cmap="RdBu_r", vmin=-1, vmax=1)
            else:
                im_e = ax.imshow(np.abs(tru - pre), cmap="magma", vmin=0,
                                 vmax=1)
            if c == 0:
                ax.set_ylabel(rows[r], fontsize=13, fontweight="bold")
    for r in range(2):
        cb = fig.colorbar(im_f, cax=fig.add_subplot(gs[r, -1]))
        cb.set_label(r"$\psi$")
    cb = fig.colorbar(im_e, cax=fig.add_subplot(gs[2, -1]))
    cb.set_label(r"$|\psi_{\rm truth} - \psi_{\rm " + name + r"}|$")
    bs, md, ws = pk["best"], pk["median"], pk["worst"]
    fig.suptitle(
        f"Cahn-Hilliard {name} - {case_title(d, meta)}" + "\n"
        + f"{d['l']} x {d['l']} lattice sites, $dx = 1$; autoregressive "
        f"rollout seeded at $t = {t0:g}$, run to $t = {t1:g}$" + "\n"
        + f"best seed {bs['seed']} (mean $R^2$ {bs['r2_trajectory_mean']:.3f}"
        f", $R^2({t1:g})$ = {bs['r2_at_end']:.3f}) of {len(d['seeds'])};  "
        f"median seed {md['seed']} (mean $R^2$ "
        f"{md['r2_trajectory_mean']:.3f});  worst seed {ws['seed']} (mean "
        f"$R^2$ {ws['r2_trajectory_mean']:.3f})" + "\n"
        + "mean $R^2$ = mean of $R^2$ over every rollout time"
        + ("" if every is None else f"\nlong-horizon snapshots, every "
           f"{every:g} up to $t = {t1:g}$"),
        fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return dict(rule="highest trajectory-mean R^2", seed=bs["seed"],
                panels=shown)


def fig_pdf(plt, d, meta, path):
    """P(psi) at t_end, truth against prediction, pooled over every seed."""
    name = DRAW["model_name"]
    t_pdf = horizons(meta, d)[1]
    j = find_index(d["snap_times"], t_pdf)
    assert j is not None, (d["case"], "no snapshot at t_end", t_pdf)
    xt = d["snap_truth"][:, j].astype(np.float64).ravel()
    xp = d["snap_pred"][:, j].astype(np.float64).ravel()
    xp = xp[np.isfinite(xp)]
    lo, hi = DRAW["pdf_range"]
    edges = np.linspace(lo, hi, DRAW["pdf_bins"] + 1)
    ht, _ = np.histogram(xt, edges, density=True)
    hp, _ = np.histogram(xp, edges, density=True)
    mid = 0.5 * (edges[1:] + edges[:-1])
    tc, pc = DRAW["truth_color"], DRAW["pred_color"]
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.stairs(ht, edges, fill=True, color=tc, alpha=0.30)
    ax.stairs(ht, edges, color=tc, lw=2.0, label="ground truth (numerical)")
    ax.plot(mid, np.where(hp > 0, hp, np.nan), "-o", color=pc, lw=1.8, ms=5,
            mfc="white", mew=1.5, label=f"{name} prediction")
    for v in (-1.0, 1.0):
        ax.axvline(v, color="0.45", ls=":", lw=1.2)
    ax.set_yscale("log")
    pos = np.concatenate([ht[ht > 0], hp[hp > 0]])
    ax.set_ylim(pos.min() * 0.5, pos.max() * 2.0)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"$\psi$", fontsize=14)
    ax.set_ylabel(r"$P(\psi)$", fontsize=14)
    ax.grid(alpha=0.25, which="both", ls=":")
    ax.legend(loc="upper center", fontsize=11)
    ax.set_title(rf"$\bar\psi_0 = {d['off']:+.1f}$,  $l = {d['l']}$" + "\n"
                 + rf"$P(\psi)$ at $t$ = {t_pdf:g}, {len(d['seeds'])} seeds "
                 f"pooled ({xt.size:,} sites)", fontsize=13,
                 fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    lever = 0.5 * (1.0 + d["off"])
    return dict(time=t_pdf, n_sites=int(xt.size), bins=int(len(ht)),
                range=[lo, hi], frac_A_rich_truth=float((xt > 0).mean()),
                frac_A_rich_pred=float((xp > 0).mean()), lever_rule=lever,
                lever_shift_truth=float((xt > 0).mean() - lever),
                lever_shift_pred=float((xp > 0).mean() - lever),
                clipped_truth=int(((xt < lo) | (xt > hi)).sum()),
                clipped_pred=int(((xp < lo) | (xp > hi)).sum()),
                peak_density_truth=float(ht.max()),
                peak_density_pred=float(hp.max()))


def fig_growth(plt, d, meta, path, spans):
    """Seed-mean half-maximum L(t) against the Lifshitz-Slyozov law."""
    name = DRAW["model_name"]
    t = d["times"]
    g = growth_record(d, meta, "half")
    lt, lp = nanmean0(d["L_half_truth"]), nanmean0(d["L_half_pred"])
    st, sp = sd_of(d["L_half_truth"], 0), sd_of(d["L_half_pred"], 0)
    tc, pc = DRAW["truth_color"], DRAW["pred_color"]
    every = DRAW["markevery"]
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.fill_between(t, np.maximum(lp - sp, 1e-3), lp + sp, color=pc,
                    alpha=0.12, lw=0)
    ax.fill_between(t, np.maximum(lt - st, 1e-3), lt + st, color=tc,
                    alpha=0.10, lw=0)
    ax.loglog(t, lp, "-o", color=pc, lw=2.0, ms=5.5, markevery=every,
              label=f"{name} prediction")
    ax.loglog(t, lt, "--o", color=tc, lw=1.8, ms=7, mfc="none", mew=1.5,
              markevery=every, label="ground truth (numerical)")
    tm = float(np.sqrt(t[0] * t[-1]))
    anchor = 1.35 * float(np.interp(tm, t, lt))
    tg = np.array([t[0], t[-1]])
    ax.loglog(tg, anchor * (tg / tm) ** (1.0 / 3.0), "k--", lw=1.6,
              label=r"Lifshitz-Slyozov  $L \propto t^{1/3}$")
    ax.text(tg[-1], anchor * (tg[-1] / tm) ** (1.0 / 3.0), "  1/3",
            fontsize=13, fontweight="bold", va="center")
    mark_time(ax, meta, d, spans)
    ax.set_xlabel("$t$", fontsize=14)
    ax.set_ylabel("$L(t)$", fontsize=14)
    ax.grid(alpha=0.25, which="both", ls=":")
    ax.legend(loc="upper left", fontsize=10)
    lo, hi = g["late_window"]
    tr = ""
    if "train_window" in g:
        a, b = g["train_window"]
        tr = (f";  over $t$ = {a:g}-{b:g} (training): "
              f"{fmt(g['n_truth_train'])} / {fmt(g['n_pred_train'])}")
    ax.set_title(
        rf"$\bar\psi_0 = {d['off']:+.1f}$,  $l = {d['l']}$" + "\n"
        + r"$L$: distance over which $C(r,t)$ falls to half its maximum "
        "(Puri, Fig. 1.7)" + "\n"
        + f"mean $\\pm$ 1 sd over {len(d['seeds'])} seeds;  fitted $n$ "
        f"(truth / {name}) over $t$ = {lo:g}-{hi:g}: {fmt(g['n_truth'])} / "
        f"{fmt(g['n_pred'])}{tr}", fontsize=11, fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dict(definition="C(r,t) falls to half its maximum (Puri, Fig. 1.7)",
                **g)


def fig_kinetics(plt, d, meta, path, spans):
    """Seed-averaged 2x3 phase-ordering-kinetics panel (Puri, ch. 1)."""
    name = DRAW["model_name"]
    tc, pc = DRAW["kin_truth"], DRAW["kin_pred"]
    t = d["times"]
    t0, t1, tt = horizons(meta, d)
    n = len(d["seeds"])
    ok = d["full_len"] == d["full_len"].max()
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))
    fig.suptitle(f"{name} - phase-ordering kinetics, {case_title(d, meta)}, "
                 f"$l = {d['l']}$: average over {n} seeds "
                 f"(Puri, Kinetics of Phase Transitions, ch. 1)",
                 fontweight="bold", fontsize=15)

    ax = axes[0, 0]
    g = growth_record(d, meta, "zero")
    lt, lp = nanmean0(d["L_zero_truth"]), nanmean0(d["L_zero_pred"])
    st, sp = sd_of(d["L_zero_truth"], 0), sd_of(d["L_zero_pred"], 0)
    ax.fill_between(t, np.maximum(lt - st, 1e-3), lt + st, color=tc,
                    alpha=0.15, lw=0)
    ax.fill_between(t, np.maximum(lp - sp, 1e-3), lp + sp, color=pc,
                    alpha=0.15, lw=0)
    ax.loglog(t, lt, color=tc, lw=2.2, label="ground truth")
    ax.loglog(t, lp, color=pc, lw=2.0, ls="--", label=name)
    ax.loglog(t, lt[0] * (t / t[0]) ** (1 / 3), "k:", lw=1.6,
              label="Lifshitz–Slyozov $t^{1/3}$")
    mark_time(ax, meta, d, spans)
    lo, hi = g["late_window"]
    sub = (f"$n$ over $t$ = {lo:g}-{hi:g}: truth {fmt(g['n_truth'])}, "
           f"{name} {fmt(g['n_pred'])}")
    if "train_window" in g:
        a, b = g["train_window"]
        sub += (f"\nover $t$ = {a:g}-{b:g} (training data): truth "
                f"{fmt(g['n_truth_train'])}, {name} {fmt(g['n_pred_train'])}")
    ax.set_title("(a) domain growth, mean $\\pm$ 1 sd\n" + sub, fontsize=11)
    ax.set_xlabel("$t$"); ax.set_ylabel("$L(t)$  [zero-crossing]")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    show = [(j, x) for j, x in enumerate(d["snap_times"]) if x <= t[-1] + 1e-9]
    cols = plt.cm.viridis(np.linspace(0, 0.85, max(1, len(show))))
    snap_L = []
    for j, x in show:
        k = find_index(t, x)
        snap_L.append((float(lt[k]), float(lp[k])) if k is not None
                      else (float("nan"), float("nan")))

    ax = axes[0, 1]
    smax = 0.0
    for (j, x), c in zip(show, cols):
        kk, S = structure_factor(d["snap_truth"][:, j])
        smax = max(smax, float(S.max()))
        ax.loglog(kk, S, color=c, lw=2.0, label=f"$t={x:g}$")
        if ok.any():
            kk, S = structure_factor(d["snap_pred"][ok, j])
            ax.loglog(kk, S, color=c, lw=1.6, ls="--")
    kp = np.array([0.8, 2.5])
    ax.loglog(kp, 0.3 * smax * (kp / kp[0]) ** -3, "k:", lw=1.8,
              label="Porod $k^{-3}$")
    ax.set_ylim(smax * 1e-6, smax * 5)
    ax.set_title("(b) structure factor, seed average\n"
                 "(solid: truth, dashed: prediction)", fontsize=11)
    ax.set_xlabel("$k$"); ax.set_ylabel("$S(k,t)$")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    ax = axes[0, 2]
    smax = 0.0
    for (j, x), c, (Lt, Lp) in zip(show, cols, snap_L):
        for arr, ls, L in ((d["snap_truth"][:, j], "-", Lt),
                           (d["snap_pred"][ok, j], "--", Lp)):
            if not np.isfinite(L) or len(arr) == 0:
                continue
            kk, S = structure_factor(arr)
            smax = max(smax, float((S / L ** 2).max()))
            ax.loglog(kk * L, S / L ** 2, color=c, lw=1.8, ls=ls,
                      label=f"$t={x:g}$" if ls == "-" else None)
    if smax > 0:
        ax.set_ylim(smax * 1e-6, smax * 5)
    ax.set_title("(c) dynamical scaling  $S = L^d f(kL)$  (Eq. 1.90)\n"
                 "L = seed-mean zero-crossing length at each time",
                 fontsize=11)
    ax.set_xlabel("$kL$"); ax.set_ylabel("$L^{-d}S(k,t)$")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    for (j, x), c, (Lt, Lp) in zip(show, cols, snap_L):
        for arr, ls, L in ((d["snap_truth"][:, j], "-", Lt),
                           (d["snap_pred"][ok, j], "--", Lp)):
            if not np.isfinite(L) or len(arr) == 0:
                continue
            rr, C = pair_correlation(arr)
            ax.plot(rr / L, C, color=c, lw=1.8, ls=ls,
                    label=f"$t={x:g}$" if ls == "-" else None)
    ax.axhline(0, color="k", lw=0.8); ax.set_xlim(0, 3)
    ax.set_title("(d) scaling function  $C = g(r/L)$  (Eq. 1.88), "
                 "seed average", fontsize=11)
    ax.set_xlabel("$r/L$"); ax.set_ylabel("$C(r,t)$")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if "fe_truth" in d:
        ft, fp = nanmean0(d["fe_truth"]), nanmean0(d["fe_pred"])
        ax.plot(t, ft, color=tc, lw=2.2, label="ground truth")
        ax.plot(t, fp, color=pc, lw=2.0, ls="--", label=name)
        fe_src = "every rollout time"
        sub = "seed mean at every rollout time"
    else:
        xs = np.array([x for _, x in show])
        ft = np.array([free_energy_density(
            d["snap_truth"][:, j].astype(np.float64)).mean() for j, _ in show])
        fp = np.array([free_energy_density(
            d["snap_pred"][ok, j].astype(np.float64)).mean() for j, _ in show])
        ax.plot(xs, ft, "-o", color=tc, lw=2.2, ms=7, label="ground truth")
        ax.plot(xs, fp, "--s", color=pc, lw=2.0, ms=6, label=name)
        fe_src = "snapshot frames only"
        sub = ("from the snapshot frames only (this cache predates the "
               "per-time free energy)")
    mono = bool(np.all(np.diff(fp) < 1e-12))
    mark_time(ax, meta, d, spans)
    ax.set_xscale("log")
    ax.set_title(f"(e) free-energy decay, {sub}\n"
                 f"prediction monotone: {'yes' if mono else 'no'}",
                 fontsize=11)
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$\langle f\rangle$")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    comp = composition(d)
    mt, mp = d["mean_psi_truth"].mean(axis=0), d["mean_psi_pred"].mean(axis=0)
    psi0 = float(mt[0])
    ax.plot(t, mt, color=tc, lw=2.2, label="ground truth")
    ax.plot(t, mp, color=pc, lw=2.0, ls="--", label=name)
    ax.axhline(psi0, color="0.35", lw=1.0, ls=":",
               label=f"initial mean = {psi0:.4f}")
    swing = float(np.abs(np.concatenate([mt, mp]) - psi0).max())
    half = max(1.3 * swing, 1e-3)
    ax.set_ylim(psi0 - half, psi0 + half)
    mark_time(ax, meta, d, spans)
    sec = ax.secondary_yaxis("right", functions=(psi_to_a, a_to_psi))
    sec.set_ylabel("material A content  $c_A$  (%)")
    a_t, a_p = comp["A_percent_end_truth"], comp["A_percent_end_pred"]
    ax.set_title(
        "(f) mean order parameter, seed mean\n"
        f"composition at $t$ = {t1:g} -- truth  A {a_t:.2f} / "
        f"B {100 - a_t:.2f} %   model  A {a_p:.2f} / B {100 - a_p:.2f} %",
        fontsize=11)
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$\langle\psi\rangle$")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return dict(zero_crossing=g, free_energy_source=fe_src,
                free_energy_monotone_pred=mono,
                snapshot_L_zero=[dict(t=float(x), truth=a, pred=b)
                                 for (_, x), (a, b) in zip(show, snap_L)])


def fig_composition(plt, d, meta, path):
    """Seed-mean <psi>(t), truth against prediction, with the A / B scale."""
    name = DRAW["model_name"]
    tc, pc = DRAW["kin_truth"], DRAW["kin_pred"]
    t = d["times"]
    t0, t1, _ = horizons(meta, d)
    tru, pre = d["mean_psi_truth"], d["mean_psi_pred"]
    mt, mp = nanmean0(tru), nanmean0(pre)
    st, sp = sd_of(tru, 0), sd_of(pre, 0)
    psi0 = float(mt[0])
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.fill_between(t, mt - st, mt + st, color=tc, alpha=0.18, lw=0)
    ax.fill_between(t, mp - sp, mp + sp, color=pc, alpha=0.18, lw=0)
    ax.plot(t, mt, color=tc, lw=2.2, label="ground truth, mean $\\pm$ 1 sd")
    ax.plot(t, mp, color=pc, lw=2.0, ls="--",
            label=f"{name}, mean $\\pm$ 1 sd")
    ax.axhline(psi0, color="0.35", lw=1.0, ls=":",
               label=f"initial mean = {psi0:.4f}")
    swing = float(np.nanmax(np.abs(np.concatenate([mt, mp]) - psi0)
                            + np.concatenate([st, sp])))
    half = max(1.15 * swing, 1e-3)
    ax.set_ylim(psi0 - half, psi0 + half)
    ax.set_xscale("log")
    mark_time(ax, meta, d, [])
    sec = ax.secondary_yaxis("right", functions=(psi_to_a, a_to_psi))
    sec.set_ylabel("material A content  $c_A$  (%)")
    a0t, b0t = ab_pct(float(mt[0]))
    a0p, b0p = ab_pct(float(mp[0]))
    a1t, b1t = ab_pct(float(mt[-1]))
    a1p, b1p = ab_pct(float(mp[-1]))
    ax.set_title(
        f"Composition, {case_title(d, meta)}: mean over {len(d['seeds'])} "
        f"seeds\n$t$ = {t0:g}: truth A {a0t:.3f} / B {b0t:.3f} %,  {name} "
        f"A {a0p:.3f} / B {b0p:.3f} %\n$t$ = {t1:g}: truth A {a1t:.3f} / "
        f"B {b1t:.3f} %,  {name} A {a1p:.3f} / B {b1p:.3f} %",
        fontsize=11)
    ax.set_xlabel("$t$  (log)")
    ax.set_ylabel(r"$\langle\psi\rangle$,  $\psi = c_A - c_B$")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_composition_all(plt, comps, meta, path):
    """c_A %(t) of every composition on one axis: truth solid, UNO dashed."""
    name = DRAW["model_name"]
    t0, t1, tt = horizons(meta)
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for i, d in enumerate(comps):
        c = PAPER_C[i % len(PAPER_C)]
        lab = rf"$\bar\psi_0 = {d['off']:+.1f}$"
        ax.plot(d["times"], psi_to_a(nanmean0(d["mean_psi_truth"])), color=c,
                lw=2.2, label=f"{lab} truth")
        ax.plot(d["times"], psi_to_a(nanmean0(d["mean_psi_pred"])), color=c,
                lw=1.8, ls="--", label=f"{lab} {name}")
    if tt and t0 < tt < t1:
        ax.axvline(tt, color="0.25", ls=":", lw=1.4,
                   label=f"training horizon $t$ = {tt:g}")
    ax.set_xscale("log")
    ax.set_xlabel("$t$  (log)")
    ax.set_ylabel("material A content  $c_A$  (%)")
    ax.set_title(f"Cahn-Hilliard {name} - mixture composition of the five "
                 f"quenches\nseed mean, solid: ground truth, dashed: "
                 f"{name}; $c_A = (1 + \\psi)/2$", fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.14))
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def sweep_panel(ax, series, meta, label, thr):
    t0, t1, tt = horizons(meta)
    vals = []
    for i, (lab, d) in enumerate(series):
        m, s = d["r2"].mean(axis=0), sd_of(d["r2"], 0)
        c = PAPER_C[i % len(PAPER_C)]
        ax.fill_between(d["times"], m - s, m + s, color=c, alpha=0.13, lw=0)
        ax.plot(d["times"], m, color=c, marker=PAPER_M[i % len(PAPER_M)],
                markevery=DRAW["markevery"], label=lab)
        vals.append(m - s)
    ax.axhline(thr, color="0.35", ls="--", lw=0.9,
               label=f"threshold {thr:.2f}")
    if tt and t0 < tt < t1:
        ax.axvline(tt, color="0.25", ls=":", lw=1.0,
                   label=f"training horizon {tt:g}")
    ax.set_ylim(*r2_limits(vals))
    ax.set_xlim(t0, t1)
    ax.set_ylabel("$R^2$")
    ax.text(0.97, 0.95, label, transform=ax.transAxes, ha="right", va="top")
    ax.legend(loc="lower left", borderaxespad=0.6, ncol=2, columnspacing=1.0)


def fig_r2_sweep(plt, comps, boxes, meta, base):
    """Mean R^2(t) +- 1 sd: (a) compositions, (b) box sizes."""
    thr = float(meta["threshold"])
    top = [(rf"$\bar\psi_0 = {d['off']:+.1f}$", d) for d in comps]
    bot = [(f"$l = {d['l']}$", d) for d in boxes]
    with plt.rc_context(PAPER_RC):
        fig, axs = plt.subplots(2 if bot else 1, 1, figsize=(4.4, 6.6 if bot
                                                              else 3.4),
                                sharex=True, squeeze=False)
        fig.subplots_adjust(hspace=0.09)
        sweep_panel(axs[0, 0], top, meta, "(a)", thr)
        if bot:
            sweep_panel(axs[1, 0], bot, meta, "(b)", thr)
        axs[-1, 0].set_xlabel("$t$")
        fig.savefig(base + ".png", dpi=DRAW["sweep_dpi"])
        fig.savefig(base + ".pdf")
        plt.close(fig)


def fig_comparison(plt, runs, path):
    """Critical mean R^2(t) of every run on one log-t axis."""
    name = DRAW["model_name"]
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    vals = []
    ends: list[Any] = []
    for r in runs:
        _, _, tt = horizons(r["meta"], r["crit"])
        if tt not in ends:
            ends.append(tt)
    for r in runs:
        d, meta = r["crit"], r["meta"]
        _, t1, tt = horizons(meta, d)
        lap = run_variant(r["label"], meta) == "lap"
        c = RUN_COLORS[ends.index(tt) % len(RUN_COLORS)]
        m = d["r2"].mean(axis=0)
        vals.append(m)
        ax.plot(d["times"], m, lw=2.2, color=c, ls="-" if lap else "--",
                label=f"{r['label']}: trained to $t$ = {fmt(tt, 'g')} "
                      f"({variant_text(run_variant(r['label'], meta))}), "
                      f"{len(d['seeds'])} seeds, $R^2({t1:g})$ = {m[-1]:.3f}"
                      f"{d.get('crit_note', '')}")
        if tt:
            ax.axvline(tt, color=c, ls=":", lw=1.4)
    thr = float(runs[0]["meta"]["threshold"])
    ax.axhline(thr, color="0.4", ls="--", lw=1.2,
               label=f"threshold $R^2$ = {thr:.2f}")
    ax.set_xscale("log")
    ax.set_ylim(*r2_limits(vals))
    ax.set_xlabel("time  $t$  (log)", fontsize=12)
    ax.set_ylabel(r"mean $R^2$ vs. ground truth", fontsize=12)
    ax.grid(alpha=0.3, ls=":", which="both")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"Cahn-Hilliard {name} - critical quench, every run "
                 f"(solid: {LAP_NAME}, dashed: base; dotted: training "
                 f"horizon)", fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_run(run, snaps=True):
    """Every case of a complete run: (compositions, boxes incl. l128)."""
    meta, cdir = run["meta"], run["cdir"]
    comps, boxes = [], []
    for case, c in meta["cases"].items():
        d = load_case(cdir, case, case_seeds(meta, case), meta, snaps=snaps)
        d["kind"] = c.get("kind", "composition")
        (comps if d["kind"] == "composition" else boxes).append(d)
    comps.sort(key=lambda d: d["off"])
    boxes.sort(key=lambda d: d["l"])
    box_off = float(meta.get("box_off", 0.0))
    base = [d for d in comps if abs(d["off"] - box_off) < 1e-9]
    return comps, base + boxes


def draw_run(plt, run, results):
    """Every figure of one run folder, its manifest and report.txt."""
    label, meta = run["label"], run["meta"]
    out = os.path.join(results, label)
    os.makedirs(out, exist_ok=True)
    print(f"[draw] {label} -> {out}", flush=True)
    comps, boxes = load_run(run)
    thr = float(meta["threshold"])
    cases: dict[str, Any] = {}
    files: list[str] = []
    for d in comps + [b for b in boxes if b["kind"] == "box"]:
        case = d["case"]
        c = meta["cases"][case]
        st = case_stats(d, thr)
        spans, first = finite_size(d)
        paths = {k: os.path.join(out, f"{k}_{case}.png") for k in FIG_KINDS}
        main, long, every = snapshot_sets(d, meta)
        snap = fig_snapshots(plt, d, meta, st, paths["snapshots"], times=main)
        if len(long) >= 2:
            snap_long = fig_snapshots(plt, d, meta, st,
                                      paths["snapshots_long"], times=long,
                                      every=every)
        else:
            snap_long = None
            del paths["snapshots_long"]
            print(f"[draw]   {case}: snapshots_long skipped, the cache holds "
                  f"{len(long)} long snapshot time(s) (< 2)", flush=True)
        fig_composition(plt, d, meta, paths["composition"])
        pdf = fig_pdf(plt, d, meta, paths["psi_distribution"])
        grw = fig_growth(plt, d, meta, paths["domain_growth"], spans)
        kin = fig_kinetics(plt, d, meta, paths["phase_ordering_kinetics"],
                           spans)
        files += [os.path.basename(p) for p in paths.values()]
        sch = case_schedule(meta, c)
        cases[case] = dict(
            kind=d["kind"], lattice=d["l"], psi0=d["off"],
            schedule=dict(alpha=d["alpha"], dt_min=sch.get("dt_min"),
                          dt_max=d["dt_max"], steps=st["steps"],
                          source=sch.get("source")),
            **st, growth=dict(half_max=grw, zero_crossing=kin["zero_crossing"]),
            composition=composition(d),
            finite_size=dict(L_limit=d["l"] / DRAW["min_domains"],
                             first_time_mean_L_zero_above=first,
                             spans=spans),
            composition_series=composition_series(d, meta),
            snapshots=snap, snapshots_long=snap_long, psi_pdf=pdf,
            snapshot_times_main=main, snapshot_times_long=long,
            free_energy=dict(source=kin["free_energy_source"],
                             monotone_pred=kin["free_energy_monotone_pred"]),
            snapshot_L_zero=kin["snapshot_L_zero"],
            gt_devices=c.get("gt_devices"))
        print(f"[draw]   {case}: R2({d['times'][-1]:g}) = "
              f"{st['r2_at_end']['mean']:.4f} +- {st['r2_at_end']['sd']:.4f}"
              f", best seed {st['seeds']['best']['seed']}", flush=True)
    fig_r2_sweep(plt, comps, boxes, meta, os.path.join(out, "r2_sweep"))
    files += ["r2_sweep.png", "r2_sweep.pdf"]
    fig_composition_all(plt, comps, meta, os.path.join(out,
                                                       "composition_all.png"))
    write_composition_csv(os.path.join(out, "composition.csv"), cases)
    files += ["composition_all.png", "composition.csv"]
    t0, t1, tt = horizons(meta)
    aliases = {f"l{b['l']}": b["case"] for b in boxes
               if b["kind"] == "composition"}
    man = dict(
        label=label, cache=run["cdir"], checkpoint=meta.get("checkpoint"),
        checkpoint_fingerprint=meta.get("fingerprint"), train_t_end=tt,
        t_start=t0, t_end=t1, pdf_time=t1, lattice=meta.get("base_l"),
        box_off=meta.get("box_off"), threshold=thr,
        snapshot_times=meta.get("snapshots"),
        snapshot_times_main={k: c["snapshot_times_main"]
                             for k, c in cases.items()},
        snapshot_times_long={k: c["snapshot_times_long"]
                             for k, c in cases.items()},
        seeds=meta_seeds(meta),
        n_seeds=len(meta_seeds(meta)),
        schedules={k: case_schedule(meta, c) for k, c in meta["cases"].items()},
        best_seed_rule=("per case, the seed with the highest trajectory-mean "
                        "R^2 (mean of R^2 over every rollout time); median "
                        "and worst by the same measure"),
        figures=dict(best_seed=["snapshots_<case>.png",
                                "snapshots_long_<case>.png"],
                     seed_average=["phase_ordering_kinetics_<case>.png",
                                   "domain_growth_<case>.png",
                                   "psi_distribution_<case>.png",
                                   "composition_<case>.png",
                                   "composition_all.png",
                                   "r2_sweep.png", "r2_sweep.pdf"],
                     data=["composition.csv"]),
        aliases=aliases, cases=cases, gt_devices=meta.get("gt_devices"),
        files=files + ["manifest.json", "report.txt"],
        written=f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    dump_json(os.path.join(out, "manifest.json"), man)
    rpath = results_json(label, run["work"])
    res = read_json(rpath) if os.path.exists(rpath) else None
    write_report(os.path.join(out, "report.txt"),
                 model_card(meta, run["work"], label),
                 man, res)
    return man


def model_card(meta, work, label=""):
    """meta, plus n_params and training_config from the model folder.

    Caches written before uno_predict recorded them (the migrated t300 one)
    lack both.  They are then read from Work/models/train_t<END>[_lap]/, but only
    when that folder's uno_ch_best.pt is the checkpoint the cache was built
    from (same fingerprint): its uno_ch_train_config.json and the last
    parameter count in its uno_ch_train.log.
    """
    if meta.get("n_params") and meta.get("training_config"):
        return meta
    tt = meta.get("train_t_end")
    if not tt:
        return meta
    variant = run_variant(label, meta)
    mdir = os.path.join(work, "models", train_dir_name(tt, variant))
    best = os.path.join(mdir, "uno_ch_best.pt")
    if not (os.path.isfile(best) and fingerprint(best) == meta.get("fingerprint")):
        return meta
    out = dict(meta)
    cpath = os.path.join(mdir, "uno_ch_train_config.json")
    if not out.get("training_config") and os.path.isfile(cpath):
        tcfg = read_json(cpath)
        out["training_config"] = {k: tcfg[k] for k in TRAIN_KEYS if k in tcfg}
    lpath = os.path.join(mdir, "uno_ch_train.log")
    if not out.get("n_params") and os.path.isfile(lpath):
        with open(lpath, errors="replace") as f:
            hits = [ln.split()[1] for ln in f
                    if ln.startswith("[model]") and " parameters" in ln]
        if hits:
            out["n_params"] = int(hits[-1].replace(",", ""))
    return out


def wrap_text(text, width=74):
    return textwrap.wrap(str(text), width)


def write_report(path, meta, man, res):
    """Plain-text model card, conservation, solve time, summary table.

    Built from the rollout meta, the manifest (numbers from the rollout
    cache) and, when uno_predict wrote one, results.json (diagnostics).
    """
    lines: list[str] = []
    w = lines.append
    res = res or {}
    t0, t1, tt = man["t_start"], man["t_end"], man["train_t_end"]
    w("=" * 78)
    w(f" UNO / Cahn-Hilliard -- MODEL REPORT   {man['label']}")
    w(f" generated {datetime.now():%Y-%m-%d %H:%M:%S}")
    w("=" * 78)
    w("")
    w("1. TRAINED MODEL (MODEL CARD)")
    w("-" * 78)
    w(f"  checkpoint                {meta.get('checkpoint')}")
    w(f"  fingerprint (sha256[:12]) {meta.get('fingerprint')}")
    n_par = meta.get("n_params") or res.get("n_params")
    w(f"  trainable parameters      {n_par:,}" if n_par else
      "  trainable parameters      n/a")
    w(f"  precision                 {meta.get('precision')}")
    w(f"  training data ends at     t = {fmt(tt, 'g')}")
    w(f"  model variant             "
      f"{variant_text(run_variant(man['label'], meta))}"
      + ("  (the head predicts M, the increment is its 5-point Laplacian)"
         if run_variant(man["label"], meta) == "lap" else
         "  (the head predicts the increment directly)"))
    w(f"  evaluated on lattice      l = {meta.get('base_l')}  (box sizes "
      + ", ".join(str(c["lattice"]) for c in man["cases"].values()
                  if c["kind"] == "box") + ")")
    w(f"  rollout horizon           t = {t0:g} -> {t1:g}")
    w(f"  test seeds                {man['n_seeds']}")
    w(f"  ground truth solved on    {meta.get('gt_devices') or 'n/a'}")
    w("")
    w("  architecture")
    for k, v in (meta.get("arch") or {}).items():
        w(f"      {k:<22}{v}")
    met = meta.get("training_metrics") or {}
    if met:
        w("")
        w("  checkpoint selection (held-out 128-box validation rollout)")
        for k in ("select", "epoch", "survives", "per_seed", "r2_in_range",
                  "in_range_t1", "rollout", "val1"):
            if k in met:
                v = met[k]
                if isinstance(v, list):
                    v = [round(float(x), 1) for x in v]
                elif isinstance(v, float):
                    v = round(v, 4)
                w(f"      {k:<22}{v}")
    tcfg = meta.get("training_config") or (res.get("training") or {}).get(
        "config") or {}
    if tcfg:
        w("")
        w("  training configuration")
        for k, v in tcfg.items():
            w(f"      {k:<22}{v}")
    w("")
    w("2. MIXTURE COMPOSITION AND ITS CONSERVATION")
    w("-" * 78)
    w("  The system is a binary mixture of two materials, A and B, whose")
    w("  phase separation this model predicts. psi is their composition")
    w("  contrast, psi = c_A - c_B with c_A + c_B = 1, so")
    w("      c_A = (1 + psi)/2        c_B = (1 - psi)/2")
    w("  Cahn-Hilliard is Model B, so <psi> is a constant of motion: the")
    w("  mixture cannot change composition as it separates. The value of")
    w("  <psi> at the final time is therefore the composition the model")
    w("  has ended up simulating, and it should equal the one it started")
    w(f"  from. Values below are averaged over the test seeds; the end is")
    w(f"  t = {t1:g}.")
    w("")
    w(f"    {'case':>10}  {'A / B at start':>18}  {'A / B at end':>18}"
      f"  {'shift in A':>13}")
    rows = []
    for case, c in man["cases"].items():
        cp = c["composition"]
        a0, a1 = cp["A_percent_start_truth"], cp["A_percent_end_pred"]
        rows.append((a1 - a0, cp))
        w(f"    {case:>10}  {a0:>7.3f} / {100 - a0:<8.3f}"
          f"  {a1:>7.3f} / {100 - a1:<8.3f}  {a1 - a0:>+12.5f} pp")
    if rows:
        worst = max(rows, key=lambda r: abs(r[0]))
        w("")
        w(f"  largest composition shift              {worst[0]:+.5f} "
          f"percentage points")
        w(f"  largest drift |<psi>(t) - psi0|        "
          f"{max(r[1]['max_abs_psi_drift_pred'] for r in rows):.3e}")
        w(f"  drift rate d<psi>/dt (linear fit)      "
          f"{max((r[1]['psi_drift_rate_pred'] for r in rows), key=abs):.3e}"
          f" per unit t")
        w(f"  conserve_mass in this checkpoint       "
          f"{(meta.get('arch') or {}).get('conserve_mass')}")
        w(f"  laplacian_output in this checkpoint    "
          f"{(meta.get('arch') or {}).get('laplacian_output', False)}")
        w("")
        w("  'pp' is percentage points of material A. A shift of +1 pp")
        w("  means the model is separating a mixture 1 % richer in A than")
        w("  the one it was given.")
        w("")
        w("  Convention: psi = c_A - c_B and c_A + c_B = 1, so c_A = (1 +")
        w("  psi)/2 and c_B = (1 - psi)/2: psi = 0 is 50 / 50 and psi =")
        w("  +0.05 is 52.5 % A / 47.5 % B.")
        w("")
        w("  Seed-mean composition over time, A % / B % (the same rows are")
        w("  in composition.csv):")
        for case, c in man["cases"].items():
            ser = c.get("composition_series") or []
            if not ser:
                continue
            w("")
            w(f"    {case}   (psi0 = {ser[0]['psi0']:+.4f})")
            w(f"    {'t':>8}  {'truth A / B':>19}  {'UNO A / B':>19}")
            for r in ser:
                w(f"    {r['t']:>8g}  {r['A_pct_truth']:>8.3f} / "
                  f"{r['B_pct_truth']:<8.3f}  {r['A_pct_pred']:>8.3f} / "
                  f"{r['B_pct_pred']:.3f}")
    w("")
    w("3. SOLVE TIME -- UNO SURROGATE vs GROUND-TRUTH SOLVER")
    w("-" * 78)
    sp = res.get("speedup")
    if not res:
        w("  not measured: no results.json from uno_predict for this run")
        w(f"  (expected at {results_json(man['label'], '<work>')}).")
    elif sp and sp.get("skipped"):
        w(f"  not measured ({sp['skipped']}).")
        w("  Timing the float64 solver and an MPS surrogate on different")
        w("  devices would report the MPS speedup as if it were the")
        w("  surrogate's. Measure this on the GPU node instead.")
        if res.get("speedup_note"):
            w("")
            for ln in wrap_text(res["speedup_note"]):
                w(f"  {ln}")
    elif not sp:
        w("  not measured (CONFIG['do_speedup'] is False)")
    else:
        pad = " " * 26
        w(f"  ground-truth solver     {sp['solver_steps']:,} explicit steps of "
          f"dt = {res.get('solver_dt', 0.01):g}")
        w(f"{pad}{sp['solver_seconds_per_trajectory']:.3f} s per trajectory")
        w("")
        w(f"  UNO surrogate           {sp['uno_steps']} autoregressive steps")
        w(f"{pad}{sp['uno_seconds_per_trajectory']:.3f} s per trajectory "
          f"(one at a time)")
        w(f"{pad}{sp['uno_seconds_per_trajectory_batched']:.3f} s per "
          f"trajectory (batch of {sp['batch_size']})")
        w("")
        w(f"  speedup                 {sp['speedup']:.1f}x like-for-like "
          f"-- quote this one, both unbatched")
        w(f"{pad}{sp['speedup_batched']:.1f}x with the surrogate batched")
        w(f"{pad}{sp['solver_steps'] / max(1, sp['uno_steps']):.0f}x "
          f"fewer time steps")
        w("")
        w("  The baseline is the explicit dt = 0.01 scheme ch.py integrates,")
        w("  not a semi-implicit spectral solver, which would need far fewer")
        w("  steps. Both timings are measured on the device below.")
        w(f"  device                  {res.get('device')}")
    w("")
    w(f"4. SUMMARY -- every case, t = {t0:g} -> {t1:g}, R^2 threshold "
      f"{man['threshold']:.2f}")
    w("-" * 78)
    w(f"  {'case':>10} {'alpha':>5} {'dt_max':>6} {'steps':>5} "
      f"{'R2(t_end)':>16} {'median':>7} {'best':>6} {'worst':>6} "
      f"{'mean R2':>15} {'survives to t':>15} {'never':>6} {'n pred':>7}")
    for case, c in man["cases"].items():
        e, m, s = c["r2_at_end"], c["r2_trajectory_mean"], c["survival"]
        sc = c["schedule"]
        w(f"  {case:>10} {sc['alpha']:5.2f} {sc['dt_max']:6.1f} "
          f"{c['steps']:5d} {e['mean']:8.4f} +-{e['sd']:.4f} "
          f"{e['median']:7.4f} {e['best']:6.3f} {e['worst']:6.3f} "
          f"{m['mean']:7.4f} +-{m['sd']:.4f} {s['mean']:7.0f} +-{s['sd']:<5.0f}"
          f" {s['n_survive_to_end']:>2d}/{c['n_seeds']:<3d}"
          f" {fmt(c['growth']['zero_crossing']['n_pred'], '7.4f')}")
    w("")
    w("  R2(t_end): mean +- sd over seeds of each seed's R^2 at the end of")
    w("  the rollout. mean R2: the trajectory mean (mean of R^2 over every")
    w("  rollout time), the measure the best seed is picked by. survives:")
    w("  first time R^2 drops below the threshold (t_end if never); never:")
    w("  seeds that stay above it to t_end. n pred: fitted growth exponent")
    w("  of the zero-crossing L over the late window [t_end/4, t_end].")
    diag = [k for k in ("transfer", "drift", "predictability") if res.get(k)]
    if diag:
        w("")
        w("5. DIAGNOSTICS (reference trajectory, see predict.log)")
        w("-" * 78)
        if res.get("results_note"):
            for ln in wrap_text(res["results_note"]):
                w(f"  {ln}")
            w("")
        tr = res.get("transfer") or {}
        if "tile_invariance" in tr:
            w(f"  tile invariance            {tr['tile_invariance']:.3e}")
        dr = res.get("drift") or {}
        for k in ("mean_bias_inside_pct", "mean_bias_outside_pct"):
            if k in dr:
                w(f"  {k:<27}{dr[k]:+.1f} %")
        for k, v in (res.get("predictability") or {}).items():
            w(f"  predictability {k:<12}"
              + "  ".join(f"{kk} {vv:.5f}" for kk, vv in v.items()))
    lk = res.get("lowk")
    if lk:
        w("")
        w("6. LOW-k INCREMENT RESPONSE (teacher-forced, per wavenumber shell)")
        w("-" * 78)
        for ln in wrap_text(
                "gain g = sum Re(conj(dP) dT) / sum |dT|^2 over the Fourier "
                "modes of a shell and over the tune seeds, dP = model "
                "increment, dT = true increment over dt = "
                f"{lk['dt']:g} from the ground-truth frame at t0. A "
                "well-trained model has g ~ 1 in every shell; the base "
                "(direct-increment) models showed g = 2.4-3.5 in the lowest "
                "shell, the 8 modes of the l = 128 box below the l = 64 "
                "training box's fundamental."):
            w(f"  {ln}")
        w("")
        shells = lk.get("shells") or list(next(iter(lk["gain"].values())))
        w(f"  psi0 = {lk['off']:+.1f}, l = {lk['l']}, "
          f"{len(lk['seeds'])} seed(s); below = |n| in {{1, sqrt 2}}, "
          f"first = [2, 2.9], mid = [3, 8], high = rest")
        w(f"  {'t0':>7}" + "".join(f"{k:>9}" for k in shells))
        for t0, g in lk["gain"].items():
            w(f"  {t0:>7}" + "".join(f"{g[k]:9.3f}" for k in shells))
    w("")
    w("=" * 78)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def run_summary(run):
    """Headline numbers of a complete run, from its cache (no snapshots)."""
    comps, boxes = load_run(run, snaps=False)
    thr = float(run["meta"]["threshold"])
    box_off = float(run["meta"].get("box_off", 0.0))
    crit = next((d for d in comps if abs(d["off"] - box_off) < 1e-9),
                comps[0] if comps else boxes[0])
    crit["crit_note"] = ("" if abs(crit["off"] - box_off) < 1e-9 else
                         f" (psi0 = {crit['off']:+.1f}: no critical case)")
    return dict(label=run["label"], meta=run["meta"], crit=crit,
                comps=[(d, case_stats(d, thr)) for d in comps],
                boxes=[(d, case_stats(d, thr)) for d in boxes])


def write_readme(summaries, pending, results, work, path):
    name = DRAW["model_name"]
    L = [f"# Results - Cahn-Hilliard {name}", "",
         "Report material only. Every file here is regenerated from the "
         "rollout", f"caches in `{rollout_root(work)}/` by "
         "`python make_submission.py`; no model is", "loaded and no "
         "trajectory is re-solved. One folder per run, named", "`train_t<END>"
         "[_lap]_pred_t<T>`: a model trained on data to t = END,", "rolled "
         "out to t = T. `_lap` marks the Laplacian-output model (the head "
         "predicts", "M and the increment is its 5-point Laplacian); runs "
         "without it are the base", "model (direct increment) and are kept "
         "for comparison.", "", "## Runs", "",
         "| run | model | trained to | predicted to | seeds | critical "
         "R2(t_end) | folder |", "|---|---|---|---|---|---|---|"]
    for s in summaries:
        m = s["meta"]
        _, t1, tt = horizons(m)
        e = case_stats(s["crit"], float(m["threshold"]))
        L.append(f"| {s['label']} | "
                 f"{variant_text(run_variant(s['label'], m))} | "
                 f"t = {fmt(tt, 'g')} | t = {t1:g} | "
                 f"{len(s['crit']['seeds'])} | R2({t1:g}) = "
                 f"{e['r2_at_end']['mean']:.4f} +- {e['r2_at_end']['sd']:.4f}"
                 f"{s['crit'].get('crit_note', '')} | `{s['label']}/` |")
    for lab, why in pending:
        L.append(f"| {lab} | - | - | - | - | not complete: {why} | - |")
    L += ["", "R2(t_end) is the mean over seeds of each seed's R2 against "
          "the ground", "truth at the end of that run's rollout, +- one "
          "standard deviation", "across seeds. Runs with different horizons "
          "are not comparable at", "their end points; `comparison.png` puts "
          "the critical curves of every", "run on one log-t axis instead. "
          "In the per-run tables, max and min", "are the extreme seeds' "
          "R2(t_end); the best seed is chosen by", "trajectory-mean R2, so "
          "it need not be the one with the max.", ""]
    for s in summaries:
        m = s["meta"]
        _, t1, tt = horizons(m)
        L += [f"## {s['label']}", "",
              f"Trained to t = {fmt(tt, 'g')} "
              f"({variant_text(run_variant(s['label'], m))}), rolled out "
              f"from t = {float(m['t_start']):g} to t = {t1:g}; checkpoint "
              f"`{m.get('fingerprint')}`.", "",
              f"| case | R2(t = {t1:g}) | median | max | min "
              f"| trajectory-mean R2 | best seed (mean R2) |",
              "|---|---|---|---|---|---|---|"]
        rows = [(d["case"], st) for d, st in s["comps"]]
        rows += [(d["case"], st) for d, st in s["boxes"]
                 if d["kind"] == "box"]
        for case, st in rows:
            e, tm = st["r2_at_end"], st["r2_trajectory_mean"]
            b = st["seeds"]["best"]
            L.append(f"| {case} | {e['mean']:.4f} +- {e['sd']:.4f} | "
                     f"{e['median']:.4f} | {e['best']:.4f} | "
                     f"{e['worst']:.4f} | {tm['mean']:.4f} +- {tm['sd']:.4f}"
                     f" | {b['seed']} ({b['r2_trajectory_mean']:.4f}) |")
        L.append("")
    L += ["## What each run folder holds", "",
          "Cases: `psi0m0.40` ... `psi0p0.40` are the five compositions on "
          "the", "128 x 128 test box; `l160`, `l192`, `l256` are the "
          "critical quench on", "larger boxes with the critical quench's "
          "own stride schedule. In the", "box-size plots `l128` is the "
          "critical composition `psi0p0.00`.", "",
          "Best seed (one seed per case, the highest trajectory-mean R2, "
          "i.e.", "the mean of R2 over every rollout time):", "",
          "- `snapshots_<case>.png` - ground truth, prediction and absolute "
          "error", "  at the main snapshot times (100, 500, 1000, 2000 for "
          "a new run); the", "  title names the best, median and worst "
          "seed with their mean R2",
          "- `snapshots_long_<case>.png` - the same rows at the long "
          "snapshots,", "  every 2000 up to t_end (2000, 4000, ..., 10000); "
          "absent for older", "  caches that did not store them", "",
          "Seed averages (every test seed):", "",
          "- `r2_sweep.png` / `.pdf` - mean R2(t) +- 1 sd, (a) by "
          "composition,", "  (b) by box size; dashed: threshold, dotted: "
          "training horizon",
          "- `phase_ordering_kinetics_<case>.png` - Puri ch. 1: (a) "
          "zero-crossing", "  L(t) +- 1 sd with fitted exponents, (b) S(k,t)"
          ", (c) scaling collapse,", "  (d) C(r,t) vs r/L, (e) free energy, "
          "(f) <psi>(t) and the A/B", "  composition",
          "- `composition_<case>.png` - seed-mean <psi>(t), truth vs "
          "prediction", "  +- 1 sd, with the A / B percent axis and the "
          "A / B split at", "  t_start and t_end; `composition_all.png` - "
          "c_A %(t) of the five", "  compositions on one axis; "
          "`composition.csv` - the same numbers at the", "  snapshot times",
          "- `domain_growth_<case>.png` - Puri's half-maximum L(t) +- 1 sd "
          "against", "  the Lifshitz-Slyozov t^(1/3) law",
          "- `psi_distribution_<case>.png` - P(psi) at t_end, all seeds "
          "pooled", "",
          "Plus `report.txt` (model card, A/B composition and its "
          "conservation,", "solve time, summary table, low-k increment "
          "response) and `manifest.json` "
          "(every number the", "figures show, the seeds, schedules and the "
          "best / median / worst", "seed ids). Growth exponents are fitted "
          "over [t_end/4, t_end] and", "over the training window [t_start, "
          "train_t_end].", "",
          f"`train_t0300_pred_t02000/{EARLIER}/` holds graphs kept from the",
          "Sep 8 run (10 seeds, single-seed physics panels, box curves on "
          "the", "+0.4 schedule); see its NOTE.md. The drawing code never "
          "touches it.", "",
          "## Finite-size regime", "",
          "Time axes are shaded where the seed-mean zero-crossing length "
          "exceeds", f"l/{DRAW['min_domains']}, i.e. fewer than "
          f"{DRAW['min_domains']} domains across the box. From there on "
          "the", "box, not the physics, limits coarsening: l = 128 enters "
          "it around", "t ~ 3400. Nothing is dropped; the larger boxes "
          "show the trend. The", "manifest records the first such time per "
          "case.", "",
          "## Regenerate", "",
          "    python make_submission.py                  redraw every "
          "complete run",
          "    python make_submission.py --run LABEL      redraw one run",
          "    python uno_predict.py --end N              tune, roll out, "
          "diagnose, draw",
          "    python make_submission.py --rollout --end N   refill a "
          "rollout cache", "    (add --variant base for the old "
          "architecture; default is lap)", "",
          "## Where everything else lives", "",
          f"- `{work}/rollout_cache/<run>/` - per-seed rollout cache the "
          "figures are", "  drawn from (`<case>/seed<N>.npz` + `meta.json`)",
          f"- `{work}/predict/<run>/` - `results.json` and `predict.log`",
          f"- `{work}/models/train_t<END>[_lap]/` - checkpoints, training log "
          "and memmaps",
          f"- `{work}/gt_cache/` - the ground-truth cache shared by every "
          "run",
          f"- `{work}/data/` - datasets written by `ch_gpu.py`",
          f"- `{ARCHIVE}/` - earlier Results/ and Submission/ folders, "
          "intact", ""]
    with open(path, "w") as f:
        f.write("\n".join(L).rstrip() + "\n")


def draw(labels=None, work=WORK, results=RESULTS):
    """Draw runs from <work>/rollout_cache/ into <results>/.

    labels=None draws every complete run.  README.md always covers every
    complete run; comparison.png is written when there are two or more.
    Returns the labels drawn.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = discover(work)
    for r in runs:
        r["work"] = work
    by = {r["label"]: r for r in runs}
    if labels:
        for lab in labels:
            if lab not in by:
                raise SystemExit(f"[draw] no rollout cache "
                                 f"{os.path.join(rollout_root(work), lab)}/"
                                 f"meta.json")
            if by[lab]["why"]:
                raise SystemExit(f"[draw] {lab} is not complete: "
                                 f"{by[lab]['why']}")
        todo = list(labels)
    else:
        todo = [r["label"] for r in runs if not r["why"]]
    os.makedirs(results, exist_ok=True)
    for lab in todo:
        draw_run(plt, by[lab], results)
    done = [r for r in runs if not r["why"]]
    pending = [(r["label"], r["why"]) for r in runs if r["why"]]
    for lab, why in pending:
        print(f"[draw] {lab} skipped: {why}", flush=True)
    sums = [run_summary(r) for r in done]
    if len(sums) >= 2:
        fig_comparison(plt, sums, os.path.join(results, "comparison.png"))
        print(f"[draw] comparison.png: {len(sums)} runs", flush=True)
    write_readme(sums, pending, results, work,
                 os.path.join(results, "README.md"))
    print(f"[draw] done: {len(todo)} run folder(s) drawn, README.md lists "
          f"{len(sums)} run(s) in {results}/", flush=True)
    return todo


def main():
    ap = argparse.ArgumentParser(
        description="Draw Results/ from the rollout caches (the drawing "
                    "stage); --rollout fills a cache first.")
    ap.add_argument("--run", default="",
                    help="redraw only this run label, e.g. "
                         "train_t0300_pred_t02000")
    ap.add_argument("--rollout", action="store_true",
                    help="load the checkpoint and fill the rollout cache of "
                         "run_label(--end, --t-end) first, then draw it")
    ap.add_argument("--end", type=float, default=None,
                    help="training horizon END of the model (--rollout)")
    ap.add_argument("--t-end", type=float, default=0.0,
                    help="prediction horizon (--rollout; default PAIRS[END])")
    ap.add_argument("--seeds", type=int, default=0,
                    help="roll out only the first K test seeds (default all)")
    ap.add_argument("--schedule-from", default="",
                    help="results.json whose stride schedules to reuse; "
                         "'auto' = Work/predict/<label>/results.json, "
                         "accepted only for the same checkpoint fingerprint")
    ap.add_argument("--variant", choices=("lap", "base"), default=None,
                    help="--rollout: lap (default) = Laplacian-output model "
                         "Work/models/train_t<END>_lap/, base = the old "
                         "architecture in train_t<END>/")
    ap.add_argument("--ckpt", default="", help="checkpoint to roll out")
    ap.add_argument("--smoke", action="store_true",
                    help="use Work/smoke/ and Work/smoke/Results/")
    ap.add_argument("--work", default="",
                    help="draw from <WORK>/rollout_cache/ (default Work)")
    ap.add_argument("--results", default="",
                    help="draw into this folder (default Results)")
    a = ap.parse_args()
    for k in ("ckpt", "schedule_from", "work", "results"):
        v = getattr(a, k)
        if v and v != "auto":
            setattr(a, k, os.path.abspath(v))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    work = a.work or (SMOKE_ROOT if a.smoke else WORK)
    results = a.results or (SMOKE_RESULTS if a.smoke else RESULTS)
    labels = [a.run] if a.run else None
    if a.rollout:
        if a.work:
            raise SystemExit("--work applies to the draw only; --rollout "
                             "writes Work/ (or Work/smoke/ with --smoke)")
        import uno_predict as up
        label = up.rollout_only(end=a.end, t_end=a.t_end, smoke=a.smoke,
                                ckpt=a.ckpt, seeds=a.seeds,
                                schedule_from=a.schedule_from,
                                variant=a.variant)
        labels = [label]
    draw(labels, work=work, results=results)


if __name__ == "__main__":
    sys.exit(main())
