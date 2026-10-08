#!/usr/bin/env python
"""Move the outputs of the old layout into Results/ + Work/ + Archive/.

    python migrate_layout.py            print the plan; changes nothing
    python migrate_layout.py --apply    carry the plan out

Before 2026-09-29 everything lived in Results/ (checkpoints, logs, memmaps,
the ground-truth cache and the figures), the per-seed rollout cache in
result_cache/train_t300/, the dataset in ./test.npz and the curated figures
in Submission/.  This script moves them to where the current code looks:

    Results/cache/gt_*.npz              -> Work/gt_cache/
    Results/uno_ch_best.pt, uno_ch_best_prev.pt, uno_ch_state.pt,
      uno_ch_state_stale.pt, uno_ch_train.log, uno_ch_train_config.json
                                        -> Work/models/train_t0300/
    Results/cache/train_X_f32.npy, val_X_f32.npy
                                        -> Work/models/train_t0300/cache/
    test.npz                            -> Work/data/test_t0300.npz
    result_cache/train_t300/            -> Work/rollout_cache/
                                           train_t0300_pred_t02000/
                                           (meta.json "label" rewritten)
    Results/uno_ch_results.json         copied to Work/predict/
                                           train_t0300_pred_t02000/results.json
    Results/uno_ch*_phase_ordering_kinetics.png, uno_ch_r2_sweep.png, .pdf
                                        -> Results/train_t0300_pred_t02000/
                                           earlier_run_2026-09-08/ + NOTE.md
    everything else in Results/         -> Archive/Results_2026-09-08/
    Submission/                         -> Archive/Submission_2026-09-23/

The top-level cache/, gt_cache/ and logs/ folders are reported and left alone.

Files are only moved (os.replace, or shutil.move when the destination sits on
another filesystem), so bytes and mtimes are kept and the checkpoint sha256
fingerprints stay valid.  Nothing is deleted and no existing destination is
overwritten: a clash is reported and that item skipped, the rest goes on.
Running it again is safe; what is already in place is reported as done.

Identity checks (t = 300): the model files move only when their memmaps hold
2990 frames (END = 300), test.npz only when its train_X does, and the rollout
cache only when its meta.json says t_start 100, t_end 2000, train_t_end 300.
The copied results.json gains the checkpoint's "fingerprint" when the file it
names is provably the one being moved, so `--schedule-from auto` accepts it.
"""

from __future__ import annotations

import argparse
import errno
import fnmatch
import json
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime

import numpy as np
from numpy.lib import format as npformat

from make_submission import dump_json, fingerprint

END = 300
T_START = 100.0
T_END = 2000.0
FRAMES = 10 * END - 10
LABEL = "train_t0300_pred_t02000"
OLD = "Results"
OLD_CACHE = os.path.join(OLD, "cache")
OLD_ROLLOUT = os.path.join("result_cache", "train_t300")
OLD_RES_JSON = os.path.join(OLD, "uno_ch_results.json")
OLD_DATA = "test.npz"
MODEL = os.path.join("Work", "models", "train_t0300")
MODEL_FILES = ("uno_ch_best.pt", "uno_ch_best_prev.pt", "uno_ch_state.pt",
               "uno_ch_state_stale.pt", "uno_ch_train.log",
               "uno_ch_train_config.json")
MEMMAPS = ("train_X_f32.npy", "val_X_f32.npy")
GT_DIR = os.path.join("Work", "gt_cache")
DATA = os.path.join("Work", "data", "test_t0300.npz")
ROLLOUT = os.path.join("Work", "rollout_cache", LABEL)
PREDICT = os.path.join("Work", "predict", LABEL)
EARLIER = os.path.join(OLD, LABEL, "earlier_run_2026-09-08")
KEPT = ("uno_ch*_phase_ordering_kinetics.png", "uno_ch_r2_sweep.png",
        "uno_ch_r2_sweep.pdf")
ARCH_RESULTS = os.path.join("Archive", "Results_2026-09-08")
ARCH_SUB = os.path.join("Archive", "Submission_2026-09-23")
SUBMISSION = "Submission"
LEGACY = ("cache", "gt_cache", "logs")
NEW_LAYOUT = re.compile(r"^(train_t\d{4}(?:_lap)?_pred_t\d{5}|README\.md|"
                        r"comparison\.png)$")

NOTE = """# Graphs kept from the run of 2026-09-08

These files were in `Results/` before the layout change of 2026-09-29 and were
moved here unchanged by `migrate_layout.py`:

- `uno_ch_phase_ordering_kinetics.png` (critical quench),
  `uno_ch_off<psi0>_phase_ordering_kinetics.png` (the four off-critical
  quenches) and `uno_ch_L<l>_phase_ordering_kinetics.png` (boxes 160/192/256)
- `uno_ch_r2_sweep.png` / `.pdf`

File times, kept by the move:

{files}

What they are:

- drawn by the `uno_predict.py` of that time from the t = 300 checkpoint
  (sha256[:12] `{fp}`) with **10 test seeds**, not the 50 the current figures
  use. In the Mac copy of the project that is the MPS run of 2026-09-08 13:54
  (`Archive/Results_2026-09-08/uno_ch_predict.log`); a cluster copy may hold
  the graphs of A100 job 399765 (2026-09-07) instead, which the file times
  above tell apart. Both runs share the caveats below;
- the phase-ordering-kinetics panels show **one seed** (the first test seed,
  99999), not a seed average, and their titles do not say so;
- the box-size curves (the `L<l>` panels and panel (b) of the sweep) and every
  sweep curve used **psi0 = +0.4's stride schedule** (alpha 0.10, dt_max 45),
  the last one that run tuned, not each case's own.

The seed-averaged figures in the parent folder replace them. The drawing code
(`make_submission.py`) never reads or writes this subfolder. The rest of that
run (rollout-evaluation figures, report, results.json, log) is in
`Archive/Results_2026-09-08/`.
"""


def log(*a):
    print(*a, flush=True)


def size_of(path):
    if not os.path.lexists(path):
        return 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    if os.path.islink(path):
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return total


def human(n):
    for unit, k in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= k:
            return f"{n / k:.2f} {unit}"
    return f"{n} B"


def npz_shape(path, member="train_X.npy"):
    """Shape of one member of an .npz, read from its header alone."""
    with zipfile.ZipFile(path) as zf, zf.open(member) as f:
        version = npformat.read_magic(f)
        read = (npformat.read_array_header_1_0 if version == (1, 0)
                else npformat.read_array_header_2_0)
        return tuple(read(f)[0])


def first(*paths):
    return next((p for p in paths if os.path.lexists(p)), None)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def near(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


class Plan:
    """An ordered list of steps; each knows its status before it runs."""

    def __init__(self):
        self.steps = []

    def move(self, src, dst, gate=None, group=None):
        if os.path.lexists(src) and os.path.lexists(dst):
            status = "clash"
        elif os.path.lexists(dst):
            status = "done"
        elif not os.path.lexists(src):
            status = "absent"
        elif gate is not None and not gate[0]:
            status = "refused"
        else:
            status = "todo"
        step = dict(kind="move", src=src, dst=dst, status=status,
                    why=gate[1] if gate is not None else "", group=group,
                    size=size_of(src if os.path.lexists(src) else dst))
        self.steps.append(step)
        return step

    def other(self, kind, src, dst, status, why="", **extra):
        step = dict(kind=kind, src=src, dst=dst, status=status, why=why,
                    group=None, size=0, **extra)
        self.steps.append(step)
        return step


def run_checks():
    """The t = 300 identity checks, as (model, data, cache) gates + facts."""
    out = {}
    ck = first(os.path.join(OLD, MODEL_FILES[0]),
               os.path.join(MODEL, MODEL_FILES[0]))
    out["ckpt"] = ck
    out["fp"] = fingerprint(ck) if ck else None
    log(f"[check] checkpoint {ck or 'not found'}"
        + (f": sha256[:12] {out['fp']}" if ck else ""))

    mms = {m: first(os.path.join(OLD_CACHE, m), os.path.join(MODEL, "cache", m))
           for m in MEMMAPS}
    frames = {m: (np.load(p, mmap_mode="r").shape if p else None)
              for m, p in mms.items()}
    for m, p in mms.items():
        log(f"[check] memmap {p or m + ' not found'}"
            + (f": shape {frames[m]}" if p else ""))

    meta_path = first(os.path.join(OLD_ROLLOUT, "meta.json"),
                      os.path.join(ROLLOUT, "meta.json"))
    meta = read_json(meta_path) if meta_path else {}
    out["meta"] = meta
    if meta_path:
        log(f"[check] rollout meta {meta_path}: label {meta.get('label')!r},"
            f" t_start {meta.get('t_start')}, t_end {meta.get('t_end')}, "
            f"train_t_end {meta.get('train_t_end')}, fingerprint "
            f"{meta.get('fingerprint')}")
    else:
        log("[check] rollout meta: not found")

    have = [f for f in frames.values() if f is not None]
    if have and all(len(f) > 1 and f[1] == FRAMES for f in have):
        out["model"] = (True, f"memmaps hold {FRAMES} frames = END {END}")
    elif have:
        out["model"] = (False, f"memmap frames {[f[1] for f in have]} are "
                               f"not {FRAMES}: not the END = {END} model")
    elif (meta and near(meta.get("train_t_end"), END) and out["fp"]
          and meta.get("fingerprint") == out["fp"]):
        out["model"] = (True, f"no memmap, but the rollout cache of this "
                              f"checkpoint says train_t_end {END}")
    else:
        out["model"] = (False, "no memmap and no rollout cache to prove the "
                               f"model is END = {END}")
    log(f"[check] model is t = {END}: "
        f"{'OK' if out['model'][0] else 'NO'} ({out['model'][1]})")

    if os.path.lexists(OLD_DATA):
        try:
            shape = npz_shape(OLD_DATA)
        except (OSError, KeyError, ValueError, zipfile.BadZipFile) as e:
            shape = None
            out["data"] = (False, f"cannot read train_X of {OLD_DATA}: {e}")
        if shape is not None:
            ok = len(shape) > 1 and shape[1] == FRAMES
            out["data"] = (ok, f"train_X {shape}"
                           + ("" if ok else f", not {FRAMES} frames"))
        link = (f" (symlink to {os.readlink(OLD_DATA)})"
                if os.path.islink(OLD_DATA) else "")
        log(f"[check] {OLD_DATA}{link}: "
            f"{'OK' if out['data'][0] else 'NO'} ({out['data'][1]})")
    else:
        out["data"] = (False, f"{OLD_DATA} not found")

    if meta:
        bad = [f"{k} {meta.get(k)} != {v:g}"
               for k, v in (("t_start", T_START), ("t_end", T_END),
                            ("train_t_end", END)) if not near(meta.get(k), v)]
        out["cache"] = (not bad, "; ".join(bad) or
                        f"t {T_START:g} -> {T_END:g}, trained to {END}")
        log(f"[check] rollout cache is {LABEL}: "
            f"{'OK' if not bad else 'NO'} ({out['cache'][1]})")
        if out["fp"] and meta.get("fingerprint") != out["fp"]:
            log(f"[check] WARNING: the rollout cache fingerprint "
                f"{meta.get('fingerprint')} differs from the checkpoint's "
                f"{out['fp']}; uno_predict will refuse to extend that cache")
        elif out["fp"]:
            log("[check] rollout cache fingerprint = checkpoint fingerprint")
    else:
        out["cache"] = (False, "no rollout meta.json")
    return out


def results_copy_plan(plan, facts):
    """Copy uno_ch_results.json to Work/predict/<label>/, maybe + fingerprint."""
    src = first(OLD_RES_JSON, os.path.join(ARCH_RESULTS,
                                           os.path.basename(OLD_RES_JSON)))
    dst = os.path.join(PREDICT, "results.json")
    if os.path.lexists(dst):
        return plan.other("copy", src or OLD_RES_JSON, dst, "done")
    if src is None:
        return plan.other("copy", OLD_RES_JSON, dst, "absent")
    res = read_json(src)
    fp, ck = facts["fp"], facts["ckpt"]
    add, why = None, ""
    if res.get("fingerprint"):
        why = f"it already records fingerprint {res['fingerprint']}"
    elif not fp:
        why = "no checkpoint found, so no fingerprint added"
    else:
        rec = os.path.normpath(str(res.get("checkpoint") or ""))
        tail = os.path.join(OLD, MODEL_FILES[0])
        same = (rec.endswith(os.sep + tail) or rec == tail) and (
            os.path.getmtime(ck) <= os.path.getmtime(src))
        if same:
            add = fp
            why = (f"+ fingerprint {fp}: it names {tail}, which is unchanged "
                   f"since the file was written")
        else:
            why = (f"no fingerprint added: it names {rec or 'no checkpoint'}, "
                   f"not provably {ck}")
    notes = results_notes(src, res)
    if notes.get("speedup_note"):
        why += "; + the A100 speedup note"
    return plan.other("copy", src, dst, "todo", why, add_fp=add, notes=notes)


A100_JSON = os.path.join("a100_job399765", "uno_ch_results.json")


def results_notes(src, res):
    """Provenance notes for the copied results.json (report sections 3, 5).

    The Mac file's speedup was skipped (MPS).  The A100 job's results.json,
    which rolled out with the same stride schedule, holds a measurement; it
    is quoted with its archive path, not merged in as if it were this run's.
    """
    when = f"{datetime.fromtimestamp(os.path.getmtime(src)):%Y-%m-%d %H:%M}"
    notes = dict(results_note=(
        f"copied by migrate_layout.py from {src} (written {when}): its "
        f"diagnostics come from that earlier uno_predict run, "
        f"not from the rollout cache this report is drawn from"))
    sp = res.get("speedup") or {}
    a100 = first(os.path.join(OLD, A100_JSON),
                 os.path.join(ARCH_RESULTS, A100_JSON))
    if not sp.get("skipped") or a100 is None:
        return notes
    other = read_json(a100)
    osp = other.get("speedup") or {}
    if "speedup" in osp and other.get("schedule") == res.get("schedule"):
        notes["speedup_note"] = (
            f"measured on the A100 by job 399765 with the same stride "
            f"schedule: {osp['speedup']:.1f}x like-for-like, "
            f"{osp.get('speedup_batched', float('nan')):.1f}x batched "
            f"({os.path.join(ARCH_RESULTS, A100_JSON)})")
    return notes


def is_gt(name):
    return (name.startswith("gt_") and name.endswith(".npz")
            and not name.endswith(".partial.npz"))


def needed(name):
    """A file in Results/cache/ that the current code still reads."""
    return is_gt(name) or name in MEMMAPS


def cache_dir_plan(plan, moved):
    """Archive Results/cache/ once nothing the code needs is left in it."""
    if not os.path.isdir(OLD_CACHE):
        return None
    left = [n for n in sorted(os.listdir(OLD_CACHE))
            if os.path.join(OLD_CACHE, n) not in moved]
    keep = [n for n in left if needed(n)]
    if keep:
        return plan.other("leave", OLD_CACHE, "", "leave",
                          f"still holds {len(keep)} file(s) the code needs "
                          f"(first {keep[0]}); archive it by hand later")
    step = plan.move(OLD_CACHE, os.path.join(ARCH_RESULTS, "cache"))
    step["size"] = sum(size_of(os.path.join(OLD_CACHE, n)) for n in left)
    step["why"] = "once the files above have left it"
    if left:
        step["why"] = f"with {len(left)} leftover file(s): {left[:4]}"
    return step


def build_plan(facts):
    plan = Plan()
    gts = []
    if os.path.isdir(OLD_CACHE):
        gts = sorted(n for n in os.listdir(OLD_CACHE) if is_gt(n))
    for n in gts:
        plan.move(os.path.join(OLD_CACHE, n), os.path.join(GT_DIR, n),
                  group="gt")
    for n in MODEL_FILES:
        plan.move(os.path.join(OLD, n), os.path.join(MODEL, n),
                  gate=facts["model"])
    for n in MEMMAPS:
        plan.move(os.path.join(OLD_CACHE, n), os.path.join(MODEL, "cache", n),
                  gate=facts["model"])
    plan.move(OLD_DATA, DATA,
              gate=facts["data"] if os.path.lexists(OLD_DATA) else None)
    roll = plan.move(OLD_ROLLOUT, ROLLOUT,
                     gate=facts["cache"] if os.path.lexists(OLD_ROLLOUT)
                     else None)
    mpath = os.path.join(ROLLOUT, "meta.json")
    if roll["status"] == "todo":
        plan.other("label", mpath, "", "todo",
                   f"{facts['meta'].get('label')!r} -> {LABEL!r}, after the "
                   f"move")
    elif roll["status"] == "done" and os.path.exists(mpath):
        old = read_json(mpath).get("label")
        plan.other("label", mpath, "", "done" if old == LABEL else "todo",
                   f"{old!r} -> {LABEL!r}")
    results_copy_plan(plan, facts)

    names = sorted(os.listdir(OLD)) if os.path.isdir(OLD) else []
    kept = [n for n in names if any(fnmatch.fnmatch(n, k) for k in KEPT)]
    for n in kept:
        plan.move(os.path.join(OLD, n), os.path.join(EARLIER, n),
                  group="kept")
    note = os.path.join(EARLIER, "NOTE.md")
    if os.path.lexists(note):
        plan.other("note", "", note, "done")
    elif kept or os.path.isdir(EARLIER):
        plan.other("note", "", note, "todo", fp=facts["fp"])

    moved = {s["src"] for s in plan.steps
             if s["kind"] == "move" and s["status"] == "todo"}
    cache_dir_plan(plan, moved)
    for n in names:
        if (n == "cache" or n in MODEL_FILES or n in kept
                or NEW_LAYOUT.match(n)):
            continue
        plan.move(os.path.join(OLD, n), os.path.join(ARCH_RESULTS, n),
                  group="rest")
    if os.path.lexists(SUBMISSION) or os.path.lexists(ARCH_SUB):
        plan.move(SUBMISSION, ARCH_SUB)
    for n in LEGACY:
        if os.path.isdir(n):
            k = len(os.listdir(n))
            plan.other("leave", n, "", "leave",
                       f"{k} entr{'y' if k == 1 else 'ies'}, "
                       f"{human(size_of(n))}; read by no code, left in place")
    return plan


TAGS = dict(todo="MOVE", done="DONE", clash="CLASH", refused="REFUSED",
            absent="ABSENT", leave="LEAVE")


def show(plan):
    groups = {}
    for s in plan.steps:
        if s["group"] == "gt":
            groups.setdefault(s["status"], []).append(s)
    for status, steps in groups.items():
        n, size = len(steps), sum(s["size"] for s in steps)
        log(f"  {TAGS[status]:8s} {n} ground-truth files {OLD_CACHE}/gt_*.npz "
            f"-> {GT_DIR}/  ({human(size)})")
        if status == "clash":
            for s in steps:
                log(f"           clash: {s['dst']} exists; "
                    f"{s['src']} stays")
    for s in plan.steps:
        if s["group"] == "gt" or s["status"] == "absent":
            continue
        tag = TAGS.get(s["status"], s["status"].upper())
        if s["kind"] == "move":
            size = f"  ({human(s['size'])})" if s["size"] else ""
            why = f"  [{s['why']}]" if s["why"] else ""
            log(f"  {tag:8s} {s['src']} -> {s['dst']}{size}{why}")
        elif s["kind"] == "copy":
            tag = "COPY" if s["status"] == "todo" else tag
            why = f"  [{s['why']}]" if s["why"] else ""
            log(f"  {tag:8s} {s['src']} -> {s['dst']}{why}")
        elif s["kind"] == "label":
            tag = "REWRITE" if s["status"] == "todo" else tag
            log(f"  {tag:8s} {s['src']} \"label\": {s['why']}")
        elif s["kind"] == "note":
            tag = "WRITE" if s["status"] == "todo" else tag
            log(f"  {tag:8s} {s['dst']}")
        else:
            log(f"  {tag:8s} {s['src']}/  [{s['why']}]")
    absent = [s["src"] for s in plan.steps if s["status"] == "absent"]
    if absent:
        log(f"  (not present, nothing to do: {', '.join(absent)})")
    count = {}
    for s in plan.steps:
        count[s["status"]] = count.get(s["status"], 0) + 1
    words = dict(todo="to do", done="already done", clash="clash(es)",
                 refused="refused", absent="not present", leave="left alone")
    log("  steps: " + ", ".join(f"{v} {words.get(k, k)}"
                                for k, v in sorted(count.items())))


def do_move(src, dst):
    """Move one file, folder or symlink; never overwrite, never delete."""
    if os.path.lexists(dst):
        return f"CLASH    {dst} exists; {src} left where it is"
    if not os.path.lexists(src):
        return f"GONE     {src} is no longer there"
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.islink(src) and not os.path.isabs(os.readlink(src)):
        target = os.path.join(os.path.dirname(src), os.readlink(src))
        os.symlink(os.path.relpath(target, os.path.dirname(dst) or "."), dst)
        os.unlink(src)
        return f"moved    {src} -> {dst} (relative symlink re-pointed)"
    try:
        os.replace(src, dst)
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        shutil.move(src, dst)
        return f"moved    {src} -> {dst} (copied across filesystems)"
    return f"moved    {src} -> {dst}"


def note_files():
    """(name, mtime) of the graphs in the earlier-run folder."""
    if not os.path.isdir(EARLIER):
        return []
    out = []
    for n in sorted(os.listdir(EARLIER)):
        if n != "NOTE.md":
            t = datetime.fromtimestamp(os.path.getmtime(
                os.path.join(EARLIER, n)))
            out.append((n, f"{t:%Y-%m-%d %H:%M}"))
    return out


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def apply(plan):
    n_gt = 0
    for s in plan.steps:
        if s["status"] != "todo":
            continue
        kind = s["kind"]
        if kind == "move" and s["src"] == OLD_CACHE:
            left = ([n for n in os.listdir(OLD_CACHE) if needed(n)]
                    if os.path.isdir(OLD_CACHE) else [])
            if left:
                log(f"LEAVE    {OLD_CACHE}/ still holds {left[0]} ...")
                continue
        if kind == "move":
            msg = do_move(s["src"], s["dst"])
            if s["group"] == "gt" and msg.startswith("moved"):
                n_gt += 1
            else:
                log(msg)
        elif kind == "label":
            if not os.path.exists(s["src"]):
                log(f"GONE     {s['src']} is not there; label not rewritten")
                continue
            meta = read_json(s["src"])
            old = meta.get("label")
            if old != LABEL:
                meta["label"] = LABEL
                dump_json(s["src"], meta)
            log(f"rewrote  {s['src']} \"label\": {old!r} -> {LABEL!r}")
        elif kind == "copy":
            if os.path.lexists(s["dst"]):
                log(f"CLASH    {s['dst']} exists; not copied")
                continue
            os.makedirs(os.path.dirname(s["dst"]), exist_ok=True)
            if s.get("add_fp") or s.get("notes"):
                res = read_json(s["src"])
                if s.get("add_fp"):
                    res["fingerprint"] = s["add_fp"]
                    res["fingerprint_note"] = (
                        f"added by migrate_layout.py on "
                        f"{datetime.now():%Y-%m-%d}: sha256[:12] of the "
                        f"checkpoint named above, unchanged since this file "
                        f"was written; now at "
                        f"{os.path.join(MODEL, MODEL_FILES[0])}")
                res.update(s.get("notes") or {})
                dump_json(s["dst"], res)
            else:
                tmp = s["dst"] + ".partial"
                shutil.copy2(s["src"], tmp)
                os.replace(tmp, s["dst"])
            log(f"copied   {s['src']} -> {s['dst']}  [{s['why']}]")
        elif kind == "note":
            if not os.path.lexists(s["dst"]):
                files = "\n".join(f"- `{n}` (modified {t})"
                                   for n, t in note_files())
                write_text(s["dst"], NOTE.format(fp=s.get("fp") or "?",
                                                 files=files or "- (none)"))
                log(f"wrote    {s['dst']}")
    if n_gt:
        log(f"moved    {n_gt} ground-truth files -> {GT_DIR}/")


def main():
    ap = argparse.ArgumentParser(
        description="Move the old Results/ layout into Results/ + Work/ + "
                    "Archive/ (dry run unless --apply).")
    ap.add_argument("--apply", action="store_true",
                    help="carry the plan out (default: only print it)")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    log(f"migrate_layout: {os.getcwd()}  "
        f"({'APPLY' if a.apply else 'dry run, nothing is changed'})")
    log("")
    facts = run_checks()
    plan = build_plan(facts)
    log("")
    log("plan:")
    show(plan)
    log("")
    todo = [s for s in plan.steps if s["status"] == "todo"]
    if not a.apply:
        log(f"dry run: {len(todo)} step(s) to do. Carry them out with "
            f"`python migrate_layout.py --apply`; afterwards run "
            f"`python make_submission.py`.")
        return 0
    if not todo:
        log("nothing to do: the layout is already migrated.")
    else:
        apply(plan)
    log("")
    log("next: `python make_submission.py` redraws Results/"
        f"{LABEL}/ from the moved rollout cache (python3 on the Mac).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
