"""Read-only investigation of MIT-BIH before cleaning (plan step 0).

Quantifies the three legitimate cleaning levers so the cleaning thresholds and the
report numbers are grounded in data, not guessed:
  1. per-record lead mapping (which channel is MLII) -> confirm only 114 is swapped
  2. RR-interval contamination (beats whose neighbour annotation is a non-beat marker)
  3. signal-quality annotations ('~' aux semantics, '[' ']' flutter episodes) ->
     how many beats fall in bad-quality segments, per split
  4. R-peak offset: how far the annotated sample sits from the local |signal| extremum

Run:  PYTHONPATH=. ./env/bin/python tools/inspect_dataset.py
Nothing is written; this only prints tables.
"""
from __future__ import annotations

import collections

import numpy as np
import wfdb

from src import config
from src.data import aami_binary_label, bandpass_filter

DB = config.MITBIH_DIR
ALL_RECORDS = config.DS1_RECORDS + config.DS2_RECORDS
SET_OF = {}
for r in config.DS1_TRAIN_RECORDS:
    SET_OF[r] = "DS1-train"
for r in config.DS1_VAL_RECORDS:
    SET_OF[r] = "DS1-val"
for r in config.DS2_RECORDS:
    SET_OF[r] = "DS2"

# beat-type symbols actually scored (anything that aami_binary_label maps to 0/1)
def is_beat(sym: str) -> bool:
    return aami_binary_label(sym) is not None


def sec(title: str) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ---------------- 1. lead mapping ----------------
def lead_mapping():
    sec("1. LEAD MAPPING (per record, all DS1+DS2)")
    swapped = []
    print(f"  {'rec':<5} {'set':<10} {'ch0':<8} {'ch1':<8}")
    for r in ALL_RECORDS:
        h = wfdb.rdheader(str(DB / r))
        names = list(h.sig_name)
        ch0 = names[0]
        ch1 = names[1] if len(names) > 1 else "-"
        flag = "" if ch0 == "MLII" else "  <-- NOT MLII on ch0"
        if ch0 != "MLII":
            swapped.append((r, names))
        print(f"  {r:<5} {SET_OF[r]:<10} {ch0:<8} {ch1:<8}{flag}")
    print(f"\n  records without MLII on channel 0: {swapped}")
    # also flag any record lacking MLII entirely
    for r in ALL_RECORDS:
        h = wfdb.rdheader(str(DB / r))
        if "MLII" not in h.sig_name:
            print(f"  NOTE {r} has NO MLII lead at all: {list(h.sig_name)}")


# ---------------- 2. RR contamination ----------------
def rr_contamination():
    sec("2. RR CONTAMINATION (beat whose prev/next annotation is a non-beat marker)")
    tot = contam = 0
    per_set = collections.Counter()
    per_set_tot = collections.Counter()
    for r in ALL_RECORDS:
        ann = wfdb.rdann(str(DB / r), "atr")
        syms = ann.symbol
        for i, s in enumerate(syms):
            if not is_beat(s):
                continue
            per_set_tot[SET_OF[r]] += 1
            tot += 1
            prev_bad = i > 0 and not is_beat(syms[i - 1])
            next_bad = i + 1 < len(syms) and not is_beat(syms[i + 1])
            if prev_bad or next_bad:
                contam += 1
                per_set[SET_OF[r]] += 1
    print(f"  overall: {contam}/{tot} = {100*contam/tot:.2f}% beats have a contaminated RR")
    for st in ("DS1-train", "DS1-val", "DS2"):
        print(f"    {st:<10} {per_set[st]:>6}/{per_set_tot[st]:<6} "
              f"= {100*per_set[st]/max(per_set_tot[st],1):.2f}%")


# ---------------- 3. signal-quality annotations ----------------
def quality_annotations():
    sec("3a. '~' SIGNAL-QUALITY annotation: aux_note / num / subtype samples")
    aux_counter = collections.Counter()
    for r in ALL_RECORDS:
        ann = wfdb.rdann(str(DB / r), "atr")
        for i, s in enumerate(ann.symbol):
            if s == "~":
                aux = (ann.aux_note[i] or "").strip("\x00").strip()
                aux_counter[(aux, int(ann.num[i]) if ann.num is not None else -1)] += 1
    print("  (aux_note, num) -> count   [num is the signal-quality bitmap in MIT-BIH]")
    for (aux, num), c in aux_counter.most_common():
        print(f"    aux={aux!r:<12} num={num:<3} -> {c}")

    sec("3b. BEATS inside bad-quality segments, per split")
    # Build bad intervals two ways and count beats inside:
    #   (A) flutter/fib episodes: between '[' and ']'
    #   (B) noise: from '~' whose num != 0 (some lead noisy) until next '~' with num==0
    per_set_beats = collections.Counter()
    per_set_flutter = collections.Counter()
    per_set_noise = collections.Counter()
    for r in ALL_RECORDS:
        ann = wfdb.rdann(str(DB / r), "atr")
        samp, syms = ann.sample, ann.symbol
        nums = ann.num if ann.num is not None else [0] * len(syms)
        # flutter intervals
        flut = []
        start = None
        for i, s in enumerate(syms):
            if s == "[":
                start = samp[i]
            elif s == "]" and start is not None:
                flut.append((start, samp[i])); start = None
        # noise intervals from '~' num bitmap (num>0 => at least one lead noisy)
        noise = []
        nstart = None
        for i, s in enumerate(syms):
            if s == "~":
                if int(nums[i]) != 0 and nstart is None:
                    nstart = samp[i]
                elif int(nums[i]) == 0 and nstart is not None:
                    noise.append((nstart, samp[i])); nstart = None
        def inside(x, ivs):
            return any(a <= x <= b for a, b in ivs)
        for i, s in enumerate(syms):
            if not is_beat(s):
                continue
            per_set_beats[SET_OF[r]] += 1
            if inside(samp[i], flut):
                per_set_flutter[SET_OF[r]] += 1
            if inside(samp[i], noise):
                per_set_noise[SET_OF[r]] += 1
    print(f"  {'set':<10} {'beats':>8} {'in_flutter':>12} {'in_noise':>10}")
    for st in ("DS1-train", "DS1-val", "DS2"):
        tot = per_set_beats[st]
        print(f"  {st:<10} {tot:>8} {per_set_flutter[st]:>12} {per_set_noise[st]:>10}"
              f"   ({100*per_set_flutter[st]/max(tot,1):.2f}% / {100*per_set_noise[st]/max(tot,1):.2f}%)")


# ---------------- 4. R-peak offset ----------------
def rpeak_offset():
    sec("4. R-PEAK OFFSET: |annotated sample - local |signal| max| within +/-15")
    W = 15
    offs = []
    for r in ALL_RECORDS[:8]:  # a representative subset is enough for the distribution
        rec = wfdb.rdrecord(str(DB / r), channels=[0])
        sig = bandpass_filter(rec.p_signal[:, 0].astype(np.float32), fs=int(rec.fs))
        ann = wfdb.rdann(str(DB / r), "atr")
        for i, s in enumerate(ann.symbol):
            if not is_beat(s):
                continue
            c = ann.sample[i]
            lo, hi = max(0, c - W), min(len(sig), c + W)
            if hi - lo < 2:
                continue
            local = lo + int(np.argmax(np.abs(sig[lo:hi])))
            offs.append(abs(local - c))
    offs = np.array(offs)
    print(f"  n={len(offs)} beats (8 records)")
    for p in (50, 75, 90, 95, 99):
        print(f"    p{p:<2} offset = {np.percentile(offs, p):.1f} samples")
    print(f"    mean={offs.mean():.2f}  max={offs.max()}  "
          f"%==0 (already on peak): {100*(offs==0).mean():.1f}%")
    print(f"  -> suggests re-align window W ~= p99 rounded up")


if __name__ == "__main__":
    lead_mapping()
    rr_contamination()
    quality_annotations()
    rpeak_offset()
    print()
