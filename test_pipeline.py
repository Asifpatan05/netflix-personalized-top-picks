#!/usr/bin/env python3
"""Torch-free checks of the data/eval logic. Run:  python test_pipeline.py --catalog title_catalog.csv

Uses synthetic events on your real catalog, so it verifies the windows, histories,
leakage guarantees, popularity baseline and Hit Rate@10 without needing PyTorch or
your member data.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from recsys_data import (HOLDOUT_END, HOLDOUT_START, TRAIN_START, VAL_START, build_item_features,
                         compare, load_catalog, load_events, make_eval_set, make_train_set,
                         popularity_counts, seen_before, topk_unseen)


def synth_events(cat, n_members=500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genres = sorted(cat["primary_genre"].unique())
    pop_w = rng.pareto(1.5, len(cat)) + 0.1
    rows = []
    days = pd.date_range("2026-08-17", "2026-11-16")
    for m in range(n_members):
        fav = rng.choice(genres, 2, replace=False)
        w = pop_w * np.where(cat["primary_genre"].isin(fav), 5.0, 1.0)
        w = w / w.sum()
        for _ in range(rng.integers(3, 30)):
            t = cat["title_id"][rng.choice(len(cat), p=w)]
            if rng.random() < 0.03:
                t = rng.choice(["TT-0301", "TT-0302"])  # outside the catalog
            rows.append({"member_id": f"M{m:04d}", "title_id": t,
                         "event_datetime": str(rng.choice(days.strftime("%Y-%m-%d"))),
                         "completed": bool(rng.random() < 0.6),
                         "completed_validation": rng.choice(["consistent", "below_90pct_runtime"],
                                                            p=[0.4, 0.6])})
    df = pd.DataFrame(rows)
    df.insert(0, "event_id", [f"E{i:06d}" for i in range(len(df))])
    return df


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return bool(cond)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    a = ap.parse_args()
    cat, cat_index = load_catalog(a.catalog)
    n = len(cat)
    ok = []

    cat_ids, cat_sizes, text = build_item_features(cat)
    ok.append(check("item features", cat_ids.shape == (n, 6) and text.shape[0] == n
                    and np.isfinite(text).all(), f"cat {cat_ids.shape}, text {text.shape}"))

    raw = synth_events(cat)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ev.csv"
        raw.to_csv(p, index=False)
        ev = load_events(p, cat_index, "completed")
        ev_c = load_events(p, cat_index, "completed_consistent")
    ok.append(check("consistent rule is stricter", ev_c["positive"].sum() < ev["positive"].sum(),
                    f"{ev['positive'].sum()} vs {ev_c['positive'].sum()}"))
    ok.append(check("unmatched titles never positive",
                    not ev.loc[ev["item"] < 0, "positive"].any()))

    # ---- train set: histories only from strictly earlier days, no rewatch targets ----
    ts = make_train_set(ev, TRAIN_START, VAL_START, n)
    ok.append(check("no target inside its own seen mask",
                    not ts.seen[np.arange(len(ts)), ts.target].any(), f"{len(ts)} examples"))
    # brute-force re-derivation for a handful of members
    pos = ev[ev["positive"] & (ev["date"] < VAL_START)]
    bad = 0
    for m in pos["member_id"].unique()[:40]:
        g = pos[pos["member_id"] == m]
        rows = [(r.date, r.item) for r in g.itertuples()]
        expect = 0
        for d_, it in set(rows):
            earlier = {i for dd, i in rows if dd < d_}
            if it not in earlier:
                expect += 1
        # compare against make_train_set restricted to this member
        sub = make_train_set(ev[ev["member_id"] == m], TRAIN_START, VAL_START, n)
        if len(sub) != expect:
            bad += 1
    ok.append(check("train examples match brute-force derivation", bad == 0, f"{bad} mismatches"))

    # ---- leakage: scramble every holdout row; nothing built from train may move ----
    ev2 = ev.copy()
    h = ev2["date"] >= HOLDOUT_START
    rng = np.random.default_rng(1)
    ev2.loc[h, "item"] = rng.integers(0, n, h.sum())
    ev2.loc[h, "member_id"] = rng.permutation(ev2.loc[h, "member_id"].values)
    ev2["positive"] = ev2["flag"] & (ev2["item"] >= 0)
    ts2 = make_train_set(ev2, TRAIN_START, HOLDOUT_START, n)
    ts1 = make_train_set(ev, TRAIN_START, HOLDOUT_START, n)
    ok.append(check("train set unchanged when holdout is scrambled",
                    np.array_equal(ts1.target, ts2.target) and np.array_equal(ts1.hist_pad, ts2.hist_pad)))
    ok.append(check("popularity unchanged when holdout is scrambled",
                    np.array_equal(popularity_counts(ev, TRAIN_START, HOLDOUT_START, n),
                                   popularity_counts(ev2, TRAIN_START, HOLDOUT_START, n))))
    s1, s2 = seen_before(ev, HOLDOUT_START), seen_before(ev2, HOLDOUT_START)
    ok.append(check("histories at the cutoff unchanged when holdout is scrambled", s1 == s2))

    # ---- eval set ----
    es = make_eval_set(ev, HOLDOUT_START, HOLDOUT_END, n)
    print("   holdout stats:", es.stats)
    ok.append(check("eligible members have a prior event and a new target",
                    all(len(t) > 0 for t in es.targets) and len(es.members) == es.stats["eligible_members"]))
    ok.append(check("targets never include already-seen titles",
                    all(not es.seen[i, list(t)].any() for i, t in enumerate(es.targets))))

    pop = popularity_counts(ev, TRAIN_START, HOLDOUT_START, n)

    # oracle scorer must hit 1.0 for everyone
    oracle = np.zeros(es.seen.shape)
    for i, t in enumerate(es.targets):
        oracle[i, list(t)] = 10.0
    r = compare(oracle, es, pop)
    ok.append(check("oracle scorer -> Hit Rate 1.0", r["model_hit_rate@10"] == 1.0))

    # popularity used as the "model" must reproduce the baseline exactly
    r = compare(np.broadcast_to(pop, es.seen.shape).copy(), es, pop, n_boot=500)
    ok.append(check("popularity-as-model equals baseline (diff 0)",
                    r["model_hit_rate@10"] == r["popularity_hit_rate@10"]
                    and abs(r["hit_rate_diff_model_minus_popularity"]) < 1e-12))
    ok.append(check("random ranker below popularity", r["random_hit_rate@10"] < r["popularity_hit_rate@10"],
                    f"random {r['random_hit_rate@10']:.3f} < pop {r['popularity_hit_rate@10']:.3f}"))

    # picks never contain already-seen titles when enough unseen titles exist
    picks = topk_unseen(np.random.default_rng(2).random(es.seen.shape), es.seen)
    ok.append(check("picks exclude seen titles",
                    not np.take_along_axis(es.seen, picks, 1).any() and picks.shape[1] == 10))

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
