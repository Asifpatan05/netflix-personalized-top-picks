#!/usr/bin/env python3
"""Score the holdout week ONCE: Hit Rate@10 of the model vs the popularity top 10.

Hit rule: an eligible member scores 1 if any of their 10 picks is a title they
completed during 2026-11-10..11-16 that they had not already completed before.
  * eligible = at least one event before 2026-11-10 and at least one such new completion
  * picks    = top 10 of the 300 catalog titles, already-completed titles removed
  * popularity top 10 = most-completed titles in 2026-08-17..11-09 only, scored on
    the same members with the same removal
The positive rule (raw `completed` or `completed_consistent`) is read from the
locked training run, so it cannot differ between training and scoring.

Refuses to run a second time unless --force is given, so the holdout number is
not shopped around.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from recsys_data import (HOLDOUT_END, HOLDOUT_START, TRAIN_START, build_item_features, compare,
                         load_catalog, load_env_file, load_events, make_eval_set,
                         popularity_counts)
from train import TwoTower, init_wandb, score_members


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="cleaned_viewing_history.csv",
                    help="must include the holdout week (cleaned_viewing_history.csv does)")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out-dir", default="stage2_output")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--project", default="netflix-personalized-top-picks")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-score the holdout (logged as a re-run)")
    a = ap.parse_args()

    out = Path(a.out_dir)
    result_path = out / "holdout_results.json"
    if result_path.exists() and not a.force:
        raise SystemExit(f"{result_path} already exists: the holdout has been scored. "
                         "Use --force only for a deliberate re-run.")
    locked = json.loads((out / "locked_epoch.json").read_text())
    ckpt = torch.load(out / "model_final.pt", map_location="cpu")
    positive = locked["positive"]
    if ckpt["positive"] != positive:
        raise SystemExit("checkpoint and locked_epoch.json disagree on the positive rule")

    cat, cat_index = load_catalog(a.catalog)
    n_items = len(cat)
    cat_ids, cat_sizes, text = build_item_features(cat)
    model = TwoTower(cat_ids, cat_sizes, text, n_items)
    model.load_state_dict(ckpt["state"])

    ev = load_events(a.events, cat_index, positive)
    es = make_eval_set(ev, HOLDOUT_START, HOLDOUT_END, n_items)
    pop = popularity_counts(ev, TRAIN_START, HOLDOUT_START, n_items)  # train window only
    res = compare(score_members(model, es.hist_pad), es, pop, n_boot=a.n_boot)

    report = {"positive_rule": positive, "locked_epoch": locked["best_epoch"],
              "rerun_with_force": bool(a.force and result_path.exists()),
              "window": es.stats, "metrics": res}
    out.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    lo, hi = res["hit_rate_diff_ci95"]
    print(f"\nHit Rate@10  model {res['model_hit_rate@10']:.3f}  |  "
          f"popularity {res['popularity_hit_rate@10']:.3f}  |  random {res['random_hit_rate@10']:.3f}"
          f"  (n={res['eligible_members']})")
    print(f"model - popularity = {res['hit_rate_diff_model_minus_popularity']:+.3f}  "
          f"(95% CI {lo:+.3f} to {hi:+.3f})")

    load_env_file()
    run = init_wandb(not a.no_wandb, a.project, f"holdout-{positive}", report)
    if run is not None:
        run.log({f"holdout/{k}": v for k, v in res.items() if isinstance(v, (int, float))})
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
