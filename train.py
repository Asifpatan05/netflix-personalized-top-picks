#!/usr/bin/env python3
"""Train the two-tower model (64-d towers) with early stopping, then lock the epoch count.

Phase A  fit on 2026-08-17..11-02, validate on 11-03..11-09, stop when validation
         Recall@10 has not improved for `patience` epochs (cap: 30 epochs).
Phase B  refit on 2026-08-17..11-09 for exactly the locked epoch count and save
         the checkpoint. The holdout week is never read here: events on or after
         2026-11-10 are dropped right after loading. evaluate.py scores it once.

Usage:
    python train.py --events cleaned_viewing_history.csv --catalog title_catalog.csv
W&B logging turns on only if WANDB_API_KEY is set in the environment (or a
local, git-ignored .env file); otherwise the run is local-only.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recsys_data import (HOLDOUT_START, MAX_HIST, POSITIVE_MODES, TRAIN_START, VAL_START,
                         build_item_features, compare, load_catalog, load_env_file,
                         load_events, make_eval_set, make_train_set, popularity_counts)

EMB_DIM = 64  # both towers output 64 dimensions


class FeatureEmbed(nn.Module):
    """Content features for every catalog item: categorical embeddings + text vector."""

    def __init__(self, cat_ids, cat_sizes, text, dim=16):
        super().__init__()
        self.register_buffer("cat_ids", torch.as_tensor(cat_ids))
        self.register_buffer("text", torch.as_tensor(text))
        self.embs = nn.ModuleList([nn.Embedding(s, dim) for s in cat_sizes])
        self.out_dim = dim * len(cat_sizes) + text.shape[1]

    def forward(self):  # -> [n_items, out_dim]
        parts = [e(self.cat_ids[:, j]) for j, e in enumerate(self.embs)]
        return torch.cat(parts + [self.text], dim=1)


class ItemTower(nn.Module):
    def __init__(self, feats, n_items, id_dim=16, id_dropout=0.5):
        super().__init__()
        self.feats, self.p = feats, id_dropout
        self.id_emb = nn.Embedding(n_items, id_dim)
        self.mlp = nn.Sequential(nn.Linear(feats.out_dim + id_dim, 128), nn.ReLU(),
                                 nn.Linear(128, EMB_DIM))

    def forward(self):  # -> [n_items, 64], L2-normalised
        ids = self.id_emb.weight
        if self.training:  # ID dropout so new titles still work from content alone
            keep = (torch.rand(ids.size(0), 1, device=ids.device) > self.p).float()
            ids = ids * keep / (1 - self.p)
        return F.normalize(self.mlp(torch.cat([self.feats(), ids], 1)), dim=-1)


class UserTower(nn.Module):
    def __init__(self, feats, n_items, id_dim=16):
        super().__init__()
        self.feats = feats
        self.id_emb = nn.Embedding(n_items, id_dim)
        self.mlp = nn.Sequential(nn.Linear(feats.out_dim + id_dim + 1, 128), nn.ReLU(),
                                 nn.Linear(128, EMB_DIM))

    def forward(self, hist_pad):  # [B, H] item ids, -1 = padding -> [B, 64]
        m = (hist_pad >= 0).unsqueeze(-1).float()
        idx = hist_pad.clamp(min=0)
        h = torch.cat([self.feats()[idx], self.id_emb(idx)], dim=-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        has_hist = (m.sum(1) > 0).float()  # lets the tower tell "no history" apart
        return F.normalize(self.mlp(torch.cat([pooled, has_hist], -1)), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, cat_ids, cat_sizes, text, n_items):
        super().__init__()
        self.item = ItemTower(FeatureEmbed(cat_ids, cat_sizes, text), n_items)
        self.user = UserTower(FeatureEmbed(cat_ids, cat_sizes, text), n_items)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    def logits(self, hist_pad):  # full catalog: [B, n_items]
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * self.user(hist_pad) @ self.item().T


@torch.no_grad()
def score_members(model, hist_pad: np.ndarray) -> np.ndarray:
    model.eval()
    return model.logits(torch.as_tensor(hist_pad)).cpu().numpy()


def init_wandb(enabled, project, name, config):
    load_env_file()
    if not enabled:
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("W&B: WANDB_API_KEY not set -> logging off (local run only)")
        return None
    try:
        import wandb
    except ImportError:
        print("W&B: package not installed (pip install wandb) -> logging off")
        return None
    return wandb.init(project=project, name=name, config=config)


def fit(ev, feats, n_items, train_end, epochs, seed, bs, lr, wd,
        val=None, patience=3, run=None, tag=""):
    """Train with full-catalog softmax. With `val=(eval_set, pop_counts)` use early
    stopping on validation Recall@10 and restore the best epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cat_ids, cat_sizes, text = feats
    ts = make_train_set(ev, TRAIN_START, train_end, n_items)
    target = torch.as_tensor(ts.target)
    hist = torch.as_tensor(ts.hist_pad)
    seen = torch.as_tensor(ts.seen)
    print(f"[{tag}] {len(ts):,} training examples, up to {epochs} epochs")

    model = TwoTower(cat_ids, cat_sizes, text, n_items)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    gen = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state, bad, log = -1.0, 0, None, 0, []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(ts), generator=gen)
        total = 0.0
        for i in range(0, len(ts), bs):
            b = perm[i:i + bs]
            logits = model.logits(hist[b]).masked_fill(seen[b], float("-inf"))
            loss = F.cross_entropy(logits, target[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        row = {"epoch": epoch, "train_loss": total / len(ts)}

        if val is not None:
            es, pop = val
            m = compare(score_members(model, es.hist_pad), es, pop)
            row.update({"val_recall@10": m["model_recall@10"],
                        "val_hit_rate@10": m["model_hit_rate@10"],
                        "val_popularity_hit_rate@10": m["popularity_hit_rate@10"]})
            if m["model_recall@10"] > best:
                best, best_epoch, bad = m["model_recall@10"], epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
        log.append(row)
        print(f"[{tag}] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in row.items()))
        if run is not None:
            run.log({f"{tag}/{k}": v for k, v in row.items() if k != "epoch"}, step=epoch)
        if val is not None and bad >= patience:
            print(f"[{tag}] early stop: no val Recall@10 gain for {patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="cleaned_viewing_history.csv")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out-dir", default="stage2_output")
    ap.add_argument("--positive", choices=POSITIVE_MODES, default="completed")
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="netflix-personalized-top-picks")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cat, cat_index = load_catalog(a.catalog)
    n_items = len(cat)
    feats = build_item_features(cat)

    ev = load_events(a.events, cat_index, a.positive)
    ev = ev[ev["date"] < HOLDOUT_START].copy()  # the holdout week is never loaded past here
    print(f"{len(ev):,} events before {HOLDOUT_START.date()}; positive rule: {a.positive}")

    config = {**vars(a), "emb_dim": EMB_DIM, "max_hist": MAX_HIST}
    run = init_wandb(not a.no_wandb, a.project, f"train-{a.positive}-seed{a.seed}", config)

    # ---- Phase A: early stopping on the validation week ----
    es = make_eval_set(ev, VAL_START, HOLDOUT_START, n_items)
    pop_val = popularity_counts(ev, TRAIN_START, VAL_START, n_items)
    print("validation window:", es.stats)
    model, best_epoch, log = fit(ev, feats, n_items, VAL_START, a.max_epochs, a.seed,
                                 a.batch_size, a.lr, a.weight_decay,
                                 val=(es, pop_val), patience=a.patience, run=run, tag="A")
    val_final = compare(score_members(model, es.hist_pad), es, pop_val, n_boot=1000)
    print(f"locked epoch count: {best_epoch}")
    print("validation:", json.dumps(val_final, indent=2))

    (out / "locked_epoch.json").write_text(json.dumps({
        "best_epoch": best_epoch, "positive": a.positive, "seed": a.seed,
        "validation_window": es.stats, "validation_metrics": val_final,
        "config": config, "epoch_log": log}, indent=2))

    # ---- Phase B: refit on the full train window for the locked epoch count ----
    final, _, _ = fit(ev, feats, n_items, HOLDOUT_START, best_epoch, a.seed,
                      a.batch_size, a.lr, a.weight_decay, val=None, run=run, tag="B")
    torch.save({"state": final.state_dict(), "best_epoch": best_epoch,
                "positive": a.positive, "n_items": n_items}, out / "model_final.pt")
    print(f"saved {out / 'model_final.pt'}; now run evaluate.py (once).")
    if run is not None:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
