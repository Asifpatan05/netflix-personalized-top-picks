"""Torch-free data handling, time windows, popularity baseline and Hit Rate@10.

Everything that decides what counts as a positive, a history, a held-out target
or a hit lives here, so the model and the popularity baseline are always scored
by exactly the same code.

Windows (date-only events; end dates are exclusive):
    train       2026-08-17 .. 2026-11-02   (phase A fit)
    validation  2026-11-03 .. 2026-11-09   (early stopping only)
    holdout     2026-11-10 .. 2026-11-16   (scored once, in evaluate.py)
Phase B refits on 2026-08-17 .. 2026-11-09 for the locked epoch count.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_START = pd.Timestamp("2026-08-17")
VAL_START = pd.Timestamp("2026-11-03")
HOLDOUT_START = pd.Timestamp("2026-11-10")
HOLDOUT_END = pd.Timestamp("2026-11-17")  # exclusive
K = 10
MAX_HIST = 50
POSITIVE_MODES = ("completed", "completed_consistent")


# ----------------------------------------------------------------------------
# Environment (W&B key is read from the environment, never from code)
# ----------------------------------------------------------------------------
def load_env_file(path: str = ".env") -> None:
    """Load KEY=VALUE lines from a local .env without overriding real env vars."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ----------------------------------------------------------------------------
# Catalog and item features
# ----------------------------------------------------------------------------
def load_catalog(path) -> tuple[pd.DataFrame, dict]:
    cat = pd.read_csv(path)
    cat["title_id"] = cat["title_id"].astype(str).str.strip().str.upper()
    if cat["title_id"].duplicated().any():
        raise ValueError("Duplicate title_id values in catalog")
    cat = cat.reset_index(drop=True)
    return cat, {t: i for i, t in enumerate(cat["title_id"])}


def build_item_features(cat: pd.DataFrame, text_dim: int = 32):
    """Categorical ids [n_items, 6], their vocab sizes, and a text matrix.

    Text features are fitted on catalog descriptions only, never on events.
    """
    def codes(s: pd.Series) -> np.ndarray:
        return pd.factorize(s.fillna("none").astype(str))[0]

    fields = [codes(cat[c]) for c in
              ["primary_genre", "secondary_genre", "title_type", "maturity_rating"]]
    rt = pd.cut(cat["runtime_minutes"], [0, 30, 45, 60, 90, 120, 10_000], labels=False)
    fields.append(rt.fillna(6).astype(int).to_numpy())
    yr = pd.to_datetime(cat["release_date"], errors="coerce").dt.year
    fields.append(yr.sub(2015).clip(0, 12).fillna(13).astype(int).to_numpy())
    cat_ids = np.stack(fields, axis=1).astype(np.int64)
    cat_sizes = [int(cat_ids[:, j].max()) + 1 for j in range(cat_ids.shape[1])]

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(stop_words="english", min_df=2, max_features=2000).fit_transform(
        cat["description"].fillna(""))
    k = max(1, min(text_dim, X.shape[1] - 1, X.shape[0] - 1))
    Z = TruncatedSVD(k, random_state=0).fit_transform(X)
    Z = Z / np.clip(np.linalg.norm(Z, axis=1, keepdims=True), 1e-8, None)
    return cat_ids, cat_sizes, Z.astype(np.float32)


# ----------------------------------------------------------------------------
# Events
# ----------------------------------------------------------------------------
def load_events(path, cat_index: dict, positive: str = "completed") -> pd.DataFrame:
    """Load events and mark positives.

    positive = "completed"              -> raw completed flag (default)
    positive = "completed_consistent"   -> completed AND completed_validation == consistent
    The SAME flag defines training positives, held-out hits and the popularity
    baseline. Events on titles outside the catalog can never be positives.
    """
    if positive not in POSITIVE_MODES:
        raise ValueError(f"positive must be one of {POSITIVE_MODES}")
    ev = pd.read_csv(path)
    missing = {"member_id", "title_id", "event_datetime", "completed"} - set(ev.columns)
    if missing:
        raise ValueError(f"events file is missing columns: {sorted(missing)}")
    ev["member_id"] = ev["member_id"].astype(str)
    ev["title_id"] = ev["title_id"].astype(str).str.strip().str.upper()
    ev["date"] = pd.to_datetime(ev["event_datetime"], format="%Y-%m-%d")
    flag = ev["completed"].astype(str).str.strip().str.lower().eq("true")
    if positive == "completed_consistent":
        if "completed_validation" not in ev.columns:
            raise ValueError("completed_validation column needed for completed_consistent")
        flag = flag & ev["completed_validation"].astype(str).eq("consistent")
    ev["item"] = ev["title_id"].map(cat_index).fillna(-1).astype(int)
    ev["flag"] = flag
    ev["positive"] = flag & (ev["item"] >= 0)
    return ev


def seen_before(ev: pd.DataFrame, cutoff) -> dict:
    """member -> unique positive items strictly before cutoff, most recent last."""
    p = ev[ev["positive"] & (ev["date"] < cutoff)].sort_values(
        ["member_id", "date"], kind="stable")
    out = {}
    for m, g in p.groupby("member_id", sort=False):
        out[m] = list(dict.fromkeys(g["item"].tolist()[::-1]))[::-1]
    return out


def pack_histories(hists: list, n_items: int, max_hist: int = MAX_HIST):
    """Padded recent history [N, max_hist] (-1 = pad) and full seen mask [N, n_items]."""
    pad = np.full((len(hists), max_hist), -1, dtype=np.int64)
    seen = np.zeros((len(hists), n_items), dtype=bool)
    for i, h in enumerate(hists):
        if h:
            seen[i, h] = True
            t = h[-max_hist:]
            pad[i, :len(t)] = t
    return pad, seen


@dataclass
class TrainSet:
    target: np.ndarray
    hist_pad: np.ndarray
    seen: np.ndarray

    def __len__(self):
        return len(self.target)


def make_train_set(ev: pd.DataFrame, start, end, n_items: int) -> TrainSet:
    """One example per positive event in [start, end).

    History = the member's positives on STRICTLY EARLIER days. Re-watches of an
    already-completed title are skipped as targets (they cannot be hits at
    evaluation time either).
    """
    p = ev[ev["positive"] & (ev["date"] >= start) & (ev["date"] < end)].sort_values(
        ["member_id", "date"], kind="stable")
    targets, hists = [], []
    for _, g in p.groupby("member_id", sort=False):
        seen: dict = {}
        for _, gd in g.groupby("date", sort=True):
            hist = list(seen.keys())
            day_items = list(dict.fromkeys(gd["item"].tolist()))
            for it in day_items:
                if it not in seen:
                    targets.append(it)
                    hists.append(hist)
            for it in day_items:
                seen.pop(it, None)
                seen[it] = None
    if not targets:
        raise ValueError("no training examples in the requested window")
    pad, seen_mask = pack_histories(hists, n_items)
    return TrainSet(np.asarray(targets, dtype=np.int64), pad, seen_mask)


# ----------------------------------------------------------------------------
# Evaluation window
# ----------------------------------------------------------------------------
@dataclass
class EvalSet:
    members: list
    hist_pad: np.ndarray
    seen: np.ndarray
    targets: list  # list[set[int]]
    stats: dict


def make_eval_set(ev: pd.DataFrame, start, end, n_items: int) -> EvalSet:
    """Eligible member = at least one event before `start` AND at least one
    positive in [start, end) on a catalog title they had not already completed."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    hist = seen_before(ev, start)
    active = set(ev.loc[ev["date"] < start, "member_id"])
    win = ev[(ev["date"] >= start) & (ev["date"] < end)]
    wpos = win[win["positive"]]

    members, hists, targets = [], [], []
    with_pos = cold = no_new = 0
    for m, g in wpos.groupby("member_id", sort=True):
        with_pos += 1
        if m not in active:
            cold += 1
            continue
        h = hist.get(m, [])
        tg = set(g["item"].tolist()) - set(h)
        if not tg:
            no_new += 1
            continue
        members.append(m)
        hists.append(h)
        targets.append(tg)
    pad, seen = pack_histories(hists, n_items)
    stats = {
        "window": f"{start.date()} to {(end - pd.Timedelta(days=1)).date()}",
        "window_events": int(len(win)),
        "window_positive_events": int(len(wpos)),
        "positive_flag_events_on_titles_outside_catalog": int((win["flag"] & (win["item"] < 0)).sum()),
        "members_with_positives": with_pos,
        "members_dropped_no_prior_event": cold,
        "members_dropped_only_rewatches": no_new,
        "eligible_members": len(members),
    }
    return EvalSet(members, pad, seen, targets, stats)


def popularity_counts(ev: pd.DataFrame, start, end, n_items: int) -> np.ndarray:
    """Positive views per title in [start, end) -- built from history only."""
    p = ev[ev["positive"] & (ev["date"] >= start) & (ev["date"] < end)]
    return np.bincount(p["item"], minlength=n_items).astype(np.float64)


# ----------------------------------------------------------------------------
# Picks and metrics
# ----------------------------------------------------------------------------
def topk_unseen(scores: np.ndarray, seen: np.ndarray, k: int = K) -> np.ndarray:
    """Top-k items per row, never recommending already-completed titles.
    Ties break toward the lower item index, so results are deterministic."""
    s = np.where(seen, -np.inf, scores.astype(np.float64))
    s = s - np.arange(s.shape[1])[None, :] * 1e-9
    idx = np.argpartition(-s, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)


def hits_and_recall(picks: np.ndarray, targets: list):
    hit = np.array([float(len(set(p.tolist()) & t) > 0) for p, t in zip(picks, targets)])
    rec = np.array([len(set(p.tolist()) & t) / len(t) for p, t in zip(picks, targets)])
    return hit, rec


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    d = a - b
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(0, n_boot, 200):  # chunked to keep memory flat
        m = min(200, n_boot - i)
        means[i:i + m] = d[rng.integers(0, len(d), (m, len(d)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), (float(lo), float(hi))


def compare(model_scores: np.ndarray, es: EvalSet, pop: np.ndarray,
            n_boot: int = 0, seed: int = 0) -> dict:
    """Hit Rate@10 / Recall@10 for the model, the global popularity top 10
    (same members, same already-seen removal) and a seeded random ranker."""
    if len(es.members) == 0:
        raise ValueError("no eligible members in this window")
    rng = np.random.default_rng(seed)
    m_hit, m_rec = hits_and_recall(topk_unseen(model_scores, es.seen), es.targets)
    p_hit, p_rec = hits_and_recall(
        topk_unseen(np.broadcast_to(pop, es.seen.shape), es.seen), es.targets)
    r_hit, _ = hits_and_recall(topk_unseen(rng.random(es.seen.shape), es.seen), es.targets)
    out = {
        "eligible_members": len(es.members),
        "model_hit_rate@10": float(m_hit.mean()),
        "popularity_hit_rate@10": float(p_hit.mean()),
        "random_hit_rate@10": float(r_hit.mean()),
        "model_recall@10": float(m_rec.mean()),
        "popularity_recall@10": float(p_rec.mean()),
    }
    if n_boot > 0:
        diff, (lo, hi) = paired_bootstrap(m_hit, p_hit, n_boot, seed)
        out.update({"hit_rate_diff_model_minus_popularity": diff,
                    "hit_rate_diff_ci95": [lo, hi]})
    return out
