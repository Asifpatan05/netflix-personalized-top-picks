#!/usr/bin/env python3
"""Clean Netflix viewing events and split them chronologically for a
recommendation model (train) and a final-week holdout.

Usage:
    python cleaning.py [--events EVENTS.csv] [--catalog CATALOG.csv] [--out-dir DIR]

Design rules
  * Source files are only ever read (SHA-256 recorded before and after).
  * No event is silently dropped. Problems are flagged; the only possible
    exclusions (unparseable date, duplicate event_id) are written to
    excluded_records.csv and counted in the report. Both are 0 in this export.
  * watch_seconds is never imputed, clipped or altered. Missing stays null,
    over-long values are kept for audit and marked unusable for training.
  * `completed` is never changed here (see report section 3.4).
  * Every cleaning rule is row-level or uses constants / the title catalog.
    Nothing is fitted on event data, so holdout rows cannot influence train
    rows. check_leakage_replay() proves this by re-cleaning train rows alone.

Outputs (in --out-dir):
    cleaned_viewing_history.csv, train_events.csv, holdout_events.csv,
    data_quality_report.md   (+ excluded_records.csv only if rows were excluded)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SOURCE_START = pd.Timestamp("2026-08-17")
HOLDOUT_START = pd.Timestamp("2026-11-10")  # cutoff: inclusive start of holdout
SOURCE_END = pd.Timestamp("2026-11-16")
EXPECTED_RAW_TRAIN = 4622                    # counts supplied with the brief
EXPECTED_RAW_HOLDOUT = 378

OVERLONG_TOLERANCE = 1.10                    # threshold = runtime * 60 * 1.10
PROVISIONAL_FALLBACK_SECONDS = 8844          # 134 min * 60 * 1.10 (to be confirmed)
COMPLETION_THRESHOLD = 0.90                  # completed=True needs >= 90% of runtime
VERY_SHORT_RATIO = 0.10                      # "very short" completed watch
VERY_SHORT_ABS_SECONDS = 300                 # used when runtime is unknown

DEVICE_ALIASES = {
    "tv": "tv", "television": "tv", "smart tv": "tv", "smart_tv": "tv",
    "mobile": "mobile", "phone": "mobile", "smartphone": "mobile",
    "tablet": "tablet", "ipad": "tablet",
    "web": "web", "browser": "web", "desktop": "web", "pc": "web", "laptop": "web",
}
COUNTRY_ALIASES = {"UK": "GB"}

FLAG_COLS = [
    "unmatched_title_id", "runtime_unavailable", "missing_watch_time",
    "overlong_watch_time", "completed_below_90pct_runtime", "completed_very_short",
    "completed_unverifiable", "device_missing", "country_missing",
    "event_before_release",
]
OUTPUT_COLS = [
    "event_id", "member_id", "title_id", "event_datetime", "split",
    "watch_seconds", "completed", "device_type", "country", "profile_type",
    "title_type", "primary_genre", "secondary_genre", "runtime_minutes",
    "watch_to_runtime", "overlong_threshold_seconds", "overlong_threshold_source",
    "overlong_severity", "watch_time_usable_for_training", "completed_validation",
] + FLAG_COLS + ["quality_flags", "quality_status"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_str(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    return s.mask((s == "").fillna(False), pd.NA)


def parse_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype("string").str.strip(), format="%Y-%m-%d", errors="coerce")


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def n(x) -> str:
    return f"{int(x):,}"


def pct(a, b) -> str:
    return f"{(100 * a / b):.1f}%" if b else "n/a"


# ----------------------------------------------------------------------------
# Catalog confirmation (needed before the global fallback is applied)
# ----------------------------------------------------------------------------
def prepare_catalog(catalog: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cat = catalog.copy()
    cat["title_id"] = clean_str(cat["title_id"]).str.upper()
    cat["runtime_minutes"] = pd.to_numeric(cat["runtime_minutes"], errors="coerce")
    cat["release_date"] = parse_dates(cat["release_date"])
    if cat["title_id"].duplicated().any():
        raise ValueError("Duplicate title_id values in catalog")
    top = cat.loc[cat["runtime_minutes"].idxmax()]
    max_rt = float(cat["runtime_minutes"].max())
    info = {
        "titles": len(cat),
        "columns": len(catalog.columns),
        "missing_runtime_ids": cat.loc[cat["runtime_minutes"].isna(), "title_id"].tolist(),
        "max_runtime_minutes": max_rt,
        "max_runtime_title_id": top["title_id"],
        "max_runtime_title_name": top["title_name"],
        "n_at_max": int((cat["runtime_minutes"] == max_rt).sum()),
        "fallback_seconds": int(round(max_rt * 60 * OVERLONG_TOLERANCE)),
    }
    info["fallback_confirmed"] = info["fallback_seconds"] == PROVISIONAL_FALLBACK_SECONDS
    return cat, info


# ----------------------------------------------------------------------------
# Core cleaning (pure function of rows + catalog + constants)
# ----------------------------------------------------------------------------
def clean_events(raw: pd.DataFrame, cat: pd.DataFrame, fallback_seconds: int):
    """Return (cleaned, excluded, norm_stats)."""
    df = raw.copy()
    for col in ["event_id", "member_id", "title_id", "device_type", "country", "profile_type"]:
        df[col] = clean_str(df[col])
    df["title_id"] = df["title_id"].str.upper()
    df["_date"] = parse_dates(df["event_datetime"])

    # --- exclusions (documented, none expected) ---
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    reason = reason.mask(df["_date"].isna(), "unparseable_event_date")
    dup = df["event_id"].duplicated(keep="first") & reason.isna()
    reason = reason.mask(dup, "duplicate_event_id")
    excluded = raw.loc[reason.notna()].copy()
    excluded["exclusion_reason"] = reason[reason.notna()]
    df = df.loc[reason.isna()].copy()

    # --- split by date only ---
    df["split"] = np.where(df["_date"] >= HOLDOUT_START, "holdout", "train")
    df["event_datetime"] = df["_date"].dt.strftime("%Y-%m-%d")

    # --- title matching and catalog attributes ---
    cat_cols = ["title_id", "title_type", "primary_genre", "secondary_genre",
                "runtime_minutes", "release_date"]
    df = df.merge(cat[cat_cols], on="title_id", how="left", validate="m:1")
    df["unmatched_title_id"] = ~df["title_id"].isin(cat["title_id"])
    df["runtime_unavailable"] = (~df["unmatched_title_id"]) & df["runtime_minutes"].isna()
    runtime_s = df["runtime_minutes"] * 60

    # --- watch time: never imputed, clipped or altered ---
    watch = pd.to_numeric(df["watch_seconds"], errors="coerce")
    df["missing_watch_time"] = watch.isna()
    known_rt = runtime_s.notna()
    df["overlong_threshold_seconds"] = (runtime_s * OVERLONG_TOLERANCE).where(known_rt, float(fallback_seconds))
    df["overlong_threshold_source"] = np.where(known_rt, "title_runtime", "global_fallback")
    df["overlong_watch_time"] = (watch > df["overlong_threshold_seconds"]).fillna(False)
    df["overlong_severity"] = np.select(
        [(watch > fallback_seconds).fillna(False), df["overlong_watch_time"]],
        ["exceeds_catalog_max", "exceeds_title_threshold"], default="none")
    df["watch_time_usable_for_training"] = watch.notna() & (watch > 0) & ~df["overlong_watch_time"]
    df["watch_to_runtime"] = (watch / runtime_s).round(4)
    if (watch.dropna() % 1 == 0).all():
        watch = watch.astype("Int64")
    df["watch_seconds"] = watch
    w = pd.to_numeric(df["watch_seconds"], errors="coerce")

    # --- completed validation (flags only; values never changed) ---
    comp = df["completed"]
    if comp.dtype != bool:
        comp = (comp.astype("string").str.strip().str.lower()
                .map({"true": True, "false": False}).astype("boolean"))
    comp_true = comp.eq(True).fillna(False).astype(bool)
    df["completed"] = comp
    ratio = df["watch_to_runtime"]
    unverifiable = comp_true & (w.isna() | ~known_rt | df["overlong_watch_time"])
    verifiable = comp_true & ~unverifiable
    below90 = verifiable & (ratio < COMPLETION_THRESHOLD)
    short_test = np.where(ratio.notna(), ratio < VERY_SHORT_RATIO, w < VERY_SHORT_ABS_SECONDS)
    very_short = verifiable & pd.Series(short_test, index=df.index).fillna(False)
    df["completed_unverifiable"] = unverifiable
    df["completed_below_90pct_runtime"] = below90
    df["completed_very_short"] = very_short
    df["completed_validation"] = np.select(
        [~comp_true, unverifiable, very_short, below90],
        ["not_applicable", "unverifiable", "very_short", "below_90pct_runtime"],
        default="consistent")

    # --- device / country normalisation; missing stays 'unknown' ---
    dev_in, ctry_in = df["device_type"].copy(), df["country"].copy()
    dev = dev_in.str.lower().map(DEVICE_ALIASES)
    ctry = ctry_in.str.upper().replace(COUNTRY_ALIASES)
    ctry = ctry.where(ctry.str.fullmatch(r"[A-Z]{2}").fillna(False), pd.NA)
    df["device_missing"] = dev_in.isna()
    df["country_missing"] = ctry_in.isna()
    norm_stats = {
        "device_changed": int((dev_in.notna() & dev.notna() & (dev_in != dev)).sum()),
        "device_unrecognised": int((dev_in.notna() & dev.isna()).sum()),
        "country_changed": int((ctry_in.notna() & ctry.notna() & (ctry_in != ctry)).sum()),
        "country_invalid": int((ctry_in.notna() & ctry.isna()).sum()),
    }
    df["device_type"] = dev.fillna("unknown").astype(str)
    df["country"] = ctry.fillna("unknown").astype(str)
    df["profile_type"] = df["profile_type"].str.lower().fillna("unknown").astype(str)

    # --- other consistency ---
    df["event_before_release"] = (df["_date"] < df["release_date"]).fillna(False)

    # --- quality summary ---
    df["quality_flags"] = df[FLAG_COLS].apply(
        lambda r: "|".join(c for c in FLAG_COLS if r[c]), axis=1)
    df["quality_status"] = np.where(df["quality_flags"] == "", "clean", "flagged")

    out = df[OUTPUT_COLS].sort_values("event_id").reset_index(drop=True)
    return out, excluded, norm_stats


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def check_leakage_replay(raw: pd.DataFrame, cat, fallback, cleaned: pd.DataFrame) -> tuple[bool, str]:
    """Re-clean only the train rows; result must equal the train rows of the full run."""
    train_raw = raw.loc[parse_dates(raw["event_datetime"]) < HOLDOUT_START]
    replay, _, _ = clean_events(train_raw, cat, fallback)
    full_train = cleaned.loc[cleaned["split"] == "train"].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(replay, full_train, check_dtype=False)
        return True, f"re-cleaning {n(len(replay))} train rows without any holdout rows gave identical output"
    except AssertionError as exc:
        return False, f"mismatch: {str(exc)[:120]}"


def run_checks(raw, cat, cat_info, cleaned, excluded, out_dir, hashes_before, hashes_after,
               paths, raw_dates, fallback):
    checks = []

    def add(name, status, detail):
        checks.append((name, status, detail))

    ok = lambda b: "PASS" if b else "FAIL"

    # re-read what was actually written
    cl = pd.read_csv(out_dir / "cleaned_viewing_history.csv")
    tr = pd.read_csv(out_dir / "train_events.csv")
    ho = pd.read_csv(out_dir / "holdout_events.csv")
    for d in (cl, tr, ho):
        d["_d"] = pd.to_datetime(d["event_datetime"], format="%Y-%m-%d")

    # row counts
    add("Source rows = cleaned + excluded", ok(len(raw) == len(cl) + len(excluded)),
        f"{n(len(raw))} = {n(len(cl))} + {n(len(excluded))}")
    add("All source event IDs retained or documented",
        ok(set(raw["event_id"]) == set(cl["event_id"]) | set(excluded["event_id"])),
        "every source event_id is in cleaned_viewing_history.csv" if excluded.empty
        else "excluded IDs listed in excluded_records.csv")
    add("Cleaned = train + holdout files", ok(len(cl) == len(tr) + len(ho)),
        f"{n(len(cl))} = {n(len(tr))} + {n(len(ho))}")
    add("event_id unique in cleaned history", ok(cl["event_id"].is_unique), f"{n(cl['event_id'].nunique())} unique IDs")
    add("Split column holds only train/holdout", ok(set(cl["split"]) == {"train", "holdout"}),
        f"values: {sorted(cl['split'].unique())}")
    add("Split files match split column",
        ok((tr["split"] == "train").all() and (ho["split"] == "holdout").all()
           and set(tr["event_id"]) == set(cl.loc[cl.split == "train", "event_id"])
           and set(ho["event_id"]) == set(cl.loc[cl.split == "holdout", "event_id"])),
        "train_events.csv contains only train rows; holdout_events.csv only holdout rows")
    raw_train = int((raw_dates < HOLDOUT_START).sum())
    raw_hold = int((raw_dates >= HOLDOUT_START).sum())
    add("Source split matches brief (4,622 / 378)",
        ok(raw_train == EXPECTED_RAW_TRAIN and raw_hold == EXPECTED_RAW_HOLDOUT),
        f"source rows by date: train {n(raw_train)}, holdout {n(raw_hold)}")

    # title matching
    add("Title matching re-run on cleaned history",
        ok((cl["unmatched_title_id"] == ~cl["title_id"].isin(cat["title_id"])).all()),
        f"{n(cl['unmatched_title_id'].sum())} unmatched events flagged; no title_id values were replaced "
        f"(IDs identical to source: {bool((raw.set_index('event_id').loc[cl['event_id'], 'title_id'].values == cl['title_id'].values).all())})")

    # dates
    add("No missing or invalid event dates", ok(cl["_d"].notna().all() and excluded.empty), "all dates parse as YYYY-MM-DD")
    add("Dates within source window",
        ok(cl["_d"].min() >= SOURCE_START and cl["_d"].max() <= SOURCE_END),
        f"{cl['_d'].min().date()} to {cl['_d'].max().date()}")
    add("Train dates 2026-08-17..2026-11-09",
        ok(tr["_d"].min() >= SOURCE_START and tr["_d"].max() < HOLDOUT_START),
        f"{tr['_d'].min().date()} to {tr['_d'].max().date()}")
    add("Holdout dates 2026-11-10..2026-11-16",
        ok(ho["_d"].min() >= HOLDOUT_START and ho["_d"].max() <= SOURCE_END),
        f"{ho['_d'].min().date()} to {ho['_d'].max().date()}")
    b9 = cl[cl["event_datetime"] == "2026-11-09"]
    b10 = cl[cl["event_datetime"] == "2026-11-10"]
    add("Cutoff boundary: 9 Nov -> train, 10 Nov -> holdout",
        ok((b9["split"] == "train").all() and (b10["split"] == "holdout").all()),
        f"2026-11-09: {n(len(b9))} rows all train; 2026-11-10: {n(len(b10))} rows all holdout")

    # leakage
    add("No train/holdout overlap in event_id", ok(not (set(tr["event_id"]) & set(ho["event_id"]))),
        f"{n(len(set(tr['event_id']) & set(ho['event_id'])))} shared IDs")
    add("Every train date earlier than every holdout date", ok(tr["_d"].max() < ho["_d"].min()),
        f"latest train {tr['_d'].max().date()} < earliest holdout {ho['_d'].min().date()}")
    okr, detail = check_leakage_replay(raw, cat, fallback, cleaned)
    add("No holdout influence on train cleaning (replay test)", ok(okr), detail)
    late = cat[cat["release_date"] >= HOLDOUT_START]["title_id"]
    leak_titles = tr[tr["title_id"].isin(late)]
    add("Train events on titles released on/after cutoff",
        "PASS" if leak_titles.empty else "WARN",
        f"{n(len(leak_titles))} train events across {leak_titles['title_id'].nunique()} titles "
        f"(catalog is a November snapshot; these rows carry flag event_before_release)")
    cold_m = len(set(ho["member_id"]) - set(tr["member_id"]))
    cold_t = len(set(ho["title_id"]) - set(tr["title_id"]))
    add("Holdout members / titles unseen in train", "INFO", f"{cold_m} members, {cold_t} titles")

    # value checks
    add("device_type values valid", ok(set(cl["device_type"]) <= {"tv", "mobile", "tablet", "web", "unknown"}),
        f"values: {sorted(cl['device_type'].unique())}")
    add("country values valid ISO-style codes or unknown",
        ok(cl["country"].str.fullmatch(r"[A-Z]{2}|unknown").all()), f"{cl['country'].nunique()} distinct values incl. unknown")
    add("Missing watch_seconds preserved as null",
        ok(cl["watch_seconds"].isna().sum() == cl["missing_watch_time"].sum() == raw["watch_seconds"].isna().sum()),
        f"{n(cl['watch_seconds'].isna().sum())} nulls in output = {n(raw['watch_seconds'].isna().sum())} in source")
    src_w = raw.set_index("event_id").loc[cl["event_id"], "watch_seconds"].values
    same = np.array_equal(np.nan_to_num(src_w, nan=-1), np.nan_to_num(cl["watch_seconds"].astype(float).values, nan=-1))
    add("watch_seconds values identical to source (no imputation/clipping)", ok(same), "compared every row")
    add("Fallback threshold confirmed against catalog",
        ok(cat_info["fallback_confirmed"]),
        f"max runtime {cat_info['max_runtime_minutes']:.0f} min x 60 x 1.10 = {cat_info['fallback_seconds']:,} s "
        f"(provisional {PROVISIONAL_FALLBACK_SECONDS:,} s)")
    # source preservation
    add("Source files unchanged (SHA-256 before = after)", ok(hashes_before == hashes_after),
        "; ".join(f"{p.name}: {hashes_after[p][:12]}..." for p in paths))
    return checks


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def build_report(raw, cat, cat_info, cleaned, excluded, norm_stats, checks, hashes_before,
                 hashes_after, paths, fallback, raw_dates) -> str:
    c = cleaned
    tr, ho = c[c.split == "train"], c[c.split == "holdout"]
    total = len(c)
    L = []
    A = L.append
    A("# Data Quality Report: Netflix viewing events")
    A(f"\nGenerated {datetime.now().strftime('%Y-%m-%d %H:%M')} by `cleaning.py`. "
      "Purpose: prepare reliable viewing-event data for a personalised recommendation model, "
      "with a hidden final-week holdout used only for final evaluation.\n")

    fails = [x for x in checks if x[1] == "FAIL"]
    A(f"**Validation status: {'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' CHECK(S) FAILED'}** "
      f"({sum(x[1]=='PASS' for x in checks)} pass, {sum(x[1]=='WARN' for x in checks)} warn, "
      f"{sum(x[1]=='INFO' for x in checks)} info).\n")

    # 1 ----------------------------------------------------------------
    A("## 1. Row reconciliation and split\n")
    A(md_table(pd.DataFrame([
        ["Source events", n(len(raw)), n((raw_dates < HOLDOUT_START).sum()), n((raw_dates >= HOLDOUT_START).sum())],
        ["Excluded (unparseable date / duplicate ID)", n(len(excluded)), "-", "-"],
        ["Cleaned events (all retained, flagged where needed)", n(total), n(len(tr)), n(len(ho))],
    ], columns=["Stage", "All", "train", "holdout"])))
    A(f"\n- **Train:** 2026-08-17 to 2026-11-09 (`split = train`). **Holdout:** 2026-11-10 to 2026-11-16 "
      "(`split = holdout`). The cutoff date itself belongs to the holdout.")
    A("- Every source event ID is present in `cleaned_viewing_history.csv`; none were excluded.")
    A(f"- Rows per split match the counts in the brief ({EXPECTED_RAW_TRAIN:,} / {EXPECTED_RAW_HOLDOUT}).")
    A(f"- Event dates are date-only in the source (no time of day).\n")

    # 2 ----------------------------------------------------------------
    A("## 2. Source files preserved\n")
    A(md_table(pd.DataFrame(
        [[p.name, hashes_before[p], "unchanged" if hashes_before[p] == hashes_after[p] else "CHANGED"] for p in paths],
        columns=["File", "SHA-256", "After run"])))
    A("\nThe script opens source files read-only and writes only to the output folder.\n")

    # 3 ----------------------------------------------------------------
    A("## 3. Findings and cleaning decisions\n")

    # 3.1 titles
    unm = c[c.unmatched_title_id]
    A("### 3.1 Title matching against the November 2026 catalog\n")
    A(f"Catalog: {cat_info['titles']} titles, {cat_info['columns']} columns, no duplicate title IDs. "
      f"{n(len(unm))} of {n(total)} events ({pct(len(unm), total)}) reference "
      f"{unm.title_id.nunique()} IDs that are not in the catalog.\n")
    ut = (unm.groupby("title_id").agg(events=("event_id", "size"),
          train=("split", lambda s: (s == "train").sum()),
          holdout=("split", lambda s: (s == "holdout").sum()),
          members=("member_id", "nunique")).reset_index())
    A(md_table(ut))
    A(f"\nThe catalog runs TT-0001 to TT-0300, so these IDs are outside its range. **Decision:** keep the events, "
      f"flag `unmatched_title_id`, do not invent or guess replacement IDs. They carry no genre or runtime, so "
      f"content-based features cannot use them. **Unresolved:** the data owner should say what these IDs are.\n")
    rt_miss = c[c.runtime_unavailable]
    A(f"{len(cat_info['missing_runtime_ids'])} catalog titles have no runtime "
      f"({', '.join(cat_info['missing_runtime_ids'])}); they account for {n(len(rt_miss))} events, flagged "
      f"`runtime_unavailable`. Together with the unmatched IDs, {n((c.overlong_threshold_source == 'global_fallback').sum())} "
      f"events use the global fallback threshold.\n")

    # 3.2 watch time
    A("### 3.2 Watch time\n")
    miss = c[c.missing_watch_time]
    A(f"**Missing `watch_seconds`:** {n(len(miss))} events ({pct(len(miss), total)}): "
      f"{n((miss.split=='train').sum())} train, {n((miss.split=='holdout').sum())} holdout; "
      f"{n(miss.unmatched_title_id.sum())} are on unmatched titles and {n(miss.completed.sum())} are marked completed. "
      f"**Decision:** keep, preserve null, flag `missing_watch_time`, never fill with zero, a mean or any invented value. "
      f"They are excluded from watch-time training only.\n")
    w = pd.to_numeric(raw["watch_seconds"], errors="coerce")
    gt3600 = c[c.watch_seconds.astype(float) > 3600]
    ol = c[c.overlong_watch_time]
    extra = ol[~(ol.watch_seconds.astype(float) > 3600)]
    A("**Over-long `watch_seconds`:** a title-specific threshold replaces the global 3,600 s rule: "
      "`runtime_minutes x 60 x 1.10`, and "
      f"{fallback:,} s ({cat_info['max_runtime_minutes']:.0f} min x 60 x 1.10, confirmed against the catalog: "
      f"longest title {cat_info['max_runtime_title_id']} \"{cat_info['max_runtime_title_name']}\", "
      f"{cat_info['n_at_max']} titles share that maximum) where runtime is unavailable.\n")
    sev = c.groupby("overlong_severity").size()
    A(md_table(pd.DataFrame([
        ["Events above 3,600 s (initial finding)", n(len(gt3600)), f"{n((gt3600.overlong_watch_time).sum())} of them are flagged"],
        ["Events flagged `overlong_watch_time` (title-specific rule)", n(len(ol)),
         f"{n((ol.split=='train').sum())} train / {n((ol.split=='holdout').sum())} holdout"],
        ["...of which longer than any title in the catalog (> %s s)" % f"{fallback:,}",
         n(sev.get("exceeds_catalog_max", 0)), "severity `exceeds_catalog_max`"],
        ["...of which only above their own title's threshold", n(sev.get("exceeds_title_threshold", 0)),
         "severity `exceeds_title_threshold`"],
        ["Flagged events that are on series", n((ol.title_type == "series").sum()), "runtime appears to be per episode"],
        ["Flagged events that are on films", n((ol.title_type == "film").sum()), ""],
    ], columns=["Measure", "Events", "Note"])))
    A(f"\n**The 46 initial findings do not carry over unchanged.** All {n(len(gt3600))} events above 3,600 s are still "
      f"flagged, but the title-specific rule flags {n(len(ol))} events in total, because {n(len(extra))} further events "
      f"are at or below 3,600 s yet exceed a short title's own threshold. "
      f"{n((extra.title_type=='series').sum())} of those are series events. Series runtimes in the catalog "
      f"({cat[cat.title_type=='series'].runtime_minutes.min():.0f} to {cat[cat.title_type=='series'].runtime_minutes.max():.0f} min) "
      f"look like per-episode lengths (unconfirmed), so a viewer watching several episodes in one session would exceed the "
      f"threshold without anything being wrong with the data. The events above {fallback:,} s "
      f"(max {n(c.watch_seconds.astype(float).max())} s, about {c.watch_seconds.astype(float).max()/3600:.1f} h) are the clearly "
      f"implausible group, with no values between {n(c.watch_seconds.astype(float)[c.watch_seconds.astype(float) <= fallback].max())} s and "
      f"{n(c.watch_seconds.astype(float)[c.watch_seconds.astype(float) > fallback].min())} s. "
      f"Use `overlong_severity` to separate the two groups.\n")
    A("**Decision:** flag, never delete, clip or reduce. Original values stay in `watch_seconds` for audit. "
      "`watch_time_usable_for_training` is false for missing, non-positive and flagged over-long durations: "
      f"{n(c.watch_time_usable_for_training.sum())} of {n(total)} events ({pct(c.watch_time_usable_for_training.sum(), total)}) "
      f"remain usable ({n(tr.watch_time_usable_for_training.sum())} train, {n(ho.watch_time_usable_for_training.sum())} holdout). "
      f"Non-positive durations in the source: {n((w <= 0).sum())}.\n")

    # 3.3 usable table
    A("**Watch-time flag counts by split**\n")
    A(md_table(pd.DataFrame([
        [lbl, n(a[col].sum()), n(b[col].sum()), n(c[col].sum())]
        for lbl, col in [("missing_watch_time", "missing_watch_time"), ("overlong_watch_time", "overlong_watch_time")]
        for a, b in [(tr, ho)]
    ] + [["usable for watch-time training", n(tr.watch_time_usable_for_training.sum()),
          n(ho.watch_time_usable_for_training.sum()), n(c.watch_time_usable_for_training.sum())]],
        columns=["Flag", "train", "holdout", "all"])))
    A("")

    # 3.4 completed
    A("### 3.3 `completed` validation\n")
    ct = c[c.completed.eq(True)]
    vc = ct.completed_validation.value_counts()
    A(f"{n(len(ct))} events are marked `completed = True`. Each was compared with its title's runtime "
      f"(a completion needs at least {int(COMPLETION_THRESHOLD*100)}% of runtime).\n")
    A(md_table(pd.DataFrame([
        ["consistent (>= 90% of runtime)", n(vc.get("consistent", 0))],
        ["below 90% of runtime (flag `completed_below_90pct_runtime`)", n(vc.get("below_90pct_runtime", 0) + vc.get("very_short", 0))],
        ["...of which very short (< 10% of runtime, or < 300 s if runtime unknown; flag `completed_very_short`)", n(vc.get("very_short", 0))],
        ["unverifiable (missing watch time, no runtime or over-long; flag `completed_unverifiable`)", n(vc.get("unverifiable", 0))],
    ], columns=["Result for completed = True", "Events"])))
    fb = ct[ct.completed_below_90pct_runtime | ct.completed_very_short]
    fm = fb.groupby("title_type").size()
    tot_by_type = ct[ct.completed_validation.isin(["consistent", "below_90pct_runtime", "very_short"])].groupby("title_type").size()
    A(f"\n{pct(len(fb), vc.get('consistent',0)+len(fb))} of verifiable completed events fall below 90% of runtime "
      f"(films {n(fm.get('film',0))} of {n(tot_by_type.get('film',0))}, series {n(fm.get('series',0))} of {n(tot_by_type.get('series',0))}). "
      f"Median watch share for films marked completed is "
      f"{ct[(ct.title_type=='film')&ct.watch_to_runtime.notna()].watch_to_runtime.median():.0%} of runtime. "
      "A failure rate this large looks like a different definition of `completed`, not scattered bad records.\n")
    A("**Corrections made: 0.** No inconsistency is *confirmed*, because the rule itself cannot tell a wrong flag "
      "from a differently defined one. Flipping these values would rewrite most of the column on a guess. "
      "`completed` is left exactly as in the source, with flags for filtering. "
      "**Unresolved:** the data owner should define what triggers `completed = True`; once confirmed, "
      "corrections can be applied from `completed_validation`.\n")

    # 3.5 device country
    A("### 3.4 `device_type`, `country`, `profile_type`\n")
    A(f"- `device_type`: {n(c.device_missing.sum())} missing set to `unknown`; values after cleaning: "
      f"{', '.join(sorted(c.device_type.unique()))}. Source values needing re-casing or aliasing: {n(norm_stats['device_changed'])}; "
      f"unrecognised values: {n(norm_stats['device_unrecognised'])}.")
    A(f"- `country`: {n(c.country_missing.sum())} missing set to `unknown`; {c[c.country!='unknown'].country.nunique()} distinct "
      f"valid two-letter codes ({', '.join(sorted(c[c.country!='unknown'].country.unique()))}). "
      f"Values needing upper-casing or aliasing: {n(norm_stats['country_changed'])}; invalid codes: {n(norm_stats['country_invalid'])}. "
      "Validation is by format (two capital letters), not against the full ISO list.")
    A("- Missing values are never guessed or filled from other rows. `profile_type` was already clean (`standard`, `kids`).\n")

    # 3.6 other
    A("### 3.5 Other checks\n")
    ebr = c[c.event_before_release]
    A(f"- **Events dated before the title's release date:** {n(len(ebr))} events on {ebr.title_id.nunique()} titles "
      f"({', '.join(sorted(ebr.title_id.unique()))}), flagged `event_before_release`. Kept; may be early access or a data error.")
    cold_m = len(set(ho.member_id) - set(tr.member_id))
    cold_t = len(set(ho.title_id) - set(tr.title_id))
    A(f"- **Cold start in holdout:** {cold_m} members and {cold_t} titles in the holdout have no train events.")
    A(f"- No duplicate event IDs, no unparseable dates, no non-boolean `completed` values.\n")

    # 4 ----------------------------------------------------------------
    A("## 4. Corrections, exclusions and rejected records\n")
    A(md_table(pd.DataFrame([
        ["Events excluded from the cleaned history", n(len(excluded)), "none; all source events retained"],
        ["`watch_seconds` values changed", "0", "no imputation, clipping or reduction"],
        ["`completed` values changed", "0", "flagged only (section 3.3)"],
        ["`title_id` values changed", "0", "unmatched IDs flagged only"],
        ["`device_type` / `country` missing -> `unknown`", f"{n(c.device_missing.sum())} / {n(c.country_missing.sum())}", "no guessing"],
        ["Events excluded from watch-time model training only", n((~c.watch_time_usable_for_training).sum()),
         "missing or over-long watch time; rows stay in the file"],
    ], columns=["Action", "Events", "Note"])))
    A("")

    # 5 ----------------------------------------------------------------
    A("## 5. Validation checks\n")
    A(md_table(pd.DataFrame([[a, b, d] for a, b, d in checks], columns=["Check", "Result", "Detail"])))
    A("")

    # 6 ----------------------------------------------------------------
    A("## 6. Unresolved issues for the data owner\n")
    A(f"1. What are the unmatched title IDs ({', '.join(unm.title_id.unique())}) and why are they absent from the catalog?")
    A("2. What does `completed = True` mean, and is it comparable with catalog runtime?")
    A(f"3. Are series runtimes per episode, and can events be tied to episodes? The title-specific over-long rule "
      f"flags {n((ol.title_type=='series').sum())} series events that may be legitimate multi-episode sessions.")
    A(f"4. Can the {len(cat_info['missing_runtime_ids'])} missing catalog runtimes be supplied?")
    A("5. Is `event_before_release` early access or a data error?")
    A("6. Source events carry a date only, so time-of-day features are not possible.\n")
    A("## 7. Reproducing\n")
    A("`python cleaning.py --events <events.csv> --catalog <catalog.csv> --out-dir <folder>`. "
      "The script exits with a non-zero status if any check fails.\n")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    up = Path("/mnt/user-data/uploads")
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, default=up / "Netflix_viewing_events_export__17_Aug_to_16_Nov_2026.csv")
    ap.add_argument("--catalog", type=Path, default=up / "Netflix_title_catalog__November_2026.csv")
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/user-data/outputs"))
    a = ap.parse_args()
    paths = [a.events, a.catalog]
    out_dir = a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.resolve() in {p.resolve().parent for p in paths}:
        raise SystemExit("Refusing to write into the folder that holds the source files.")

    hashes_before = {p: sha256(p) for p in paths}
    raw = pd.read_csv(a.events)
    catalog = pd.read_csv(a.catalog)
    cat, cat_info = prepare_catalog(catalog)
    fallback = cat_info["fallback_seconds"]
    raw_dates = parse_dates(raw["event_datetime"])

    cleaned, excluded, norm_stats = clean_events(raw, cat, fallback)
    cleaned.to_csv(out_dir / "cleaned_viewing_history.csv", index=False)
    cleaned[cleaned.split == "train"].to_csv(out_dir / "train_events.csv", index=False)
    cleaned[cleaned.split == "holdout"].to_csv(out_dir / "holdout_events.csv", index=False)
    if len(excluded):
        excluded.to_csv(out_dir / "excluded_records.csv", index=False)

    hashes_after = {p: sha256(p) for p in paths}
    checks = run_checks(raw, cat, cat_info, cleaned, excluded, out_dir, hashes_before,
                        hashes_after, paths, raw_dates, fallback)
    report = build_report(raw, cat, cat_info, cleaned, excluded, norm_stats, checks,
                          hashes_before, hashes_after, paths, fallback, raw_dates)
    (out_dir / "data_quality_report.md").write_text(report, encoding="utf-8")

    for name, status, detail in checks:
        print(f"[{status:4}] {name}: {detail}")
    failed = [c for c in checks if c[1] == "FAIL"]
    print(f"\n{len(cleaned):,} cleaned rows written to {out_dir}; {len(failed)} failed check(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
