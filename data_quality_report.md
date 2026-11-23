# Data Quality Report: Netflix viewing events

Generated 2026-11-23 10:29 by `cleaning.py`. Purpose: prepare reliable viewing-event data for a personalised recommendation model, with a hidden final-week holdout used only for final evaluation.

**Validation status: ALL CHECKS PASSED** (23 pass, 0 warn, 1 info).

## 1. Row reconciliation and split

| Stage | All | train | holdout |
|---|---|---|---|
| Source events | 5,000 | 4,622 | 378 |
| Excluded (unparseable date / duplicate ID) | 0 | - | - |
| Cleaned events (all retained, flagged where needed) | 5,000 | 4,622 | 378 |

- **Train:** 2026-08-17 to 2026-11-09 (`split = train`). **Holdout:** 2026-11-10 to 2026-11-16 (`split = holdout`). The cutoff date itself belongs to the holdout.
- Every source event ID is present in `cleaned_viewing_history.csv`; none were excluded.
- Rows per split match the counts in the brief (4,622 / 378).
- Event dates are date-only in the source (no time of day).

## 2. Source files preserved

| File | SHA-256 | After run |
|---|---|---|
| Netflix_viewing_events_export__17_Aug_to_16_Nov_2026.csv | 6a424686f72496adcb88dda5b5004e1cb6e28574872e6d9d9b82fb6345715af1 | unchanged |
| Netflix_title_catalog__November_2026.csv | 7ea62a9e569dbf54f4356e6b3f3a72a230c293d13cff86b7dc2daa16a4ab6f47 | unchanged |

The script opens source files read-only and writes only to the output folder.

## 3. Findings and cleaning decisions

### 3.1 Title matching against the November 2026 catalog

Catalog: 300 titles, 10 columns, no duplicate title IDs. 180 of 5,000 events (3.6%) reference 3 IDs that are not in the catalog.

| title_id | events | train | holdout | members |
|---|---|---|---|---|
| TT-0347 | 61 | 54 | 7 | 58 |
| TT-0412 | 65 | 60 | 5 | 62 |
| TT-0588 | 54 | 47 | 7 | 52 |

The catalog runs TT-0001 to TT-0300, so these IDs are outside its range. **Decision:** keep the events, flag `unmatched_title_id`, do not invent or guess replacement IDs. They carry no genre or runtime, so content-based features cannot use them. **Unresolved:** the data owner should say what these IDs are.

10 catalog titles have no runtime (TT-0008, TT-0060, TT-0097, TT-0119, TT-0143, TT-0170, TT-0194, TT-0239, TT-0262, TT-0288); they account for 104 events, flagged `runtime_unavailable`. Together with the unmatched IDs, 284 events use the global fallback threshold.

### 3.2 Watch time

**Missing `watch_seconds`:** 74 events (1.5%): 62 train, 12 holdout; 5 are on unmatched titles and 10 are marked completed. **Decision:** keep, preserve null, flag `missing_watch_time`, never fill with zero, a mean or any invented value. They are excluded from watch-time training only.

**Over-long `watch_seconds`:** a title-specific threshold replaces the global 3,600 s rule: `runtime_minutes x 60 x 1.10`, and 8,844 s (134 min x 60 x 1.10, confirmed against the catalog: longest title TT-0129 "The Ballad of Clara Vane", 2 titles share that maximum) where runtime is unavailable.

| Measure | Events | Note |
|---|---|---|
| Events above 3,600 s (initial finding) | 46 | 46 of them are flagged |
| Events flagged `overlong_watch_time` (title-specific rule) | 827 | 765 train / 62 holdout |
| ...of which longer than any title in the catalog (> 8,844 s) | 46 | severity `exceeds_catalog_max` |
| ...of which only above their own title's threshold | 781 | severity `exceeds_title_threshold` |
| Flagged events that are on series | 812 | runtime appears to be per episode |
| Flagged events that are on films | 15 |  |

**The 46 initial findings do not carry over unchanged.** All 46 events above 3,600 s are still flagged, but the title-specific rule flags 827 events in total, because 781 further events are at or below 3,600 s yet exceed a short title's own threshold. 781 of those are series events. Series runtimes in the catalog (22 to 58 min) look like per-episode lengths (unconfirmed), so a viewer watching several episodes in one session would exceed the threshold without anything being wrong with the data. The events above 8,844 s (max 85,372 s, about 23.7 h) are the clearly implausible group, with no values between 3,598 s and 31,300 s. Use `overlong_severity` to separate the two groups.

**Decision:** flag, never delete, clip or reduce. Original values stay in `watch_seconds` for audit. `watch_time_usable_for_training` is false for missing, non-positive and flagged over-long durations: 4,099 of 5,000 events (82.0%) remain usable (3,795 train, 304 holdout). Non-positive durations in the source: 0.

**Watch-time flag counts by split**

| Flag | train | holdout | all |
|---|---|---|---|
| missing_watch_time | 62 | 12 | 74 |
| overlong_watch_time | 765 | 62 | 827 |
| usable for watch-time training | 3,795 | 304 | 4,099 |

### 3.3 `completed` validation

2,448 events are marked `completed = True`. Each was compared with its title's runtime (a completion needs at least 90% of runtime).

| Result for completed = True | Events |
|---|---|
| consistent (>= 90% of runtime) | 254 |
| below 90% of runtime (flag `completed_below_90pct_runtime`) | 1,477 |
| ...of which very short (< 10% of runtime, or < 300 s if runtime unknown; flag `completed_very_short`) | 68 |
| unverifiable (missing watch time, no runtime or over-long; flag `completed_unverifiable`) | 717 |

85.3% of verifiable completed events fall below 90% of runtime (films 902 of 902, series 575 of 829). Median watch share for films marked completed is 38% of runtime. A failure rate this large looks like a different definition of `completed`, not scattered bad records.

**Corrections made: 0.** No inconsistency is *confirmed*, because the rule itself cannot tell a wrong flag from a differently defined one. Flipping these values would rewrite most of the column on a guess. `completed` is left exactly as in the source, with flags for filtering. **Unresolved:** the data owner should define what triggers `completed = True`; once confirmed, corrections can be applied from `completed_validation`.

### 3.4 `device_type`, `country`, `profile_type`

- `device_type`: 135 missing set to `unknown`; values after cleaning: mobile, tablet, tv, unknown, web. Source values needing re-casing or aliasing: 0; unrecognised values: 0.
- `country`: 49 missing set to `unknown`; 12 distinct valid two-letter codes (AU, BR, CA, DE, ES, FR, GB, IN, JP, KR, MX, US). Values needing upper-casing or aliasing: 0; invalid codes: 0. Validation is by format (two capital letters), not against the full ISO list.
- Missing values are never guessed or filled from other rows. `profile_type` was already clean (`standard`, `kids`).

### 3.5 Other checks

- **Events dated before the title's release date:** 27 events on 3 titles (TT-0163, TT-0195, TT-0258), flagged `event_before_release`. Kept; may be early access or a data error.
- **Cold start in holdout:** 0 members and 0 titles in the holdout have no train events.
- No duplicate event IDs, no unparseable dates, no non-boolean `completed` values.

## 4. Corrections, exclusions and rejected records

| Action | Events | Note |
|---|---|---|
| Events excluded from the cleaned history | 0 | none; all source events retained |
| `watch_seconds` values changed | 0 | no imputation, clipping or reduction |
| `completed` values changed | 0 | flagged only (section 3.3) |
| `title_id` values changed | 0 | unmatched IDs flagged only |
| `device_type` / `country` missing -> `unknown` | 135 / 49 | no guessing |
| Events excluded from watch-time model training only | 901 | missing or over-long watch time; rows stay in the file |

## 5. Validation checks

| Check | Result | Detail |
|---|---|---|
| Source rows = cleaned + excluded | PASS | 5,000 = 5,000 + 0 |
| All source event IDs retained or documented | PASS | every source event_id is in cleaned_viewing_history.csv |
| Cleaned = train + holdout files | PASS | 5,000 = 4,622 + 378 |
| event_id unique in cleaned history | PASS | 5,000 unique IDs |
| Split column holds only train/holdout | PASS | values: ['holdout', 'train'] |
| Split files match split column | PASS | train_events.csv contains only train rows; holdout_events.csv only holdout rows |
| Source split matches brief (4,622 / 378) | PASS | source rows by date: train 4,622, holdout 378 |
| Title matching re-run on cleaned history | PASS | 180 unmatched events flagged; no title_id values were replaced (IDs identical to source: True) |
| No missing or invalid event dates | PASS | all dates parse as YYYY-MM-DD |
| Dates within source window | PASS | 2026-08-17 to 2026-11-16 |
| Train dates 2026-08-17..2026-11-09 | PASS | 2026-08-17 to 2026-11-09 |
| Holdout dates 2026-11-10..2026-11-16 | PASS | 2026-11-10 to 2026-11-16 |
| Cutoff boundary: 9 Nov -> train, 10 Nov -> holdout | PASS | 2026-11-09: 53 rows all train; 2026-11-10: 58 rows all holdout |
| No train/holdout overlap in event_id | PASS | 0 shared IDs |
| Every train date earlier than every holdout date | PASS | latest train 2026-11-09 < earliest holdout 2026-11-10 |
| No holdout influence on train cleaning (replay test) | PASS | re-cleaning 4,622 train rows without any holdout rows gave identical output |
| Train events on titles released on/after cutoff | PASS | 0 train events across 0 titles (catalog is a November snapshot; these rows carry flag event_before_release) |
| Holdout members / titles unseen in train | INFO | 0 members, 0 titles |
| device_type values valid | PASS | values: ['mobile', 'tablet', 'tv', 'unknown', 'web'] |
| country values valid ISO-style codes or unknown | PASS | 13 distinct values incl. unknown |
| Missing watch_seconds preserved as null | PASS | 74 nulls in output = 74 in source |
| watch_seconds values identical to source (no imputation/clipping) | PASS | compared every row |
| Fallback threshold confirmed against catalog | PASS | max runtime 134 min x 60 x 1.10 = 8,844 s (provisional 8,844 s) |
| Source files unchanged (SHA-256 before = after) | PASS | Netflix_viewing_events_export__17_Aug_to_16_Nov_2026.csv: 6a424686f724...; Netflix_title_catalog__November_2026.csv: 7ea62a9e569d... |

## 6. Unresolved issues for the data owner

1. What are the unmatched title IDs (TT-0412, TT-0347, TT-0588) and why are they absent from the catalog?
2. What does `completed = True` mean, and is it comparable with catalog runtime?
3. Are series runtimes per episode, and can events be tied to episodes? The title-specific over-long rule flags 812 series events that may be legitimate multi-episode sessions.
4. Can the 10 missing catalog runtimes be supplied?
5. Is `event_before_release` early access or a data error?
6. Source events carry a date only, so time-of-day features are not possible.

## 7. Reproducing

`python cleaning.py --events <events.csv> --catalog <catalog.csv> --out-dir <folder>`. The script exits with a non-zero status if any check fails.
