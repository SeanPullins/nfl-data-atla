#!/usr/bin/env python3
"""Build a leakage-safe, pre-draft QB feature table.

One row per (canonical_name, draft_season) for every QB in
qb_draft_profiles.csv. Only pre-draft college/PFF/CFBD/combine information
and draft market data (pick/round) are included. No NFL outcome data.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "data/external/NFLQBs/data/processed/qb_draft_profiles.csv"
COLLEGE_SEASONS = ROOT / "data/external/NFLQBs/data/processed/pff_qb_college.csv"
DRAFT_MASTER = ROOT / "data/raw/open_prospect/nflverse_draft_player_master_SAFE.csv"
OUT_CSV = ROOT / "data/processed/qb_model_features.csv"
OUT_META = ROOT / "data/processed/qb_model_features_metadata.json"
LABELED_ERA_MAX = 2023

# Final (most recent college) season PFF summary.
FINAL_CORE = [
    "final_grades_grades_pass", "final_grades_grades_offense", "final_grades_accuracy_percent",
    "final_grades_btt_rate", "final_grades_twp_rate", "final_grades_ypa",
    "final_grades_avg_depth_of_target", "final_grades_avg_time_to_throw",
    "final_grades_completion_percent", "final_grades_sack_percent",
    "final_grades_pressure_to_sack_rate", "final_grades_positive_epa_percent",
    "final_grades_epa", "final_grades_dropbacks", "final_grades_qb_rating",
]
CAREER_CORE = [
    "career_grades_grades_pass", "career_grades_grades_offense", "career_grades_accuracy_percent",
    "career_grades_btt_rate", "career_grades_twp_rate", "career_grades_ypa",
    "career_grades_avg_depth_of_target", "career_grades_avg_time_to_throw",
    "career_grades_completion_percent", "career_grades_sack_percent",
    "career_grades_pressure_to_sack_rate", "career_grades_positive_epa_percent",
    "career_grades_epa", "career_grades_dropbacks", "career_grades_qb_rating",
]
PRESSURE_SPLITS = [
    "final_pressure_pressure_accuracy_percent", "final_pressure_pressure_grades_pass",
    "final_pressure_pressure_twp_rate", "final_pressure_pressure_sack_percent",
    "final_pressure_no_pressure_accuracy_percent", "final_pressure_no_pressure_grades_pass",
    "final_pressure_blitz_accuracy_percent", "final_pressure_blitz_grades_pass",
]
DEPTH_SPLITS = [
    "final_depth_deep_attempts_percent", "final_depth_deep_accuracy_percent",
    "final_depth_deep_btt_rate", "final_depth_deep_ypa",
    "final_depth_medium_accuracy_percent", "final_depth_short_accuracy_percent",
]
CFBD_FEATURES = ["final_average_ppa_pass", "career_average_ppa_pass"]
COMBINE_FEATURES = [
    # combine_ras_score is excluded: 0/4123 non-null in the source table (dead column).
    "combine_height", "combine_weight", "combine_forty", "combine_vertical",
    "combine_broad", "combine_three_cone", "combine_short_shuttle", "combine_bmi",
]
# OL/pressure-context source columns from qb_draft_profiles.csv. These come
# from PFF's "allowed-pressure" report, which despite its name is really the
# QB's own full-season passing line plus one genuinely new signal --
# def_gen_pressures, the raw count of pressures the defense generated on
# that QB across the season (i.e. how much heat the QB's own offensive line
# let through). Most of the other allowed_* columns duplicate FINAL_CORE/
# CAREER_CORE/PRESSURE_SPLITS already captured above, so we only pull the
# columns needed to build supporting-cast-quality signals: pressure
# exposure rate, get-hit rate, and the raw pressure-to-sack conversion rate.
ALLOWED_CONTEXT_RAW = [
    "final_allowed_def_gen_pressures", "final_allowed_dropbacks", "final_allowed_hit_as_threw",
    "final_allowed_pressure_to_sack_rate",
    "career_allowed_def_gen_pressures", "career_allowed_dropbacks", "career_allowed_hit_as_threw",
    "career_allowed_pressure_to_sack_rate",
]
# Passing-concept splits (PFF passing-concept__QB__*.csv, verified against
# raw files: pa = play-action dropbacks, npa = non-play-action dropbacks,
# no_screen = dropbacks excluding screens, screen = screen-pass dropbacks;
# no_screen_dropbacks + screen_dropbacks == pa_dropbacks + npa_dropbacks ==
# concept_dropbacks, confirmed exactly equal across all non-null rows) and
# time-in-pocket splits (time-in-pocket__QB__*.csv: "less" = time-to-throw
# < 2.5s, "more" = time-to-throw >= 2.5s, confirmed via avg_time_to_throw
# ~1.7-1.9s for "less" vs ~3.3-3.9s for "more" in the raw files) that were
# previously unused. Source columns pulled to build the derived
# concept_*/pocket_* features below.
CONCEPT_POCKET_RAW = [
    "final_concept_no_screen_grades_pass", "final_concept_no_screen_accuracy_percent",
    "final_concept_no_screen_ypa",
    "final_concept_pa_grades_pass", "final_concept_npa_grades_pass",
    "final_concept_pa_accuracy_percent", "final_concept_npa_accuracy_percent",
    "final_concept_screen_dropbacks", "final_concept_dropbacks",
    "final_pocket_more_grades_pass", "final_pocket_more_accuracy_percent",
    "final_pocket_less_accuracy_percent",
    "career_concept_no_screen_grades_pass",
    "career_concept_pa_grades_pass", "career_concept_npa_grades_pass",
    "career_pocket_more_grades_pass", "career_pocket_less_accuracy_percent",
]

PASSTHROUGH_KEYS = ["canonical_name", "join_name", "draft_season", "colleges", "pff_player_id"]

# Single-season college table (pff_qb_college.csv) columns used for trajectory.
TRAJ_COLS = ["grades_grades_pass", "grades_ypa", "grades_twp_rate", "grades_accuracy_percent"]
TRAJ_DROPBACK_COL = "grades_dropbacks"


GENERATIONAL_SUFFIXES = {"ii", "iii", "iv", "v", "jr", "sr"}


def normalize(value: object) -> str:
    """Lowercase, strip periods/apostrophes, collapse whitespace, and strip
    a trailing generational suffix token (Jr/Sr/II/III/IV/V) so names like
    'Gardner Minshew II' match 'Gardner Minshew' across data sources."""
    tokens = str(value).lower().replace(".", "").replace("'", "").split()
    if tokens and tokens[-1] in GENERATIONAL_SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens)


def build_trajectory(profile: pd.DataFrame) -> pd.DataFrame:
    """Season-over-season college trajectory features keyed by pff_player_id."""
    seasons = pd.read_csv(COLLEGE_SEASONS, low_memory=False)
    keep_cols = ["player_id", "season"] + TRAJ_COLS + [TRAJ_DROPBACK_COL]
    keep_cols = [c for c in keep_cols if c in seasons.columns]
    seasons = seasons[keep_cols].copy()
    seasons = seasons.sort_values(["player_id", "season"])
    seasons["dropbacks_150plus"] = (
        pd.to_numeric(seasons[TRAJ_DROPBACK_COL], errors="coerce") >= 150
    ).astype(float)

    rows = []
    for player_id, grp in seasons.groupby("player_id"):
        grp = grp.sort_values("season")
        last = grp.iloc[-1]
        prior = grp.iloc[-2] if len(grp) >= 2 else None
        rec = {"pff_player_id": player_id}
        rec["college_career_dropbacks_total"] = pd.to_numeric(
            grp[TRAJ_DROPBACK_COL], errors="coerce"
        ).sum()
        rec["college_seasons_150plus_dropbacks"] = grp["dropbacks_150plus"].sum()
        for col in TRAJ_COLS:
            last_val = pd.to_numeric(last.get(col), errors="coerce")
            if prior is not None:
                prior_val = pd.to_numeric(prior.get(col), errors="coerce")
                rec[f"traj_delta_{col}"] = last_val - prior_val
            else:
                rec[f"traj_delta_{col}"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def build_ol_context(profile: pd.DataFrame) -> pd.DataFrame:
    """OL/pressure supporting-cast-quality features from the allowed_* PFF
    columns. Pressure exposure and get-hit rate are normalized to
    per-dropback rates so they're comparable across QBs with different
    workloads; the raw pressure_to_sack_rate columns are already rates.
    """
    ctx = pd.DataFrame(index=profile.index)
    for col in ALLOWED_CONTEXT_RAW:
        ctx[col] = pd.to_numeric(profile[col], errors="coerce")

    final_db = ctx["final_allowed_dropbacks"].replace(0, np.nan)
    career_db = ctx["career_allowed_dropbacks"].replace(0, np.nan)

    out = pd.DataFrame(index=profile.index)
    out["final_ol_pressure_rate"] = ctx["final_allowed_def_gen_pressures"] / final_db
    out["career_ol_pressure_rate"] = ctx["career_allowed_def_gen_pressures"] / career_db
    out["final_ol_hit_rate"] = ctx["final_allowed_hit_as_threw"] / final_db
    out["career_ol_hit_rate"] = ctx["career_allowed_hit_as_threw"] / career_db
    out["final_ol_pressure_to_sack_rate"] = ctx["final_allowed_pressure_to_sack_rate"]
    out["career_ol_pressure_to_sack_rate"] = ctx["career_allowed_pressure_to_sack_rate"]

    # High-pressure-context flag: final-season pressure rate above the
    # (non-null) median, i.e. this QB's own line let through more heat than
    # a typical draft-eligible QB's line did. Used only to build a simple
    # interaction below -- kept as a derived flag rather than a raw source
    # column per the task's "keep simple" guidance.
    median_rate = out["final_ol_pressure_rate"].median()
    out["high_pressure_context"] = np.where(
        out["final_ol_pressure_rate"].isna(), np.nan,
        (out["final_ol_pressure_rate"] > median_rate).astype("float"),
    )

    final_grades_pass = pd.to_numeric(profile["final_grades_grades_pass"], errors="coerce")
    out["grades_pass_x_high_pressure_context"] = final_grades_pass * out["high_pressure_context"]

    return out


def parse_first_college_season(value: object) -> float:
    """Parse the earliest year out of a 'School:Year;School:Year;...'
    college_seasons string. Returns NaN if unparseable/missing."""
    if pd.isna(value):
        return np.nan
    years = []
    for token in str(value).split(";"):
        token = token.strip()
        if ":" not in token:
            continue
        year_part = token.rsplit(":", 1)[-1]
        try:
            years.append(int(float(year_part)))
        except ValueError:
            continue
    return float(min(years)) if years else np.nan


def build_declare_context(profile: pd.DataFrame) -> pd.DataFrame:
    """Draft age / early-declare proxies from the college_seasons string and
    college_seasons_count. college_seasons_count is already an existing
    feature (see base['college_seasons_count'] below); these are new
    derived columns built on top of it.
    """
    out = pd.DataFrame(index=profile.index)
    seasons_count = pd.to_numeric(profile["college_seasons_count"], errors="coerce")
    out["first_college_season"] = profile["college_seasons"].map(parse_first_college_season)
    draft_season = pd.to_numeric(profile["draft_season"], errors="coerce")
    out["seasons_since_first"] = draft_season - out["first_college_season"]
    out["early_declare"] = (seasons_count <= 3).astype("float")
    return out


def build_concept_pocket(profile: pd.DataFrame) -> pd.DataFrame:
    """Straight-dropback / play-action / screen-rate / time-in-pocket
    features mined from PFF's passing-concept and time-in-pocket reports.

    - concept_*_no_screen_*: performance on straight-dropback (non-screen)
      passing plays, i.e. play design stripped out.
    - concept_*_pa_minus_npa_*: play-action minus non-play-action deltas --
      how much a QB's grade/accuracy benefits from play-action scheme help.
    - concept_final_screen_rate: share of final-season dropbacks that were
      screens -- a proxy for scheme dependence (high screen rate = more
      easy completions manufactured by the offense).
    - pocket_*_more_*: performance on dropbacks with time-to-throw >= 2.5s,
      i.e. processing/pocket-management under extended pressure.
    - pocket_*_less_accuracy_percent: quick-game (< 2.5s) accuracy.
    """
    src = pd.DataFrame(index=profile.index)
    for col in CONCEPT_POCKET_RAW:
        src[col] = pd.to_numeric(profile[col], errors="coerce")

    out = pd.DataFrame(index=profile.index)

    # Straight-dropback (no-play-action-scheme, no-screen) performance.
    out["concept_final_no_screen_grades_pass"] = src["final_concept_no_screen_grades_pass"]
    out["concept_final_no_screen_accuracy_percent"] = src["final_concept_no_screen_accuracy_percent"]
    out["concept_final_no_screen_ypa"] = src["final_concept_no_screen_ypa"]
    out["concept_career_no_screen_grades_pass"] = src["career_concept_no_screen_grades_pass"]

    # Play-action minus non-play-action deltas.
    out["concept_final_pa_minus_npa_grades_pass"] = (
        src["final_concept_pa_grades_pass"] - src["final_concept_npa_grades_pass"]
    )
    out["concept_final_pa_minus_npa_accuracy_percent"] = (
        src["final_concept_pa_accuracy_percent"] - src["final_concept_npa_accuracy_percent"]
    )
    out["concept_career_pa_minus_npa_grades_pass"] = (
        src["career_concept_pa_grades_pass"] - src["career_concept_npa_grades_pass"]
    )

    # Screen-rate: share of dropbacks that were screens (scheme dependence).
    final_concept_db = src["final_concept_dropbacks"].replace(0, np.nan)
    out["concept_final_screen_rate"] = src["final_concept_screen_dropbacks"] / final_concept_db

    # Time-in-pocket: 2.5s+ (extended pocket) vs <2.5s (quick game).
    out["pocket_final_more_grades_pass"] = src["final_pocket_more_grades_pass"]
    out["pocket_final_more_accuracy_percent"] = src["final_pocket_more_accuracy_percent"]
    out["pocket_final_less_accuracy_percent"] = src["final_pocket_less_accuracy_percent"]
    out["pocket_career_more_grades_pass"] = src["career_pocket_more_grades_pass"]
    out["pocket_career_less_accuracy_percent"] = src["career_pocket_less_accuracy_percent"]

    return out


def build_market(profile_keys: pd.DataFrame) -> pd.DataFrame:
    draft_master = pd.read_csv(DRAFT_MASTER, low_memory=False)
    # Restrict to QBs only: a bare name+season join without a position filter
    # can match a non-QB draftee with the same normalized name and assign the
    # wrong pick.
    draft_master = draft_master[draft_master["position"] == "QB"].copy()
    draft_master["join_name"] = draft_master["pfr_player_name"].map(normalize)
    draft_master["draft_season"] = pd.to_numeric(draft_master["season"], errors="coerce")
    market = draft_master[["join_name", "draft_season", "pick", "round"]].copy()
    market["pick"] = pd.to_numeric(market["pick"], errors="coerce")
    market["round"] = pd.to_numeric(market["round"], errors="coerce")
    market = market.dropna(subset=["pick"]).drop_duplicates(["join_name", "draft_season"])
    return market


def dedupe_profile(profile: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep, per (canonical_name, draft_season), the row with most non-null PFF fields."""
    pff_cols = [c for c in profile.columns if c.startswith("final_") or c.startswith("career_")]
    profile = profile.copy()
    profile["_nonnull_pff"] = profile[pff_cols].notna().sum(axis=1)
    before = len(profile)
    profile = profile.sort_values("_nonnull_pff", ascending=False)
    deduped = profile.drop_duplicates(["canonical_name", "draft_season"], keep="first")
    deduped = deduped.drop(columns="_nonnull_pff").sort_index()
    dropped = before - len(deduped)
    return deduped, dropped


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    profile = pd.read_csv(PROFILES, low_memory=False)
    print(f"Loaded profiles: {profile.shape}")

    dup_count = profile.duplicated(["canonical_name", "draft_season"]).sum()
    print(f"Duplicate (canonical_name, draft_season) rows before dedupe: {dup_count}")
    profile, dropped = dedupe_profile(profile)
    print(f"Dropped {dropped} duplicate rows; {len(profile)} remain.")
    assert not profile.duplicated(["canonical_name", "draft_season"]).any(), "duplicates remain"

    profile["join_name"] = profile["canonical_name"].map(normalize)

    all_feature_cols = (
        FINAL_CORE + CAREER_CORE + PRESSURE_SPLITS + DEPTH_SPLITS + CFBD_FEATURES + COMBINE_FEATURES
    )
    missing_in_profile = [c for c in all_feature_cols if c not in profile.columns]
    if missing_in_profile:
        raise RuntimeError(f"Expected columns missing from profiles: {missing_in_profile}")

    base = profile[PASSTHROUGH_KEYS + all_feature_cols].copy()

    # CFBD final-vs-career delta.
    base["cfbd_ppa_pass_delta"] = pd.to_numeric(
        profile["final_average_ppa_pass"], errors="coerce"
    ) - pd.to_numeric(profile["career_average_ppa_pass"], errors="coerce")

    # Age/experience proxy.
    base["college_seasons_count"] = pd.to_numeric(profile["college_seasons_count"], errors="coerce")

    # Trajectory features from per-season college table.
    traj = build_trajectory(profile)
    base = base.merge(traj, on="pff_player_id", how="left")

    # OL/pressure supporting-cast context (new). base's index was reset to a
    # fresh RangeIndex by the merge above, but row order still matches
    # `profile` 1:1 (traj has zero duplicate pff_player_id, so the left
    # join above cannot add/drop/reorder rows) -- reset_index here keeps
    # these frames aligned with base by position instead of by the old
    # profile index labels.
    ol_context = build_ol_context(profile).reset_index(drop=True)
    base = pd.concat([base, ol_context], axis=1)

    # Draft age / early-declare proxies (new).
    declare_context = build_declare_context(profile).reset_index(drop=True)
    base = pd.concat([base, declare_context], axis=1)

    # Passing-concept / time-in-pocket splits (new).
    concept_pocket = build_concept_pocket(profile).reset_index(drop=True)
    base = pd.concat([base, concept_pocket], axis=1)

    # Draft market.
    market = build_market(base)
    base = base.merge(market, on=["join_name", "draft_season"], how="left")
    base["log_pick"] = np.log(base["pick"])
    base["day"] = pd.cut(
        base["round"], bins=[0, 3, 6, 7], labels=[1, 2, 3], include_lowest=True
    ).astype("float")

    numeric_feature_cols = [
        c for c in base.columns if c not in ("canonical_name", "join_name", "colleges", "pff_player_id")
    ]
    for col in numeric_feature_cols:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    feature_list = [c for c in base.columns if c not in PASSTHROUGH_KEYS]
    print(f"Total feature columns: {len(feature_list)}")

    base.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({base.shape[0]} rows, {base.shape[1]} cols)")

    labeled_era = base[base["draft_season"] <= LABELED_ERA_MAX]
    coverage_overall = {c: round(float(base[c].notna().mean()), 4) for c in feature_list}
    coverage_labeled = {c: round(float(labeled_era[c].notna().mean()), 4) for c in feature_list}

    burrow = base[(base["canonical_name"].str.contains("Burrow", case=False, na=False)) & (base["draft_season"] == 2020)]
    burrow_check = {}
    if not burrow.empty:
        row = burrow.iloc[0]
        burrow_check = {
            "canonical_name": row["canonical_name"],
            "career_grades_ypa": row.get("career_grades_ypa"),
            "final_grades_ypa": row.get("final_grades_ypa"),
            "pick": row.get("pick"),
            "early_declare": row.get("early_declare"),
            "college_seasons_count": row.get("college_seasons_count"),
        }
        print("Burrow spot check:", burrow_check)
    else:
        print("WARNING: Joe Burrow 2020 not found in output.")

    sample_rows = base.sample(min(3, len(base)), random_state=42)[
        ["canonical_name", "draft_season", "pick", "career_grades_ypa", "final_grades_grades_pass"]
    ].to_dict(orient="records")
    print("Sample rows:", json.dumps(sample_rows, default=str, indent=2))

    metadata = {
        "row_count": int(len(base)),
        "duplicate_rows_dropped": int(dropped),
        "feature_list": feature_list,
        "feature_groups": {
            "final_core_pff": FINAL_CORE,
            "career_core_pff": CAREER_CORE,
            "pressure_splits": PRESSURE_SPLITS,
            "depth_splits": DEPTH_SPLITS,
            "cfbd": CFBD_FEATURES + ["cfbd_ppa_pass_delta"],
            "combine": COMBINE_FEATURES,
            "trajectory": [c for c in traj.columns if c != "pff_player_id"],
            "market": ["pick", "round", "log_pick", "day"],
            "experience": ["college_seasons_count"],
            "ol_pressure_context": list(ol_context.columns),
            "declare_context": list(declare_context.columns),
            "concept_pocket": list(concept_pocket.columns),
        },
        "coverage_overall": coverage_overall,
        "coverage_labeled_era_2015_2023": coverage_labeled,
        "labeled_era_row_count": int(len(labeled_era)),
        "burrow_spot_check": burrow_check,
        "notes": (
            "final_* = final college season (draft_season-1); career_* = college career "
            "aggregates. Market pick/round sourced only from nflverse draft master "
            "(pick/round/team/season columns only, no outcomes)."
        ),
    }
    OUT_META.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print(f"Wrote {OUT_META}")


if __name__ == "__main__":
    main()
