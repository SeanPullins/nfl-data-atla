#!/usr/bin/env python3
"""Leakage-safe QB draft translation validation lab.

All features are frozen college/PFF/CFBD/combine inputs available before the
player's draft. NFL outcome labels are used only as targets. Each evaluation
year is predicted by models trained exclusively on earlier draft classes.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "data/external/NFLQBs/data/processed/qb_draft_profiles.csv"
LABELS = ROOT / "data/raw/qbs/nfl_qb_outcomes.csv"
DRAFT_MASTER = ROOT / "data/raw/open_prospect/nflverse_draft_player_master_SAFE.csv"
REPORTS = ROOT / "reports/qb_validation"
PROJECTIONS = ROOT / "data/qb_projections"

# Explicit pre-draft feature allowlist. Do not add NFL outcome or post-draft fields.
TALENT_FEATURES = [
    "final_grades_accuracy_percent", "final_grades_big_time_throws",
    "final_grades_btt_rate", "final_grades_grades_pass",
    "final_grades_pressure_to_sack_rate", "final_grades_twp_rate",
    "final_grades_ypa", "career_grades_btt_rate", "career_grades_pass",
    "career_grades_pressure_to_sack_rate", "career_grades_twp_rate",
    "career_grades_ypa", "final_average_ppa_pass", "career_average_ppa_pass",
    "combine_height", "combine_weight", "combine_forty", "combine_vertical",
    "combine_broad", "combine_three_cone", "combine_short_shuttle",
    "combine_ras_score",
]
MARKET_FEATURE = "pick"

def normalize(value: object) -> str:
    return " ".join(str(value).lower().replace(".", "").replace("'", "").split())

def pipeline(columns: list[str]) -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer([("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), columns)], remainder="drop")),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)),
    ])

def metrics(frame: pd.DataFrame, probability: str) -> dict:
    y, p = frame["hit"].astype(int), frame[probability].clip(0.001, 0.999)
    result = {"n": int(len(frame)), "positive_rate": round(float(y.mean()), 4),
              "brier": round(float(brier_score_loss(y, p)), 4),
              "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 4)}
    result["auc"] = round(float(roc_auc_score(y, p)), 4) if y.nunique() == 2 else None
    return result

def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PROJECTIONS.mkdir(parents=True, exist_ok=True)
    profile = pd.read_csv(PROFILES, low_memory=False)
    labels = pd.read_csv(LABELS)
    profile["join_name"] = profile["canonical_name"].map(normalize)
    draft_master = pd.read_csv(DRAFT_MASTER, low_memory=False)
    draft_master["join_name"] = draft_master["pfr_player_name"].map(normalize)
    draft_master["draft_season"] = pd.to_numeric(draft_master["season"], errors="coerce")
    draft_market = draft_master[["join_name", "draft_season", "pick"]].dropna(subset=["pick"]).drop_duplicates(["join_name", "draft_season"])
    labels["join_name"] = labels["player"].map(normalize)
    labels["draft_year"] = pd.to_numeric(labels["draft_year"], errors="coerce")
    labels["hit"] = pd.to_numeric(labels["hit"], errors="coerce")
    # Only mature, finalized labels can participate in validation.
    labels = labels[(labels["label_status"] == "final") & (labels["draft_year"] <= 2023)].copy()
    data = profile.merge(labels[["join_name", "draft_year", "pick", "round", "hit"]],
                         left_on=["join_name", "draft_season"],
                         right_on=["join_name", "draft_year"], how="inner",
                         suffixes=("", "_label"))
    data["pick"] = pd.to_numeric(data["pick"], errors="coerce")
    features = [x for x in TALENT_FEATURES if x in data.columns]
    data = data.dropna(subset=["draft_season", "pick", "hit"]).copy()
    data["draft_season"] = data["draft_season"].astype(int)
    records = []
    min_train_year, max_eval_year = 2015, 2023
    for year in range(min_train_year + 3, max_eval_year + 1):
        train = data[data["draft_season"] < year].copy()
        test = data[data["draft_season"] == year].copy()
        if len(test) < 2 or train["hit"].nunique() < 2:
            continue
        baseline = pipeline([MARKET_FEATURE]).fit(train, train["hit"])
        talent = pipeline(features).fit(train, train["hit"])
        market = pipeline(features + [MARKET_FEATURE]).fit(train, train["hit"])
        test["baseline_prob"] = baseline.predict_proba(test)[:, 1]
        test["talent_prob"] = talent.predict_proba(test)[:, 1]
        test["talent_market_prob"] = market.predict_proba(test)[:, 1]
        records.append(test)
    if not records:
        raise RuntimeError("No eligible walk-forward QB folds were created.")
    oof = pd.concat(records, ignore_index=True)
    oof.to_csv(REPORTS / "qb_walk_forward_predictions.csv", index=False)
    summary = {
        "design": {
            "target": "curated mature-QB hit label",
            "training": "strictly earlier draft years for every test year",
            "held_out_classes": "2024-2026 never used in fitting or feature discovery",
            "feature_policy": "explicit college PFF/CFBD/combine allowlist; no NFL outcomes as features",
            "match_rule": "canonical player name plus draft year",
        },
        "features_used": features,
        "walk_forward": {
            "years": sorted(oof["draft_season"].unique().tolist()),
            "baseline_draft_pick_only": metrics(oof, "baseline_prob"),
            "talent_only": metrics(oof, "talent_prob"),
            "talent_plus_market": metrics(oof, "talent_market_prob"),
        },
        "coverage": {
            "labeled_qbs": int(len(data)),
            "out_of_fold_qbs": int(len(oof)),
            "unmatched_profile_or_label_rows": int(len(labels) - data[["join_name", "draft_year"]].drop_duplicates().shape[0]),
        },
    }
    (REPORTS / "qb_walk_forward_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Freeze a final model on mature classes only; score 2024-2026 separately.
    frozen_train = data[data["draft_season"] <= 2023].copy()
    future = profile[profile["draft_season"].isin([2024, 2025, 2026])].copy()
    future = future.merge(draft_market, on=["join_name", "draft_season"], how="left")
    future["pick"] = pd.to_numeric(future["pick"], errors="coerce")
    eligible = future.dropna(subset=["pick"]).copy()
    if not eligible.empty:
        talent = pipeline(features).fit(frozen_train, frozen_train["hit"])
        market = pipeline(features + [MARKET_FEATURE]).fit(frozen_train, frozen_train["hit"])
        base = pipeline([MARKET_FEATURE]).fit(frozen_train, frozen_train["hit"])
        eligible["baseline_prob"] = base.predict_proba(eligible)[:, 1]
        eligible["talent_prob"] = talent.predict_proba(eligible)[:, 1]
        eligible["talent_market_prob"] = market.predict_proba(eligible)[:, 1]
        eligible["market_edge"] = eligible["talent_prob"] - eligible["baseline_prob"]
        keep = ["canonical_name", "draft_season", "pick", "colleges", "baseline_prob", "talent_prob", "talent_market_prob", "market_edge"]
        eligible[[x for x in keep if x in eligible.columns]].sort_values(["draft_season", "talent_prob"], ascending=[True, False]).to_csv(PROJECTIONS / "qb_frozen_2024_2026_board.csv", index=False)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
