#!/usr/bin/env python3
"""Audit whether the QB data is ready for a credible residual-edge experiment."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "data/external/NFLQBs/data/processed/qb_draft_profiles.csv"
LABELS = ROOT / "data/raw/qbs/nfl_qb_outcomes.csv"
OUT = ROOT / "reports/qb_validation/qb_edge_readiness.json"

FEATURE_BLOCKS = {
  "decision": ["final_grades_accuracy_percent", "final_grades_btt_rate", "final_grades_twp_rate", "final_grades_grades_pass", "final_grades_ypa"],
  "pressure": ["final_grades_pressure_to_sack_rate", "final_pressure_pressure_to_sack_rate", "final_pressure_pressure_btt_rate", "final_pressure_pressure_twp_rate"],
  "creation": ["final_average_ppa_rush", "career_average_ppa_rush", "final_grades_scrambles", "combine_forty"],
  "experience": ["college_seasons_count", "final_grades_dropbacks", "career_grades_dropbacks", "combine_ras_score"],
}

def norm(x):
    return " ".join(str(x).lower().replace(".", "").replace("'", "").split())

def main():
    profile = pd.read_csv(PROFILE, low_memory=False)
    labels = pd.read_csv(LABELS)
    profile["name_key"] = profile["canonical_name"].map(norm)
    labels["name_key"] = labels["player"].map(norm)
    labels["draft_year"] = pd.to_numeric(labels["draft_year"], errors="coerce")
    final = labels[labels["label_status"].eq("final")].copy()
    joined = final.merge(profile, left_on=["name_key", "draft_year"], right_on=["name_key", "draft_season"], how="left", indicator=True)
    by_year = []
    for year, frame in final.groupby("draft_year"):
        joined_year = joined[joined["draft_year"] == year]
        by_year.append({"draft_year": int(year), "labels": int(len(frame)), "matched_profiles": int((joined_year["_merge"] == "both").sum()), "hits": int(pd.to_numeric(frame["hit"], errors="coerce").fillna(0).sum())})
    blocks = {}
    for name, features in FEATURE_BLOCKS.items():
        present = [x for x in features if x in profile.columns]
        coverage = {x: round(float(pd.to_numeric(joined.loc[joined["_merge"] == "both", x], errors="coerce").notna().mean()), 3) for x in present}
        blocks[name] = {"candidate_features": features, "available_features": present, "mature_match_coverage": coverage}
    audit = {
      "decision": "Do not attempt a broad all-feature model. Run only compact residual challengers after this audit passes.",
      "mature_label_definition": "label_status=final; provisional 2023 and projection-target 2024-2026 are excluded from model selection.",
      "mature_cohort": {"labels": int(len(final)), "matched_profiles": int((joined["_merge"] == "both").sum()), "years": by_year},
      "feature_blocks": blocks,
      "promotion_gate": "A block must improve class-clustered out-of-fold Brier and log loss versus a nonlinear pick baseline, with non-negative calibration and a bootstrap confidence interval excluding zero.",
      "frozen_holdouts": [2024, 2025, 2026],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()
