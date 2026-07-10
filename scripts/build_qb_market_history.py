#!/usr/bin/env python3
"""Build a deep-history QB draft-market table used ONLY to fit a
draft-pick -> hit-probability prior.

This table is intentionally separate from qb_model_features.csv: its NFL
outcome columns (w_av, dr_av, games, seasons_started, probowls, allpro,
proxy_hit) must never be joined onto test/labeled players as features. It
exists solely to let the market-prior component of the model estimate
P(hit | pick) from ~40 years of draft history instead of the ~86-player
2015-2022 labeled sample.

Source: data/raw/open_prospect/nflverse_draft_player_master_SAFE.csv,
filtered to position == 'QB', draft seasons 1980-2026.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DRAFT_MASTER = ROOT / "data/raw/open_prospect/nflverse_draft_player_master_SAFE.csv"
LABELS = ROOT / "data/raw/qbs/nfl_qb_outcomes.csv"
OUT_CSV = ROOT / "data/processed/qb_market_history.csv"
OUT_META = ROOT / "data/processed/qb_market_history_metadata.json"

# proxy_hit threshold, chosen and validated below against curated labels.
PROXY_WAV_THRESHOLD = 30.0

MATURE_MAX = 2019
PROVISIONAL_MAX = 2022


def normalize(value: object) -> str:
    """Lowercase, strip periods/apostrophes, collapse whitespace."""
    return " ".join(str(value).lower().replace(".", "").replace("'", "").split())


def label_maturity(draft_season: float) -> str:
    if pd.isna(draft_season):
        return "unknown"
    if draft_season <= MATURE_MAX:
        return "mature"
    if draft_season <= PROVISIONAL_MAX:
        return "provisional"
    return "immature"


def build_market_history() -> pd.DataFrame:
    df = pd.read_csv(DRAFT_MASTER, low_memory=False)
    qb = df[df["position"] == "QB"].copy()
    print(f"Raw QB draft rows: {len(qb)}")

    qb["join_name"] = qb["pfr_player_name"].map(normalize)
    qb["draft_season"] = pd.to_numeric(qb["season"], errors="coerce")
    qb["pick"] = pd.to_numeric(qb["pick"], errors="coerce")
    qb["round"] = pd.to_numeric(qb["round"], errors="coerce")

    for col in ["w_av", "dr_av", "games", "seasons_started", "probowls", "allpro"]:
        qb[col] = pd.to_numeric(qb[col], errors="coerce")

    # --- Missing w_av / games rule -----------------------------------
    # For drafted QBs, a NaN w_av (and paired NaN games) means Pro-Football-
    # Reference / nflverse has no recorded NFL AV for that player, which in
    # this table always coincides with seasons_started == 0, probowls == 0,
    # allpro == 0 (verified: every 2015-2022 QB with NaN w_av is a
    # never/rarely-active backup such as Christian Hackenberg, Brandon
    # Doughty, Matt Corral). Known counter-check: recent stars (Patrick
    # Mahomes, Josh Allen, Lamar Jackson) all have non-null, large w_av, and
    # old-era never-played QBs (e.g. 1980s late-round QBs with
    # seasons_started == 0) are also NaN. So NaN -> 0 is safe for proxy
    # purposes only; the raw NaN is preserved in w_av/dr_av output columns
    # and a separate *_filled column drives proxy_hit.
    qb["w_av_filled"] = qb["w_av"].fillna(0.0)
    qb["games_filled"] = qb["games"].fillna(0.0)

    qb["proxy_hit"] = (qb["w_av_filled"] >= PROXY_WAV_THRESHOLD).astype(int)
    qb["label_maturity"] = qb["draft_season"].map(label_maturity)

    out_cols = [
        "join_name", "draft_season", "pick", "round",
        "w_av", "dr_av", "games", "seasons_started", "probowls", "allpro",
        "proxy_hit", "label_maturity",
    ]
    out = qb[out_cols].dropna(subset=["pick", "draft_season"]).copy()
    out = out.sort_values(["draft_season", "pick"]).reset_index(drop=True)
    return out


def validate_proxy_hit(out: pd.DataFrame) -> dict:
    """Compare proxy_hit (2015-2022) against curated final labels."""
    labels = pd.read_csv(LABELS)
    labels = labels[labels["label_status"] == "final"].copy()
    labels["join_name"] = labels["player"].map(normalize)
    labels["draft_year"] = pd.to_numeric(labels["draft_year"], errors="coerce")
    labels["hit"] = pd.to_numeric(labels["hit"], errors="coerce")

    merged = out.merge(
        labels[["join_name", "draft_year", "hit", "player"]],
        left_on=["join_name", "draft_season"],
        right_on=["join_name", "draft_year"],
        how="inner",
    )

    pred = merged["proxy_hit"]
    actual = merged["hit"]
    tp = int(((pred == 1) & (actual == 1)).sum())
    fp = int(((pred == 1) & (actual == 0)).sum())
    tn = int(((pred == 0) & (actual == 0)).sum())
    fn = int(((pred == 0) & (actual == 1)).sum())
    agreement = (tp + tn) / len(merged) if len(merged) else float("nan")

    unmatched = len(labels) - len(merged)

    curated_hit_rate = float(labels["hit"].mean())
    proxy_hit_rate_matched = float(pred.mean())

    old = out[(out["draft_season"] >= 1990) & (out["draft_season"] <= 2014)]
    proxy_hit_rate_1990_2014 = float(old["proxy_hit"].mean())

    result = {
        "proxy_hit_definition": f"w_av (NaN->0) >= {PROXY_WAV_THRESHOLD}",
        "validation_window": "draft_year 2015-2022, label_status == 'final'",
        "curated_labels_total": int(len(labels)),
        "matched_to_market_history": int(len(merged)),
        "unmatched_labels": int(unmatched),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "agreement_rate": round(agreement, 4),
        "curated_hit_rate_2015_2022": round(curated_hit_rate, 4),
        "proxy_hit_rate_2015_2022_matched": round(proxy_hit_rate_matched, 4),
        "proxy_hit_rate_1990_2014": round(proxy_hit_rate_1990_2014, 4),
        "notes": (
            "False positives are almost all curated success_tier==2 "
            "(real-career but not a true 'hit') players (e.g. Jameis "
            "Winston, Marcus Mariota, Mitchell Trubisky, Daniel Jones, "
            "Justin Fields, Mac Jones) -- proxy_hit is deliberately a "
            "coarser 'had a real starting career' signal for a market "
            "prior, not a replacement for the curated label. Zero false "
            "negatives: proxy never misses a curated hit. One curated "
            "label (Gardner Minshew, 2019) fails to join because the "
            "draft master lists him as 'Gardner Minshew II'."
        ),
    }
    return result


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out = build_market_history()

    print(f"Total QB rows written: {len(out)}")
    for season_lo, season_hi, label in [
        (1980, 2026, "1980-2026 (all)"),
        (1990, 2014, "1990-2014"),
        (2015, 2022, "2015-2022"),
    ]:
        n = len(out[(out["draft_season"] >= season_lo) & (out["draft_season"] <= season_hi)])
        print(f"  {label}: {n} QBs")

    print(out["label_maturity"].value_counts())

    validation = validate_proxy_hit(out)
    print("Proxy-hit validation:", json.dumps(validation, indent=2))

    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({out.shape[0]} rows, {out.shape[1]} cols)")

    metadata = {
        "row_count": int(len(out)),
        "columns": list(out.columns),
        "draft_season_range": [int(out["draft_season"].min()), int(out["draft_season"].max())],
        "proxy_hit": validation,
        "label_maturity_rule": (
            "'mature' for draft_season <= 2019 (careers fully played out), "
            "'provisional' for 2020-2022 (still resolving), "
            "'immature' for 2023+ (kept in the CSV but not intended for "
            "prior-fitting without modeler review)."
        ),
        "missing_w_av_rule": (
            "NaN w_av/games in the source table corresponds to QBs with no "
            "recorded NFL production (seasons_started == 0, probowls == 0, "
            "allpro == 0 in every observed case). NaN is filled with 0 only "
            "for the derived w_av_filled/proxy_hit computation; the raw "
            "w_av/dr_av/games columns in the output CSV retain NaN so "
            "downstream users can distinguish 'no data' from 'zero AV' if "
            "needed."
        ),
        "usage_warning": (
            "This table is for fitting a draft-pick -> hit-rate prior only. "
            "Its outcome columns (w_av, dr_av, games, seasons_started, "
            "probowls, allpro, proxy_hit) must never be merged onto "
            "qb_model_features.csv rows as features -- that would leak NFL "
            "outcomes into pre-draft features."
        ),
    }
    OUT_META.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print(f"Wrote {OUT_META}")


if __name__ == "__main__":
    main()
