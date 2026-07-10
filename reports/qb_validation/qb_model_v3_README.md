# QB Hit Model v3 — deep market prior + talent residual

Pipeline: `scripts/build_qb_market_history.py` (607 drafted QBs 1980–2026,
validated w_av≥30 hit proxy) + `scripts/build_qb_features.py` (82 pre-draft
features) → `scripts/run_qb_model_v3.py`.

## Result (pooled out-of-fold, test years 2018–2022, n=54 / 12 hits, identical folds)

| model | AUC | Brier | log loss |
| --- | --- | --- | --- |
| pick-only baseline | 0.8115 | 0.1733 | 0.5438 |
| v2 blend | 0.8214 | 0.1262 | 0.4010 |
| **v3 (chosen)** | **0.8492** | **0.1242** | **0.3921** |

v3 = deep market prior (log-pick logistic fit per fold on ~500 historical
QBs drafted ≥4 years earlier, Platt-recalibrated from the hot w_av proxy to
the curated hit level on training-fold labels) blended at a fixed a-priori
0.85 anchor with an aux-talent residual (Ridge on log1p(w_av) over the
college PFF/CFBD composite). Inner-CV weights independently corroborate the
anchor (fold mean 0.881). All 8 candidates and the selection rule are in
`qb_model_v3_summary.json`.

- vs baseline: P(better) = 0.97 (AUC), 0.94 (Brier), 0.98 (log loss).
- vs v2: point estimates better on all three; AUC P(better)=0.96, the
  calibration edges are small and their CIs straddle zero at n=54.
- 2023 provisional class (14 QBs, never trained on): Brier 0.0705 vs
  baseline 0.1725, log loss 0.2471 vs 0.5197.
- Component attribution: the deep prior alone already beats v2 on all three
  metrics — the stable pick-curve shape is the main lever; the talent arm
  adds rank separation (AUC 0.833→0.849) at ~15% weight. The new
  OL-context/early-declare features did NOT earn weight and were excluded
  (kept in the feature table for future use).

## Verification

Independent audit: temporal discipline traced per fold for every fitted
object (deep prior, Platt shift, aux Ridge, composites, inner CV) — pass;
anchor weight confirmed hard-coded a priori, not tuned on OOF — pass;
2023 labels never in any fit — pass; deterministic double rerun — pass;
metrics independently recomputed from predictions CSV to 4 decimals — pass.
The audit found one real bug: the frozen board's aux pool was not
year-filtered, letting 2024–2025 rookies' partial-season w_av into the
board's talent fit (~1e-4 probability impact, OOF metrics unaffected).
Fixed at `run_qb_model_v3.py` (`build_board` now filters
`draft_season < min(BOARD_YEARS)`) and the board was regenerated clean.

## Outputs

- `qb_model_v3_summary.json` — protocol, all candidates, CIs, per-year
  breakdown, 2023 holdout, disagreement sanity lists.
- `qb_model_v3_predictions.csv` — pooled OOF predictions, all candidates.
- `data/qb_projections/qb_model_v3_2024_2026_board.csv` — frozen board;
  note v3 probs sit below the (over-warm) balanced baseline across the
  board — the edge column is mostly recalibration plus within-tier talent
  separation (e.g. Caleb Williams 0.72 vs J.J. McCarthy 0.36 despite both
  being round-1 picks).
