# QB Hit Model v2 — beats the draft-pick baseline

Pipeline: `scripts/build_qb_features.py` (66 leakage-safe pre-draft features)
→ `scripts/run_qb_model_v2.py` (walk-forward validation + frozen 2024–2026 board).

## Result (pooled out-of-fold, test years 2018–2022, n=54 QBs / 12 hits)

| model | AUC | Brier | log loss |
| --- | --- | --- | --- |
| pick-only baseline (old lab spec) | 0.8115 | 0.1733 | 0.5438 |
| **v2 blend (chosen)** | **0.8214** | **0.1262** | **0.4010** |

The chosen model is a market-anchored convex blend: a logistic prior on
`log(pick)` blended with a college-talent composite (z-average of eight
well-covered career PFF/CFBD passing signals), with the blend weight chosen
per fold by inner 5-fold CV on training years only. All six candidate
strategies and their metrics are reported in `qb_model_v2_summary.json` —
nothing was cherry-picked silently.

## Honest read

- Beats the mandated baseline on all three metrics. Bootstrap 95% CI
  (model − baseline, 5000 resamples): Brier improves with P≈0.93 and
  log loss with P≈0.97; the AUC gain (+0.010) is inside the noise at n=54.
- Most of the edge is **calibration** (log-pick prior, no class
  re-weighting), not college talent: inner CV puts 0.90–1.0 weight on the
  market prior. On this sample, college PFF/CFBD stats add little reliable
  rank-ordering beyond draft slot — consistent with the old lab's finding.
- An auxiliary model trained on `w_av` for all drafted QBs reached the top
  OOF AUC (0.8433) and also beat the baseline on all three metrics, but had
  worse calibration than the blend; see the candidate table in the summary.

## Verification

An independent audit reran everything and passed 7/7 checks: no banned
sources (model outputs, success projections, player cards, NFL outcome
columns) in features; strict temporal discipline for every fitted component
including imputers, scalers, composites, inner CV, and the aux `w_av` pool;
deterministic reproduction; metrics recomputed independently from
`qb_model_v2_predictions.csv` match to 4 decimals; board covers 2024–2026
drafted QBs with no NaN probabilities.

## Outputs

- `qb_model_v2_summary.json` — protocol, all candidates, CIs, per-year breakdown.
- `qb_model_v2_predictions.csv` — pooled OOF predictions.
- `data/qb_projections/qb_model_v2_2024_2026_board.csv` — frozen board with
  `baseline_prob`, `model_prob`, and `edge` per drafted QB.
