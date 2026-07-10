# NFL Data Atlas

One integration workspace for SeanPullins' NFL draft, prospect, QB, and
team-building research. It deliberately keeps source datasets separate at the
raw layer and builds an auditable SQLite atlas rather than forcing unrelated
grains into one giant CSV.

## Included research sources

| Source repository | Role in atlas | Primary grain | Licensing / handling |
| --- | --- | --- | --- |
| `nfl-open-prospect-dataset` | public historical draft master | player-draft | public-safe |
| `nfl-prospect-dataset-private` | PFF-enriched prospect overlay and V6 projections | player-draft / player-season | private; never publish PFF-derived fields |
| `NFLModel` | APEX feature and backtest outputs | player-draft | public, model-derived |
| `NFLQBs` | QB college-to-pro pipeline and labels | QB-draft / QB-season | mixed; PFF fields private |
| `DraftTool` | browser projection inputs and prospect board UX | player-draft | derived inputs; inspect sources before import |
| `DraftLens` | normalized gold-layer contract and public model exports | player-draft | pipeline / derived outputs |
| `BrownsCap`, `BrownsDraft.com` | team-specific decision context | team-season / prospect | optional, not player-master inputs |
| `MaddenProject` | simulation product | simulation | excluded until its raw schema is reviewed |
| `NFL-Press-Conferences`, `browns-intel-bot`, `ClevelandBrowns-intel-Bot` | NFL/Browns content collection | article / transcript | reviewed; not a structured player-data source |

`nfl-scouting` could not be read through the connected GitHub account and is
listed as an unresolved source in the catalog.

## Why this is not one flat table

There are four genuinely different data grains: player-draft, college
player-season, NFL player-season, and team-season. Flattening them would repeat
draft outcomes across seasons and create leakage risks. The build creates one
SQLite file with normalized tables and a documented identity bridge.

## Build

1. Place source exports under `sources/<source-key>/` (the source keys are in
   `catalog/sources.json`). Do not put PFF data in a public clone.
2. Run `python3 scripts/build_atlas.py`.
3. Open `data/atlas/nfl_data_atlas.sqlite` with DB Browser for SQLite, DBeaver,
   or Python.

The build inventories every CSV/CSV.GZ, stores it as a source table, captures
column-level lineage, and creates a `player_identity_bridge` from the best
available ID columns. It does **not** claim name-only matches are canonical.

## First use recommendation

Start with the public player-draft master as the backbone. Add private PFF
features only for local modeling. Keep the APEX and QB model outputs as
separate experiment tables until we decide the exact project question and
target; otherwise current model scores can leak into future validation.
