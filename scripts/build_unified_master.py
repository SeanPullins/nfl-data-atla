#!/usr/bin/env python3
"""Create a one-row-per-player-draft unified master from imported NFL sources."""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
PRIVATE = RAW / "private_prospect" / "NFL_master_dataset_WITH_PFF_PRIVATE.csv"
PUBLIC = RAW / "open_prospect" / "nflverse_draft_player_master_SAFE.csv"

def value(row, options):
    for name in options:
        if row.get(name):
            return row[name].strip()
    return ""

def key(row):
    stable = value(row, ["gsis_id", "pfr_player_id", "player_pfr_id", "cfb_player_id", "player_espn_id"])
    if stable:
        return "id:" + stable
    name = value(row, ["player_name_clean", "player_name", "pfr_player_name", "player_display_name"]).lower()
    year = value(row, ["player_draft_year", "draft_year", "season"])
    return "name:" + name + "|" + year

def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def main():
    if not PRIVATE.exists() or not PUBLIC.exists():
        raise SystemExit("Missing imported source master files.")
    private_rows, public_rows = read(PRIVATE), read(PUBLIC)
    merged = {key(row): dict(row) for row in public_rows}
    matched = 0
    for row in private_rows:
        existing = merged.get(key(row), {})
        if existing: matched += 1
        # PFF/private values win only when present; preserve public-only fields.
        merged[key(row)] = {**existing, **{k: v for k, v in row.items() if v not in ("", None)}}
        merged[key(row)]["atlas_source_private_pff"] = "1"
    for row in merged.values():
        row.setdefault("atlas_source_private_pff", "0")
        row["atlas_entity_grain"] = "player_draft"
    fields = sorted({field for row in merged.values() for field in row})
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "unified_player_draft_master.csv"
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(merged.values())
    (OUT / "unified_player_draft_master_metadata.json").write_text(json.dumps({
        "grain": "one player-draft record",
        "private_pff_overlay": True,
        "public_rows": len(public_rows),
        "private_rows": len(private_rows),
        "stable_or_name_year_matches": matched,
        "output_rows": len(merged),
        "join_rule": "stable ID first; normalized-name-plus-draft-year fallback",
        "warning": "Review name-year fallback matches before model training."
    }, indent=2) + "\n")
    print(f"Wrote {target} ({len(merged)} rows)")
if __name__ == "__main__":
    main()
