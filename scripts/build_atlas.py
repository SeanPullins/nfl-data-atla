#!/usr/bin/env python3
"""Build an auditable local SQLite atlas from staged NFL research exports."""
from __future__ import annotations

import csv
import gzip
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "sources"
OUT = ROOT / "data" / "atlas" / "nfl_data_atlas.sqlite"
CATALOG = ROOT / "catalog" / "sources.json"

def clean_identifier(value: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in value).strip("_") or "unnamed"

def open_csv(path: Path):
    return gzip.open(path, "rt", encoding="utf-8-sig", newline="") if path.suffix == ".gz" else path.open("r", encoding="utf-8-sig", newline="")

def build() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists(): OUT.unlink()
    db = sqlite3.connect(OUT)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
      CREATE TABLE source_registry (source_key TEXT, repository TEXT, visibility TEXT, priority TEXT, imported_at_utc TEXT);
      CREATE TABLE table_registry (source_key TEXT, source_file TEXT, table_name TEXT, row_count INTEGER, imported_at_utc TEXT);
      CREATE TABLE column_registry (table_name TEXT, ordinal INTEGER, source_column TEXT, atlas_column TEXT);
      CREATE TABLE player_identity_bridge (source_table TEXT, source_row INTEGER, gsis_id TEXT, pfr_player_id TEXT, cfb_player_id TEXT, espn_id TEXT, pff_id TEXT, player_name TEXT, draft_year TEXT);
    """)
    catalog = json.loads(CATALOG.read_text())
    now = datetime.now(timezone.utc).isoformat()
    for spec in catalog["source_repositories"]:
        db.execute("INSERT INTO source_registry VALUES (?,?,?,?,?)", (spec["key"], spec["repo"], spec["visibility"], spec["priority"], now))
    for source_dir in sorted(p for p in SOURCES.glob("*") if p.is_dir()):
        for file in sorted(list(source_dir.rglob("*.csv")) + list(source_dir.rglob("*.csv.gz"))):
            table = "raw__" + clean_identifier(source_dir.name) + "__" + clean_identifier(file.name.replace(".csv.gz", "").replace(".csv", ""))
            with open_csv(file) as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames: continue
                columns = [clean_identifier(c) for c in reader.fieldnames]
                db.execute(f'CREATE TABLE "{table}" (_source_row INTEGER, ' + ", ".join(f'"{c}" TEXT' for c in columns) + ")")
                db.executemany("INSERT INTO column_registry VALUES (?,?,?,?)", [(table, i, raw, clean) for i, (raw, clean) in enumerate(zip(reader.fieldnames, columns), 1)])
                count = 0
                for count, row in enumerate(reader, 1):
                    values = [row.get(raw, "") for raw in reader.fieldnames]
                    db.execute(f'INSERT INTO "{table}" VALUES (' + ",".join("?" for _ in range(len(values)+1)) + ")", [count, *values])
                    lookup = {clean_identifier(k): v for k, v in row.items()}
                    db.execute("INSERT INTO player_identity_bridge VALUES (?,?,?,?,?,?,?,?,?)", (table, count, lookup.get("gsis_id", ""), lookup.get("pfr_player_id", "") or lookup.get("pfr_id", ""), lookup.get("cfb_player_id", ""), lookup.get("player_espn_id", "") or lookup.get("espn_id", ""), lookup.get("pff_id", ""), lookup.get("player_display_name", "") or lookup.get("player_name", "") or lookup.get("pfr_player_name", ""), lookup.get("player_draft_year", "") or lookup.get("draft_year", "") or lookup.get("season", "")))
            db.execute("INSERT INTO table_registry VALUES (?,?,?,?,?)", (source_dir.name, str(file.relative_to(SOURCES)), table, count, now))
    db.commit(); db.close()
    print(f"Built {OUT}")

if __name__ == "__main__": build()
