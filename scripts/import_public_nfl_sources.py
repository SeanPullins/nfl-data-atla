#!/usr/bin/env python3
"""Mirror data artifacts from SeanPullins' public NFL research repositories."""
from __future__ import annotations
import json, shutil, subprocess, tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external"
REPOS = {
    "nfl-open-prospect-dataset": "https://github.com/SeanPullins/nfl-open-prospect-dataset.git",
    "NFLModel": "https://github.com/SeanPullins/NFLModel.git",
    "NFLQBs": "https://github.com/SeanPullins/NFLQBs.git",
    "DraftTool": "https://github.com/SeanPullins/DraftTool.git",
    "DraftLens": "https://github.com/SeanPullins/DraftLens.git",
    "BrownsCap": "https://github.com/SeanPullins/BrownsCap.git",
    "ClevelandBrowns-intel-Bot": "https://github.com/SeanPullins/ClevelandBrowns-intel-Bot.git",
    "browns-intel-bot": "https://github.com/SeanPullins/browns-intel-bot.git",
    "NFL-Press-Conferences": "https://github.com/SeanPullins/NFL-Press-Conferences.git",
    "MaddenProject": "https://github.com/SeanPullins/MaddenProject.git",
}
DATA_SUFFIXES = {".csv", ".json", ".parquet", ".xlsx", ".xls", ".tsv", ".sqlite", ".db", ".gz"}

def is_data(path: Path) -> bool:
    return path.suffix.lower() in DATA_SUFFIXES and ".git" not in path.parts

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"built_at_utc": datetime.now(timezone.utc).isoformat(), "repositories": []}
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        for name, url in REPOS.items():
            clone = temp / name
            row = {"repository": name, "url": url, "status": "ok", "files": []}
            try:
                subprocess.run(["git", "clone", "--depth", "1", url, str(clone)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                destination = OUT / name
                for path in clone.rglob("*"):
                    if path.is_file() and is_data(path):
                        relative = path.relative_to(clone)
                        target = destination / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, target)
                        row["files"].append({"path": str(relative), "bytes": path.stat().st_size})
            except subprocess.CalledProcessError as error:
                row["status"] = "clone_failed"
                row["detail"] = error.stderr[-1000:]
            manifest["repositories"].append(row)
    (OUT / "import_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
