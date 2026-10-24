from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware


# backend/main.py is inside CP_new/backend.
BASE_DIR = Path(__file__).resolve().parent.parent  # CP_new
REPO_ROOT = BASE_DIR.parent  # folder containing the outer .git and .dvc

# CSV files created/used by the scripts in CP_new/data.
VERSIONED_DATASET_PATH = BASE_DIR / "data" / "test_dataset.csv"
SOURCE_DATASET_PATH = BASE_DIR / "data" / "data.csv"
DVC_TARGETS = [
    str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    for path in (SOURCE_DATASET_PATH, VERSIONED_DATASET_PATH)
]
GITHUB_REMOTE_URL = (
    "https://github.com/Shrawani-Darde-24/Mlops-new-airsense.git"
)

# Keep event history in the repo, but keep the bulky comparison baseline
# inside .git so it is never accidentally committed.
TRACKER_DIR = REPO_ROOT / ".git" / "dataset-versioning" / "test_dataset_tracker"
BASELINE_PATH = TRACKER_DIR / "previous_dataset.csv"
CHANGE_HISTORY_PATH = BASE_DIR / "dataset-versioning" / "test_dataset_change_history.json"

KEY_COLUMNS = ["station_id", "timestamp"]


app = FastAPI(
    title="AirSense AI Dataset Version API",
    description="Dataset statistics and change history for test_dataset.csv.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    """Read a CSV, treating a zero-byte file as an empty dataset."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def ensure_tracker() -> None:
    """Create the change-history files and initial baseline if needed."""
    if not VERSIONED_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {VERSIONED_DATASET_PATH}"
        )

    TRACKER_DIR.mkdir(parents=True, exist_ok=True)
    CHANGE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CHANGE_HISTORY_PATH.exists():
        CHANGE_HISTORY_PATH.write_text("[]", encoding="utf-8")

    if not BASELINE_PATH.exists():
        shutil.copy2(VERSIONED_DATASET_PATH, BASELINE_PATH)


def file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def rows_by_key(data: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    """Index dataset rows by station ID and timestamp."""
    if data.empty:
        return {}

    missing = [column for column in KEY_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            "Dataset is missing key columns: " + ", ".join(missing)
        )

    result = {}
    for _, row in data.iterrows():
        key = (str(row["station_id"]), str(row["timestamp"]))
        result[key] = row.to_dict()

    return result


def values_differ(left: Any, right: Any) -> bool:
    """Compare CSV cell values while treating two empty values as equal."""
    left_missing = pd.isna(left)
    right_missing = pd.isna(right)

    if left_missing and right_missing:
        return False
    if left_missing != right_missing:
        return True

    return str(left) != str(right)


def check_dataset_for_changes() -> dict[str, Any]:
    """Compare test_dataset.csv with its previous saved baseline."""
    ensure_tracker()

    old_data = read_csv_or_empty(BASELINE_PATH)
    new_data = read_csv_or_empty(VERSIONED_DATASET_PATH)

    old_columns = set(old_data.columns)
    new_columns = set(new_data.columns)

    added_columns = sorted(new_columns - old_columns)
    removed_columns = sorted(old_columns - new_columns)

    old_rows = rows_by_key(old_data)
    new_rows = rows_by_key(new_data)

    old_keys = set(old_rows)
    new_keys = set(new_rows)

    added_keys = new_keys - old_keys
    removed_keys = old_keys - new_keys
    shared_keys = old_keys & new_keys

    common_columns = sorted(
        (old_columns & new_columns) - set(KEY_COLUMNS)
    )

    changed_keys = set()
    changed_columns = set()

    for key in shared_keys:
        for column in common_columns:
            if values_differ(
                old_rows[key].get(column),
                new_rows[key].get(column),
            ):
                changed_keys.add(key)
                changed_columns.add(column)

    has_changes = bool(
        added_keys
        or removed_keys
        or changed_keys
        or added_columns
        or removed_columns
    )

    if not has_changes:
        return {
            "changed": False,
            "message": "No dataset changes detected.",
        }

    event = {
        "detected_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "dataset": VERSIONED_DATASET_PATH.name,
        "path": str(VERSIONED_DATASET_PATH.relative_to(BASE_DIR)),
        "old_rows": int(len(old_data)),
        "new_rows": int(len(new_data)),
        "added_rows": len(added_keys),
        "removed_rows": len(removed_keys),
        "changed_records": len(changed_keys),
        "changed_columns": sorted(changed_columns),
        "added_columns": added_columns,
        "removed_columns": removed_columns,
    }

    try:
        history = json.loads(CHANGE_HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        history = []

    if not isinstance(history, list):
        history = []

    history.insert(0, event)
    CHANGE_HISTORY_PATH.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save the new baseline after recording the change.
    shutil.copy2(VERSIONED_DATASET_PATH, BASELINE_PATH)

    return {
        "changed": True,
        "message": "Dataset changes detected and recorded.",
        "event": event,
    }


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AirSense AI · Dataset Versions</title>
<style>
:root{--bg:#f5f7fb;--card:#fff;--text:#172033;--muted:#738096;--line:#e5eaf2;--teal:#0f766e}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Segoe UI,Arial,sans-serif}main{max-width:1200px;margin:auto;padding:28px}.hero{display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;margin-bottom:20px}.eyebrow{color:#0d9488;font-size:11px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase}h1{font-size:32px;margin:5px 0}.sub,.muted{color:var(--muted)}button{background:var(--teal);color:white;border:0;border-radius:10px;padding:12px 17px;font-weight:750;cursor:pointer}button:disabled{opacity:.6}.live{color:var(--muted);font-size:12px;margin-right:12px}.dot{color:#10b981}.notice{padding:12px 15px;border-radius:10px;background:#e0f2fe;color:#075985;margin-bottom:15px}.error{background:#fff1f2;color:#be123c}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}.card,.section{background:var(--card);border:1px solid var(--line);border-radius:15px;box-shadow:0 5px 20px #16233b06}.card{padding:18px}.label{color:var(--muted);font-size:12px}.value{font-size:30px;font-weight:800;margin:8px 0}.foot{color:var(--muted);font-size:11px}.section{margin-bottom:18px;overflow:hidden}.head{padding:17px 20px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}h2{font-size:16px;margin:0}.details{padding:18px 20px;display:grid;grid-template-columns:1.4fr 1fr;gap:12px}.detail{border:1px solid var(--line);border-radius:10px;padding:13px;overflow-wrap:anywhere}.small{font-size:10px;color:var(--muted);margin-bottom:5px}.pill{display:inline-block;border-radius:99px;background:#eef2f7;padding:6px 10px;font-size:11px;font-weight:700}.green{color:#047857;background:#d1fae5}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;min-width:760px;font-size:12px;text-align:left}th,td{padding:12px 15px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#f8fafc;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.6px}.empty{padding:28px;text-align:center;color:var(--muted)}.footer{display:flex;justify-content:space-between;color:var(--muted);font-size:11px}@media(max-width:760px){main{padding:15px}.stats{grid-template-columns:repeat(2,1fr)}.details{grid-template-columns:1fr}}@media(max-width:460px){.stats{grid-template-columns:1fr}}
</style></head><body><main>
<header class="hero"><div><div class="eyebrow">AirSense AI / MLOps</div><h1>Dataset Versions</h1><p class="sub">Sync DVC snapshots and review dataset changes over time.</p></div><div><span class="live"><b class="dot">●</b> Live · 5 sec</span><button id="sync">↻ Sync to GitHub + DVC</button></div></header>
<div id="notice" class="notice" hidden></div>
<div class="stats"><div class="card"><div class="label">Dataset records</div><div class="value" id="rows">—</div><div class="foot">Rows in data/test_dataset.csv</div></div><div class="card"><div class="label">Feature columns</div><div class="value" id="columns">—</div><div class="foot">Available fields</div></div><div class="card"><div class="label">Duplicate pairs</div><div class="value" id="duplicates">—</div><div class="foot">Station and timestamp matches</div></div><div class="card"><div class="label">Change events</div><div class="value" id="events">0</div><div class="foot">Recorded changes</div></div></div>
<section class="section"><div class="head"><div><h2>Dataset health</h2><span class="muted">Current source and quality checks</span></div><span id="health" class="pill">Loading</span></div><div class="details"><div class="detail"><div class="small">DATASET</div><b id="dataset">—</b><div class="small" style="margin-top:10px">PATH</div><span id="path"></span></div><div class="detail"><div class="small">TRACKING</div><b>DVC snapshots + Git commits</b><div class="small" style="margin-top:10px">REFRESH</div><b>Every 5 seconds</b></div></div></section>
<section class="section"><div class="head"><div><h2>Dataset change history</h2><span class="muted">Records added, removed, or modified in test_dataset.csv</span></div><span id="eventBadge" class="pill">0 events</span></div><div id="changes" class="empty">Loading history…</div></section>
<section class="section"><div class="head"><div><h2>DVC version history</h2><span class="muted">Git commits that changed the DVC pointer files</span></div><span id="versionBadge" class="pill">0 versions</span></div><div id="versions" class="empty">Loading versions…</div></section>
<footer class="footer"><span>AirSense AI · Dataset observability</span><span>Last checked: <span id="checked">—</span></span></footer>
</main><script>
const el=id=>document.getElementById(id), fmt=n=>Number(n??0).toLocaleString('en-IN'), esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])), dt=s=>{if(!s)return'—';const d=new Date(s);return Number.isNaN(d.getTime())?esc(s):d.toLocaleString('en-IN',{day:'2-digit',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit'})};
async function req(url,opts){const r=await fetch(url,opts),v=await r.json();if(!r.ok)throw Error(v.detail||`Request failed (${r.status})`);return v}function notice(msg,bad=false){el('notice').textContent=msg;el('notice').className=bad?'notice error':'notice';el('notice').hidden=false}
function renderChanges(a){el('events').textContent=fmt(a.length);el('eventBadge').textContent=`${fmt(a.length)} events`;if(!a.length){el('changes').className='empty';el('changes').textContent='No dataset changes recorded yet.';return}el('changes').className='tablewrap';el('changes').innerHTML=`<table><thead><tr><th>Detected at</th><th>Rows before → after</th><th>Added</th><th>Removed</th><th>Modified</th><th>Changed columns</th></tr></thead><tbody>${a.map(x=>`<tr><td>${dt(x.detected_at)}</td><td><b>${fmt(x.old_rows)} → ${fmt(x.new_rows)}</b></td><td>${fmt(x.added_rows)}</td><td>${fmt(x.removed_rows)}</td><td>${fmt(x.changed_records)}</td><td>${esc([...(x.added_columns||[]),...(x.removed_columns||[]),...(x.changed_columns||[])].join(', ')||'—')}</td></tr>`).join('')}</tbody></table>`}
function renderVersions(p){const a=p.versions||[];el('versionBadge').textContent=`${fmt(p.version_count??a.length)} versions`;if(!a.length){el('versions').className='empty';el('versions').textContent='No DVC snapshots yet. Click Sync to create one.';return}el('versions').className='tablewrap';el('versions').innerHTML=`<table><thead><tr><th>Dataset</th><th>Commit</th><th>Date</th><th>Message</th></tr></thead><tbody>${a.map(x=>`<tr><td><b>${esc(x.dataset||x.pointer_file)}</b></td><td>${esc(x.short_commit||x.commit)}</td><td>${dt(x.date)}</td><td>${esc(x.message)}</td></tr>`).join('')}</tbody></table>`}
async function refresh(){try{const [s,c,v]=await Promise.all([req('/api/v1/dataset-versions/stats'),req('/api/v1/dataset-versions/changes'),req('/api/v1/dataset-versions/history')]);el('rows').textContent=fmt(s.rows);el('columns').textContent=fmt(s.columns);el('duplicates').textContent=fmt(s.duplicate_station_timestamp_pairs);el('dataset').textContent=s.dataset||'test_dataset.csv';el('path').textContent=s.path||'data/test_dataset.csv';el('health').textContent=s.validation_status||'Unavailable';el('health').className='pill '+(String(s.validation_status).toLowerCase()==='healthy'?'green':'');renderChanges(Array.isArray(c)?c:[]);renderVersions(v||{});el('checked').textContent=new Date().toLocaleTimeString()}catch(e){notice(e.message||'Could not load dataset history.',true)}}
el('sync').onclick=async()=>{const b=el('sync');b.disabled=true;b.textContent='◌ Syncing…';try{const r=await req('/api/v1/dataset-versions/sync',{method:'POST'});notice(r.message);await refresh()}catch(e){notice(e.message||'Sync failed.',true)}finally{b.disabled=false;b.textContent='↻ Sync to GitHub + DVC'}};refresh();setInterval(refresh,5000);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
@app.get("/dataset-versions", response_class=HTMLResponse)
def dataset_versions_page():
    return HTMLResponse(DASHBOARD_HTML)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AirSense AI Dataset Version API",
    }


@app.get("/api/v1/dataset-versions/stats")
def get_dataset_version_stats():
    if not VERSIONED_DATASET_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found: {VERSIONED_DATASET_PATH}",
        )

    try:
        data = read_csv_or_empty(VERSIONED_DATASET_PATH)
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read dataset: {exc}",
        ) from exc

    rows = len(data)
    columns = len(data.columns)

    if "station_id" in data.columns:
        missing_station_ids = int(data["station_id"].isna().sum())
    else:
        missing_station_ids = rows

    if "timestamp" in data.columns:
        missing_timestamps = int(data["timestamp"].isna().sum())
    else:
        missing_timestamps = rows

    if all(column in data.columns for column in KEY_COLUMNS):
        duplicate_pairs = int(
            data.duplicated(subset=KEY_COLUMNS).sum()
        )
    else:
        duplicate_pairs = 0

    healthy = (
        rows > 0
        and all(column in data.columns for column in KEY_COLUMNS)
        and duplicate_pairs == 0
        and missing_station_ids == 0
        and missing_timestamps == 0
    )

    return {
        "dataset": VERSIONED_DATASET_PATH.name,
        "path": str(VERSIONED_DATASET_PATH.relative_to(BASE_DIR)),
        "rows": rows,
        "columns": columns,
        "duplicate_station_timestamp_pairs": duplicate_pairs,
        "missing_station_ids": missing_station_ids,
        "missing_timestamps": missing_timestamps,
        "column_names": data.columns.tolist(),
        "validation_status": "healthy" if healthy else "warning",
    }


@app.get("/api/v1/dataset-versions/history")
def get_dataset_version_history():
    """Return Git commits that changed DVC pointers for the two data files."""
    versions = []
    datasets = (("source", SOURCE_DATASET_PATH), ("test_dataset", VERSIONED_DATASET_PATH))

    for dataset_name, data_path in datasets:
        pointer_path = str(data_path.relative_to(REPO_ROOT)).replace("\\", "/") + ".dvc"
        result = subprocess.run(
            [
                "git", "log", "--date=iso-strict",
                "--format=%H%x1f%h%x1f%ad%x1f%s", "--", pointer_path,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            parts = line.split("\x1f", 3)
            if len(parts) != 4:
                continue
            commit, short_commit, date, message = parts
            versions.append({
                "dataset": dataset_name,
                "pointer_file": pointer_path,
                "commit": commit,
                "short_commit": short_commit,
                "date": date,
                "message": message,
            })

    versions.sort(key=lambda item: (item["date"], item["dataset"]), reverse=True)
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "project": "AirSense AI",
        "source": "Git history of DVC pointer files",
        "version_count": len(versions),
        "versions": versions,
    }


@app.post("/api/v1/dataset-versions/sync")
def sync_dataset():
    """DVC-add datasets, commit changes, and push to configured remotes."""
    if not SOURCE_DATASET_PATH.exists() or not VERSIONED_DATASET_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Both data/data.csv and data/test_dataset.csv must exist.",
        )

    def run_command(args: list[str]) -> str:
        try:
            result = subprocess.run(
                args,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Command not found: {args[0]}. Install it and restart the backend."
            ) from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"{' '.join(args)} failed: {detail}")
        return result.stdout.strip()

    try:
        # Set the repo URL supplied by the user if origin has not been added yet.
        origin = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if origin.returncode != 0:
            run_command(["git", "remote", "add", "origin", GITHUB_REMOTE_URL])
        elif origin.stdout.strip().rstrip("/").removesuffix(".git").lower() != (
            GITHUB_REMOTE_URL.rstrip("/").removesuffix(".git").lower()
        ):
            raise RuntimeError(
                "Git origin points to a different repository: "
                + origin.stdout.strip()
            )

        # Detect and persist the current CSV change before staging all project changes.
        change_result = check_dataset_for_changes()
        change_event = change_result.get("event")
        if change_event is None and CHANGE_HISTORY_PATH.exists():
            try:
                saved_events = json.loads(CHANGE_HISTORY_PATH.read_text(encoding="utf-8"))
                change_event = saved_events[0] if saved_events else None
            except (OSError, json.JSONDecodeError, TypeError):
                change_event = None
        added_rows = int((change_event or {}).get("added_rows", 0))

        # DVC cannot add files that are still tracked directly by Git.
        # --cached removes only the Git index entry and preserves each CSV on disk.
        for target in DVC_TARGETS:
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", target],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if tracked.returncode == 0:
                run_command(["git", "rm", "-f", "--cached", "--", target])

        run_command(["dvc", "add", *DVC_TARGETS])

        remote_result = subprocess.run(
            ["dvc", "remote", "list"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if remote_result.returncode != 0:
            raise RuntimeError(remote_result.stderr.strip() or "Could not list DVC remotes.")
        has_dvc_remote = bool(remote_result.stdout.strip())
        if has_dvc_remote:
            run_command(["dvc", "push"])

        # Per the user's request, stage all changes in the outer repository.
        run_command(["git", "add", "-A", "--", "CP_new", ".dvc/config", ".dvcignore"])
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        if staged.returncode == 1:
            run_command([
                "git", "commit", "-m",
                f"Sync dataset snapshot ({added_rows} new records)",
            ])
            commit = run_command(["git", "rev-parse", "--short", "HEAD"])
        elif staged.returncode == 0:
            commit = None
        else:
            raise RuntimeError("Could not inspect staged Git changes.")

        run_command(["git", "push", "-u", "origin", "HEAD"])
        return {
            "success": True,
            "message": (
                "Dataset committed and pushed to GitHub."
                if has_dvc_remote
                else "GitHub sync completed. DVC data is cached locally; configure a DVC remote to upload it."
            ),
            "commit": commit,
            "dvc_uploaded": has_dvc_remote,
            "dvc_targets": DVC_TARGETS,
            "rows": len(read_csv_or_empty(VERSIONED_DATASET_PATH)),
        }
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/dataset-versions/check")
def check_dataset_changes_endpoint():
    """Manually check the CSV for changes."""
    try:
        return check_dataset_for_changes()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset check failed: {exc}",
        ) from exc


@app.get("/api/v1/dataset-versions/changes")
def get_dataset_change_history():
    """Check for updates, then return the recorded change events."""
    try:
        check_dataset_for_changes()
        history = json.loads(CHANGE_HISTORY_PATH.read_text(encoding="utf-8"))
        return history if isinstance(history, list) else []
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load dataset change history: {exc}",
        ) from exc





