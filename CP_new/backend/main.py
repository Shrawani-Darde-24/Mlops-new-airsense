from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# backend/main.py is inside CP_new/backend.
BASE_DIR = Path(__file__).resolve().parent.parent  # CP_new

# CSV created by data/append_hourly_data.py
VERSIONED_DATASET_PATH = BASE_DIR / "data" / "test_dataset.csv"

# Optional Git/DVC version-history JSON.
VERSION_HISTORY_PATH = (
    BASE_DIR / "dataset-versioning" / "version_history.json"
)

# Keep this tracker separate from other dataset trackers.
TRACKER_DIR = BASE_DIR / "dataset-versioning" / "test_dataset_tracker"
BASELINE_PATH = TRACKER_DIR / "previous_dataset.csv"
CHANGE_HISTORY_PATH = TRACKER_DIR / "change_history.json"

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
    """Return optional Git/DVC history for the page's version-history table."""
    if not VERSION_HISTORY_PATH.exists():
        return {
            "version_count": 0,
            "versions": [],
        }

    try:
        payload = json.loads(VERSION_HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read version history: {exc}",
        ) from exc

    if isinstance(payload, list):
        return {
            "version_count": len(payload),
            "versions": payload,
        }

    if isinstance(payload, dict):
        payload.setdefault("versions", [])
        payload.setdefault("version_count", len(payload["versions"]))
        return payload

    return {
        "version_count": 0,
        "versions": [],
    }


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