from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


# Keep this script, data.csv, and test_dataset.csv in the same folder.
FOLDER = Path(__file__).resolve().parent
SOURCE_CSV = FOLDER / "data.csv"
OUTPUT_CSV = FOLDER / "test_dataset.csv"

INTERVAL_SECONDS = 5 * 60
RANDOM_SEED = 42

COLUMNS = [
    "timestamp", "station_id", "station_name", "city", "state",
    "latitude", "longitude", "PM2.5", "PM10", "NO2", "SO2", "CO",
    "O3", "NH3", "data_source", "temperature_2m_c",
    "relative_humidity_2m_pct", "precipitation_mm", "pressure_msl_hpa",
    "cloud_cover_pct", "wind_speed_10m_ms", "wind_direction_10m_deg",
    "wind_gusts_10m_ms", "hour", "day_of_week", "day_of_month", "month",
    "is_weekend", "season", "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_6h",
    "pm25_lag_12h", "pm25_lag_24h", "pm25_lag_48h",
    "pm25_roll_mean_3h", "pm25_roll_mean_6h", "pm25_roll_mean_24h",
    "pm25_roll_std_24h", "target_1h", "target_6h", "target_24h",
]

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]

WEATHER = [
    "temperature_2m_c",
    "relative_humidity_2m_pct",
    "precipitation_mm",
    "pressure_msl_hpa",
    "cloud_cover_pct",
    "wind_speed_10m_ms",
    "wind_direction_10m_deg",
    "wind_gusts_10m_ms",
]


def season_name(month: int) -> str:
    if month in (3, 4, 5):
        return "PreMonsoon"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    if month in (10, 11):
        return "PostMonsoon"
    return "Winter"


def make_station_row(
    station_history: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:
    """Generate one synthetic row for a station's next hourly timestamp."""
    station_history = station_history.sort_values("_sort_time")
    last = station_history.iloc[-1]

    timestamp = pd.Timestamp(last["timestamp"]) + pd.Timedelta(hours=1)
    row = last[COLUMNS].to_dict()
    row["timestamp"] = timestamp.isoformat(sep=" ")

    # Synthetic demonstration values. Replace these with sensor/API readings
    # if you have real hourly observations.
    for column in POLLUTANTS:
        previous = float(last[column])
        noise = max(abs(previous) * 0.12, 0.1)
        row[column] = max(0.0, previous + rng.normal(0, noise))

    for column in WEATHER:
        previous = float(last[column])

        if column == "wind_direction_10m_deg":
            row[column] = float((previous + rng.normal(0, 25)) % 360)
        elif column in ("relative_humidity_2m_pct", "cloud_cover_pct"):
            row[column] = float(
                np.clip(previous + rng.normal(0, 5), 0, 100)
            )
        elif column == "precipitation_mm":
            row[column] = float(max(0, previous + rng.normal(0, 0.5)))
        else:
            noise = max(abs(previous) * 0.03, 0.1)
            row[column] = float(previous + rng.normal(0, noise))

    row["data_source"] = "synthetic_generated"
    row["hour"] = timestamp.hour
    row["day_of_week"] = timestamp.dayofweek
    row["day_of_month"] = timestamp.day
    row["month"] = timestamp.month
    row["is_weekend"] = int(timestamp.dayofweek >= 5)
    row["season"] = season_name(timestamp.month)

    past_pm25 = station_history["PM2.5"].astype(float).tolist()

    for lag in (1, 3, 6, 12, 24, 48):
        row[f"pm25_lag_{lag}h"] = (
            past_pm25[-lag] if len(past_pm25) >= lag else np.nan
        )

    pm25_values = past_pm25 + [float(row["PM2.5"])]

    for window in (3, 6, 24):
        row[f"pm25_roll_mean_{window}h"] = float(
            np.mean(pm25_values[-window:])
        )

    recent_pm25 = pm25_values[-24:]
    row["pm25_roll_std_24h"] = (
        float(np.std(recent_pm25, ddof=1))
        if len(recent_pm25) > 1
        else 0.0
    )

    # Targets are filled as matching future rows get generated.
    row["target_1h"] = np.nan
    row["target_6h"] = np.nan
    row["target_24h"] = np.nan

    return row


def fill_available_targets(data: pd.DataFrame) -> pd.DataFrame:
    """Fill target values when future rows exist in test_dataset.csv."""
    for _, indexes in data.groupby("station_id", sort=False).groups.items():
        station = data.loc[indexes]
        times = pd.to_datetime(station["timestamp"], utc=True)
        pm25_by_time = dict(zip(times, station["PM2.5"].astype(float)))

        for horizon in (1, 6, 24):
            future_times = times + pd.Timedelta(hours=horizon)
            data.loc[indexes, f"target_{horizon}h"] = [
                pm25_by_time.get(future_time, np.nan)
                for future_time in future_times
            ]

    return data


def append_batch() -> int:
    if not SOURCE_CSV.exists() or SOURCE_CSV.stat().st_size == 0:
        raise FileNotFoundError(
            f"Station history is missing or empty: {SOURCE_CSV}"
        )

    source = pd.read_csv(SOURCE_CSV)

    missing_source_columns = [
        column for column in COLUMNS
        if column not in source.columns
    ]
    if missing_source_columns:
        raise ValueError(
            "data.csv is missing required columns: "
            + ", ".join(missing_source_columns)
        )

    # An empty test_dataset.csv is okay; the first batch creates its header.
    if OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0:
        output = pd.read_csv(OUTPUT_CSV)

        missing_output_columns = [
            column for column in COLUMNS
            if column not in output.columns
        ]
        if missing_output_columns:
            raise ValueError(
                "test_dataset.csv is missing required columns: "
                + ", ".join(missing_output_columns)
            )
    else:
        output = pd.DataFrame(columns=COLUMNS)

    source["station_id"] = source["station_id"].astype(str)
    if not output.empty:
        output["station_id"] = output["station_id"].astype(str)

    # Combine source history and generated rows only in memory.
    # Only the new generated rows are written to test_dataset.csv.
    history = pd.concat(
        [source[COLUMNS], output[COLUMNS]],
        ignore_index=True,
    )
    history["station_id"] = history["station_id"].astype(str)
    history["_sort_time"] = pd.to_datetime(
        history["timestamp"],
        utc=True,
    )

    station_count = max(source["station_id"].nunique(), 1)
    batches_already_written = len(output) // station_count
    rng = np.random.default_rng(RANDOM_SEED + batches_already_written)

    new_rows = []
    for station_id in source["station_id"].drop_duplicates():
        station_history = history[
            history["station_id"] == station_id
        ].sort_values("_sort_time")

        new_rows.append(make_station_row(station_history, rng))

    additions = pd.DataFrame(new_rows).reindex(columns=COLUMNS)
    updated_output = pd.concat(
        [output[COLUMNS], additions],
        ignore_index=True,
    )
    updated_output = fill_available_targets(updated_output)

    # Only test_dataset.csv is written. data.csv is used as read-only history.
    temp_path = OUTPUT_CSV.with_name(OUTPUT_CSV.name + ".tmp")
    updated_output[COLUMNS].to_csv(temp_path, index=False)
    temp_path.replace(OUTPUT_CSV)

    return len(additions)


def main() -> None:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Could not find station history: {SOURCE_CSV}")

    print(f"Reading station history from: {SOURCE_CSV}")
    print(f"Writing only generated rows to: {OUTPUT_CSV}")
    print("Appending now, then every 5 minutes. Press Ctrl+C to stop.")

    while True:
        rows_added = append_batch()
        print(f"Appended {rows_added} rows to test_dataset.csv.")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()