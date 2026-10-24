"""
Generate a synthetic hourly air-quality + weather dataset for every station
listed in an existing CSV.

Example (Windows PowerShell):
    pip install pandas numpy
    python generate_air_quality_data.py --stations-csv test_dataset.csv --start 2024-01-01 --end 2025-12-31 --output air_quality_weather_generated.csv

The end date is inclusive. Rows are generated for every hour from 00:00 on
--start through 23:00 on --end, for every unique station in the input CSV.

IMPORTANT:
This creates SYNTHETIC data for development/testing, not real observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "timestamp", "station_id", "station_name", "city", "state",
    "latitude", "longitude",
    "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3", "data_source",
    "temperature_2m_c", "relative_humidity_2m_pct", "precipitation_mm",
    "pressure_msl_hpa", "cloud_cover_pct", "wind_speed_10m_ms",
    "wind_direction_10m_deg", "wind_gusts_10m_ms",
    "hour", "day_of_week", "day_of_month", "month", "is_weekend", "season",
    "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_6h", "pm25_lag_12h",
    "pm25_lag_24h", "pm25_lag_48h",
    "pm25_roll_mean_3h", "pm25_roll_mean_6h", "pm25_roll_mean_24h",
    "pm25_roll_std_24h",
    "target_1h", "target_6h", "target_24h",
]


def get_season(month: int) -> str:
    # Broad Indian seasonal categories; adjust to your project definition if needed.
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "PreMonsoon"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "PostMonsoon"


def load_station_registry(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = [
        "station_id", "station_name", "city", "state", "latitude", "longitude"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing station metadata columns: {missing}"
        )

    stations = df[required].drop_duplicates("station_id").copy()
    stations = stations.dropna(subset=["station_id", "latitude", "longitude"])
    if stations.empty:
        raise ValueError("No valid stations found in the input CSV.")
    return stations.reset_index(drop=True)


def generate_station_data(
    station: pd.Series,
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
    station_number: int,
) -> pd.DataFrame:
    n = len(timestamps)
    hour = timestamps.hour.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()
    month = timestamps.month.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()
    is_weekend = (weekday >= 5).astype(int)

    # Station-specific differences make stations distinct and repeatable.
    station_baseline = 24 + (station_number % 9) * 3.0
    seasonal_temp = 5.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    daily_temp = 4.0 * np.sin(2 * np.pi * (hour - 8) / 24)
    temperature = 27 + seasonal_temp + daily_temp + rng.normal(0, 1.8, n)

    humidity = 66 - 0.75 * (temperature - 27) + rng.normal(0, 8, n)
    humidity = np.clip(humidity, 20, 100)

    monsoon = np.isin(month, [6, 7, 8, 9])
    rain_probability = np.where(monsoon, 0.24, 0.055)
    is_raining = rng.random(n) < rain_probability
    precipitation = np.where(
        is_raining, rng.gamma(shape=1.5, scale=np.where(monsoon, 2.2, 0.8)),
        0.0
    )
    pressure = 1012 - 0.7 * (humidity - 60) / 10 + rng.normal(0, 2.2, n)
    cloud = np.clip(
        20 + humidity * 0.65 + precipitation * 4 + rng.normal(0, 12, n),
        0, 100
    )
    wind_speed = np.clip(
        rng.gamma(shape=2.0, scale=1.3, size=n) + np.where(monsoon, 1.0, 0),
        0, 18
    )
    wind_direction = rng.uniform(0, 360, n)
    wind_gusts = np.clip(wind_speed + rng.gamma(1.5, 1.0, n), 0, 28)

    # Hourly emissions pattern: traffic peaks in morning/evening, lower overnight.
    traffic = (
        0.8
        + 0.75 * np.exp(-((hour - 8) / 2.4) ** 2)
        + 0.9 * np.exp(-((hour - 19) / 3.0) ** 2)
    )
    weekday_effect = np.where(is_weekend == 1, 0.82, 1.0)
    ventilation = np.clip(1.15 - wind_speed * 0.045, 0.35, 1.15)
    rain_washout = np.exp(-0.09 * precipitation)

    # Generate PM2.5 as a correlated process rather than independent random rows.
    pm25 = np.zeros(n, dtype=float)
    previous = station_baseline
    for i in range(n):
        seasonal_pollution = 1.0 + (0.20 if month[i] in (11, 12, 1) else 0.0)
        desired = (
            station_baseline * traffic[i] * weekday_effect[i]
            * ventilation[i] * rain_washout[i] * seasonal_pollution
        )
        previous = 0.78 * previous + 0.22 * desired + rng.normal(0, 4.0)
        pm25[i] = previous
    pm25 = np.clip(pm25, 2, 350)

    pm10 = np.clip(pm25 * rng.normal(1.55, 0.16, n) + rng.normal(4, 5, n), 3, 600)
    no2 = np.clip(
        8 + 0.42 * pm25 + 11 * (traffic - 1) * weekday_effect
        + rng.normal(0, 5, n), 1, 180
    )
    so2 = np.clip(1.2 + 0.07 * pm25 + rng.normal(0, 1.8, n), 0.1, 80)
    co = np.clip(0.18 + 0.009 * pm25 + 0.12 * (traffic - 1) + rng.normal(0, 0.08, n), 0.05, 8)
    o3 = np.clip(
        8 + 0.65 * np.maximum(temperature - 20, 0)
        + 8 * np.sin(2 * np.pi * (hour - 7) / 24)
        - 0.12 * no2 + rng.normal(0, 5, n), 1, 180
    )
    nh3 = np.clip(2.0 + 0.12 * pm25 + rng.normal(0, 2.0, n), 0.1, 100)

    out = pd.DataFrame({
        "timestamp": timestamps,
        "station_id": station["station_id"],
        "station_name": station["station_name"],
        "city": station["city"],
        "state": station["state"],
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "PM2.5": np.round(pm25, 2),
        "PM10": np.round(pm10, 2),
        "NO2": np.round(no2, 2),
        "SO2": np.round(so2, 2),
        "CO": np.round(co, 3),
        "O3": np.round(o3, 2),
        "NH3": np.round(nh3, 2),
        "data_source": "synthetic",
        "temperature_2m_c": np.round(temperature, 2),
        "relative_humidity_2m_pct": np.round(humidity, 2),
        "precipitation_mm": np.round(precipitation, 3),
        "pressure_msl_hpa": np.round(pressure, 2),
        "cloud_cover_pct": np.round(cloud, 2),
        "wind_speed_10m_ms": np.round(wind_speed, 2),
        "wind_direction_10m_deg": np.round(wind_direction, 1),
        "wind_gusts_10m_ms": np.round(wind_gusts, 2),
        "hour": hour,
        "day_of_week": weekday,
        "day_of_month": timestamps.day,
        "month": month,
        "is_weekend": is_weekend,
        "season": [get_season(int(m)) for m in month],
    })

    # Compute history features independently within each station.
    s = out["PM2.5"]
    for lag in (1, 3, 6, 12, 24, 48):
        out[f"pm25_lag_{lag}h"] = s.shift(lag)

    out["pm25_roll_mean_3h"] = s.rolling(3, min_periods=3).mean()
    out["pm25_roll_mean_6h"] = s.rolling(6, min_periods=6).mean()
    out["pm25_roll_mean_24h"] = s.rolling(24, min_periods=24).mean()
    out["pm25_roll_std_24h"] = s.rolling(24, min_periods=24).std()

    # Targets represent future PM2.5 at +1, +6, and +24 hours.
    out["target_1h"] = s.shift(-1)
    out["target_6h"] = s.shift(-6)
    out["target_24h"] = s.shift(-24)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic hourly air-quality/weather features for every station."
    )
    parser.add_argument("--stations-csv", default="test_dataset.csv",
                        help="CSV containing station metadata; defaults to test_dataset.csv")
    parser.add_argument("--start", default="2024-01-01",
                        help="First date to generate, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31",
                        help="Last date to generate, inclusive (YYYY-MM-DD)")
    parser.add_argument("--output", default="air_quality_weather_generated.csv",
                        help="Output CSV filename")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible data")
    parser.add_argument("--timezone", default="Asia/Kolkata",
                        help="Timezone for timestamps; default Asia/Kolkata")
    parser.add_argument("--drop-incomplete", action="store_true",
                        help="Drop rows with missing lag/rolling features or future targets")
    args = parser.parse_args()

    input_path = Path(args.stations_csv)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find {input_path.resolve()}. Put the station CSV in this folder "
            "or pass its path with --stations-csv."
        )

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if end < start:
        raise ValueError("--end must be the same as or later than --start")

    # Hourly timestamps in the requested local timezone, inclusive of the full end date.
    timestamps = pd.date_range(
        start=start.strftime("%Y-%m-%d") + " 00:00",
        end=end.strftime("%Y-%m-%d") + " 23:00",
        freq="h",
        tz=args.timezone,
    )

    stations = load_station_registry(input_path)
    rng = np.random.default_rng(args.seed)

    frames = []
    for i, (_, station) in enumerate(stations.iterrows()):
        # Use a station-specific RNG stream so station order is reproducible.
        station_rng = np.random.default_rng(args.seed + i * 1009)
        frames.append(generate_station_data(station, timestamps, station_rng, i))

    result = pd.concat(frames, ignore_index=True)
    result = result[OUTPUT_COLUMNS]

    if args.drop_incomplete:
        feature_cols = [
            c for c in OUTPUT_COLUMNS
            if c.startswith("pm25_lag_") or c.startswith("pm25_roll_")
            or c.startswith("target_")
        ]
        result = result.dropna(subset=feature_cols).reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("\nDataset generated successfully.")
    print(f"Output file : {output_path.resolve()}")
    print(f"Stations    : {stations['station_id'].nunique():,}")
    print(f"Hours/station: {len(timestamps):,}")
    print(f"Rows        : {len(result):,}")
    print(f"Columns     : {len(result.columns)}")
    print(f"Date range  : {timestamps[0]} to {timestamps[-1]}")
    print(f"Missing cells: {int(result.isna().sum().sum()):,}")
    print("\nFirst 5 rows:")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
