"""
Pipeline penyiapan data dan pelatihan model untuk dashboard Kualitas Udara.

Skrip ini mereproduksi tahap Scrub pada notebook (integrasi OpenAQ + Global Air
Pollution lewat inner join kota/negara), tetapi MEMPERTAHANKAN nama negara, nama
kota, dan koordinat agar dapat dipakai pada visualisasi dashboard. Outputnya:

    data/processed/air_quality_clean.csv   -> dataset siap pakai untuk dashboard
    models/aqi_model.pkl                   -> model terbaik + daftar fitur
    models/metrics.json                    -> metrik semua model + feature importance

Jalankan sekali sebelum menjalankan aplikasi:  python prepare_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"
MODELS = BASE / "models"

# Sumber data di Google Drive (dipakai sebagai fallback unduh bila file mentah hilang)
DRIVE_IDS = {
    "pollution.csv": "1OdvRDj-Teu5Li0pjGZD68lsyXWbPxTt-",
    "openaq_global_air_quality.csv": "1gy8ZQQ46u9IKG6y_b9b2dlexdiWeWWr7",
    "global_air_quality_data_10000.csv": "1PPZcMPLciyTQ8-qf_SkiPqyolULKFQhJ",
}

# Pemetaan kode negara OpenAQ ke nama negara agar cocok dengan dataset Global Air Pollution
COUNTRY_CODE_MAP = {
    "AU": "Australia", "BR": "Brazil", "CA": "Canada", "CN": "China", "FR": "France",
    "DE": "Germany", "IN": "India", "MX": "Mexico", "RU": "Russia", "ZA": "South Africa",
    "ES": "Spain", "TH": "Thailand", "TR": "Turkey", "AE": "UAE", "GB": "UK", "US": "USA",
    "AD": "Andorra", "AR": "Argentina", "AT": "Austria", "BA": "Bosnia and Herzegovina",
    "BE": "Belgium", "BG": "Bulgaria", "CH": "Switzerland", "CL": "Chile", "CO": "Colombia",
    "CW": "Curacao", "CY": "Cyprus", "CZ": "Czech Republic", "DK": "Denmark", "EC": "Ecuador",
    "EE": "Estonia", "FI": "Finland", "GR": "Greece", "GT": "Guatemala", "HK": "Hong Kong",
    "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IL": "Israel", "IT": "Italy",
    "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "ME": "Montenegro",
    "MK": "North Macedonia", "MT": "Malta", "NL": "Netherlands", "NO": "Norway", "NP": "Nepal",
    "PE": "Peru", "PL": "Poland", "PT": "Portugal", "QA": "Qatar", "RO": "Romania",
    "RS": "Serbia", "SE": "Sweden", "SK": "Slovakia", "TW": "Taiwan", "UZ": "Uzbekistan",
    "XK": "Kosovo",
}

POLLUTANT_AQI_COLS = ["CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"]
MODEL_FEATURES = POLLUTANT_AQI_COLS + [
    "traffic_pollution_proxy", "value", "latitude", "longitude",
    "hour", "day_of_week", "month", "averaged_over_in_hours",
]
TARGET = "AQI Value"


def _ensure_raw(name: str) -> Path:
    path = RAW / name
    if not path.exists():
        import gdown
        RAW.mkdir(parents=True, exist_ok=True)
        gdown.download(id=DRIVE_IDS[name], output=str(path), quiet=False)
    return path


def build_clean_dataset() -> pd.DataFrame:
    df_global = pd.read_csv(_ensure_raw("pollution.csv"))
    df_openaq = pd.read_csv(_ensure_raw("openaq_global_air_quality.csv"))

    df_openaq["country_name"] = df_openaq["country"].map(COUNTRY_CODE_MAP)
    df_global = df_global.rename(columns={"Country": "country"})

    df_openaq["city_clean"] = df_openaq["city"].str.strip().str.lower()
    df_global["city_clean"] = df_global["City"].str.strip().str.lower()

    merged = pd.merge(
        df_global, df_openaq,
        left_on=["city_clean", "country"],
        right_on=["city_clean", "country_name"],
        how="inner", suffixes=("_global", "_openaq"),
    )

    # OpenAQ menyumbang kolom 'city' (huruf kecil) yang tidak dipakai; buang agar
    # tidak bentrok dengan 'City' dari Global Air Pollution yang akan dijadikan 'city'.
    merged = merged.drop(columns=["city"]).rename(columns={"country_global": "country", "City": "city"})

    ts = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged["hour"] = ts.dt.hour
    merged["day_of_week"] = ts.dt.dayofweek
    merged["month"] = ts.dt.month
    merged["traffic_pollution_proxy"] = merged["CO AQI Value"] + merged["NO2 AQI Value"]

    keep = [
        "country", "city", "latitude", "longitude",
        "AQI Value", "AQI Category", *POLLUTANT_AQI_COLS,
        "CO AQI Category", "Ozone AQI Category", "NO2 AQI Category", "PM2.5 AQI Category",
        "pollutant", "value", "unit", "averaged_over_in_hours",
        "hour", "day_of_week", "month", "traffic_pollution_proxy",
    ]
    clean = merged[keep].dropna(subset=["AQI Value", "AQI Category", "latitude", "longitude"])
    return clean.reset_index(drop=True)


def train_models(df: pd.DataFrame) -> dict:
    data = df[MODEL_FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
    X, y = data[MODEL_FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
    }

    results, fitted = [], {}

    def record(name, pred):
        results.append({
            "Model": name,
            "MAE": round(float(mean_absolute_error(y_test, pred)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
            "R2": round(float(r2_score(y_test, pred)), 4),
        })

    record("Baseline Mean", np.full(len(y_test), y_train.mean()))
    record("PM2.5 Only", X_test["PM2.5 AQI Value"].values)
    record("Max Pollutant Rule", X_test[POLLUTANT_AQI_COLS].max(axis=1).values)

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        record(name, model.predict(X_test))

    results.sort(key=lambda r: r["RMSE"])
    best_name = results[0]["Model"]
    best_model = fitted.get(best_name, fitted["Random Forest"])
    if best_name not in fitted:
        best_name = "Random Forest"

    importances = {}
    if hasattr(best_model, "feature_importances_"):
        importances = {
            f: round(float(i), 4)
            for f, i in sorted(zip(MODEL_FEATURES, best_model.feature_importances_),
                               key=lambda kv: kv[1], reverse=True)
        }

    joblib.dump({"model": best_model, "features": MODEL_FEATURES, "name": best_name}, MODELS / "aqi_model.pkl")

    metrics = {
        "results": results,
        "best_model": best_name,
        "feature_importance": importances,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return metrics


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    print("Membangun dataset bersih (merge OpenAQ + Global Air Pollution)...")
    clean = build_clean_dataset()
    out_csv = PROCESSED / "air_quality_clean.csv"
    clean.to_csv(out_csv, index=False)
    print(f"  -> {out_csv}  ({clean.shape[0]:,} baris x {clean.shape[1]} kolom)")
    print(f"  -> {clean['country'].nunique()} negara, {clean['city'].nunique()} kota")

    print("Melatih model regresi AQI...")
    metrics = train_models(clean)
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  -> model terbaik: {metrics['best_model']}")
    for r in metrics["results"]:
        print(f"     {r['Model']:<20} MAE={r['MAE']:<10} RMSE={r['RMSE']:<10} R2={r['R2']}")


if __name__ == "__main__":
    main()
