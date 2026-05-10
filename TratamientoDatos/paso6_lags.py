import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
INPUT_SERIE_METEO = "serie_horaria_calendario_meteo.csv"  # sep=";"
OUT_DATASET = "dataset_modelo.csv"

DATE_FMT = "%Y-%m-%d %H:%M:%S"
FLOAT_FMT = "%.3f"

# Lags que usaremos
LAGS_Y = [1, 2, 3, 6, 12, 24, 48, 168]
LAGS_X = [1, 24, 168]  # para variables "observadas" (ocupación, etc.)

ROLL_WINDOWS = [6, 24, 168]  # horas

# =========================
# LOAD
# =========================
df = pd.read_csv(INPUT_SERIE_METEO, sep=";", encoding="utf-8-sig", low_memory=False)

df["TIMESTAMP_HORA"] = pd.to_datetime(df["TIMESTAMP_HORA"], errors="coerce")
df = df[df["TIMESTAMP_HORA"].notna()].copy()
df = df.sort_values("TIMESTAMP_HORA").reset_index(drop=True)

# Aseguramos columnas base numéricas
df["LLEGADAS_HORA"] = pd.to_numeric(df["LLEGADAS_HORA"], errors="coerce").fillna(0).astype(int)

# =========================
# 1) Missing flags + rellenos meteo (sin inventar, con trazabilidad)
# =========================
# Precipitación
if "prec_mm" in df.columns:
    df["PREC_MM_MISSING"] = df["prec_mm"].isna().astype(int)
    df["prec_mm"] = pd.to_numeric(df["prec_mm"], errors="coerce").fillna(0.0)

# Horas de sol
if "sol" in df.columns:
    df["SOL_MISSING"] = df["sol"].isna().astype(int)
    df["sol"] = pd.to_numeric(df["sol"], errors="coerce")
    # mediana por mes (si MES existe; si no, mediana global)
    if "MES" in df.columns:
        med_by_month = df.groupby("MES")["sol"].transform("median")
        df["sol"] = df["sol"].fillna(med_by_month)
    df["sol"] = df["sol"].fillna(df["sol"].median())

# Presión
for c in ["presMax", "presMin"]:
    if c in df.columns:
        df[f"{c}_MISSING"] = df[c].isna().astype(int)
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

# =========================
# 2) Features cíclicas (hora/día semana)
# =========================
# (no inventa nada, son transformaciones matemáticas)
if "HORA" in df.columns:
    df["HORA_SIN"] = np.sin(2 * np.pi * df["HORA"] / 24.0)
    df["HORA_COS"] = np.cos(2 * np.pi * df["HORA"] / 24.0)

if "DIA_SEMANA" in df.columns:
    df["DOW_SIN"] = np.sin(2 * np.pi * df["DIA_SEMANA"] / 7.0)
    df["DOW_COS"] = np.cos(2 * np.pi * df["DIA_SEMANA"] / 7.0)

# Tendencia (índice temporal)
df["T_IDX"] = np.arange(len(df), dtype=int)

# =========================
# 3) LAGS del target (LLEGADAS_HORA)
# =========================
for lag in LAGS_Y:
    df[f"LLEGADAS_LAG_{lag}H"] = df["LLEGADAS_HORA"].shift(lag)

# Rolling del target (USANDO shift(1) para no filtrar la hora actual)
for w in ROLL_WINDOWS:
    df[f"LLEGADAS_ROLL_MEAN_{w}H"] = df["LLEGADAS_HORA"].shift(1).rolling(window=w, min_periods=w).mean()
    df[f"LLEGADAS_ROLL_STD_{w}H"] = df["LLEGADAS_HORA"].shift(1).rolling(window=w, min_periods=w).std()

# =========================
# 4) Lags de variables "observadas" (evitar leakage usando solo pasado)
# =========================
base_x = [
    "PACIENTES_PRESENTES",
    "EXCESO_SOBRE_BOXES_NOMINALES",
    "RATIO_PRESENTES_BOXES_NOMINALES",
    "INGRESOS_HORA",
    "PCT_INGRESO",
    "TRIAGE_DELAY_MEDIA",
    "TRIAGE_DELAY_P95",
    "PCT_TRIAJE_ALTO_I_II",
]
# solo las que existan
base_x = [c for c in base_x if c in df.columns]

for c in base_x:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    for lag in LAGS_X:
        df[f"{c}_LAG_{lag}H"] = df[c].shift(lag)

# (Opcional) rolling de pacientes presentes, también con shift(1)
if "PACIENTES_PRESENTES" in df.columns:
    for w in [6, 24]:
        df[f"PRESENTES_ROLL_MEAN_{w}H"] = df["PACIENTES_PRESENTES"].shift(1).rolling(w, min_periods=w).mean()

# =========================
# 5) Limpiar filas iniciales con NaN por lags
# =========================
needed = [f"LLEGADAS_LAG_{lag}H" for lag in LAGS_Y] + [f"LLEGADAS_ROLL_MEAN_{w}H" for w in ROLL_WINDOWS]
df_model = df.dropna(subset=needed).copy()

# =========================
# 6) Guardar dataset final
# =========================
df_model.to_csv(
    OUT_DATASET,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    na_rep="",
    date_format=DATE_FMT,
    float_format=FLOAT_FMT
)

print("OK ->", OUT_DATASET, df_model.shape)
print("Rango:", df_model["TIMESTAMP_HORA"].min(), "->", df_model["TIMESTAMP_HORA"].max())