import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
INPUT_SERIE_CAL = "serie_horaria_calendario.csv"  # tu salida del paso 3 (sep=";")
INPUT_AEMET = "Meterologia/DatosAEMET.csv"                    # AEMET (sep=",")
OUT_SERIE_METEO = "serie_horaria_calendario_meteo.csv"

DATE_FMT = "%Y-%m-%d %H:%M:%S"
FLOAT_FMT = "%.3f"

# =========================
# 1) Cargar serie horaria
# =========================
df = pd.read_csv(INPUT_SERIE_CAL, sep=";", encoding="utf-8-sig", low_memory=False)
df["TIMESTAMP_HORA"] = pd.to_datetime(df["TIMESTAMP_HORA"], errors="coerce")
df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce").dt.normalize()

df = df[df["TIMESTAMP_HORA"].notna() & df["FECHA"].notna()].copy()

# =========================
# 2) Cargar AEMET (diario)
# =========================
aemet = pd.read_csv(INPUT_AEMET, sep=",", encoding="utf-8-sig", low_memory=False)

# Normalizar nombres de columnas
aemet.columns = [c.strip() for c in aemet.columns]

# Parse fecha
aemet["fecha"] = pd.to_datetime(aemet["fecha"], errors="coerce").dt.normalize()
aemet = aemet.dropna(subset=["fecha"]).copy()

# Si hubiese varias estaciones (por si acaso), nos quedamos con la más frecuente
if "indicativo" in aemet.columns:
    top_station = aemet["indicativo"].value_counts().idxmax()
    aemet = aemet[aemet["indicativo"] == top_station].copy()

# =========================
# 3) Limpiar / convertir variables meteorológicas
# =========================
# Precipitación: puede venir como "Ip" (inapreciable)
prec_raw = aemet["prec"].astype(str).str.strip() if "prec" in aemet.columns else pd.Series([np.nan]*len(aemet))
aemet["PREC_IP"] = prec_raw.str.upper().eq("IP").astype(int)

# Convertimos a numérico: "Ip" -> 0.0 (y mantenemos flag PREC_IP)
prec_clean = prec_raw.replace({"Ip": "0", "IP": "0", "nan": np.nan, "": np.nan})
aemet["prec_mm"] = pd.to_numeric(prec_clean, errors="coerce")

# Convertir numéricos típicos (si existen)
num_cols = [
    "tmed", "tmin", "tmax",
    "velmedia", "racha",
    "sol",
    "presMax", "presMin",
    "hrMedia", "hrMax", "hrMin",
    "altitud"
]
for c in num_cols:
    if c in aemet.columns:
        aemet[c] = pd.to_numeric(aemet[c], errors="coerce")

# Derivadas simples (no inventan nada)
if "tmax" in aemet.columns and "tmin" in aemet.columns:
    aemet["TEMP_RANGO"] = aemet["tmax"] - aemet["tmin"]
else:
    aemet["TEMP_RANGO"] = np.nan

aemet["LLUVIA_FLAG"] = (aemet["prec_mm"] > 0).astype(int)

# =========================
# 4) Selección de columnas a unir
# =========================
keep_cols = ["fecha", "prec_mm", "PREC_IP", "LLUVIA_FLAG", "TEMP_RANGO"]

# Añadimos si existen:
for c in ["tmed", "tmin", "tmax", "hrMedia", "hrMax", "hrMin", "velmedia", "racha", "sol", "presMax", "presMin"]:
    if c in aemet.columns:
        keep_cols.append(c)

aemet_join = aemet[keep_cols].copy().rename(columns={"fecha": "FECHA"})

# =========================
# 5) Merge por FECHA (diario -> horario)
# =========================
out = df.merge(aemet_join, on="FECHA", how="left")

# Flag de “meteo faltante” (no rellenamos inventando)
out["METEO_MISSING"] = out["tmed"].isna().astype(int) if "tmed" in out.columns else out["prec_mm"].isna().astype(int)

# =========================
# 6) Guardar
# =========================
out.to_csv(
    OUT_SERIE_METEO,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    na_rep="",
    date_format=DATE_FMT,
    float_format=FLOAT_FMT
)

# Diagnóstico rápido
pct_missing = float(out["METEO_MISSING"].mean() * 100)
print("OK ->", OUT_SERIE_METEO, out.shape)
print(f"METEO_MISSING: {pct_missing:.2f}% de horas sin meteo")