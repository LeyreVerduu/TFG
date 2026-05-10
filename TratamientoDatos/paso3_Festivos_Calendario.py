import pandas as pd
import numpy as np

INPUT_SERIE = "serie_horaria_base.csv"
INPUT_FESTIVOS = "Festivo/FestivosCuenca.csv"
OUT_SERIE_CAL = "serie_horaria_calendario.csv"

DATE_FMT = "%Y-%m-%d %H:%M:%S"
FLOAT_FMT = "%.3f"

df = pd.read_csv(INPUT_SERIE, sep=";", encoding="utf-8-sig", low_memory=False)
df["TIMESTAMP_HORA"] = pd.to_datetime(df["TIMESTAMP_HORA"], errors="coerce")
df = df[df["TIMESTAMP_HORA"].notna()].copy()

df["FECHA"] = df["TIMESTAMP_HORA"].dt.normalize()

# calendario
df["ANIO"] = df["TIMESTAMP_HORA"].dt.year
df["MES"] = df["TIMESTAMP_HORA"].dt.month
df["DIA"] = df["TIMESTAMP_HORA"].dt.day
df["HORA"] = df["TIMESTAMP_HORA"].dt.hour
df["DIA_SEMANA"] = df["TIMESTAMP_HORA"].dt.dayofweek
df["ES_FIN_SEMANA"] = (df["DIA_SEMANA"] >= 5).astype(int)
df["SEMANA_ISO"] = df["TIMESTAMP_HORA"].dt.isocalendar().week.astype(int)
df["TRIMESTRE"] = df["TIMESTAMP_HORA"].dt.quarter

# festivos
fest = pd.read_csv(INPUT_FESTIVOS, sep=",", encoding="utf-8-sig")
fest.columns = [c.strip() for c in fest.columns]

fest["Fecha"] = pd.to_datetime(fest["Fecha"], errors="coerce").dt.normalize()
fest["Motivo del festivo"] = fest["Motivo del festivo"].astype(str).str.strip().str.strip(",")
fest["Tipo de festivo"] = fest["Tipo de festivo"].astype(str).str.strip().str.strip(",")

fest_agg = (
    fest.dropna(subset=["Fecha"])
        .groupby("Fecha", as_index=False)
        .agg({
            "Motivo del festivo": lambda s: " | ".join(sorted(set(s.dropna()))),
            "Tipo de festivo": lambda s: " | ".join(sorted(set(s.dropna()))),
        })
        .rename(columns={
            "Fecha": "FECHA",
            "Motivo del festivo": "FESTIVO_MOTIVO",
            "Tipo de festivo": "FESTIVO_TIPO",
        })
)

df = df.merge(fest_agg, on="FECHA", how="left")
df["ES_FESTIVO"] = df["FESTIVO_MOTIVO"].notna().astype(int)

# flags diarios
cal = df[["FECHA"]].drop_duplicates().sort_values("FECHA").copy()
fest_set = set(fest_agg["FECHA"].dropna().tolist())

cal["ES_FESTIVO_DIA"] = cal["FECHA"].isin(fest_set).astype(int)
cal["ES_PRE_FESTIVO"] = (cal["FECHA"] + pd.Timedelta(days=1)).isin(fest_set).astype(int)
cal["ES_POST_FESTIVO"] = (cal["FECHA"] - pd.Timedelta(days=1)).isin(fest_set).astype(int)

dow = cal["FECHA"].dt.dayofweek
between_two = (
    (cal["ES_FESTIVO_DIA"] == 0) &
    ((cal["FECHA"] - pd.Timedelta(days=1)).isin(fest_set)) &
    ((cal["FECHA"] + pd.Timedelta(days=1)).isin(fest_set))
)
mon_bridge = (cal["ES_FESTIVO_DIA"] == 0) & (dow == 0) & ((cal["FECHA"] + pd.Timedelta(days=1)).isin(fest_set))
fri_bridge = (cal["ES_FESTIVO_DIA"] == 0) & (dow == 4) & ((cal["FECHA"] - pd.Timedelta(days=1)).isin(fest_set))

cal["ES_PUENTE"] = (between_two | mon_bridge | fri_bridge).astype(int)

df = df.merge(cal[["FECHA", "ES_PRE_FESTIVO", "ES_POST_FESTIVO", "ES_PUENTE"]], on="FECHA", how="left")

# flags int
for col in ["ES_FESTIVO", "ES_PRE_FESTIVO", "ES_POST_FESTIVO", "ES_PUENTE"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# texto sin NaN
if "FESTIVO_TIPO" in df.columns:
    df["FESTIVO_TIPO"] = df["FESTIVO_TIPO"].astype("string")
    df["FESTIVO_TIPO"] = np.where(df["ES_FESTIVO"] == 1, df["FESTIVO_TIPO"].fillna("Festivo"), "No festivo")
    df["FESTIVO_TIPO"] = pd.Series(df["FESTIVO_TIPO"]).astype("string")

if "FESTIVO_MOTIVO" in df.columns:
    df["FESTIVO_MOTIVO"] = df["FESTIVO_MOTIVO"].astype("string")
    df["FESTIVO_MOTIVO"] = np.where(df["ES_FESTIVO"] == 1, df["FESTIVO_MOTIVO"].fillna("Festivo"), "No festivo")
    df["FESTIVO_MOTIVO"] = pd.Series(df["FESTIVO_MOTIVO"]).astype("string")

# tipo simple
if "FESTIVO_TIPO" in df.columns:
    df["FESTIVO_TIPO_SIMPLE"] = df["FESTIVO_TIPO"].replace(
        {
            "No festivo": "No festivo",
            "Nacional": "Nacional",
            "Autonómica": "Autonómica",
            "Autonomica": "Autonómica",
            "Regional": "Regional",
            "Local": "Local",
        }
    )
    df["FESTIVO_TIPO_SIMPLE"] = df["FESTIVO_TIPO_SIMPLE"].fillna("Festivo")

# reorden
first_cols = ["TIMESTAMP_HORA", "FECHA", "LLEGADAS_HORA"]
cols_rest = [c for c in df.columns if c not in first_cols]
df = df[first_cols + cols_rest]

df.to_csv(
    OUT_SERIE_CAL,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    na_rep="",
    date_format=DATE_FMT,
    float_format=FLOAT_FMT
)

print("OK ->", OUT_SERIE_CAL, df.shape)
print("Festivos detectados (días):", int(cal["ES_FESTIVO_DIA"].sum()))
print("Puentes detectados (días):", int(cal["ES_PUENTE"].sum()))