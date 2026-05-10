import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================
# CONFIG
# =========================
INPUT_XLSX = r"Hospital/DatosOriginalesHospital.xlsx"
OUT_CSV = "visitas_urgencias_limpiaPaso1.csv"

DATE_FMT = "%Y-%m-%d %H:%M:%S"  # evita nanosegundos
FLOAT_FMT = "%.3f"              # mismo formato de decimales

LOS_Q = 0.9995  # p99.95 (winsorización solo para ocupación)

DIAG_DIR = "diagnosticos_paso1"
os.makedirs(DIAG_DIR, exist_ok=True)

# =========================
# LOAD
# =========================
df = pd.read_excel(INPUT_XLSX, sheet_name=0)

cols = [
    "ID",
    "ACTO_CLINICO_FECHA_INICIO",
    "ACTO_CLINICO_FECHA_FIN",
    "TRIAJE_FECHA_HORA_INICIO",
    "TRIAJE_CLASIFICACION",
    "ACTO_CLINICO_TIEMPO_ESTANCIA",
    "ACTO_CLINICO_DESTINO_FINAL",
    "SEXO",
    "EDAD",
    "PROCEDENCIA",
]
df = df[[c for c in cols if c in df.columns]].copy()

# =========================
# FECHAS
# =========================
for c in ["ACTO_CLINICO_FECHA_INICIO", "ACTO_CLINICO_FECHA_FIN", "TRIAJE_FECHA_HORA_INICIO"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# =========================
# FIN USADO
# =========================
est_min = pd.to_numeric(df.get("ACTO_CLINICO_TIEMPO_ESTANCIA", np.nan), errors="coerce")

df["FECHA_FIN_USADA"] = df.get("ACTO_CLINICO_FECHA_FIN")
missing_fin = df["FECHA_FIN_USADA"].isna() & df["ACTO_CLINICO_FECHA_INICIO"].notna()
df.loc[missing_fin, "FECHA_FIN_USADA"] = (
    df.loc[missing_fin, "ACTO_CLINICO_FECHA_INICIO"]
    + pd.to_timedelta(est_min.loc[missing_fin].fillna(60), unit="m")
)

bad = (df["FECHA_FIN_USADA"] <= df["ACTO_CLINICO_FECHA_INICIO"])
df.loc[bad, "FECHA_FIN_USADA"] = df.loc[bad, "ACTO_CLINICO_FECHA_INICIO"] + pd.to_timedelta(30, unit="m")

# =========================
# VARIABLES BASE
# =========================
df["TIMESTAMP_HORA"] = df["ACTO_CLINICO_FECHA_INICIO"].dt.floor("h")
df["LOS_MIN"] = (df["FECHA_FIN_USADA"] - df["ACTO_CLINICO_FECHA_INICIO"]).dt.total_seconds() / 60

if "TRIAJE_FECHA_HORA_INICIO" in df.columns:
    df["TRIAGE_DELAY_MIN"] = (df["TRIAJE_FECHA_HORA_INICIO"] - df["ACTO_CLINICO_FECHA_INICIO"]).dt.total_seconds() / 60
    df.loc[df["TRIAGE_DELAY_MIN"] < 0, "TRIAGE_DELAY_MIN"] = np.nan

# =========================
# LIMPIEZA CATEGÓRICAS
# =========================
if "TRIAJE_CLASIFICACION" in df.columns:
    df["TRIAJE_CLASIFICACION"] = df["TRIAJE_CLASIFICACION"].astype(str).str.strip().str.upper()
    valid = {"I", "II", "III", "IV", "V"}
    df.loc[~df["TRIAJE_CLASIFICACION"].isin(valid), "TRIAJE_CLASIFICACION"] = np.nan

if "PROCEDENCIA" in df.columns:
    df["PROCEDENCIA"] = (
        df["PROCEDENCIA"].astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace({"nan": np.nan, "": np.nan})
    )

if "SEXO" in df.columns:
    df["SEXO"] = df["SEXO"].astype(str).str.strip().str.upper().replace({"NAN": np.nan, "": np.nan})
    df.loc[~df["SEXO"].isin(["H", "M"]), "SEXO"] = "OTRO"

# =========================
# EDAD
# =========================
if "EDAD" in df.columns:
    df = df.rename(columns={"EDAD": "EDAD_RANGO"})
    edad = df["EDAD_RANGO"].astype(str).str.strip().replace({"nan": np.nan, "": np.nan, "None": np.nan})

    mm = edad.str.extract(r"^\s*(\d{1,3})\s*-\s*(\d{1,3})\s*$")
    df["EDAD_MIN"] = pd.to_numeric(mm[0], errors="coerce")
    df["EDAD_MAX"] = pd.to_numeric(mm[1], errors="coerce")

    is_gt90 = edad.str.match(r"^>\s*90\s*$", na=False)
    df.loc[is_gt90, "EDAD_MIN"] = 91
    df.loc[is_gt90, "EDAD_MAX"] = np.nan

    edad_label = edad.copy()
    edad_label = np.where(mm[0].notna(), mm[0].astype(str) + "-" + mm[1].astype(str) + " años", edad_label)
    edad_label = np.where(is_gt90, ">90 años", edad_label)
    df["EDAD_RANGO"] = pd.Series(edad_label, index=df.index)

    orden = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", ">90"]
    map_ord = {k: i + 1 for i, k in enumerate(orden)}

    edad_base = df["EDAD_RANGO"].astype(str).str.replace(" años", "", regex=False).str.strip()
    edad_base = edad_base.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    df["EDAD_ORD"] = edad_base.map(map_ord)

    df["EDAD_MIN"] = df["EDAD_MIN"].astype("Int64")
    df["EDAD_MAX"] = df["EDAD_MAX"].astype("Int64")

# =========================
# OUTLIERS (serio y defendible)
# =========================
los_cap = df["LOS_MIN"].quantile(LOS_Q)
df["LOS_OUTLIER_99_95"] = df["LOS_MIN"] > los_cap
df["LOS_MIN_CAP_99_95"] = df["LOS_MIN"].clip(upper=los_cap)
df["FECHA_FIN_CAP_99_95"] = df["ACTO_CLINICO_FECHA_INICIO"] + pd.to_timedelta(df["LOS_MIN_CAP_99_95"], unit="m")

if "TRIAGE_DELAY_MIN" in df.columns:
    triage_cap = df["TRIAGE_DELAY_MIN"].quantile(LOS_Q)
    df["TRIAGE_DELAY_OUTLIER_99_95"] = df["TRIAGE_DELAY_MIN"] > triage_cap
else:
    triage_cap = np.nan

# =========================
# GRÁFICOS + REPORTE (para memoria)
# =========================
print("\nGenerando gráficos y reporte de justificación de outliers...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sns.histplot(df["LOS_MIN"].dropna(), bins=100, ax=ax1, color="skyblue", kde=False)
ax1.axvline(los_cap, color="red", linestyle="--", linewidth=2, label=f"Corte p99.95: {los_cap:.1f} min")
ax1.set_title("Distribución de Tiempos de Estancia (LOS)")
ax1.set_xlabel("Minutos de Estancia")
ax1.set_ylabel("Número de Pacientes")
ax1.legend()
ax1.set_yscale("log")

sample = df["LOS_MIN"].dropna()
if len(sample) > 50000:
    sample = sample.sample(50000, random_state=2026)
sns.boxplot(x=sample, ax=ax2, color="lightgreen")
ax2.axvline(los_cap, color="red", linestyle="--", linewidth=2, label=f"Corte p99.95: {los_cap:.1f} min")
ax2.set_title("Boxplot mostrando Outliers de Estancia (submuestra)")
ax2.set_xlabel("Minutos de Estancia")
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(DIAG_DIR, "justificacion_outliers_LOS.png"), dpi=300)
plt.close()

# --- NUEVO: histograma zoom 0–2000 ---
plt.figure(figsize=(10, 5))
x = df["LOS_MIN"].dropna()
plt.hist(x[x <= 2000], bins=200)
plt.axvline(los_cap, color="red", linestyle="--", linewidth=2, label=f"Corte p99.95: {los_cap:.1f} min")
plt.title("LOS_MIN (zoom 0–2000 min)")
plt.xlabel("Minutos de Estancia")
plt.ylabel("Frecuencia")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DIAG_DIR, "justificacion_outliers_LOS_zoom_0_2000.png"), dpi=300)
plt.close()

# --- NUEVO: top 10 outliers ---
top10 = df.nlargest(10, "LOS_MIN")[["ID", "ACTO_CLINICO_FECHA_INICIO", "FECHA_FIN_USADA", "LOS_MIN"]].copy()
top10.to_csv(os.path.join(DIAG_DIR, "top10_outliers_LOS.csv"), index=False, sep=";", encoding="utf-8-sig")

total_pacientes = len(df)
outliers_los = int(df["LOS_OUTLIER_99_95"].sum())
max_original = float(df["LOS_MIN"].max())

with open(os.path.join(DIAG_DIR, "memoria_justificacion_outliers.txt"), "w", encoding="utf-8") as f:
    f.write("========================================================\n")
    f.write("   REPORTE DE JUSTIFICACIÓN DE TRATAMIENTO DE OUTLIERS  \n")
    f.write("========================================================\n\n")

    f.write("1. METODOLOGÍA EMPLEADA:\n")
    f.write("Se emplea Winsorización (capping) en el percentil 99.95 del tiempo de estancia (LOS) "
            "y variables indicadoras (flags). No se imputan outliers por la media para evitar "
            "deformar la distribución asimétrica.\n\n")

    f.write("2. ESTADÍSTICAS LOS_MIN:\n")
    f.write(f"- Total registros: {total_pacientes}\n")
    f.write(f"- Máximo original: {max_original:.2f} min ({max_original/60:.2f} h)\n")
    f.write(f"- Umbral p99.95: {los_cap:.2f} min ({los_cap/60:.2f} h)\n")
    f.write(f"- Outliers detectados: {outliers_los} ({(outliers_los/total_pacientes)*100:.4f}% del total)\n\n")

    if "TRIAGE_DELAY_MIN" in df.columns:
        outliers_tri = int(df.get("TRIAGE_DELAY_OUTLIER_99_95", pd.Series(False)).sum())
        max_tri = float(df["TRIAGE_DELAY_MIN"].max())
        f.write("3. ESTADÍSTICAS TRIAGE_DELAY_MIN:\n")
        f.write(f"- Máximo original: {max_tri:.2f} min\n")
        f.write(f"- Umbral p99.95: {triage_cap:.2f} min\n")
        f.write(f"- Outliers detectados: {outliers_tri}\n\n")

    f.write("4. NOTA SOBRE OCUPACIÓN:\n")
    f.write("La variable LOS_MIN_CAP_99_95 se utiliza únicamente para cálculos de ocupación simultánea "
            "(pacientes activos). Los valores originales se conservan.\n\n")

    f.write("5. ARCHIVOS GENERADOS (ANEXOS):\n")
    f.write(f"- {os.path.join(DIAG_DIR, 'justificacion_outliers_LOS.png')}\n")
    f.write(f"- {os.path.join(DIAG_DIR, 'justificacion_outliers_LOS_zoom_0_2000.png')}\n")
    f.write(f"- {os.path.join(DIAG_DIR, 'top10_outliers_LOS.csv')}\n")

print(f"-> Gráficos y reporte guardados en: {DIAG_DIR}")

# =========================
# DEDUP + EXPORT
# =========================
if "ID" in df.columns:
    df = df.drop_duplicates(subset=["ID"])

df.to_csv(
    OUT_CSV,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    na_rep="",
    date_format=DATE_FMT,
    float_format=FLOAT_FMT
)

print("\nPROCESO COMPLETADO:")
print("OK ->", OUT_CSV, df.shape)
print("LOS cap (p99.95) =", round(float(los_cap), 2), "min")