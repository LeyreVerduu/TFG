"""
PASO 5 — Construcción del dataset PREDICTIVO final.

Toma la salida del paso 6 (dataset_modelo.csv, que contiene TODA la información
horaria con lags y features ya calculados) y produce el dataset listo para
entrenar el modelo de predicción de llegadas a t+1h.

Decisiones tomadas en este paso (todas justificables en memoria):

1. TARGET = LLEGADAS_HORA desplazado a t+1h.
   El modelo predice cuántos pacientes llegarán en la hora siguiente.

2. Se añade LLEGADAS_HORA_T como feature: cuando se realiza la predicción para
   t+1, las llegadas de la hora actual (t) ya son conocidas y son la señal
   autoregresiva más fuerte disponible. NO incluirla supondría desperdiciar
   información operativa real.

3. Se EXCLUYEN del predictivo las variables observadas en t que no se conocerían
   en el momento real de predicción (OCUPACION_SIMULTANEA, INGRESOS_HORA,
   TRIAJE_*, RATIO_BOXES, OVERFLOW_BOXES, TRIAGE_DELAY_*, PCT_*). Sí se incluyen
   sus LAGS, que sí están disponibles.

4. Se EXCLUYEN variables constantes que no aportan poder predictivo:
   - CAP_BOXES: constante = 9 en todo el periodo.
   - ENFERMERAS_TURNO: constante = 8 en todo el periodo.
   - MEDICOS_TURNO: redundante con TURNO (función determinista de TURNO).
   Se documenta su existencia en la memoria pero no entran al modelo.

5. Se conservan variables de calendario, meteo, festivos y todos los lags/rolling.

6. Se eliminan filas iniciales con NaN en lags críticos (que el paso 6 ya
   debería haber eliminado, pero verificamos).
"""

import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
INPUT_FILE = "dataset_modelo.csv"
OUTPUT_FILE = "dataset_predictivo.csv"

DATE_FMT = "%Y-%m-%d %H:%M:%S"
FLOAT_FMT = "%.3f"

# =========================
# 1) Cargar
# =========================
df = pd.read_csv(INPUT_FILE, sep=";", encoding="utf-8-sig", low_memory=False)
df["TIMESTAMP_HORA"] = pd.to_datetime(df["TIMESTAMP_HORA"], errors="coerce")
df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
df = df.sort_values("TIMESTAMP_HORA").reset_index(drop=True)

print(f"Cargado: {df.shape}")
print(f"Rango: {df['TIMESTAMP_HORA'].min()} -> {df['TIMESTAMP_HORA'].max()}")

# =========================
# 2) Crear target a t+1h
# =========================
df["TARGET_LLEGADAS_T_PLUS_1H"] = df["LLEGADAS_HORA"].shift(-1)

# =========================
# 3) Feature autoregresiva en t
#    (las llegadas de la hora actual SÍ se conocen al predecir t+1)
# =========================
df["LLEGADAS_HORA_T"] = df["LLEGADAS_HORA"].astype(float)

# =========================
# 4) Selección de columnas para el modelo predictivo
# =========================
columnas_modelo = [
    # IDs temporales (para diagnóstico, no entran al modelo)
    "TIMESTAMP_HORA", "FECHA",

    # TARGET
    "TARGET_LLEGADAS_T_PLUS_1H",

    # Feature autoregresiva en t (NUEVA — disponible al predecir t+1)
    "LLEGADAS_HORA_T",

    # Contexto operativo NO constante
    "TURNO",  # MAÑANA / TARDE / NOCHE — los demás (CAP_BOXES, MEDICOS, ENFERMERAS)
              # son constantes o redundantes y se excluyen.

    # Calendario
    "ANIO", "MES", "DIA", "HORA", "DIA_SEMANA",
    "ES_FIN_SEMANA", "SEMANA_ISO", "TRIMESTRE",

    # Festivos
    "FESTIVO_TIPO", "FESTIVO_TIPO_SIMPLE",
    "ES_FESTIVO", "ES_PRE_FESTIVO", "ES_POST_FESTIVO", "ES_PUENTE",
    # OJO: FESTIVO_MOTIVO se excluye del modelo (texto libre, alta cardinalidad,
    # no aporta al árbol y obliga a one-hot enorme). Se mantiene en explicativo
    # para análisis descriptivo.

    # Meteo (diaria de AEMET propagada a horario — limitación documentada)
    "prec_mm", "PREC_IP", "LLUVIA_FLAG", "TEMP_RANGO",
    "tmed", "tmin", "tmax",
    "hrMedia", "hrMax", "hrMin",
    "velmedia", "racha", "sol", "presMax", "presMin",
    "METEO_MISSING", "PREC_MM_MISSING", "SOL_MISSING",
    "presMax_MISSING", "presMin_MISSING",

    # Codificación cíclica + tendencia
    "HORA_SIN", "HORA_COS", "DOW_SIN", "DOW_COS", "T_IDX",

    # Lags del target
    "LLEGADAS_LAG_1H", "LLEGADAS_LAG_2H", "LLEGADAS_LAG_3H",
    "LLEGADAS_LAG_6H", "LLEGADAS_LAG_12H", "LLEGADAS_LAG_24H",
    "LLEGADAS_LAG_48H", "LLEGADAS_LAG_168H",

    # Rolling del target (calculados con shift(1), sin leakage)
    "LLEGADAS_ROLL_MEAN_6H", "LLEGADAS_ROLL_STD_6H",
    "LLEGADAS_ROLL_MEAN_24H", "LLEGADAS_ROLL_STD_24H",
    "LLEGADAS_ROLL_MEAN_168H", "LLEGADAS_ROLL_STD_168H",

    # Lags de variables operativas observadas (solo pasado, sin leakage)
    "PACIENTES_PRESENTES_LAG_1H",
    "PACIENTES_PRESENTES_LAG_24H",
    "PACIENTES_PRESENTES_LAG_168H",
    "PRESENTES_ROLL_MEAN_6H", "PRESENTES_ROLL_MEAN_24H",
]

# Quedarnos solo con las que existan
columnas_modelo = [c for c in columnas_modelo if c in df.columns]
df_pred = df[columnas_modelo].copy()

print(f"\nColumnas seleccionadas: {len(columnas_modelo)}")

# =========================
# 5) Verificación de constantes (sanity check)
# =========================
print("\nVerificando que no quedan constantes en el modelo...")
constantes_detectadas = []
for c in df_pred.columns:
    if c in ("TIMESTAMP_HORA", "FECHA", "TARGET_LLEGADAS_T_PLUS_1H"):
        continue
    if df_pred[c].dtype in (object, "string"):
        if df_pred[c].nunique(dropna=False) <= 1:
            constantes_detectadas.append(c)
    else:
        if df_pred[c].nunique(dropna=False) <= 1:
            constantes_detectadas.append(c)

if constantes_detectadas:
    print(f"  AVISO: columnas constantes encontradas (se eliminan): {constantes_detectadas}")
    df_pred = df_pred.drop(columns=constantes_detectadas)
else:
    print("  OK: ninguna columna constante.")

# =========================
# 6) Imputación suave en meteo (por si quedaran NaN tras paso 6)
# =========================
cols_meteo_num = [
    "prec_mm", "TEMP_RANGO", "tmed", "tmin", "tmax",
    "hrMedia", "hrMax", "hrMin", "velmedia", "racha", "sol",
    "presMax", "presMin"
]
for c in cols_meteo_num:
    if c in df_pred.columns:
        df_pred[c] = pd.to_numeric(df_pred[c], errors="coerce")
        if df_pred[c].isna().any():
            df_pred[c] = df_pred[c].interpolate(limit_direction="both")

# =========================
# 7) Eliminar filas con NaN en columnas críticas
# =========================
columnas_criticas = [
    "TARGET_LLEGADAS_T_PLUS_1H",
    "LLEGADAS_HORA_T",
    "LLEGADAS_LAG_1H",
    "LLEGADAS_LAG_24H",
    "LLEGADAS_LAG_168H",
    "LLEGADAS_ROLL_MEAN_168H",
]
columnas_criticas = [c for c in columnas_criticas if c in df_pred.columns]

n_antes = len(df_pred)
df_pred = df_pred.dropna(subset=columnas_criticas).reset_index(drop=True)
print(f"\nFilas con NaN crítico eliminadas: {n_antes - len(df_pred)}")

# =========================
# 8) Comprobaciones finales
# =========================
print("\n=== COMPROBACIONES FINALES ===")
print(f"Shape final: {df_pred.shape}")
print(f"Rango: {df_pred['TIMESTAMP_HORA'].min()} -> {df_pred['TIMESTAMP_HORA'].max()}")

# Continuidad temporal
diffs = df_pred["TIMESTAMP_HORA"].diff().dropna()
gaps = (diffs != pd.Timedelta("1h")).sum()
print(f"Gaps temporales (diferentes de 1h): {gaps}")
if gaps > 0:
    print("  AVISO: hay discontinuidad horaria. Revisar.")

# NaN restantes
nulos = df_pred.isna().sum()
nulos = nulos[nulos > 0].sort_values(ascending=False)
if len(nulos) > 0:
    print(f"\nColumnas con NaN restantes (no críticos):")
    print(nulos)
else:
    print("Sin NaN en el dataset final.")

# Descriptivo del target
print(f"\nTARGET descriptivo:")
print(df_pred["TARGET_LLEGADAS_T_PLUS_1H"].describe())

# =========================
# 9) Guardar
# =========================
df_pred.to_csv(
    OUTPUT_FILE,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    na_rep="",
    date_format=DATE_FMT,
    float_format=FLOAT_FMT,
)
print(f"\nOK -> {OUTPUT_FILE}  {df_pred.shape}")
