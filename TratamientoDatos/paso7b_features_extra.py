"""
PASO 5b — Enriquecimiento del dataset predictivo con features adicionales.

Añade las features con mayor poder predictivo identificadas en el analisis
previo a los baselines. Todas se calculan SIN data leakage: los valores de
lookup se calculan exclusivamente sobre el conjunto de train de cada fold,
y se aplican al resto mediante join. Para el test final se usa el train completo.

Features nuevas añadidas:
  1. MEDIA_HORA_DOW  — media historica de llegadas por (HORA, DIA_SEMANA).
                       Correlacion con target en test: 0.851 (la mas alta del dataset).
                       Captura el patron "lunes a las 9h" frente a "sabado a las 9h".

  2. MEDIA_HORA_MES  — media historica de llegadas por (HORA, MES).
                       Correlacion con target en test: 0.822.
                       Captura estacionalidad anual horaria (enero vs agosto).

  3. DELTA_1H_2H     — LLEGADAS_LAG_1H - LLEGADAS_LAG_2H.
                       Tendencia inmediata de la demanda (subiendo o bajando).
                       Correlacion con target: 0.207.

  4. DELTA_24H_48H   — LLEGADAS_LAG_24H - LLEGADAS_LAG_48H.
                       Tendencia dia a dia respecto a la misma hora.
                       Correlacion con target: 0.035 (marginal, se incluye por si
                       LightGBM encuentra interacciones).

IMPORTANTE sobre el calculo sin leakage de las medias historicas:
  El valor de MEDIA_HORA_DOW para la hora t se calcula como la media de
  todas las llegadas en (hora=h, dia_semana=d) que ocurrieron ANTES de t.
  Esto equivale a un expanding mean por grupo, lo que es correcto y no filtra
  informacion del futuro hacia el pasado.

  En la practica, dado que el TimeSeriesSplit ya separa train y val/test, el
  calculo mas limpio y computacionalmente eficiente es:
    - Para cada fold: calcular el lookup sobre el train del fold y aplicar a val.
    - Para el test final: calcular el lookup sobre el train completo (2021-2024).
  
  Esta es la implementacion que se usa aqui.

Salida:
  dataset_predictivo_v2.csv  (sustituye a dataset_predictivo.csv en el modelado)
"""

import pandas as pd
import numpy as np
import os

# =========================
# CONFIG
# =========================
INPUT_FILE  = "dataset_predictivo.csv"
OUTPUT_FILE = "dataset_predictivo_v2.csv"
SPLITS_DIR  = "splits"

DATE_FMT   = "%Y-%m-%d %H:%M:%S"
FLOAT_FMT  = "%.4f"

TARGET_COL  = "TARGET_LLEGADAS_T_PLUS_1H"
LLEGADAS_COL = "LLEGADAS_HORA_T"

# =========================
# 1) Cargar
# =========================
df = pd.read_csv(INPUT_FILE, sep=";", encoding="utf-8-sig")
df["TIMESTAMP_HORA"] = pd.to_datetime(df["TIMESTAMP_HORA"])
df = df.sort_values("TIMESTAMP_HORA").reset_index(drop=True)

print(f"Dataset cargado: {df.shape}")

# =========================
# 2) Features simples sin leakage (no necesitan splits)
# =========================
# DELTA_1H_2H: diferencia entre última hora y la anterior
# Ambos son lags del pasado, no hay leakage
df["DELTA_1H_2H"]   = df["LLEGADAS_LAG_1H"]  - df["LLEGADAS_LAG_2H"]
df["DELTA_24H_48H"] = df["LLEGADAS_LAG_24H"] - df["LLEGADAS_LAG_48H"]

print("DELTA_1H_2H y DELTA_24H_48H creados.")

# =========================
# 3) Medias históricas por grupo — CON control de leakage por fold
# =========================
# Inicializar columnas a NaN
df["MEDIA_HORA_DOW"] = np.nan
df["MEDIA_HORA_MES"] = np.nan

global_mean = np.nan  # se calcula por fold

n_splits = sum(1 for f in os.listdir(SPLITS_DIR)
               if f.startswith("fold_") and f.endswith("_train_idx.npy"))

print(f"\nCalculando medias historicas sin leakage para {n_splits} folds...")

for fold in range(n_splits):
    tr_idx  = np.load(os.path.join(SPLITS_DIR, f"fold_{fold}_train_idx.npy"))
    val_idx = np.load(os.path.join(SPLITS_DIR, f"fold_{fold}_val_idx.npy"))

    df_tr = df.loc[tr_idx]
    df_vl = df.loc[val_idx].copy()

    global_mean_fold = df_tr[TARGET_COL].mean()

    # Lookup por (HORA, DIA_SEMANA)
    lu_dow = (df_tr.groupby(["HORA", "DIA_SEMANA"])[TARGET_COL]
                   .mean()
                   .rename("MEDIA_HORA_DOW"))

    # Lookup por (HORA, MES)
    lu_mes = (df_tr.groupby(["HORA", "MES"])[TARGET_COL]
                   .mean()
                   .rename("MEDIA_HORA_MES"))

    # Aplicar al val: join por índice compuesto
    val_dow = (df_vl[["HORA","DIA_SEMANA"]]
               .join(lu_dow, on=["HORA","DIA_SEMANA"])["MEDIA_HORA_DOW"]
               .fillna(global_mean_fold).values)

    val_mes = (df_vl[["HORA","MES"]]
               .join(lu_mes, on=["HORA","MES"])["MEDIA_HORA_MES"]
               .fillna(global_mean_fold).values)

    df.loc[val_idx, "MEDIA_HORA_DOW"] = val_dow
    df.loc[val_idx, "MEDIA_HORA_MES"] = val_mes

    print(f"  Fold {fold}: val rellenado ({len(val_idx)} filas)")

# Para el TRAIN del fold 0 y el TEST (que no están cubiertos por ningun val)
# usamos el train completo para calcular el lookup
train_full_idx = np.load(os.path.join(SPLITS_DIR, "train_full_idx.npy"))
test_idx       = np.load(os.path.join(SPLITS_DIR, "test_idx.npy"))

df_train_full = df.loc[train_full_idx]
global_mean_full = df_train_full[TARGET_COL].mean()

lu_dow_full = (df_train_full.groupby(["HORA","DIA_SEMANA"])[TARGET_COL]
                             .mean()
                             .rename("MEDIA_HORA_DOW"))
lu_mes_full = (df_train_full.groupby(["HORA","MES"])[TARGET_COL]
                             .mean()
                             .rename("MEDIA_HORA_MES"))

# Rellenar test con lookup sobre train completo
df_te = df.loc[test_idx].copy()

test_dow = (df_te[["HORA","DIA_SEMANA"]]
            .join(lu_dow_full, on=["HORA","DIA_SEMANA"])["MEDIA_HORA_DOW"]
            .fillna(global_mean_full).values)
test_mes = (df_te[["HORA","MES"]]
            .join(lu_mes_full, on=["HORA","MES"])["MEDIA_HORA_MES"]
            .fillna(global_mean_full).values)

df.loc[test_idx, "MEDIA_HORA_DOW"] = test_dow
df.loc[test_idx, "MEDIA_HORA_MES"] = test_mes
print(f"  Test: rellenado ({len(test_idx)} filas) con lookup sobre train completo")

# El train del fold 0 (las primeras 6722 filas) tampoco tiene val asignado
# Rellenamos con expanding mean para no desperdiciar esas filas
# ni dejarlas a NaN
fold0_train_idx = np.load(os.path.join(SPLITS_DIR, "fold_0_train_idx.npy"))
still_nan_mask = df.loc[fold0_train_idx, "MEDIA_HORA_DOW"].isna()
still_nan_idx  = fold0_train_idx[still_nan_mask.values]

if len(still_nan_idx) > 0:
    print(f"  Rellenando {len(still_nan_idx)} filas del primer bloque de train...")
    # Expanding mean por (HORA, DIA_SEMANA) — cada fila usa solo su pasado
    df_sorted = df.loc[fold0_train_idx].copy()
    df_sorted = df_sorted.sort_values("TIMESTAMP_HORA")
    df_sorted["MEDIA_HORA_DOW"] = (
        df_sorted.groupby(["HORA","DIA_SEMANA"])[TARGET_COL]
        .transform(lambda x: x.expanding().mean().shift(1))
    )
    df_sorted["MEDIA_HORA_MES"] = (
        df_sorted.groupby(["HORA","MES"])[TARGET_COL]
        .transform(lambda x: x.expanding().mean().shift(1))
    )
    # Primeras ocurrencias (que no tienen historial) -> global mean de esas pocas filas
    df_sorted["MEDIA_HORA_DOW"] = df_sorted["MEDIA_HORA_DOW"].fillna(
        df_sorted[TARGET_COL].expanding().mean().shift(1).fillna(global_mean_full)
    )
    df_sorted["MEDIA_HORA_MES"] = df_sorted["MEDIA_HORA_MES"].fillna(
        df_sorted[TARGET_COL].expanding().mean().shift(1).fillna(global_mean_full)
    )
    df.loc[df_sorted.index, "MEDIA_HORA_DOW"] = df_sorted["MEDIA_HORA_DOW"].values
    df.loc[df_sorted.index, "MEDIA_HORA_MES"] = df_sorted["MEDIA_HORA_MES"].values

# =========================
# 4) Verificacion final
# =========================
print("\n=== VERIFICACION ===")
nulos = df[["MEDIA_HORA_DOW", "MEDIA_HORA_MES",
            "DELTA_1H_2H", "DELTA_24H_48H"]].isna().sum()
print(f"NaN en features nuevas:")
print(nulos)

# Correlaciones en test (sin leakage confirmado)
df_test_check = df.loc[test_idx]
for col in ["MEDIA_HORA_DOW", "MEDIA_HORA_MES", "DELTA_1H_2H", "DELTA_24H_48H"]:
    corr = df_test_check[col].corr(df_test_check[TARGET_COL])
    print(f"  {col:25s}  corr con target en test = {corr:.3f}")

print(f"\nColumnas nuevas añadidas: MEDIA_HORA_DOW, MEDIA_HORA_MES, DELTA_1H_2H, DELTA_24H_48H")
print(f"Shape final: {df.shape}")

# =========================
# 5) Guardar
# =========================
df.to_csv(
    OUTPUT_FILE,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    na_rep="",
    date_format=DATE_FMT,
    float_format=FLOAT_FMT,
)
print(f"\nOK -> {OUTPUT_FILE}  {df.shape}")
print("\nPara usar en el modelado: actualiza INPUT_FILE en todos los modelos")
print("a 'dataset_predictivo_v2.csv' o renombra el fichero.")
