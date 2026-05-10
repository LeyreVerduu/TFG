import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
# NOTA SOBRE NOMENCLATURA:
# - PACIENTES_PRESENTES = nº de pacientes con caso clinico abierto en cada hora
#   (inicio <= t < fin). NO mide ocupacion estricta de los 9 boxes nominales,
#   sino carga total del servicio (incluye sala de espera, pasillo, etc.).
# - CAP_BOXES_NOMINAL = capacidad nominal declarada por el hospital (9 boxes).
# - EXCESO_SOBRE_BOXES_NOMINALES = max(0, PACIENTES_PRESENTES - 9).
#   Mide presion asistencial sobre la capacidad nominal.
# - RATIO_PRESENTES_BOXES_NOMINALES = PACIENTES_PRESENTES / 9.
INPUT_VISITAS_CSV = "visitas_urgencias_limpiaPaso1.csv"
OUT_SERIE_CSV = "serie_horaria_base.csv"
OUT_RESUMEN_TXT = "resumen_paso2_serie_horaria.txt"

DATE_FMT = "%Y-%m-%d %H:%M:%S"
FLOAT_FMT = "%.3f"

# Datos reales del servicio
CAP_BOXES = 9
ENFERMERAS_TURNO = 8
MEDICOS_DIA = 10
MEDICOS_NOCHE = 5

def turno_from_hour(h: int) -> str:
    if 8 <= h < 15:
        return "MANANA"
    if 15 <= h < 22:
        return "TARDE"
    return "NOCHE"

def occupancy_at(times_ns: np.ndarray, starts_sorted: np.ndarray, ends_sorted: np.ndarray) -> np.ndarray:
    return (
        np.searchsorted(starts_sorted, times_ns, side="right")
        - np.searchsorted(ends_sorted, times_ns, side="right")
    ).astype(np.int32)

def main():
    df = pd.read_csv(INPUT_VISITAS_CSV, sep=";", encoding="utf-8-sig", low_memory=False)

    df["ACTO_CLINICO_FECHA_INICIO"] = pd.to_datetime(df["ACTO_CLINICO_FECHA_INICIO"], errors="coerce")
    df["LOS_MIN_CAP_99_95"] = pd.to_numeric(df.get("LOS_MIN_CAP_99_95", np.nan), errors="coerce")

    if "TRIAGE_DELAY_MIN" in df.columns:
        df["TRIAGE_DELAY_MIN"] = pd.to_numeric(df["TRIAGE_DELAY_MIN"], errors="coerce")

    # inicio obligatorio
    df = df[df["ACTO_CLINICO_FECHA_INICIO"].notna()].copy()

    # fin de ocupación robusto (recalculado)
    df["FECHA_FIN_OCUPACION"] = df["ACTO_CLINICO_FECHA_INICIO"] + pd.to_timedelta(
        df["LOS_MIN_CAP_99_95"].fillna(0), unit="m"
    )

    # seguridad: fin >= inicio
    df["FECHA_FIN_OCUPACION"] = df["FECHA_FIN_OCUPACION"].where(
        df["FECHA_FIN_OCUPACION"] >= df["ACTO_CLINICO_FECHA_INICIO"],
        df["ACTO_CLINICO_FECHA_INICIO"]
    )

    df["TIMESTAMP_HORA"] = df["ACTO_CLINICO_FECHA_INICIO"].dt.floor("h")

    # recortar hasta la última hora con llegadas (para que no quede “cola” al final)
    last_arrival_ts = df["TIMESTAMP_HORA"].max()

    # Grid horario
    t0 = df["TIMESTAMP_HORA"].min()
    t1 = df["FECHA_FIN_OCUPACION"].max().ceil("h")
    hours = pd.date_range(t0, t1, freq="h")
    hours = hours[hours <= last_arrival_ts]  # recorte

    hours_ns = hours.values.astype("datetime64[ns]").astype("int64")

    # Llegadas por hora (target)
    llegadas = df.groupby("TIMESTAMP_HORA")["ID"].size().reindex(hours, fill_value=0).astype(int)

    # Ocupación simultánea por hora
    start_ns = df["ACTO_CLINICO_FECHA_INICIO"].values.astype("datetime64[ns]").astype("int64")
    end_ns = df["FECHA_FIN_OCUPACION"].values.astype("datetime64[ns]").astype("int64")

    ocupacion = occupancy_at(hours_ns, np.sort(start_ns), np.sort(end_ns))

    # Turno + personal
    h = hours.hour.values
    turno = np.array([turno_from_hour(int(x)) for x in h], dtype=object)
    medicos = np.where(turno == "NOCHE", MEDICOS_NOCHE, MEDICOS_DIA).astype(int)
    enfermeras = np.full(len(hours), ENFERMERAS_TURNO, dtype=int)

    # Métricas derivadas sobre la capacidad nominal de boxes (9)
    # NOTA: 'ocupacion' aquí mide PACIENTES_PRESENTES en el servicio
    # (con caso clinico abierto), no boxes ocupados estrictamente.
    ratio_presentes_boxes = ocupacion / CAP_BOXES
    exceso_sobre_boxes = np.maximum(0, ocupacion - CAP_BOXES).astype(int)

    hourly = pd.DataFrame({
        "TIMESTAMP_HORA": hours,
        "LLEGADAS_HORA": llegadas.values,
        "PACIENTES_PRESENTES": ocupacion,
        "CAP_BOXES_NOMINAL": CAP_BOXES,
        "RATIO_PRESENTES_BOXES_NOMINALES": ratio_presentes_boxes,
        "EXCESO_SOBRE_BOXES_NOMINALES": exceso_sobre_boxes,
        "TURNO": turno,
        "MEDICOS_TURNO": medicos,
        "ENFERMERAS_TURNO": enfermeras,
        "PACIENTES_POR_MEDICO": ocupacion / medicos,
        "PACIENTES_POR_ENFERMERA": ocupacion / enfermeras,
    })

    # Triage delay por hora
    if "TRIAGE_DELAY_MIN" in df.columns:
        triage_mean = df.groupby("TIMESTAMP_HORA")["TRIAGE_DELAY_MIN"].mean().reindex(hours)
        triage_p95 = df.groupby("TIMESTAMP_HORA")["TRIAGE_DELAY_MIN"].quantile(0.95).reindex(hours)
        hourly["TRIAGE_DELAY_MEDIA"] = triage_mean.values
        hourly["TRIAGE_DELAY_P95"] = triage_p95.values

    # Ingresos por hora
    if "ACTO_CLINICO_DESTINO_FINAL" in df.columns:
        hosp_mask = df["ACTO_CLINICO_DESTINO_FINAL"].astype("string") == "2.Hospitalizacion"
        ingresos = df[hosp_mask].groupby("TIMESTAMP_HORA")["ID"].size().reindex(hours, fill_value=0).astype(int)
        hourly["INGRESOS_HORA"] = ingresos.values
        hourly["PCT_INGRESO"] = (ingresos / llegadas.replace(0, np.nan)).astype(float).fillna(0).values

    # Triaje I-V por hora
    if "TRIAJE_CLASIFICACION" in df.columns:
        triage_counts = (
            df.assign(TRIAJE=df["TRIAJE_CLASIFICACION"].astype("string"))
              .pivot_table(index="TIMESTAMP_HORA", columns="TRIAJE", values="ID", aggfunc="count", fill_value=0)
              .reindex(hours, fill_value=0)
        )
        for k in ["I", "II", "III", "IV", "V"]:
            if k not in triage_counts.columns:
                triage_counts[k] = 0
        triage_counts = triage_counts[["I", "II", "III", "IV", "V"]].astype(int)

        hourly["TRIAJE_I"] = triage_counts["I"].values
        hourly["TRIAJE_II"] = triage_counts["II"].values
        hourly["TRIAJE_III"] = triage_counts["III"].values
        hourly["TRIAJE_IV"] = triage_counts["IV"].values
        hourly["TRIAJE_V"] = triage_counts["V"].values

        total_triaje = hourly[["TRIAJE_I", "TRIAJE_II", "TRIAJE_III", "TRIAJE_IV", "TRIAJE_V"]].sum(axis=1).replace(0, np.nan)
        hourly["PCT_TRIAJE_ALTO_I_II"] = ((hourly["TRIAJE_I"] + hourly["TRIAJE_II"]) / total_triaje).values

    # Guardar
    hourly.to_csv(
        OUT_SERIE_CSV,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        na_rep="",
        date_format=DATE_FMT,
        float_format=FLOAT_FMT
    )

    # Resumen para memoria
    with open(OUT_RESUMEN_TXT, "w", encoding="utf-8") as f:
        f.write("=== RESUMEN PASO 2 (SERIE HORARIA) ===\n")
        f.write(f"Filas (horas): {len(hourly)}\n")
        f.write(f"Suma llegadas: {int(hourly['LLEGADAS_HORA'].sum())}\n")
        f.write(f"Llegadas max/h: {int(hourly['LLEGADAS_HORA'].max())}\n")
        f.write(f"Pacientes presentes p95: {float(hourly['PACIENTES_PRESENTES'].quantile(0.95))}\n")
        f.write(f"Pacientes presentes p99: {float(hourly['PACIENTES_PRESENTES'].quantile(0.99))}\n")
        f.write(f"% horas con exceso sobre boxes nominales (>9): "
                f"{float((hourly['EXCESO_SOBRE_BOXES_NOMINALES']>0).mean()*100):.2f}\n")

    print("OK ->", OUT_SERIE_CSV, hourly.shape)
    print("Resumen ->", OUT_RESUMEN_TXT)
    print("Suma llegadas hora =", int(hourly["LLEGADAS_HORA"].sum()), "| Visitas =", int(df.shape[0]))

if __name__ == "__main__":
    main()