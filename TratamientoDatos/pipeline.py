"""
PIPELINE COMPLETO — TFG predicción llegadas urgencias HVLZ Cuenca.

Ejecuta los 6 pasos del pipeline en orden, validando que cada paso ha
producido su salida antes de pasar al siguiente.

Uso:
    python pipeline.py            # ejecuta todo
    python pipeline.py --desde 4  # ejecuta desde el paso 4 en adelante
    python pipeline.py --solo 5   # ejecuta solo el paso 5

Estructura de archivos esperada en el mismo directorio:
    LimpiarPaso1.py
    paso2_serie_horaria.py
    paso3_Festivos_Calendario.py
    paso4_meterologia.py
    paso6_lags.py
    paso5_predictivo.py

Y los datos de entrada en sus subcarpetas:
    Hospital/DatosOriginalesHospital.xlsx
    Festivo/FestivosCuenca.csv
    Meterologia/DatosAEMET.csv
"""

import argparse
import subprocess
import sys
import os
import time

# Definición de los pasos: (numero, nombre_legible, script, salida_esperada)
PASOS = [
    (1, "Limpieza Excel original",        "LimpiarPaso1.py",            "visitas_urgencias_limpiaPaso1.csv"),
    (2, "Construccion serie horaria",     "paso2_serie_horaria.py",     "serie_horaria_base.csv"),
    (3, "Calendario y festivos",          "paso3_Festivos_Calendario.py","serie_horaria_calendario.csv"),
    (4, "Meteorologia AEMET",             "paso4_meterologia.py",       "serie_horaria_calendario_meteo.csv"),
    (5, "Lags y rolling features",        "paso6_lags.py",              "dataset_modelo.csv"),
    (6, "Dataset predictivo final",       "paso5_predictivo.py",        "dataset_predictivo.csv"),
]


def ejecutar_paso(numero, nombre, script, salida):
    print("\n" + "=" * 70)
    print(f"  PASO {numero}: {nombre}")
    print(f"  Script: {script}  ->  {salida}")
    print("=" * 70)

    if not os.path.exists(script):
        print(f"  ERROR: no se encuentra el script {script}")
        return False

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script],
            check=False,
            capture_output=False,
        )
    except Exception as e:
        print(f"  ERROR ejecutando {script}: {e}")
        return False

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FALLO: {script} terminó con código {result.returncode}")
        return False

    if not os.path.exists(salida):
        print(f"  FALLO: {script} no produjo {salida}")
        return False

    size_mb = os.path.getsize(salida) / (1024 * 1024)
    print(f"  OK ({elapsed:.1f}s) -> {salida} ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Pipeline TFG urgencias")
    parser.add_argument("--desde", type=int, default=1,
                        help="Paso desde el que empezar (1-6)")
    parser.add_argument("--solo", type=int, default=None,
                        help="Ejecutar solo este paso (1-6)")
    args = parser.parse_args()

    if args.solo is not None:
        pasos_a_ejecutar = [p for p in PASOS if p[0] == args.solo]
    else:
        pasos_a_ejecutar = [p for p in PASOS if p[0] >= args.desde]

    if not pasos_a_ejecutar:
        print("Nada que ejecutar.")
        return

    print(f"Ejecutando {len(pasos_a_ejecutar)} paso(s) del pipeline...")
    t_inicio = time.time()

    for numero, nombre, script, salida in pasos_a_ejecutar:
        ok = ejecutar_paso(numero, nombre, script, salida)
        if not ok:
            print(f"\nPipeline INTERRUMPIDO en paso {numero}.")
            sys.exit(1)

    elapsed_total = time.time() - t_inicio
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETADO EN {elapsed_total:.1f}s")
    print("=" * 70)
    print("\nDataset final listo para entrenar modelos:")
    print("  - dataset_predictivo.csv  (para forecasting de llegadas a t+1h)")
    print("  - dataset_modelo.csv      (versión completa con todas las features,")
    print("                             para análisis exploratorio)")


if __name__ == "__main__":
    main()
