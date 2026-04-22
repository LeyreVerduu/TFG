import pandas as pd

# Leer el JSON
df = pd.read_json("datosMet.json")  # tu archivo JSON

# Columnas numéricas que vamos a limpiar
columnas_numericas = [
    "altitud", "tmed", "prec", "tmin", "tmax", "velmedia", "racha",
    "sol", "presMax", "presMin", "hrMedia", "hrMax", "hrMin"
]

# Limpiar y convertir a float
for col in columnas_numericas:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors='coerce')

# Guardar CSV limpio
df.to_csv("DatosAEMET.csv", index=False)
print("CSV creado con éxito: DatosAEMET.csv")
