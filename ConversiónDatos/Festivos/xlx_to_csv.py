import pandas as pd

# Ruta del Excel
excel_path = "Festivo/FestivosCuenca.xlsx"   # si está en la misma carpeta
# excel_path = r"C:\ruta\FestivosCuenca.xlsx"  # si no

# Lee la primera hoja
df = pd.read_excel(excel_path)

# Guarda a CSV
df.to_csv("FestivosCuenca.csv", index=False, encoding="utf-8-sig")

print("✅ Listo: FestivosCuenca.csv")