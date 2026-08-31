import csv

input_file = "noticias.csv"
output_file = "noticias_filtradas.csv"

PERMITTED_OUTLETS = {"wsj"}

with open(input_file, mode="r", encoding="utf-8", newline="") as infile, \
     open(output_file, mode="w", encoding="utf-8", newline="") as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile, lineterminator="\n")
    
    for row in reader:
        # Verificar que la fila tenga al menos 3 columnas
        if len(row) >= 3:
            # Obtener el outlet (último elemento por si hay comas extras en el titular)
            outlet = row[-1].strip().lower()
            
            if outlet in PERMITTED_OUTLETS:
                # Limpiar saltos de línea dentro del titular (columna 1 hasta N-1)
                date = row[0].strip()
                headline = " ".join(" ".join(row[1:-1]).split())  # Quita saltos de línea internos
                
                writer.writerow([date, headline, outlet])

print("Procesamiento completado. Saltos de línea corregidos.")