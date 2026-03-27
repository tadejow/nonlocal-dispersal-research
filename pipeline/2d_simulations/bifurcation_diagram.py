import matplotlib.pyplot as plt
import json

# Wczytaj dane z pliku JSON
with open("../../data/desertification/bifurcation_data_K100000.json", 'r') as f:
    data = json.load(f)

# Przygotuj listy do przechowywania danych
A_values = []
avg_bio_values = []
max_bio_values = []
critical_level_values = []

# Przetwórz dane w jednej pętli
for A_str, results in data.items():
    A = float(A_str)
    A_values.append(A)
    avg_bio_values.append(results[0][0])
    max_bio_values.append(results[0][1])
    critical_level_values.append(0.45 / A)

# Stwórz wykres
plt.figure(figsize=(5, 5))

# Narysuj wszystkie punkty za jednym razem
plt.scatter(A_values, avg_bio_values, marker='o', color='g', label='AVG BIOMASS')
plt.scatter(A_values, max_bio_values, marker='o', color='y', label='MAX BIOMASS')
plt.scatter(A_values, critical_level_values, marker='x', color='r', label='CRITICAL LEVEL')

# Ustawienia wykresu
plt.xlabel('WATER SUPPLY')
plt.ylabel('BIOMASS')
plt.title('Desertification in the regime of shrinking water resources')
plt.gca().invert_xaxis()
plt.grid(True)
plt.legend()

# Zapisz i wyświetl wykres
plt.savefig("../../data/desertification/images/bifurcation_diagram_K100000.png")
plt.show()