import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

# Ajouter le chemin du répertoire parent pour importer cge.py
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from cge import CGE

def generate_trajectory_plot():
    print("Initialisation du modèle CGE pour simulation dynamique (T=5)...")
    m = CGE()
    # On simule sur 5 périodes pour avoir une belle courbe
    path = m.solve_dynamic(T=5)
    
    print("Simulation terminée. Génération des graphiques...")
    
    # Extraction des données
    years = np.arange(1, len(path) + 1)
    gdp = [p['GDP_VA'] / 1000 for p in path] # En milliards
    fbcf = [p['Invest'] / 1000 for p in path]
    cons = [p['Conso'] / 1000 for p in path] # Consommation globale
    exports = [p['Export'] / 1000 for p in path]
    imports = [p['Import'] / 1000 for p in path]
    walras = [np.abs(p['walras']) for p in path]
    
    plt.figure(figsize=(15, 10))
    
    # Graphique 1 : PIB et Consommation
    plt.subplot(2, 3, 1)
    plt.plot(years, gdp, 'b-o', linewidth=2, label="PIB (Valeur Ajoutée)")
    plt.title("Croissance du PIB", fontsize=12)
    plt.xlabel("Années", fontsize=10)
    plt.ylabel("Milliards FCFA", fontsize=10)
    plt.xticks(years)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Graphique 2 : Investissement (FBCF)
    plt.subplot(2, 3, 2)
    plt.plot(years, fbcf, 'g-s', linewidth=2, label="FBCF Totale")
    plt.title("Évolution de l'Investissement", fontsize=12)
    plt.xlabel("Années", fontsize=10)
    plt.ylabel("Milliards FCFA", fontsize=10)
    plt.xticks(years)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Graphique 3 : Commerce Extérieur
    plt.subplot(2, 3, 3)
    plt.plot(years, exports, 'r-^', linewidth=2, label="Exportations")
    plt.plot(years, imports, 'c-v', linewidth=2, label="Importations")
    plt.title("Balance Commerciale", fontsize=12)
    plt.xlabel("Années", fontsize=10)
    plt.ylabel("Milliards FCFA", fontsize=10)
    plt.xticks(years)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Graphique 4 : Consommation
    plt.subplot(2, 3, 4)
    plt.plot(years, cons, 'm-d', linewidth=2, label="Consommation (Ménages)")
    plt.title("Consommation des Ménages", fontsize=12)
    plt.xlabel("Années", fontsize=10)
    plt.ylabel("Milliards FCFA", fontsize=10)
    plt.xticks(years)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Graphique 5 : Distance de Walras
    plt.subplot(2, 3, 5)
    plt.plot(years, walras, 'k-x', linewidth=2, label="|Walras|")
    plt.title("Loi de Walras (Erreur absolue)", fontsize=12)
    plt.xlabel("Années", fontsize=10)
    plt.ylabel("Écart (FCFA)", fontsize=10)
    plt.yscale('log')
    plt.xticks(years)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'trajectoires_dynamiques.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé avec succès : {output_path}")

if __name__ == '__main__':
    generate_trajectory_plot()
