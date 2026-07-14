import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cge import CGE

m = CGE()
agriculture = ['A%d'%i for i in range(1,28)]
idx_j = [j for j,c in enumerate(m.d.J) if c in agriculture]

# Somme des consommations intermédiaires par produit pour l'agriculture
total_ic_by_product = m.aij[:, idx_j].sum(axis=1)

# Trier les produits par consommation intermédiaire totale
sorted_indices = total_ic_by_product.argsort()[::-1]

print("Top 10 inputs consumed by agriculture:")
for i in sorted_indices[:10]:
    if total_ic_by_product[i] > 0:
        print(f"Product {m.d.I[i]}: {total_ic_by_product[i]}")

# Aussi, voyons si 'P41' est consommé par un autre secteur
p41_idx = m.d.I.index('P41')
p41_ic_by_sector = m.aij[p41_idx, :]
print("\nSectors consuming P41:")
for j, val in enumerate(p41_ic_by_sector):
    if val > 0:
        print(f"Sector {m.d.J[j]}: {val}")
