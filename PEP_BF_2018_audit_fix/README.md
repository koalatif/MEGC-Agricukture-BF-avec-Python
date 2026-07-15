> **VERSION RÉPARÉE (audit du 12/07/2026).** Benchmark répliqué à ~1e-12, loi de Walras
> vérifiée à chaque solution, institutions complètes (firmes, État, RdM), marges par produit,
> capital sectoriel par défaut, homotopie `solve_path` pour les grands chocs.
> Détails : `CHANGELOG_REPARATION.md`. Validation : `python validation_finale.py`.
> **MCS réconciliée avec la macro-MCS** (Unité fictive restaurée) : `CGE(dataset='reconcilie')` (défaut,
> secteur extérieur réaliste) vs `CGE(dataset='original')` (chocs fiscaux exacts à toute ampleur).
> Anciens fichiers : `backup_avant_reparation/`.

# MEGC Burkina Faso 2018 — implémentation Python (complète)

Modèle d'équilibre général calculable **exécutable et vérifié**, dérivé de PEP-1-t.
Inclut : taxes en coin de prix, **marges commerce/transport**, **capital sectoriel**,
dynamique récursive, **reporting enrichi** (bien-être, revenus, sectoriel) et **outil de scénarios**.

## Démarrer
```bash
pip install numpy scipy openpyxl matplotlib pytest
python demo.py        # démonstration complète (scénarios + bien-être + dynamique + exports)
python cge.py         # équilibre + choc fiscal + dynamique
python scenarios.py   # exemples de scénarios
pytest tests/         # validation complète et automatisée du modèle (loi de Walras, homotopie, dynamique)
python plot_dynamique.py # générer les trajectoires économiques sur 5 ans (trajectoires_dynamiques.png)
```

## Modules
| Fichier | Rôle |
|---|---|
| `nests.py` | Blocs CES/CET cohérents (vérifiés à 1e-16). |
| `calib.py` | Chargement MCS + calibration. |
| `cge.py` | Modèle (production, commerce, **marges**, demande LES, **taxes en coin de prix**), solveur robuste, **capital mobile ou sectoriel**, dynamique récursive. |
| `report.py` | Agrégats macro, **bien-être (variation équivalente)**, revenus par ménage, tableaux sectoriels, **export Excel + graphiques**. |
| `scenarios.py` | Constructeurs de chocs (fiscal, TFP, prix mondiaux, facteurs) + exécution statique/dynamique BAU vs choc. |
| `data/` | MCS équilibrée + métadonnées (autonome). |

## Capacités

**Équilibre & fermetures**
```python
from cge import CGE
m = CGE()                    # capital mobile (défaut)
m.sec_cap = True             # capital SECTORIEL (spécifique par branche, PEP-1-t)
x, sol = m.solve()           # solveur robuste (résidus relatifs, ~1e-13)
r = m.report(x)              # PIB, prix, revenus, recettes, utilité…
```

**Scénarios** (`scenarios.py`)
```python
import scenarios as SC
SC.run_static(SC.fiscal(prefix=(62,97), add=0.10))          # +10 pts taxe agroalimentaire
SC.run_static(SC.world_price(commodities=['P60'], exp_factor=0.8))  # -20% prix mondial or
SC.run_static(SC.tfp(branch_range=(1,27), factor=1.10))     # +10% TFP agriculture
SC.run_static(SC.factor_supply(labour=1.05))                # +5% offre de travail
SC.run_dynamic(SC.tfp(branch_range=(1,27),factor=1.10), T=10)  # dynamique BAU vs choc
SC.run_static(SC.combine(SC.fiscal(prefix=(62,97),0.05), SC.tfp(branch_range=(1,27))))  # combiné
```
Leviers directs : `m.tic_sim` (taxes produit), `m.p.B_VA` (TFP), `m.pwm/m.pwe` (prix mondiaux),
`m.LS/m.KS` (facteurs), `m.ttip` (taxe production), `m.KDj` (capital sectoriel), `CGE(elas={...})`.

**Reporting** (`report.py`)
```python
import report as R
R.print_summary(m, r0, r1, "Mon scénario")     # tableau macro + bien-être
R.export_excel("resultats.xlsx", m, r0, r1, dyn_path=..., dyn_shock=...)
R.plot_dynamic(path_bau, path_choc, out="sentier.png")
R.equivalent_variation(m, r0, r1)              # variation équivalente par ménage
```


## Bouclages macroéconomiques (clôtures)

Le choix de clôture change la réponse d'un choc — c'est l'axe clé de la **sensibilité**.
On compare toujours *choc vs BAU sous la MÊME clôture* (les écarts, pas les niveaux).

```python
m = CGE()
m.closure_lab = 'chomage'      # 'plein_emploi' (défaut) | 'chomage' (salaire réel fixe, emploi flexible)
m.closure_inv = 'exogene'      # 'epargne' (défaut, I tiré par l'épargne) | 'exogene' (I réel fixe, keynésien)
```

Analyse de sensibilité automatique :
```python
import scenarios as SC
SC.compare_closures(SC.factor_supply(labour=1.05))   # même choc sous 2 clôtures
```
Exemple (choc +5% d'offre de travail) :
- **plein_emploi / epargne** (long terme néoclassique) : ΔPIB ≈ **+1,9%** (plus de travail → plus de production).
- **chomage / exogene** (court terme keynésien) : ΔPIB ≈ **0%** (l'offre ne crée pas sa demande ; la production est bornée par la demande).

Combinaison recommandée pour le BF : *plein_emploi + epargne + change fixe (CFA)* en référence
long terme, comparé à *chomage + exogene* pour le court terme.

**Solveur robuste** : `run_static(..., steps=5)` applique les gros chocs par **homotopie**
(montée progressive 0→1 avec réchauffage), ce qui stabilise la convergence.

## Validation (exécutée)
- Calibration et blocs CES/CET : **1e-16**.
- Équilibre statique (mobile & sectoriel) : résidus relatifs **~1e-13** (machine).
- Marchés des facteurs, zéro-profit : **exacts**.
- Marges endogènes, taxes en coin de prix : benchmark répliqué, réponses cohérentes (pass-through, recettes).
- Dynamique : sentiers BAU et choc cohérents.

## Structure du modèle
Production emboîtée CES (travail × capital) + Leontief ; make à parts fixes ;
CET ventes/exportations ; Armington ; **marges commerce/transport endogènes** ;
demande LES ; **taxes sur produits en coin de prix** ; État, investissement ;
marchés des biens et facteurs ; numéraire = taux de change ;
**capital mobile OU sectoriel** (investissement alloué par rendement) ;
**dynamique récursive** (accumulation du capital, croissance du travail).

## Limites documentées
1. **Capital sectoriel dynamique** : fonctionnel mais plus lent (système ~400 inconnues) ;
   petits résidus possibles sur longs horizons. Le capital mobile est robuste et rapide (défaut).
2. **Marges** : allocation uniforme (proportionnelle au volume échangé), faute de détail par bien.
3. **Produit territorial I8** exogénéisé en stocks ; résidu de base ≈ 0,1 % (comptes anomaux).
4. **Clôture chômage** : la base sous chômage se décale de ~1,6% (l'imperfection résiduelle du benchmark n'est pas ancrée sans la contrainte d'emploi) — d'où la lecture en **écarts** choc/BAU sous une même clôture. La clôture budgétaire de l'État reste *dépenses+épargne fixes* (standard PEP) : un recyclage complet des recettes fiscales exigerait de réconcilier la comptabilité des marges.
5. La version **GAMS** (PEP-1-t officiel) reste la référence institutionnelle.
