# Corrections d'audit — 15/07/2026

Copie corrigée du projet (l'original n'a pas été modifié). Fichiers repris :
code du modèle, tests, données. Les sorties (logs, xlsx, png) n'ont pas été copiées.

## 1. Correctif majeur : solveur d'homotopie (`cge.py`, `solve_path`)

**Diagnostic.** Les tests T5 (or -20%) et T6 (clôture chômage) échouaient, et le choc
fiscal T3 n'était appliqué qu'à ~20% (blocage silencieux). Cause racine : certains
produits entièrement domestiques et sans consommation finale (P50, P15) ont une offre
(CET sans exportations) et une demande (intrants Leontief + déstockage fixe) toutes
deux insensibles au prix. Sous un choc suffisant, leur prix d'équilibre s'effondre
vers zéro (bien libre : offre > demande à prix nul). Sans traitement de ce coin,
la Jacobienne devient quasi singulière et tous les solveurs (hybr, lm) stagnent.

**Corrections :**
- Pivot « biens libres » : quand un prix atteint le plancher (PLFLOOR=1e-6), il est
  épinglé et l'équation de marché est remplacée (excédent librement disposé, valorisé
  à ~0 : loi de Walras préservée). Libération si excès de demande au plancher.
  Symétrique au pivot existant pour le capital oisif (rentes nulles).
- Newton amorti maison (`CGE._newton_ls`) : moindres carrés (lstsq) + recherche
  linéaire ; robuste aux Jacobiennes mal conditionnées là où SciPy stagne.
- Prédicteur sécant le long du chemin d'homotopie (continuation d'ordre 1).
- Cascade par pas : hybr (xtol 1e-8) -> Newton niveaux -> Newton log (positivité).
- `solve_path` renvoie en plus `free` (produits à prix plancher).

**Résultats (validation complète, T1-T8 tous RÉUSSIS) :**
| Test | Avant | Après |
|---|---|---|
| T3 choc fiscal +10 pts | s=0,20 (choc tronqué), Walras=1047 | s=1,00, res=2e-15, Walras=0 |
| T5 or -20% | ÉCHEC (res 2e-3, Walras=863) | s=1,00, res=1e-14, Walras=0, 4 s |
| T6 clôture chômage | ÉCHEC (blocage s=0,39) | s=1,00, res=9e-15, Walras=0, 7 s |
| T7 dynamique T=3 | RÉUSSI (minutes) | RÉUSSI (2 s), PIB identique |

## 2. Verdicts de validation resserrés (`validation_finale.py`)
- T3 : exige désormais s>0,99 (choc appliqué à 100%) et Walras < 1% du SROW
  (avant : tolérance 20 000, soit ~5,5 fois le solde courant).
- T5/T6 : critère Walras < 1% du SROW ajouté.

## 3. Tests ajoutés (`tests/test_cge.py`)
- `test_subvention_intrants` : bouclage budgétaire de la subvention riz
  (coût ≈ -ΔSG, Walras respecté).
- `test_chocs_homotopie_completes` : garantit que solve_path applique 100% du choc.

## 4. Corrections mineures
- `scenarios.py` : print DEBUG retiré de `input_subsidy`.
- `requirements.txt` : doublons supprimés.
- `.gitignore` : réencodé UTF-8 (l'original en UTF-16 était illisible par git,
  d'où les __pycache__, .bak, logs suivis par le dépôt).
- `.gitattributes` ajouté (normalisation des fins de ligne, évite les faux diffs CRLF).

## Recommandations restantes (non appliquées ici)
- Dans le dépôt d'origine : `git rm --cached` des fichiers générés, commit du
  .gitignore corrigé.
- Résultats des scénarios riz à re-générer : les chocs étant maintenant appliqués
  à 100%, les chiffres publiés à partir des anciens runs T3-like sont sous-estimés.
