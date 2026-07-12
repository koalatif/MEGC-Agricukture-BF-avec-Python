# Réparation du MEGC BF 2018 — journal des corrections (audit du 12/07/2026)

## Constats critiques corrigés

**C1 — Réplication du benchmark.** Le résidu à l'année de base passe de 1,4e-1 à ~1e-12.
Corrections : (i) cellules make négatives (J←I, p. ex. or brut consommé par le raffinage A43,
−67,3 Mds) transférées en demande intermédiaire ; (ii) cellules de capital négatives (EBE<0)
converties en subvention à la production (TIP), compensées dans la part de l'État sur le compte
capital ; (iii) branches sans capital dotées d'un capital = travail, financé par TIP (neutre au
benchmark, donne une courbe d'offre pentue) ; (iv) marges par produit (voir M1) ; (v) suppression
de l'écrêtage des taux de taxe.

**C2 — Loi de Walras.** Institutions complètes : firmes (revenu du capital, impôts directs ttdf,
épargne SF endogène), État (budget bouclé, SG endogène), RdM (solde courant SROW FIXÉ à la PEP-1-t ;
la contrainte RdM est l'équation redondante de Walras, recalculée à chaque solution :
`_store['walras']`, ~1e-8 à l'équilibre). Le choc fiscal est désormais intégralement recyclé
(bouclage budgétaire vérifié à 1e-14).

**C4 — Institutions.** Distribution du revenu du capital à TOUTES les institutions selon la MCS
(y compris part négative du RdM — donnée conservée telle quelle, voir Limites).

**Comptes produits notionnels (P34, P60, P95).** Leurs « demandes » sont des flux fiscaux purs
(redevances, ~86 Mds dont 58 pour l'or). Convertis en prélèvements sur la production des branches
payeuses (tspec), répartis État / fonds d'investissement. L'or garde un prix CET actif :
les chocs de prix mondiaux se transmettent désormais (ΔPIB ≠ 0).

## Constats majeurs corrigés

**M1 — Marges.** Taux de marge réels par produit (tmg_i), panier commerce/transport (P126/P130),
prix d'achat PC_i = PA_i(1+tic_i) + tmg_i·PMRG. Répartition taxe/marge du coin de prix par la part
uniforme μ = pool de marges / coins positifs (67 %), faute de détail par produit dans la MCS (documenté).

**M2 — Output unique.** Ventes = coûts par branche à 7e-14 après nettoyage des données.

**M4 — Clôture chômage.** Salaire RÉEL fixé (indexation IPC, poids = consommation de base),
revenu du travail calculé sur l'emploi effectif partout (bug de report() corrigé).

**M6 — Solveur.** Variables log (positivité), replis contrôlés, tolérance honorée, alertes de
non-convergence et de violation de Walras. Nouvelle méthode `solve_path(setter, ...)` :
homotopie adaptative avec PIVOTAGE des rentes en coin (complémentarité R≥0 ⊥ KD≤K̄ :
capital sectoriel oisif possible sous grands chocs).

**m1/m3 —** Chemin codé en dur de report.py remplacé (recherche portable) ; documentation honnête.

## Choix de spécification (défauts)

- **Capital sectoriel par défaut** (`sec_cap=True`, conforme PEP-1-t) : les rentes absorbent les
  chocs ; en capital mobile, ≥5 branches quasi-100 % exportatrices sur 4 prix de facteurs rendent
  les solutions de coin obligatoires (surdétermination type HOS).
- Clôtures : SROW fixé (PEP), SG endogène, investissement tiré par l'épargne ; options
  `closure_lab='chomage'`, `closure_inv='exogene'` (cette dernière à réserver à la clôture chômage).

## Limites restantes (données, à traiter en amont)

1. Le compte extérieur de la MCS reste non plausible (imports 2,3 % de l'absorption ; revenu du
   capital versé au RdM négatif −1 985 Mds ; épargne RdM −4 201 Mds). Le modèle est désormais
   COHÉRENT avec ces données, mais les simulations commerciales doivent être interprétées avec
   prudence tant que la MCS source n'est pas réconciliée.
2. Coins de prix extrêmes hérités (P120 : 1 484 %) : rigidité réelle forte ; les grands chocs
   déclenchent des rentes nulles en cascade (géré par solve_path, mais coûteux).
3. Élasticités uniformes et LES à parts moyennes : à documenter/différencier (recommandation R6).

## Performance

Jacobien numérique sur 405 inconnues : équilibre/chocs modérés en secondes ; grands chocs et
dynamique via homotopie : minutes. `validation_finale.py` rejoue la batterie complète.

## Réconciliation avec la macro-MCS (12/07/2026, suite)

La macro-MCS fournie par l'utilisateur a révélé le compte « Unité fictive » dissous dans la
matrice désagrégée. Reconstruction (data/SMx_reconcilie.npy, 335 comptes) :
- **Importations restaurées : ~2 874 Mds** (contre 266) — les 2 608 Mds absorbés par l'UF sont
  réinjectés au prorata des cellules de stocks négatives (clé de la revente informelle).
- **Branche informelle A_UF** : offre 4 562 Mds (dont 2 608 revendus de l'import), VA capital
  1 954 Mds versée aux institutions — comble l'écart entre EBE des branches et revenus
  institutionnels de la macro-MCS.
- **Solde courant SROW = +360 Mds** (macro : +309 ; écart = déformations résiduelles de la
  désagrégation d'origine, documentées) au lieu de −4 201.
- **Revenu net des facteurs reçu du RdM : +31 Mds** (cellule d'équilibrage du compte K, lue
  par le modèle comme revenu fixe).
- PIB(VA) passe à ~9 866 Mds (revenus institutionnels de la macro-MCS > EBE des branches ;
  écart = économie informelle non ventilée).

### Choix du jeu de données : CGE(dataset=...)
| | 'reconcilie' (défaut) | 'original' |
|---|---|---|
| Secteur extérieur | réaliste (imports 2 874, CAB −360) | non plausible (imports 266) |
| Chocs de prix mondiaux / commerce | **exact (or −20 % : 6e-15, Walras=0)** | qualitativement faux |
| Chocs fiscaux | exact jusqu'à ~+3 pts (homotopie) ; au-delà précision ~1e-3, Walras jusqu'à ~0,5 % du PIB à +10 pts | **exact à toute ampleur testée (+10 pts : 3e-14, Walras=0)** |
| Cause de la limite fiscale | rigidités du bloc informel (offre A_UF inélastique + stocks fixes) × colinéarités agricoles | — |

**Recommandation** : commerce extérieur et prix mondiaux → 'reconcilie' ; fiscalité intérieure
de grande ampleur → 'original' (ou attendre le solveur AD/MCP, travail futur prioritaire).
Le choc or −20 % sur 'reconcilie' donne l'économie attendue d'une petite économie ouverte :
imports −11 %, revenu des ménages −9,2 %, FBCF −12,6 %, déflation −7,2 %, PIB −0,2 %.
