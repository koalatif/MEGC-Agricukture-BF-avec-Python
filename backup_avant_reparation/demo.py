"""Démonstration complète du MEGC BF 2018 (Python) : calibration, équilibre, scénarios,
bien-être, dynamique, export Excel + graphique."""
import numpy as np; np.seterr(all='ignore')
from cge import CGE
import report as R, scenarios as SC

print("="*66)
print(" MEGC Burkina Faso 2018 — démonstration complète (Python)")
print("="*66)

m=CGE(); xb,sol=m.solve(); r0=m.report(xb)
print(f"\n[1] Équilibre de référence : PIB(VA)={r0['GDP_VA']/1e3:.0f} Mds, résidu={np.max(np.abs(m.residual(xb))):.0e}")
print(f"    Conso={r0['Conso']/1e3:.0f}  Invest={r0['Invest']/1e3:.0f}  Export={r0['Export']/1e3:.0f}  Recettes pub={r0['GVTrev']/1e3:.0f} Mds")

print("\n[2] Scénario fiscal : +10 pts de taxe sur l'agroalimentaire (avec bien-être)")
sc=SC.run_static(SC.fiscal(prefix=(62,97), add=0.10), verbose=True)

print("\n[3] Scénario commerce extérieur : -20% du prix mondial de l'or (P60)")
SC.run_static(SC.world_price(commodities=['P60'], exp_factor=0.80), verbose=True)

print("\n[4] Dynamique récursive : BAU vs +10% TFP agriculture (permanent)")
dyn=SC.run_dynamic(SC.tfp(branch_range=(1,27), factor=1.10), T=6, verbose=True)

print("\n[5] Analyse de sensibilité aux clôtures macro (long terme vs court terme)")
print("    Choc: +5% d'offre de travail")
SC.compare_closures(SC.factor_supply(labour=1.05))

print("\n[6] Exports")
R.export_excel("resultats_scenario_fiscal.xlsx", sc['model'], r0, sc['r1'],
               dyn_path=dyn['bau'], dyn_shock=dyn.get('shock'))
R.plot_dynamic(dyn['bau'], dyn.get('shock'), out="sentier_dynamique.png",
               labels=("BAU","+10% TFP agri"))
print("    -> resultats_scenario_fiscal.xlsx  +  sentier_dynamique.png")
print("\nTerminé.")
