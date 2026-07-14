import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scenarios as SC
sys.stdout.reconfigure(encoding='utf-8')

agriculture = ['A%d'%i for i in range(1,28)]

# Choc modéré pour garantir une convergence rapide :
# -5% de productivité agricole
choc_secheresse = SC.tfp(sectors=agriculture, factor=0.95)
# +25% sur le prix d'importation des engrais (P41)
choc_prix_mondiaux = SC.world_price(commodities=['P41'], imp_factor=1.25)
crise = SC.combine(choc_secheresse, choc_prix_mondiaux)

print("\n\n" + "="*60)
print(" SCÉNARIO 1 : CRISE (Sécheresse -5% + Engrais +25%)")
print("="*60)
SC.run_static(crise, closure={'lab':'chomage'})

print("\n\n" + "="*60)
print(" SCÉNARIO 2 : RIPOSTE (Subvention Globale Intrants 20%)")
print("="*60)
# Subvention sur les intrants chimiques/engrais réellement consommés (ex: P31, P112)
plan_intrants = SC.input_subsidy(inputs=['P31', 'P112'], sectors=agriculture, sub_rate=0.50)
riposte = SC.combine(crise, plan_intrants)
res = SC.run_static(riposte, closure={'lab':'chomage'})

# ---------------------------------------------------------
# GÉNÉRATION DES RAPPORTS EXCEL ET GRAPHIQUES
# ---------------------------------------------------------
import report as R
# res contient 'model', 'r0' (BAU) et 'r1' (Choc)
R.generate_full_report(res['model'], res['r0'], res['r1'], prefix="demo_riposte")

