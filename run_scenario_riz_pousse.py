import scenarios as sc
import report as R
import numpy as np

def main():
    print("=== Scénarios : Poussée de la subvention des engrais pour le Riz ===")
    
    taux_subventions = [0.20, 0.50, 0.80]
    cloture = {'lab': 'chomage', 'inv': 'exogene'}
    
    resultats = {}
    
    for taux in taux_subventions:
        print(f"\n--- Lancement avec subvention à {int(taux*100)}% ---")
        choc = sc.input_subsidy(sectors=['A3', 'A4'], sub_rate=taux)
        # verbose=False pour éviter d'inonder la console avec les détails de chaque étape
        res = sc.run_static(choc, closure=cloture, verbose=False)
        
        ev = R.equivalent_variation(res['model'], res['r0'], res['r1'])
        resultats[taux] = {
            'PIB_VA': res['r1']['GDP_VA'],
            'Recettes': res['r1']['GVTrev'],
            'Bienetre_hrp': ev['hrp'],
            'Bienetre_hrr': ev['hrr'],
            'Bienetre_hur': ev['hur'],
            'Bienetre_hup': ev['hup'],
            'Import': res['r1']['Import']
        }
        # On extrait la valeur de base (BAU) au premier passage
        if 'bau' not in resultats:
            resultats['bau'] = {
                'PIB_VA': res['r0']['GDP_VA'],
                'Recettes': res['r0']['GVTrev'],
                'Import': res['r0']['Import']
            }

    print("\n\n=== SYNTHÈSE DES IMPACTS MACROÉCONOMIQUES (par rapport au BAU) ===")
    print(f"{'Indicateur':<25} | {'Sub. 20%':<12} | {'Sub. 50%':<12} | {'Sub. 80%':<12}")
    print("-" * 65)
    
    def var_pct(ind, val):
        return (val - resultats['bau'][ind]) / resultats['bau'][ind] * 100

    bau_pib = resultats['bau']['PIB_VA']
    print(f"{'Δ PIB (VA)':<25} | {var_pct('PIB_VA', resultats[0.2]['PIB_VA']):>11.3f}% | {var_pct('PIB_VA', resultats[0.5]['PIB_VA']):>11.3f}% | {var_pct('PIB_VA', resultats[0.8]['PIB_VA']):>11.3f}%")
    print(f"{'Δ Recettes Publiques':<25} | {var_pct('Recettes', resultats[0.2]['Recettes']):>11.3f}% | {var_pct('Recettes', resultats[0.5]['Recettes']):>11.3f}% | {var_pct('Recettes', resultats[0.8]['Recettes']):>11.3f}%")
    print(f"{'Δ Importations globales':<25} | {var_pct('Import', resultats[0.2]['Import']):>11.3f}% | {var_pct('Import', resultats[0.5]['Import']):>11.3f}% | {var_pct('Import', resultats[0.8]['Import']):>11.3f}%")

    print("\n=== SYNTHÈSE DU BIEN-ÊTRE DES MÉNAGES (Variation Équivalente en Mds FCFA) ===")
    print(f"{'Ménage':<25} | {'Sub. 20%':<12} | {'Sub. 50%':<12} | {'Sub. 80%':<12}")
    print("-" * 65)
    
    menages = {'hrp': 'Ruraux Pauvres', 'hrr': 'Ruraux Riches', 'hup': 'Urbains Pauvres', 'hur': 'Urbains Riches'}
    for k, nom in menages.items():
        print(f"{nom:<25} | {resultats[0.2]['Bienetre_'+k]:>11.3f} | {resultats[0.5]['Bienetre_'+k]:>11.3f} | {resultats[0.8]['Bienetre_'+k]:>11.3f}")


if __name__ == "__main__":
    main()
