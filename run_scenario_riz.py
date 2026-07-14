import scenarios as sc
import report as R

def main():
    print("=== Scénario 3 : Politique publique (Subvention des engrais pour le Riz) ===")
    print("Mise en place d'une subvention de 20% sur tous les intrants utilisés")
    print("par les branches Riz pluvial (A3) et Riz irrigué (A4).\n")
    
    # Création du choc: subvention de 20% sur les achats d'intrants pour A3 et A4
    choc = sc.input_subsidy(sectors=['A3', 'A4'], sub_rate=0.20)
    
    # On choisit une clôture économique classique (chômage et investissement exogène)
    # pour observer l'impact réel sur la création de valeur et l'emploi sans contraindre
    # artificiellement le marché du travail.
    cloture = {'lab': 'chomage', 'inv': 'exogene'}
    
    print("Lancement de la simulation (Modèle Statique)...")
    res = sc.run_static(choc, closure=cloture, verbose=True)
    
    print("\nSimulation terminée.")

if __name__ == "__main__":
    main()
