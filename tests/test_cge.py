import pytest
import numpy as np
import os
import sys

# Ajouter le chemin du répertoire parent pour importer cge.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cge import CGE

def test_replication_benchmark():
    """Vérifie que le modèle reproduit la MCS à l'année de base"""
    m = CGE()
    r = m.residual(m.x0_sec)
    res_max = np.max(np.abs(r))
    assert res_max < 1e-10, f"Le résidu à l'année de base est trop élevé ({res_max})"
    
    # Vérification de la loi de Walras
    walras = abs(m._store['walras'])
    assert walras < 1e-6, f"Loi de Walras non respectée ({walras})"

def test_solveur_homotopie():
    """Vérifie la robustesse du solveur sur un choc modéré (Choc TFP)"""
    m = CGE()
    agr = [j for j,c in enumerate(m.d.J) if int(''.join(filter(str.isdigit,c)) or 0)<=27]
    bva = m.p.B_VA.copy()
    def s_shock(mm, s):
        mm.p.B_VA = bva.copy()
        mm.p.B_VA[agr] *= (1 + 0.1 * s)
    xb, info = m.solve_path(s_shock, x0=m.x0_sec, max_time=300)
    assert info['s'] > 0.9, "Le solveur d'homotopie n'a pas pu achever le choc TFP modéré"

def test_dynamique_recursive_saine():
    """Vérifie que la dynamique ne diverge pas mathématiquement (pas de NaN ni d'explosion)"""
    m = CGE()
    path = m.solve_dynamic(T=3)
    assert len(path) == 3, "Le solveur dynamique n'a pas retourné 3 périodes"
    pib_t1 = path[0]['GDP_VA']
    pib_t3 = path[2]['GDP_VA']
    assert pib_t1 > 0 and pib_t3 > 0, "Le PIB calculé est négatif ou nul"
    assert pib_t3 < pib_t1 * 2, "Explosion mathématique détectée (PIB a doublé en 3 ans)"
    assert not np.isnan(pib_t3), "Le PIB contient des NaN"
