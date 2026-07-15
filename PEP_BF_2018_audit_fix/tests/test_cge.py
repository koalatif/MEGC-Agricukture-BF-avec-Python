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

def test_subvention_intrants():
    """Vérifie le bouclage de la subvention aux intrants (scénario riz) :
    coût budgétaire ≈ baisse de l'épargne publique, Walras respecté."""
    m = CGE()
    m.closure_lab = 'chomage'; m.closure_inv = 'exogene'
    xb, _ = m.solve(warn=False)
    m.Wbar = (m._store['W'] / m._store['cpi']).copy()
    r0 = m.report(xb)
    idx_j = [j for j, c in enumerate(m.d.J) if c in ('A3', 'A4')]
    m.tsub_interm = np.zeros((m.nI, m.nJ)); m.tsub_interm[:, idx_j] = 0.20
    x2, _ = m.solve(x0=xb, warn=False)
    res = np.max(np.abs(m.residual(x2)))
    assert res < 1e-8, f"Non-convergence du scénario subvention ({res:.1e})"
    r1 = m.report(x2)
    assert abs(r1['walras']) < 0.01 * abs(m.SROWo), f"Walras violé ({r1['walras']:.1f})"
    S = m._store
    cout = (m.tsub_interm * m.aij * S['PC'][:, None] * S['XST'][None, :]).sum()
    dSG = r1['SG'] - r0['SG']
    assert cout > 0, "Subvention sans coût budgétaire"
    # le coût est financé par SG (à ±25% près : effets d'équilibre général)
    assert abs(-dSG - cout) < 0.25 * cout, f"Bouclage budgétaire incohérent (coût={cout:.0f}, dSG={dSG:.0f})"

def test_chocs_homotopie_completes():
    """Vérifie que solve_path applique les chocs à 100% (correctif biens libres/pivots)."""
    m = CGE()
    xb, _ = m.solve(warn=False)
    food = [i for i, c in enumerate(m.d.I) if 62 <= int(''.join(filter(str.isdigit, c)) or 0) <= 97]
    tic0 = m.tic.copy()
    def s3(mm, s): mm.tic_sim = tic0.copy(); mm.tic_sim[food] += 0.10 * s
    x2, info = m.solve_path(s3, x0=xb, max_time=120)
    assert info['s'] > 0.99, f"Choc fiscal incomplet (s={info['s']:.2f})"
    assert info['res'] < 1e-5, f"Résidu final trop élevé ({info['res']:.1e})"
    assert abs(m._store['walras']) < 0.01 * abs(m.SROWo), f"Walras violé ({m._store['walras']:.1f})"

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
