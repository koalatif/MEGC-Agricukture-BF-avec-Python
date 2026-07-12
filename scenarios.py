"""Outil de scénarios pour le MEGC BF: définition, exécution (statique/dynamique),
comparaison BAU vs choc et sensibilité aux clôtures. Solveur robuste par homotopie.
Chocs: fiscal, productivité (TFP), prix mondiaux, transferts, facteurs.
Chaque constructeur renvoie shock(model, t, s) où s in [0,1] est l'intensité (pour l'homotopie)."""
import numpy as np
from cge import CGE
import report as R
np.seterr(all='ignore')
def _digits(s): return int(''.join(filter(str.isdigit,s)) or 0)

def fiscal(commodities=None, add=0.10, prefix=None):
    def s(m,t=0,sc=1.0):
        idx=[i for i,c in enumerate(m.d.I) if (commodities and c in commodities)]
        if prefix: idx=[i for i,c in enumerate(m.d.I) if prefix[0]<=_digits(c)<=prefix[1]]
        m.tic_sim=m.tic.copy()
        for i in idx: m.tic_sim[i]+=add*sc
    return s
def tfp(sectors=None, branch_range=None, factor=1.10):
    def s(m,t=0,sc=1.0):
        if not hasattr(m,'_bva0'): m._bva0=m.p.B_VA.copy()
        idx=[j for j,c in enumerate(m.d.J) if (sectors and c in sectors)]
        if branch_range: idx=[j for j,c in enumerate(m.d.J) if branch_range[0]<=_digits(c)<=branch_range[1]]
        m.p.B_VA=m._bva0.copy()
        for j in idx: m.p.B_VA[j]*=1+(factor-1)*sc
    return s
def world_price(commodities=None, imp_factor=1.0, exp_factor=1.0, prefix=None):
    def s(m,t=0,sc=1.0):
        idx=[i for i,c in enumerate(m.d.I) if (commodities and c in commodities)]
        if prefix: idx=[i for i,c in enumerate(m.d.I) if prefix[0]<=_digits(c)<=prefix[1]]
        m.pwm=np.ones(m.nI); m.pwe=np.ones(m.nI)
        for i in idx: m.pwm[i]*=1+(imp_factor-1)*sc; m.pwe[i]*=1+(exp_factor-1)*sc
    return s
def factor_supply(labour=1.0, capital=1.0):
    def s(m,t=0,sc=1.0): m.LS=m.LSo*(1+(labour-1)*sc); m.KS=m.KSo*(1+(capital-1)*sc)
    return s
def combine(*shocks):
    def s(m,t=0,sc=1.0):
        for sh in shocks: sh(m,t,sc)
    return s

def _apply_closure(m, closure):
    if not closure: return
    if 'lab' in closure: m.closure_lab=closure['lab']
    if 'inv' in closure: m.closure_inv=closure['inv']

def solve_homotopy(m, shock, x0, steps=5):
    """Résolution robuste: choc progressif (0->1), convergence VÉRIFIÉE à chaque pas."""
    from scipy.optimize import root
    import warnings
    x=x0.copy(); sol=None
    for sc in np.linspace(1.0/steps,1.0,steps):
        shock(m,0,sc)
        sol=root(m.residual,x,method='lm')
        rr=float(np.max(np.abs(m.residual(sol.x))))
        if rr>1e-6: warnings.warn(f"homotopie s={sc:.2f}: résidu {rr:.1e}")
        x=sol.x
    return x,sol

def run_static(shock, sec_cap=True, closure=None, steps=6, verbose=True):
    """BAU vs choc SOUS LA MÊME clôture (comparaison en écarts). Homotopie pour gros chocs."""
    m=CGE(); m.sec_cap=sec_cap; _apply_closure(m,closure)
    xb,_=m.solve(); m.Wbar=(m._store['W']/m._store['cpi']).copy(); r0=m.report(xb)  # salaire RÉEL de réf.
    m2=CGE(); m2.sec_cap=sec_cap; _apply_closure(m2,closure); m2.Wbar=m.Wbar
    if shock:
        x2,info=m2.solve_path(lambda mm,s: shock(mm,0,s), x0=xb)
        sol=info
    else:
        x2,sol=xb,None
    r1=m2.report(x2)
    if verbose: R.print_summary(m2,r0,r1,"Scénario"+(f" [clôture {closure}]" if closure else ""))
    return dict(model=m2, r0=r0, r1=r1, conv=float(np.max(np.abs(m2.residual(x2)))))

def compare_closures(shock, closures=None, verbose=True):
    """Sensibilité: même choc sous plusieurs clôtures (long terme vs court terme)."""
    closures=closures or [{'lab':'plein_emploi','inv':'epargne'},
                          {'lab':'chomage','inv':'exogene'}]
    out=[]
    for cl in closures:
        res=run_static(shock, closure=cl, verbose=False)
        d=100*(res['r1']['GDP_VA']-res['r0']['GDP_VA'])/res['r0']['GDP_VA']
        out.append((cl,d,res))
        if verbose: print(f"  clôture [{cl.get('lab','?'):12s}/{cl.get('inv','?'):8s}] : ΔPIB(VA) = {d:+.2f}%")
    return out

def run_dynamic(shock=None, T=10, sec_cap=True, verbose=True):
    mb=CGE(); mb.sec_cap=sec_cap; bau=mb.solve_dynamic(T=T)
    res=dict(bau=bau)
    if shock is not None:
        warm=[p['x'] for p in bau]
        ms=CGE(); ms.sec_cap=sec_cap; sh=ms.solve_dynamic(T=T, shock=shock, warm_path=warm); res['shock']=sh
        if verbose:
            g0=bau[-1]['GDP_VA']/1e3; g1=sh[-1]['GDP_VA']/1e3
            print(f"Dynamique T={T}: PIB(VA) final BAU={g0:.0f} vs choc={g1:.0f} Mds ({100*(g1/g0-1):+.2f}%)")
    return res

if __name__=="__main__":
    print("=== Scénarios : sensibilité aux clôtures (choc +10% TFP agriculture) ===")
    compare_closures(tfp(branch_range=(1,27), factor=1.10))
