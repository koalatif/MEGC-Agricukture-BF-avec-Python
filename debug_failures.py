import numpy as np
from cge import CGE
from scenarios import fiscal, world_price, _apply_closure

def test_shock(name, setter, sec_cap=True, closure=None):
    print(f"\n--- Debugging {name} ---")
    m = CGE()
    m.sec_cap = sec_cap
    _apply_closure(m, closure)
    xb, _ = m.solve(warn=False)
    m.Wbar = (m._store['W'] / m._store['cpi']).copy()
    
    # Run path
    xf, info = m.solve_path(setter, x0=xb, verbose=True, step0=0.2)
    
    # Analyze residual components
    r = m.residual(xf)
    r_zp, r_com, r_L, r_K = np.split(r, [m.nJ, m.nJ+m.nI, m.nJ+m.nI+m.nL])
    
    print(f"\nMax residuals at s={info['s']}:")
    print(f"  Zero Profit (r_zp): max = {np.max(np.abs(r_zp)):.2e}  argmax = {np.argmax(np.abs(r_zp))}")
    print(f"  Commodity (r_com) : max = {np.max(np.abs(r_com)):.2e}  argmax = {np.argmax(np.abs(r_com))}")
    print(f"  Labor (r_L)       : max = {np.max(np.abs(r_L)):.2e}  argmax = {np.argmax(np.abs(r_L))}")
    print(f"  Capital (r_K)     : max = {np.max(np.abs(r_K)):.2e}  argmax = {np.argmax(np.abs(r_K))}")
    print(f"  Walras            : {m._store['walras']:.2e}")
    
    return m, xf, r_zp, r_com, r_L, r_K

if __name__ == '__main__':
    # Test T3: Fiscal shock
    m3 = CGE()
    food = [i for i,c in enumerate(m3.d.I) if 62<=int(''.join(filter(str.isdigit,c)))<=97]
    tic0 = m3.tic.copy()
    def setter3(mm,s): mm.tic_sim = tic0.copy(); mm.tic_sim[food] += 0.10*s
    test_shock("T3 (Fiscal +10% food)", setter3)

    # Test T5: World price gold -20%
    def setter5(mm,s):
        mm.pwm = np.ones(mm.nI)
        mm.pwe = np.ones(mm.nI)
        # Gold is P60 usually
        idx = [i for i,c in enumerate(mm.d.I) if '60' in c]
        for i in idx:
            mm.pwm[i] *= 1 - 0.20*s
            mm.pwe[i] *= 1 - 0.20*s
    test_shock("T5 (Gold price -20%)", setter5)

    # Test T6: Unemployment closure + TFP shock
    def setter6(mm,s):
        if not hasattr(mm,'_bva0'): mm._bva0 = mm.p.B_VA.copy()
        idx = [j for j,c in enumerate(mm.d.J) if 1<=int(''.join(filter(str.isdigit,c)) or 0)<=27]
        mm.p.B_VA = mm._bva0.copy()
        for j in idx: mm.p.B_VA[j] *= 1 + (1.10 - 1)*s

    test_shock("T6 (TFP +10% with unemployment)", setter6, closure={'lab':'chomage', 'inv':'exogene'})
