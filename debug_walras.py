import numpy as np
from cge import CGE

print("Testing T3 budget leak...")
m = CGE()
xb, _ = m.solve(warn=False)

m.tic_sim = m.tic.copy()
food = [i for i,c in enumerate(m.d.I) if 62<=int(''.join(filter(str.isdigit,c)) or 0)<=97]
m.tic_sim[food] += 0.10

r = m.residual(xb)
print("Residual at xb:", np.max(np.abs(r)))

# Print budget constraints
# Evaluate at xb
def print_budgets(x):
    # Call residual to populate _store
    res = m.residual(x)
    S = m._store
    print("max res:", np.max(np.abs(res)))
    
    # 1. Households
    YH = S['YH']
    YDH = YH - m.ttdh * YH - m.trPaidH
    CTH = YDH - S['SH']
    ConsH = (S['PC'][:,None] * S['C']).sum(0)
    print("Household error:", np.max(np.abs(CTH - ConsH)))
    
    # 2. Firms
    YF = S['YF']
    print("Firm error:", np.abs(YF - m.ttdf*YF - m.trPaidF - S['SF']))
    
    # 3. Gov
    Qd = S['DIsum'] + S['C'].sum(1) + m.CGo_q + S['INV'] + m.VSTKo_q
    Qd = Qd + m.shmg * ((m.tmg * Qd).sum())
    TIC = (m.tic_sim * S['PA'] * Qd).sum()
    TIP = (m.ttip * S['PP'] * S['XST']).sum() + (m.tspec * S['PP'] * S['XST']).sum()
    TD = (m.ttdh * YH).sum() + m.TDFo
    YG = TIC + TIP + TD + m.shK_G @ S['Kinc'] + m.trRecvG
    CG = (S['PC'] * m.CGo_q).sum()
    print("Gov error:", np.abs(YG - CG - m.trPaidG - S['SG']))
    
    # 4. ROW
    IM = S['IM']
    EX = S['EX']
    SROW_imp = (m.PM * IM).sum() + m.shK_R @ S['Kinc'] + m.trRecvR - (m.PE * EX).sum() - m.trPaidR - m.KFROW.sum()
    print("ROW error (Walras):", SROW_imp - m.SROWo)
    
    # 5. Inv
    IT = S['SH'].sum() + S['SF'].sum() + S['SG'] + m.SROWo + (1.0 - m.sh_lev_gvt)*m.LEV
    INV_val = (S['PC'] * S['INV']).sum() + (S['PC'] * m.VSTKo_q).sum()
    print("Inv error:", np.abs(IT - INV_val))

print_budgets(xb)

from scipy.optimize import root
sol = root(m.residual, xb, method='lm', options={'xtol': 1e-4})
print("\nAfter solving:")
print_budgets(sol.x)

