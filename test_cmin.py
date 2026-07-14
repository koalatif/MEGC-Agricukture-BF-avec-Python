import numpy as np
from cge import CGE

m = CGE()
xb, _ = m.solve(warn=False)

food = [i for i,c in enumerate(m.d.I) if 62<=int(''.join(filter(str.isdigit,c)) or 0)<=97]

# Let's manually push the tax up and see realb
print("Testing realb vs tax...")
m.tic_sim = m.tic.copy()

for t in [0.0, 0.02, 0.05, 0.08, 0.10]:
    m.tic_sim[food] = m.tic[food] + t
    x, info = m.solve(x0=xb, warn=False)
    xb = x # warm start
    res = np.max(np.abs(m.residual(x)))
    
    # calc realb
    S = m._store
    YDH = S['YH'] - m.ttdh * S['YH'] - m.trPaidH
    CTH = YDH - S['SH']
    realb = CTH - (S['PC'][:,None] * m.Cmin).sum(0)
    
    print(f"Tax +{t*100}% -> res: {res:.2e}, min realb: {np.min(realb):.2f}, YDH: {np.mean(YDH):.0f}")
    if res > 1e-4:
        print("  FAILED to converge!")
        # Which variable is failing?
        r = m.residual(x)
        idx = np.argmax(np.abs(r))
        print("  Max residual index:", idx)
