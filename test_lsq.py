import numpy as np
from scipy.optimize import least_squares
from cge import CGE

print("Testing least_squares on T3...")
m = CGE()
xb, _ = m.solve(warn=False)

food = [i for i,c in enumerate(m.d.I) if 62<=int(''.join(filter(str.isdigit,c)) or 0)<=97]
m.tic_sim = m.tic.copy()
m.tic_sim[food] += 0.10

# Try least_squares
res = least_squares(m.residual, xb, bounds=(1e-8, np.inf), xtol=1e-6, ftol=1e-6)
rr = np.max(np.abs(m.residual(res.x)))
print(f"least_squares max residual: {rr:.2e}")
if rr < 1e-4:
    print("SUCCESS!")
else:
    print("FAILED!")
