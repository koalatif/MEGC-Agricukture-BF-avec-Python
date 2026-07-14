import numpy as np
from cge import CGE

# Try a quick test of the analytical fix for T6
print("\n--- Test T6 fix ---")
m = CGE()
from scenarios import _apply_closure, tfp
_apply_closure(m, {'lab':'chomage', 'inv':'exogene'})
xb, _ = m.solve(warn=False)

def setter6(mm,s):
    if not hasattr(mm,'_bva0'): mm._bva0 = mm.p.B_VA.copy()
    idx = [j for j,c in enumerate(mm.d.J) if 1<=int(''.join(filter(str.isdigit,c)) or 0)<=27]
    mm.p.B_VA = mm._bva0.copy()
    for j in idx: mm.p.B_VA[j] *= 1 + (1.10 - 1)*s

# We will monkey patch residual temporarily to test the T6 fix
orig_residual = m.residual
def new_residual(x):
    # This is just a test of the concept, it's hard to monkey patch the middle of residual
    pass
