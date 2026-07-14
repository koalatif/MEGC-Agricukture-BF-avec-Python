import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cge import CGE

m = CGE()
agriculture = ['A%d'%i for i in range(1,28)]

idx_i = [i for i,c in enumerate(m.d.I) if c == 'P41']
idx_j = [j for j,c in enumerate(m.d.J) if c in agriculture]

if idx_i:
    i = idx_i[0]
    total_aij = m.aij[i, idx_j].sum()
    print("Total aij for P41 in agriculture:", total_aij)
    total_cost = (m.aij[i, idx_j] * m.PC0[i] * m.XSTo[idx_j]).sum()
    print("Total cost of P41 in agriculture:", total_cost)
else:
    print("P41 not found in m.d.I")
