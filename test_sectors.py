import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cge import CGE
m = CGE()
print("Secteurs J:", m.d.J[:10])
print("Secteurs A:", [c for c in m.d.J if c.startswith('A')])
