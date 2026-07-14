import numpy as np
from cge import CGE

m = CGE()
print("T3:")
print("  r_zp argmax 31 -> J:", m.d.J[31])
print("  r_com argmax 115 -> I:", m.d.I[115])
print("T5:")
print("  r_zp argmax 40 -> J:", m.d.J[40])
print("  r_com argmax 57 -> I:", m.d.I[57])
