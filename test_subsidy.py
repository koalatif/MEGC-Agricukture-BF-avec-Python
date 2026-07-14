import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cge import CGE
import scenarios as SC

m = CGE()
agriculture = ['A%d'%i for i in range(1,28)]
plan_intrants = SC.input_subsidy(inputs=None, sectors=agriculture, sub_rate=0.20)

plan_intrants(m, 0, 1.0)
print("Sum of tsub_interm:", m.tsub_interm.sum())
print("Non-zero entries in tsub_interm:", (m.tsub_interm > 0).sum())
