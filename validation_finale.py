"""Validation du MEGC BF 2018 RÉPARÉ — à exécuter après toute modification.
Chaque test doit afficher REUSSI. Durée totale : quelques minutes."""
import numpy as np, warnings
warnings.simplefilter("ignore")
from scipy.optimize import root
from cge import CGE
ok=lambda b:"REUSSI" if b else "ECHEC"

print("T1 - Benchmark et Walras à l'année de base")
m=CGE(); r=m.residual(m.x0_sec)
print("   res=%.1e walras=%.1e -> %s"%(np.max(np.abs(r)),abs(m._store['walras']),ok(np.max(np.abs(r))<1e-8)))

print("T2 - Réplication (solve BAU)")
xb,_=m.solve(warn=False); rb=m.report(xb)
e=max(np.max(np.abs(rb['PL']-1)),np.max(np.abs((rb['XST']-m.XSTo)/np.maximum(m.XSTo,1))))
print("   écart=%.1e PIB=%.1f Mds -> %s"%(e,rb['GDP_VA']/1e3,ok(e<1e-6)))

print("T3 - Choc fiscal +10 pts agroalimentaire : recyclage budgétaire + Walras")
m2=CGE(); food=[i for i,c in enumerate(m2.d.I) if 62<=int(''.join(filter(str.isdigit,c)) or 0)<=97]
m2.tic_sim=m2.tic.copy(); m2.tic_sim[food]+=0.10
sol=root(m2.residual,xb,method='lm'); x2=sol.x; r2=m2.report(x2)
res2=np.max(np.abs(m2.residual(x2)))
dR=(r2['GVTrev']-rb['GVTrev'])/1e3; dSG=(r2['SG']-rb['SG'])/1e3; dG=(r2['Gov']-rb['Gov'])/1e3
print("   res=%.1e ΔRec=%+.1f ΔSG=%+.1f Mds bouclage=%.1e walras=%.4f -> %s"%(
  res2,dR,dSG,abs(dSG-(dR-dG)),abs(r2['walras']),ok(res2<1e-8 and abs(r2['walras'])<1 and abs(dSG-(dR-dG))<1e-6)))

print("T4 - TFP +10% agriculture (homotopie)")
m3=CGE(); agr=[j for j,c in enumerate(m3.d.J) if int(''.join(filter(str.isdigit,c)) or 0)<=27]
bva=m3.p.B_VA.copy()
def s4(mm,s): mm.p.B_VA=bva.copy(); mm.p.B_VA[agr]*=1+0.10*s
x4,i4=m3.solve_path(s4,x0=xb); r4=m3.report(x4)
print("   s=%.2f res=%.1e ΔPIB=%+.2f%% walras=%.4f -> %s"%(i4['s'],i4['res'],
  100*(r4['GDP_VA']/rb['GDP_VA']-1),abs(r4['walras']),ok(i4['ok'] and abs(r4['walras'])<1)))

print("T5 - Prix mondial de l'or -20% (homotopie, coins de rentes)")
m5=CGE(); i60=m5.d.I.index('P60')
def s5(mm,s): mm.pwe=np.ones(mm.nI); mm.pwe[i60]=1-0.20*s
x5,i5=m5.solve_path(s5,x0=xb,max_time=1200); r5=m5.report(x5)
print("   s=%.2f res=%.1e coins=%d ΔPIB=%+.2f%% walras=%.4f -> %s"%(i5['s'],i5['res'],len(i5['idle']),
  100*(r5['GDP_VA']/rb['GDP_VA']-1),abs(r5['walras']),ok(i5['s']>0.7 and i5['res']<1e-6)))

print("T6 - Clôture chômage (salaire réel fixe)")
m6=CGE(); m6.closure_lab='chomage'
x6,_=m6.solve(warn=False); m6.Wbar=(m6._store['W']/m6._store['cpi']).copy()
tic0=m6.tic.copy()
def s6(mm,s): mm.tic_sim=tic0.copy(); mm.tic_sim[food]+=0.05*s
x6b,i6=m6.solve_path(s6,x0=x6); m6.residual(x6b); S6=m6._store
wreal=S6['W']/S6['cpi']
print("   s=%.2f res=%.1e chômage=%.2f%% |w_réel-Wbar|=%.1e walras=%.4f -> %s"%(i6['s'],i6['res'],
  S6['chomage_pct'],np.max(np.abs(wreal-m6.Wbar)),abs(S6['walras']),
  ok(i6['ok'] and np.max(np.abs(wreal-m6.Wbar))<1e-6)))

print("T7 - Dynamique récursive (T=3, ~minutes)")
m7=CGE(); path=m7.solve_dynamic(T=3)
print("   PIB: "+" ".join("%.0f"%(p['GDP_VA']/1e3) for p in path)+
      "  conv=%s maxWalras=%.4f -> %s"%(all(p['conv'] for p in path),
      max(abs(p['walras']) for p in path),ok(all(p['conv'] for p in path))))

print("T8 - Robustesse au point de départ (±20%)")
m8=CGE()
x0p=np.abs(xb*(1+0.2*np.random.RandomState(0).randn(len(xb))))
x8,_=m8.solve(x0=x0p,warn=False)
rr=np.max(np.abs(m8.residual(x8)))
print("   res=%.1e |PL-1|=%.1e -> %s"%(rr,np.max(np.abs(m8._store['PL']-1)),ok(rr<1e-8)))
