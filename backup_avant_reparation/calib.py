"""Calibration cohérente (via nests.ces_calib) du coeur statique PEP-1 - MCS BF 2018."""
import numpy as np, json, os
from nests import ces_calib
np.seterr(all='ignore')
HERE=os.path.dirname(os.path.abspath(__file__))
WORK=os.path.join(HERE,"data") if os.path.exists(os.path.join(HERE,"data","SMx.npy")) else os.path.abspath(os.path.join(HERE,".."))
def load():
    SM=np.load(os.path.join(WORK,"SMx.npy"))
    order=[tuple(o) for o in json.load(open(os.path.join(WORK,"orderx.json")))]
    meta=json.load(open(os.path.join(WORK,"metax.json")))
    return SM,order,meta
class D: pass
def extract(SM,order,meta):
    pos={o:i for i,o in enumerate(order)}
    def C(a,b,c,d):
        x=(a,b);y=(c,d); return SM[pos[x],pos[y]] if x in pos and y in pos else 0.0
    d=D(); d.C=C; d.pos=pos
    d.J=meta['A']; d.I=meta['P']; d.Xset=set(meta['X'])
    d.L=['lsal','lfam']; d.K=['kagr','knag']; d.H=['hrp','hup','hrr','hur']; d.F=['firm_nf','firm_f']
    d.LDO=np.array([[C('L',l,'J',j) for j in d.J] for l in d.L])
    KDO=np.array([[C('K',k,'J',j) for j in d.J] for k in d.K])
    for j in range(KDO.shape[1]):            # cellules capital négatives -> redistribuer dans la branche
        tot=KDO[:,j].sum()
        if (KDO[:,j]<0).any() and tot>0:
            KDO[:,j]=np.maximum(KDO[:,j],0.0); KDO[:,j]*=tot/KDO[:,j].sum()
        elif (KDO[:,j]<0).any():
            KDO[:,j]=np.maximum(KDO[:,j],0.0)
    d.KDO=KDO
    d.DIO=np.array([[C('I',i,'J',j) for j in d.J] for i in d.I])
    d.DSO=np.array([[max(C('J',j,'I',i),0.0) for i in d.I] for j in d.J])
    d.EXO=np.array([[C('J',j,'X',i) for i in d.I] for j in d.J])
    d.EXDO=np.array([C('X',i,'AG','row') for i in d.I])
    d.IMO=np.array([C('AG','row','I',i) for i in d.I])
    d.CO=np.array([[C('I',i,'AG',h) for h in d.H] for i in d.I])
    d.CGO=np.array([C('I',i,'AG','gvt') for i in d.I]); d.INVO=np.array([C('I',i,'OTH','INV') for i in d.I])
    d.VSTKO=np.array([C('I',i,'OTH','VSTK') for i in d.I])
    d.TICO=np.array([C('AG','TI','I',i) for i in d.I]); d.TIPO=np.array([C('AG','gvt','J',j) for j in d.J])
    d.TDHO=np.array([C('AG','TD','AG',h) for h in d.H])
    return d
def calibrate(d, elas=None):
    e=dict(sVA=1.5,sLD=0.8,sKD=0.8,sXT=2.0,sX=2.0,sM=2.0,sXD=2.0,frisch=-1.5,sY=1.0)
    if elas: e.update(elas)
    p=D(); p.e=e
    p.beta_LD,p.B_LD,p.LDCO,_=ces_calib(d.LDO,e['sLD'])
    p.beta_KD,p.B_KD,p.KDCO,_=ces_calib(d.KDO,e['sKD'])
    p.beta_VA,p.B_VA,p.VAO,_=ces_calib(np.array([p.LDCO,p.KDCO]),e['sVA'])
    p.CIO=d.DIO.sum(0); p.XSTO=p.VAO+p.CIO+d.TIPO
    p.io=np.where(p.XSTO>0,p.CIO/p.XSTO,0.0); p.v=np.where(p.XSTO>0,p.VAO/p.XSTO,0.0)
    p.aij=np.where(p.CIO>0,d.DIO/np.where(p.CIO>0,p.CIO,1),0.0)
    p.DDO=d.DSO.sum(0)
    p.beta_M,p.B_M,p.QO,_=ces_calib(np.array([p.DDO,d.IMO]),e['sM'])
    p.d=d
    return p
def bench_check(d,p):
    from nests import ces_price, ces_dem
    o=np.ones((1,len(d.J)))
    LDC=ces_price(p.beta_LD,p.B_LD,o*0+1,p.e['sLD'])  # prix=1
    VA=ces_price(p.beta_VA,p.B_VA,np.ones((2,len(d.J))),p.e['sVA'])
    Q=ces_price(p.beta_M,p.B_M,np.ones((2,len(d.I))),p.e['sM'])
    fac=ces_dem(p.beta_VA,p.B_VA,p.VAO,np.ones(len(d.J)),np.ones((2,len(d.J))),p.e['sVA'])
    def er(a,b):
        b=np.asarray(b,float);m=np.abs(b)>1;return float(np.nanmax(np.abs((np.asarray(a)[m]-b[m])/b[m]))) if m.any() else 0.0
    return {'prix VA=1':float(np.nanmax(np.abs(VA[p.VAO>0]-1))),
            'prix Q=1':float(np.nanmax(np.abs(Q[p.QO>0]-1))),
            'demande VA->LDC':er(fac[0],p.LDCO),'demande VA->KDC':er(fac[1],p.KDCO),
            'parts VA,M in[0,1]':bool((p.beta_VA>=-1e-9).all() and (p.beta_VA<=1+1e-9).all())}
if __name__=="__main__":
    SM,order,meta=load(); d=extract(SM,order,meta); p=calibrate(d)
    print("Reconstruction base (cohérente):")
    for k,v in bench_check(d,p).items(): print(f"  {k:22s}: {v:.2e}" if isinstance(v,float) else f"  {k:22s}: {v}")
    print(f"  PIB VA={p.VAO.sum()/1e3:.0f} Mds | branches={len(d.J)} biens={len(d.I)}")
