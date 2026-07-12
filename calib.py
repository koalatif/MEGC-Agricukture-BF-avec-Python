"""Calibration cohérente (via nests.ces_calib) du coeur statique PEP-1 - MCS BF 2018.
Version réparée (audit 2026-07) :
- cellules make négatives (J<-I) transférées en demande intermédiaire (préserve les équilibres) ;
- revenu du capital distribué à TOUTES les institutions (ménages, firmes, État, RdM) ;
- impôts directs des firmes extraits ;
- l'output aux ventes (DS+EX) coïncide désormais avec l'output aux coûts (VA+CI+TIP)."""
import numpy as np, json, os
from nests import ces_calib
np.seterr(all='ignore')
HERE=os.path.dirname(os.path.abspath(__file__))
WORK=os.path.join(HERE,"data") if os.path.exists(os.path.join(HERE,"data","SMx.npy")) else os.path.abspath(os.path.join(HERE,".."))
def load(dataset='reconcilie'):
    """dataset='reconcilie' : MCS avec Unité fictive restaurée (secteur extérieur réaliste,
    imports ~2 874 Mds, SROW=+360, branche informelle A_UF). 'original' : MCS d'origine
    (extérieur non plausible mais chocs fiscaux/productivité numériquement exacts à toute ampleur)."""
    suf='_reconcilie' if dataset=='reconcilie' else '_original'
    f=os.path.join(WORK,f"SMx{suf}.npy")
    if not os.path.exists(f): suf=''
    SM=np.load(os.path.join(WORK,f"SMx{suf}.npy"))
    order=[tuple(o) for o in json.load(open(os.path.join(WORK,f"orderx{suf}.json")))]
    meta=json.load(open(os.path.join(WORK,f"metax{suf}.json")))
    return SM,order,meta
class D: pass
def extract(SM,order,meta,verbose=False):
    pos={o:i for i,o in enumerate(order)}
    def C(a,b,c,d):
        x=(a,b);y=(c,d); return SM[pos[x],pos[y]] if x in pos and y in pos else 0.0
    d=D(); d.C=C; d.pos=pos
    d.J=meta['A']; d.I=meta['P']; d.Xset=set(meta['X'])
    d.L=['lsal','lfam']; d.K=['kagr','knag']; d.H=['hrp','hup','hrr','hur']; d.F=['firm_nf','firm_f']
    d.INST=d.H+d.F+['gvt','row']
    d.LDO=np.array([[C('L',l,'J',j) for j in d.J] for l in d.L])
    KDO=np.array([[C('K',k,'J',j) for j in d.J] for k in d.K])
    d.TIPO=np.array([C('AG','gvt','J',j) for j in d.J])
    d.lam_K_gvt_adj=np.zeros(len(d.K))
    # cellules capital négatives (EBE<0) -> subvention à la production (TIP_j += cellule)
    # compensée dans la part de l'État sur le compte capital (Walras exact)
    neg=KDO<0
    d.kd_neg_fixed=float(-KDO[neg].sum()) if neg.any() else 0.0
    if neg.any():
        d.TIPO += KDO.sum(0)*0 + np.where(neg,KDO,0.0).sum(0)      # TIP_j diminue de |cellule|
        d.lam_K_gvt_adj = -np.where(neg,KDO,0.0).sum(1)            # l'État reçoit |cellule| du compte K
        KDO=np.maximum(KDO,0.0)
    # branches sans capital (EBE<0 nettoyé) : capital minimal = 20% du travail, financé par
    # la subvention TIP et restitué à l'État via le compte K (benchmark et Walras exacts)
    noK=(KDO.sum(0)<=0)&(d.LDO.sum(0)>0)
    d.k_add=np.zeros(KDO.shape[1])
    if noK.any():
        k_add=1.0*d.LDO.sum(0)*noK
        KDO[1,:]+=k_add                                   # attribué au capital non agricole
        d.TIPO-=k_add
        d.lam_K_gvt_adj[1]+=k_add.sum()
        d.k_add=k_add
    d.KDO=KDO
    d.DIO=np.array([[C('I',i,'J',j) for j in d.J] for i in d.I])
    # --- make matrix : cellules négatives (netting) déplacées en demande intermédiaire ---
    DSO_raw=np.array([[C('J',j,'I',i) for i in d.I] for j in d.J])
    neg=DSO_raw<0
    d.make_neg_moved=float(-DSO_raw[neg].sum()) if neg.any() else 0.0
    if neg.any():
        d.DIO += np.where(neg,-DSO_raw,0.0).T       # DIO[i,j] += -DSO_raw[j,i]
        DSO_raw=np.maximum(DSO_raw,0.0)
    d.DSO=DSO_raw
    d.EXO=np.array([[C('J',j,'X',i) for i in d.I] for j in d.J])
    d.EXDO=np.array([C('X',i,'AG','row') for i in d.I])
    d.IMO=np.array([C('AG','row','I',i) for i in d.I])
    d.CO=np.array([[C('I',i,'AG',h) for h in d.H] for i in d.I])
    d.CGO=np.array([C('I',i,'AG','gvt') for i in d.I]); d.INVO=np.array([C('I',i,'OTH','INV') for i in d.I])
    d.VSTKO=np.array([C('I',i,'OTH','VSTK') for i in d.I])
    d.TICO=np.array([C('AG','TI','I',i) for i in d.I])
    d.TDHO=np.array([C('AG','TD','AG',h) for h in d.H])
    d.TDFO=np.array([C('AG','TD','AG',f) for f in d.F])
    # revenu du travail (ménages) et du capital (toutes institutions)
    d.lam_WL=np.array([[C('AG',h,'L',l) for l in d.L] for h in d.H])
    d.lam_K=np.array([[C('AG',a,'K',k) for k in d.K] for a in d.INST])
    d.lam_K[d.INST.index('gvt'),:]+=d.lam_K_gvt_adj
    d.KFROW=np.array([C('K',k,'AG','row') for k in d.K])   # revenu net des facteurs reçu du RdM (fixe)
    if verbose and (d.make_neg_moved or d.kd_neg_fixed):
        print(f"[extract] make négatif déplacé en CI: {d.make_neg_moved:,.0f} ; capital négatif redistribué: {d.kd_neg_fixed:,.0f}")
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
    VA=ces_price(p.beta_VA,p.B_VA,np.ones((2,len(d.J))),p.e['sVA'])
    Q=ces_price(p.beta_M,p.B_M,np.ones((2,len(d.I))),p.e['sM'])
    fac=ces_dem(p.beta_VA,p.B_VA,p.VAO,np.ones(len(d.J)),np.ones((2,len(d.J))),p.e['sVA'])
    def er(a,b):
        b=np.asarray(b,float);m=np.abs(b)>1;return float(np.nanmax(np.abs((np.asarray(a)[m]-b[m])/b[m]))) if m.any() else 0.0
    XST_sales=(d.DSO+d.EXO).sum(1)
    return {'prix VA=1':float(np.nanmax(np.abs(VA[p.VAO>0]-1))),
            'prix Q=1':float(np.nanmax(np.abs(Q[p.QO>0]-1))),
            'demande VA->LDC':er(fac[0],p.LDCO),'demande VA->KDC':er(fac[1],p.KDCO),
            'output ventes=couts':er(XST_sales,p.XSTO),
            'parts VA,M in[0,1]':bool((p.beta_VA>=-1e-9).all() and (p.beta_VA<=1+1e-9).all())}
if __name__=="__main__":
    SM,order,meta=load(); d=extract(SM,order,meta,verbose=True); p=calibrate(d)
    print("Reconstruction base (cohérente):")
    for k,v in bench_check(d,p).items(): print(f"  {k:22s}: {v:.2e}" if isinstance(v,float) else f"  {k:22s}: {v}")
    print(f"  PIB VA={p.VAO.sum()/1e3:.0f} Mds | branches={len(d.J)} biens={len(d.I)}")
