"""Blocs CES/CET cohérents (prix de base=1 -> reproduisent les quantités de base)."""
import numpy as np
np.seterr(all='ignore')

def ces_calib(Xo, sigma, Po=None):
    """CES Q=B[sum beta X^rho]^(1/rho), rho=(sigma-1)/sigma. Xo:(n,units). Retourne beta,B,Qo."""
    Xo=np.asarray(Xo,float); 
    if Po is None: Po=np.ones_like(Xo)
    pos=Xo>0
    rho=(sigma-1)/sigma
    w=np.where(pos,(Po*Xo)**(1/sigma)*Po**0,0.0)   # Po=1 -> Xo^(1/sigma)
    w=np.where(pos,(Po**1)*(Xo)**(1/sigma),0.0)     # part = Po*Xo^(1/sigma)? use Po^(1)?  (Po=1 ici)
    # forme standard: beta_i = Po_i*Xo_i^(1/sigma) / sum ; (avec Po=1)
    beta=np.where(pos, w/np.where(w.sum(0)>0,w.sum(0),1),0.0)
    Qo=Xo.sum(0)                                     # Po=1
    inner=np.where(pos,beta*Xo**rho,0.0).sum(0)
    B=np.where(inner>0, Qo/inner**(1/rho), 1.0)
    return beta,B,Qo,rho

def ces_price(beta,B,P,sigma):
    """
    Calcule le prix composite d'un agrégat CES (ex: prix de la Valeur Ajoutée).
    La fonction de prix CES est P_CES = (1/B) * [sum beta_i^sigma * P_i^(1-sigma)]^(1/(1-sigma))
    """
    P=np.asarray(P,float)
    val=(np.where(beta>0,beta**sigma * P**(1-sigma),0.0)).sum(0)
    return (1/B)*val**(1/(1-sigma))

def ces_dem(beta,B,Q,PQ,P,sigma):
    """
    Calcule la demande dérivée pour chaque composant i d'un agrégat CES.
    D_i = Q * (B * beta_i * PQ / P_i)^sigma
    """
    out=Q*(B**(sigma-1))*(np.where(beta>0,beta,1.0)**sigma)*(PQ/P)**sigma
    return np.where(beta>0,out,0.0)

def cet_calib(Xo,sigma):
    """
    Calibre les paramètres d'une fonction de transformation CET (Constant Elasticity of Transformation).
    Z=B[sum beta X^rho]^(1/rho), rho=(sigma+1)/sigma (>1).
    """
    Xo=np.asarray(Xo,float); pos=Xo>0
    rho=(sigma+1)/sigma
    w=np.where(pos, Xo**(1/sigma),0.0)              # Po=1
    beta=np.where(pos, w/np.where(w.sum(0)>0,w.sum(0),1),0.0)
    Zo=Xo.sum(0)
    inner=np.where(pos,beta*Xo**rho,0.0).sum(0)
    B=np.where(inner>0, Zo/inner**(1/rho),1.0)
    return beta,B,Zo,rho

def cet_price(beta,B,P,sigma):
    """
    Calcule le prix composite d'un agrégat CET (ex: prix de la production vendue localement + exportée).
    P_CET = (1/B) * [sum beta_i^(-sigma) * P_i^(1+sigma)]^(1/(1+sigma))
    """
    P=np.asarray(P,float)
    val=(np.where(beta>0,beta**(-sigma)*P**(1+sigma),0.0)).sum(0)
    return (1/B)*val**(1/(1+sigma))

def cet_sup(beta,B,Z,PZ,P,sigma):
    """
    Calcule l'offre dérivée pour chaque composant i d'un agrégat CET.
    S_i = Z * (B * beta_i^(-1) * P_i / PZ)^sigma
    """
    return Z*(B**(sigma+1))*(beta**(-sigma))*(P/PZ)**sigma

if __name__=="__main__":
    # test: 2 inputs, benchmark P=1 -> price=1, demande=Xo
    Xo=np.array([[100.,40.,0.],[50.,60.,30.]]); 
    for s in [0.8,1.5,2.0]:
        beta,B,Qo,rho=ces_calib(Xo,s)
        P=np.ones_like(Xo)
        PQ=ces_price(beta,B,P,s)
        X=ces_dem(beta,B,Qo,PQ,P,s)
        X=np.where(Xo>0,X,0.0)
        print(f"CES s={s}: prix(max)={np.max(np.abs(PQ-1)):.1e}  demande(max)={np.nanmax(np.abs((X-Xo)[Xo>0]/Xo[Xo>0])):.1e}")
    for s in [2.0]:
        beta,B,Zo,rho=cet_calib(Xo,s)
        P=np.ones_like(Xo); PZ=cet_price(beta,B,P,s)
        X=cet_sup(beta,B,Zo,PZ,P,s); X=np.where(Xo>0,X,0.0)
        print(f"CET s={s}: prix(max)={np.max(np.abs(PZ-1)):.1e}  offre(max)={np.nanmax(np.abs((X-Xo)[Xo>0]/Xo[Xo>0])):.1e}")
