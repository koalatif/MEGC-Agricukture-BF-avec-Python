"""Coeur PEP-1 (BF 2018) - version avec TAXES EN COIN DE PRIX + DYNAMIQUE RECURSIVE.
Numéraire: taux de change e=1 (prix mondiaux=1). Prix de base=1, prix d'achat PC=(1+tic)*Armington."""
import numpy as np, json, os
from nests import ces_calib, ces_price, ces_dem
from calib import load, extract, calibrate
np.seterr(all='ignore')

class CGE:
    def __init__(self, elas=None):
        SM,order,meta=load(); d=extract(SM,order,meta); p=calibrate(d,elas)
        self.d=d; self.p=p; self.e=p.e
        self.nI,self.nJ,self.nH,self.nL,self.nK=len(d.I),len(d.J),len(d.H),len(d.L),len(d.K)
        self.sLD,self.sKD,self.sVA=self.e['sLD'],self.e['sKD'],self.e['sVA']
        self.sM,self.sX=self.e['sM'],self.e['sX']
        # --- production / make ---
        XSo=d.DSO+d.EXO; self.XSTo=XSo.sum(1)                    # output value (=VA+CI_ach+TIP)
        self.ms=np.where(self.XSTo[:,None]>0,XSo/np.where(self.XSTo[:,None]>0,self.XSTo[:,None],1),0)
        self.DSco=d.DSO.sum(0); self.EXco=d.EXDO; self.XTco=self.DSco+self.EXco
        self.IMo=d.IMO; self.Qo=self.DSco+self.IMo              # quantité Armington de base
        # --- marges commerce/transport explicites ---
        self.MG=(d.TICO<-1000.0)                                # biens-marge (commerce, transport)
        margin_pool=np.where(self.MG,-d.TICO,0.0)              # marge nette embarquée
        Qbase0=self.Qo.sum()-margin_pool.sum()                # demande hors services de marge
        self.pmarg_m=np.where(self.MG, margin_pool/Qbase0, 0.0)# coeff de marge par bien-marge
        # --- taxes produit (coin de prix) sur les biens NON-marge ---
        rate=np.where(self.Qo>1, d.TICO/np.where(self.Qo>1,self.Qo,1),0.0)
        self.tic=np.where(self.MG,0.0,np.clip(rate,-0.85,5.0))
        self.marginO=np.zeros(self.nI)                         # (obsolète: marges désormais endogènes)
        opt=(1.0+self.tic)                                     # PC de base (marge = demande dérivée)
        self.Qtot0=self.Qo.sum()-margin_pool.sum()            # volume échangé de référence (hors marge)
        # --- quantités de base des demandes = valeurs/(1+tic) ---
        CO_q=d.CO/opt[:,None]; CGO_q=d.CGO/opt; INVO_q=d.INVO/opt; VSTKO_q=d.VSTKO/opt
        DIO_q=d.DIO/opt[:,None]
        # coefficients Leontief sur output (aij_ij = DI_q/XST)
        self.aij=np.where(self.XSTo>0, DIO_q/np.where(self.XSTo>0,self.XSTo,1)[None,:],0.0)
        self.ttip=np.where(self.XSTo>0, d.TIPO/np.where(self.XSTo>0,self.XSTo,1),0.0)
        self.va=1.0-(self.aij*opt[:,None]).sum(0)-self.ttip     # part VA cohérente avec l'output
        # Armington (sur quantités de base)
        self.betaM,self.BM,_,_=ces_calib(np.array([self.DSco,self.IMo]),self.sM)
        # CET commodity (DS,EX) = CES(-sX)
        self.betaX,self.BX,_,_=ces_calib(np.array([self.DSco,self.EXco]),-self.sX)
        self.valid=(self.Qo>1.0)&(self.XTco>1e-9)
        # --- facteurs ---
        lam_WL=np.array([[d.C('AG',h,'L',l) for l in d.L] for h in d.H])
        lam_RK=np.array([[d.C('AG',ag,'K',k) for k in d.K] for ag in d.H])
        self.LSo=d.LDO.sum(1); self.KSo=d.KDO.sum(1)
        self.shWL=lam_WL/np.where(self.LSo>0,self.LSo,1)[None,:]
        self.shRK=lam_RK/np.where(self.KSo>0,self.KSo,1)[None,:]
        ag=d.H+d.F+['gvt','row']; self.TR={(a,b):d.C('AG',a,'AG',b) for a in ag for b in ag}
        self.trRecvH=np.array([sum(self.TR[(h,x)] for x in ag) for h in d.H])
        self.trPaidH=np.array([sum(self.TR[(x,h)] for x in ag) for h in d.H])
        self.YHo=lam_WL.sum(1)+lam_RK.sum(1)+self.trRecvH
        self.ttdh=np.where(self.YHo>0,d.TDHO/np.where(self.YHo>0,self.YHo,1),0.0)
        self.SHo=np.array([d.C('OTH','INV','AG',h) for h in d.H])
        YDHo=self.YHo-d.TDHO-self.trPaidH
        self.shh=np.where(YDHo>0,self.SHo/np.where(YDHo>0,YDHo,1),0.0)
        # LES (prix d'achat PC=1+tic à la base)
        PCo=opt; self.CTHo=d.CO.sum(0)
        self.frisch=np.full(self.nH,self.e['frisch'])
        self.gam=np.where(self.CTHo>0, d.CO/np.where(self.CTHo>0,self.CTHo,1)[None,:],0.0)   # part budgétaire (valeur)
        supernum=-self.CTHo/self.frisch
        self.Cmin=CO_q-self.gam*supernum[None,:]/PCo[:,None]
        # gvt & investissement
        self.CGo_q=CGO_q; self.VSTKo_q=VSTKO_q; self.INVo_q=INVO_q
        self.gamINV=np.where(d.INVO.sum()>0, d.INVO/d.INVO.sum(),0.0)
        self.SROWo=d.C('OTH','INV','AG','row'); self.SGo=d.C('OTH','INV','AG','gvt')
        self.SFo=sum(d.C('OTH','INV','AG',f) for f in d.F)
        # offres facteurs = demandes de base (bouclage exact)
        WC0=ces_price(p.beta_LD,p.B_LD,np.ones((self.nL,self.nJ)),self.sLD); WC0=np.where(p.LDCO>0,WC0,1.0)
        RC0=ces_price(p.beta_KD,p.B_KD,np.ones((self.nK,self.nJ)),self.sKD); RC0=np.where(p.KDCO>0,RC0,1.0)
        fac0=ces_dem(p.beta_VA,p.B_VA,self.va*self.XSTo,np.ones(self.nJ),np.array([WC0,RC0]),self.sVA)
        self.LSo=np.nansum(ces_dem(p.beta_LD,p.B_LD,fac0[0],WC0,np.ones((self.nL,self.nJ)),self.sLD),1)
        self.KSo=np.nansum(ces_dem(p.beta_KD,p.B_KD,fac0[1],RC0,np.ones((self.nK,self.nJ)),self.sKD),1)
        # capital sectoriel (mode PEP-1-t) : KD fixe par branche, rente R_kj endogène
        self.KDo=d.KDO.copy()                                  # (k,j) capital sectoriel de base
        self.sec_cap=False                                    # False=capital mobile ; True=sectoriel
        self.x0=np.concatenate([np.ones(self.nI),np.ones(self.nL),np.ones(self.nK),self.XSTo])
        self.x0_sec=np.concatenate([np.ones(self.nI),np.ones(self.nL),np.ones(self.nK*self.nJ),self.XSTo])
        self.LS=self.LSo.copy(); self.KS=self.KSo.copy()
        self.KDj=self.KDo.copy()                              # capital sectoriel courant
        self.tic_sim=self.tic.copy()
        self.pwm=np.ones(self.nI); self.pwe=np.ones(self.nI)   # prix mondiaux (import/export)
        # --- bouclages macro ---
        self.closure_lab='plein_emploi'   # 'plein_emploi' (offre fixe, salaire flexible) | 'chomage' (salaire réel fixe, emploi flexible)
        self.closure_inv='epargne'        # 'epargne' (I tiré par l'épargne, néoclassique) | 'exogene' (I réel fixe, keynésien)
        self.Wbar=np.ones(self.nL)         # salaire réel de référence (bouclage chômage)

    def unpack(self,x):
        n=self.nI
        if self.sec_cap:
            W=x[n:n+self.nL]; R=x[n+self.nL:n+self.nL+self.nK*self.nJ].reshape(self.nK,self.nJ)
            return x[:n],W,R,x[n+self.nL+self.nK*self.nJ:]
        return x[:n],x[n:n+self.nL],x[n+self.nL:n+self.nL+self.nK],x[n+self.nL+self.nK:]
    def residual(self,x):
        d=self.d; p=self.p
        PL,W,R,XST=self.unpack(x)
        PE=self.pwe.copy(); PM=self.pwm.copy()
        PA=ces_price(self.betaM,self.BM,np.array([PL,PM]),self.sM); PA=np.where(self.valid,PA,1.0)
        PC=(1+self.tic_sim)*PA
        PT=ces_price(self.betaX,self.BX,np.array([PL,PE]),-self.sX); PT=np.where(self.valid,PT,1.0)
        onesJ=np.ones((1,self.nJ))
        WC=ces_price(p.beta_LD,p.B_LD,W[:,None]*onesJ,self.sLD); WC=np.where(p.LDCO>0,WC,1.0)
        if self.sec_cap:
            RC=ces_price(p.beta_KD,p.B_KD,R,self.sKD); RC=np.where(p.KDCO>0,RC,1.0)
        else:
            RC=ces_price(p.beta_KD,p.B_KD,R[:,None]*onesJ,self.sKD); RC=np.where(p.KDCO>0,RC,1.0)
        PVA=ces_price(p.beta_VA,p.B_VA,np.array([WC,RC]),self.sVA)
        PP=(self.ms*PT[None,:]).sum(1)
        r_zp=PP*(1-self.ttip)-(self.va*PVA+(self.aij*PC[:,None]).sum(0))
        # quantités
        VA=self.va*XST
        fac=ces_dem(p.beta_VA,p.B_VA,VA,PVA,np.array([WC,RC]),self.sVA)
        LD=ces_dem(p.beta_LD,p.B_LD,fac[0],WC,W[:,None]*onesJ,self.sLD)
        Rmat=R if self.sec_cap else R[:,None]*onesJ
        KD=ces_dem(p.beta_KD,p.B_KD,fac[1],RC,Rmat,self.sKD)
        XT=(self.ms*XST[:,None]).sum(0)
        cet=ces_dem(self.betaX,self.BX,XT,PT,np.array([PL,PE]),-self.sX); DS=cet[0]
        # revenus
        Kinc=(R*self.KDj).sum(1) if self.sec_cap else R*self.KS   # revenu du capital par type k
        Lemp=LD.sum(1)                                # emploi effectif par type (=LS en plein emploi)
        YHL=self.shWL@(W*Lemp); YHK=self.shRK@Kinc
        YH=YHL+YHK+self.trRecvH
        YDH=YH-self.ttdh*YH-self.trPaidH
        SH=self.shh*YDH; CTH=YDH-SH
        realb=CTH-(PC[:,None]*self.Cmin).sum(0)
        C=self.Cmin+self.gam*realb[None,:]/PC[:,None]           # quantités
        IT=SH.sum()+self.SFo+self.SGo+self.SROWo
        GFCF=IT-(PC*self.VSTKo_q).sum()
        if self.closure_inv=='exogene':
            INV=self.INVo_q.copy()                              # investissement réel fixe (keynésien)
        else:
            INV=self.gamINV*GFCF/PC                             # I tiré par l'épargne (néoclassique)
        DIsum=(self.aij*XST[None,:]).sum(1)
        Qbase=DIsum+C.sum(1)+self.CGo_q+INV+self.VSTKo_q           # demande hors marge
        Qtot=Qbase.sum()                                           # volume échangé courant
        Qd=Qbase+self.pmarg_m*Qtot                                 # marge endogène (proportionnelle au commerce)
        arm=ces_dem(self.betaM,self.BM,Qd,PA,np.array([PL,PM]),self.sM); DD=arm[0]; IM=arm[1]
        r_com=np.where(self.valid,(DS-DD)/np.maximum(self.Qo,1.0),0.0)      # relatif
        if self.closure_lab=='chomage':
            r_L=(W-self.Wbar)/np.maximum(self.Wbar,1e-6)        # salaire réel fixe -> chômage endogène
        else:
            r_L=(LD.sum(1)-self.LS)/np.maximum(self.LS,1.0)     # plein emploi
        if self.sec_cap:
            r_K=((KD-self.KDj)/np.maximum(self.KDj,1.0)).flatten()          # fixité relative
        else:
            r_K=(KD.sum(1)-self.KS)/np.maximum(self.KS,1.0)
        # recettes publiques
        TICrev=(self.tic_sim*PA*(DD+IM)).sum()                 # taxes produit
        TIPrev=(self.ttip*PP*XST).sum()                        # taxes production
        TDrev=(self.ttdh*YH).sum()                             # impôts directs
        GVTrev=TICrev+TIPrev+TDrev
        # utilité LES (Stone-Geary) par ménage pour la variation équivalente
        util=(self.gam*np.log(np.maximum(C-self.Cmin,1e-9))).sum(0)
        self._store=dict(PL=PL,PC=PC,PA=PA,PT=PT,W=W,R=R,XST=XST,VA=VA,LD=LD,KD=KD,DS=DS,
                         EX=cet[1],IM=IM,C=C,INV=INV,Qd=Qd,YH=YH,YDH=YDH,SH=SH,CTH=CTH,IT=IT,GFCF=GFCF,
                         PVA=PVA,RC=RC,TICrev=TICrev,TIPrev=TIPrev,TDrev=TDrev,GVTrev=GVTrev,util=util,
                         unemp=float(self.LS.sum()-LD.sum()),chomage_pct=float(100*(self.LS.sum()-LD.sum())/self.LS.sum()))
        return np.concatenate([r_zp,r_com,r_L,r_K])
    def solve(self,x0=None,method='lm',tol=None):
        from scipy.optimize import root
        if x0 is None: x0=(self.x0_sec if self.sec_cap else self.x0).copy()
        best=None; bestr=np.inf
        sol=root(self.residual,x0,method=method)
        rr=float(np.max(np.abs(self.residual(sol.x)))); best,bestr=sol,rr
        if bestr>1e-2:                                         # essai: repartir de la base
            x0b=(self.x0_sec if self.sec_cap else self.x0).copy()
            sol2=root(self.residual,x0b,method=method)
            rr2=float(np.max(np.abs(self.residual(sol2.x))))
            if rr2<bestr: best,bestr=sol2,rr2
        return best.x,best
    def report(self,x):
        self.residual(x); S=self._store; d=self.d
        Kinc=(S['R']*self.KDj).sum(1) if self.sec_cap else S['R']*self.KS
        exports=float((S['EX']).sum()); imports=float((S['IM']).sum())
        cons=float((S['PC']*S['C'].sum(1)).sum()); inv=float((S['PC']*S['INV']).sum())
        gov=float((S['PC']*self.CGo_q).sum()); vstk=float((S['PC']*self.VSTKo_q).sum())
        return dict(x=x,XST=S['XST'],PL=S['PL'],PC=S['PC'],PVA=S['PVA'],W=S['W'],R=S['R'],
                    VA=S['VA'],LD=S['LD'],KD=S['KD'],EX=S['EX'],IM=S['IM'],C=S['C'],INV=S['INV'],
                    GDP_VA=float(np.nansum(S['VA'])),GFCF=float(S['GFCF']),
                    Conso=cons,Invest=inv,Gov=gov,Export=exports,Import=imports,
                    VSTK=vstk,GDP_dep=cons+inv+gov+vstk+exports-imports,
                    YH=S['YH'],YDH=S['YDH'],SH=S['SH'],YHL=self.shWL@(S['W']*self.LS),YHK=self.shRK@Kinc,
                    GVTrev=S['GVTrev'],TICrev=S['TICrev'],TIPrev=S['TIPrev'],TDrev=S['TDrev'],
                    util=S['util']) 

    def solve_dynamic(self, T=10, delta=0.03, n=0.03, shock=None, verbose=False, sigma_inv=1.0, warm_path=None):
        """Dynamique récursive T périodes. Capital mobile (sec_cap=False) OU sectoriel avec
        allocation de l'investissement selon les rendements relatifs (sec_cap=True, mode PEP-1-t)."""
        path=[]; x=(self.x0_sec if self.sec_cap else self.x0).copy()
        LS=self.LSo.copy(); KS=self.KSo.copy(); KDj=self.KDo.copy()
        for t in range(T):
            self.LS=LS.copy(); self.KS=KS.copy()
            if self.sec_cap and t>0:
                gj=KDj.sum(0)/np.maximum(self.KDj.sum(0),1e-9)      # croissance capital par branche
                nI=self.nI
                x=x.copy(); x[nI+self.nL+self.nK*self.nJ:]*=gj      # échelle XST
            self.KDj=KDj.copy()
            self.tic_sim=self.tic.copy()
            if shock is not None: shock(self,t)
            x0t=warm_path[t] if (warm_path is not None and t<len(warm_path)) else x
            x,sol=self.solve(x0=x0t)
            rep0x=None
            rep=self.report(x); rep['t']=t; rep['KS']=KS.copy(); rep['LS']=LS.copy(); rep['KDj']=KDj.copy(); rep['x']=x.copy()
            res=float(np.max(np.abs(self.residual(x)))); rep['res']=res
            rep['conv']=bool(res < 1e-5)
            self.residual(x); Rj=self._store['R']; rep['R']=Rj
            path.append(rep)
            PC=rep['PC']; Pinv=float((self.gamINV*PC).sum())
            Ireal=rep['GFCF']/max(Pinv,1e-6)
            if self.sec_cap:
                # allocation de l'investissement par rendement relatif (par type k)
                for k in range(self.nK):
                    Rk=Rj[k]; Kk=KDj[k]
                    Ik_tot=Ireal*(Kk.sum()/max(KDj.sum(),1e-9))
                    Rbar=(Rk*Kk).sum()/max(Kk.sum(),1e-9)
                    w=Kk*np.where(Rbar>0,(np.maximum(Rk,1e-6)/Rbar)**sigma_inv,1.0)
                    w=w/max(w.sum(),1e-9)
                    KDj[k]=Kk*(1-delta)+Ik_tot*w
                KS=KDj.sum(1)
            else:
                kshare=KS/max(KS.sum(),1e-9); KS=KS*(1-delta)+kshare*Ireal
            LS=LS*(1+n)
            if verbose: print(f"    t={t}: PIB(VA)={rep['GDP_VA']/1e3:.0f} Mds  K={KS.sum()/1e3:.0f}  résidu={res:.0e} ({'OK' if rep['conv'] else '~'})")
        return path

def run_demo():
    print("=== MEGC Burkina Faso 2018 - Python (taxes en coin de prix + dynamique) ===")
    m=CGE(); r=m.residual(m.x0)
    print("[0] Résidu année de base: max=%.1e (facteurs %.0e, zéro-profit %.0e)"%(
        np.max(np.abs(r)),np.max(np.abs(r[m.nJ+m.nI:])),np.max(np.abs(r[:m.nJ]))))
    print("[1] Équilibre statique de référence (BAU):")
    xb,sol=m.solve(); rb=m.report(xb)
    print("    convergence=%s  PIB(VA)=%.0f Mds"%(sol.success,rb['GDP_VA']/1e3))
    print("[2] Simulation statique - choc fiscal +10 pts taxe agroalimentaire:")
    mf=CGE(); food=[i for i,c in enumerate(mf.d.I) if 62<=int(''.join(filter(str.isdigit,c)))<=97]
    mf.tic_sim=mf.tic.copy(); mf.tic_sim[food]+=0.10
    xf,sf=mf.solve(x0=xb); rf=mf.report(xf)
    print("    prix conso agroalim: %.3f -> %.3f  | PIB %.0f Mds (conv=%s)"%(rb['PC'][food].mean(),rf['PC'][food].mean(),rf['GDP_VA']/1e3,sf.success))
    print("[3] Dynamique récursive BAU (10 périodes, delta=3%, n=3%):")
    md=CGE(); path=md.solve_dynamic(T=10,verbose=True)
    g=[p['GDP_VA'] for p in path]
    print("    Croissance PIB(VA) cumulée sur 10 périodes: %.1f%%"%(100*(g[-1]/g[0]-1)))
    print("[4] Dynamique avec choc permanent +10%% TFP agriculture (écart au BAU):")
    def shk(mm,t):
        if not hasattr(mm,'_bva0'): mm._bva0=mm.p.B_VA.copy()
        agr=[j for j in range(mm.nJ) if int(''.join(filter(str.isdigit,mm.d.J[j])))<=27]
        mm.p.B_VA=mm._bva0.copy(); mm.p.B_VA[agr]*=1.10   # +10% permanent (pas cumulatif)
    ms=CGE(); paths=ms.solve_dynamic(T=10,shock=shk)
    gs=[p['GDP_VA'] for p in paths]
    print("    PIB(VA) final: BAU %.0f vs choc %.0f Mds (écart %.2f%%)"%(g[-1]/1e3,gs[-1]/1e3,100*(gs[-1]/g[-1]-1)))

if __name__=="__main__":
    run_demo()
