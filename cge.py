"""Coeur PEP-1 (BF 2018) — VERSION RÉPARÉE (audit 2026-07).
Numéraire : taux de change e=1 (prix mondiaux exogènes). Solde courant (SROW) FIXÉ, à la PEP-1-t.
Corrections majeures vs version antérieure :
 - réplication exacte du benchmark (résidu ~1e-12 à x0) ;
 - marges commerce/transport par produit (taux réels tmg_i, biens P126/P130), coin fiscal séparé,
   sans écrêtage des taux ;
 - institutions complètes : ménages, firmes (revenu, impôts directs, épargne endogène),
   État (budget bouclé, épargne publique SG endogène), RdM (contrainte laissée hors système
   = équation redondante de Walras, vérifiée à chaque solution : _store['walras']) ;
 - biens purement exportés (ex. or P60) : prix CET actif -> chocs de prix mondiaux transmis ;
 - clôture chômage : salaire RÉEL (indexé IPC) fixé, revenu du travail sur l'emploi effectif ;
 - solveur : tolérance honorée, alerte de non-convergence."""
import numpy as np, json, os, warnings
from nests import ces_calib, ces_price, ces_dem
from calib import load, extract, calibrate
np.seterr(all='ignore')

class CGE:
    def __init__(self, elas=None, dataset='reconcilie'):
        """
        Initialise le modèle d'Equilibre Général Calculable (CGE).
        Charge les données calibrées (via calib.py) et alloue la mémoire pour toutes les 
        variables endogènes et exogènes du modèle. Configure également les paramètres 
        de bouclage macroéconomique par défaut.
        """
        SM,order,meta=load(dataset); d=extract(SM,order,meta); p=calibrate(d,elas)
        self.dataset=dataset
        self.d=d; self.p=p; self.e=p.e
        self.nI,self.nJ,self.nH,self.nL,self.nK=len(d.I),len(d.J),len(d.H),len(d.L),len(d.K)
        self.nF=len(d.F)
        self.sLD,self.sKD,self.sVA=self.e['sLD'],self.e['sKD'],self.e['sVA']
        self.sM,self.sX=self.e['sM'],self.e['sX']
        # --- production / make ---
        XSo=d.DSO+d.EXO; self.XSTo=XSo.sum(1)                    # output ventes = couts (calib réparée)
        self.ms=np.where(self.XSTo[:,None]>0,XSo/np.where(self.XSTo[:,None]>0,self.XSTo[:,None],1),0)
        self.DSco=d.DSO.sum(0); self.EXco=d.EXDO; self.XTco=self.DSco+self.EXco
        self.IMo=d.IMO; self.Qo=self.DSco+self.IMo               # absorption directe (hors marges), val. de base
        # --- marges commerce/transport PAR PRODUIT ---
        # Dans la MCS, l'offre des biens-marge est nettée par une taxe TI négative (<-1000).
        self.MG=(d.TICO<-1000.0)                                 # P126 (commerce), P130 (transport)
        margin_m=np.where(self.MG,-d.TICO,0.0); MPOOL=margin_m.sum()
        self.shmg=np.where(MPOOL>0,margin_m/max(MPOOL,1e-9),0.0) # composition du panier de marge
        # --- comptes produits NOTIONNELS (absorption ~0, ex. or P60, coton-fibre P34) :
        #     leurs "demandes" sont des flux fiscaux (redevances) -> prélèvement sur la production
        #     des branches payeuses, réparti État / fonds d'investissement (benchmark & Walras exacts).
        self.notional=(self.Qo<=1e-6)&(~self.MG)
        NOT=self.notional
        self.levDI=(d.DIO*NOT[:,None]).sum(0)                    # payé par branche (val. de base)
        self.tspec=np.where(self.XSTo>0,self.levDI/np.where(self.XSTo>0,self.XSTo,1),0.0)
        self.levH=(d.CO*NOT[:,None]).sum(0)                      # payé par ménage (fixe nominal)
        self.levG=float((d.CGO*NOT).sum())
        LEV0=float(self.levDI.sum()+self.levH.sum()+self.levG)
        TICnot=float((d.TICO*NOT).sum())
        self.sh_lev_gvt=TICnot/LEV0 if LEV0>0 else 0.0           # part État vs fonds d'inv.
        self.LEV0=LEV0
        d.DIO=d.DIO*(~NOT)[:,None]; d.CO=d.CO*(~NOT)[:,None]     # flux notionnels retirés du réel
        d.CGO=d.CGO*(~NOT); d.INVO=d.INVO*(~NOT); d.VSTKO=d.VSTKO*(~NOT)
        TICnm=np.where(self.MG|NOT,0.0,d.TICO)                   # coins de prix des biens réels
        pool_pos=np.where(TICnm>0,TICnm,0.0).sum()
        mu=MPOOL/max(pool_pos,1e-9)                              # part marge des coins POSITIFS (uniforme, faute
        wedge=np.where(self.Qo>1e-6,TICnm/np.maximum(self.Qo,1e-6),0.0)  # de détail taxe/marge par produit)
        self.tic=np.where(wedge>0,(1.0-mu)*wedge,wedge)          # taxe produit (subventions non partagées)
        self.tmg=np.where(wedge>0,mu*wedge,0.0)                  # taux de marge réel (panier/unité)
        self.PC0=1.0+self.tic+self.tmg                           # prix d'achat de base (PA=1)
        # --- quantités de base des demandes = valeurs/PC0 ---
        CO_q=d.CO/self.PC0[:,None]; CGO_q=d.CGO/self.PC0; INVO_q=d.INVO/self.PC0; VSTKO_q=d.VSTKO/self.PC0
        DIO_q=d.DIO/self.PC0[:,None]
        self.aij=np.where(self.XSTo>0, DIO_q/np.where(self.XSTo>0,self.XSTo,1)[None,:],0.0)
        self.ttip=np.where(self.XSTo>0, d.TIPO/np.where(self.XSTo>0,self.XSTo,1),0.0)
        self.va=1.0-(self.aij*self.PC0[:,None]).sum(0)-self.ttip-self.tspec
        # Armington / CET (masques séparés : biens purement exportés gardent un prix CET actif)
        self.betaM,self.BM,_,_=ces_calib(np.array([self.DSco,self.IMo]),self.sM)
        self.betaX,self.BX,_,_=ces_calib(np.array([self.DSco,self.EXco]),-self.sX)
        self.validQ=(self.Qo>1e-6); self.validX=(self.XTco>1e-6)
        self.valid=self.validQ                                   # rétro-compatibilité
        # --- facteurs & institutions ---
        self.INST=d.INST; iG=d.INST.index('gvt'); iR=d.INST.index('row')
        self.LSo=d.LDO.sum(1); self.KSo=d.KDO.sum(1)
        self.KFROW=getattr(d,'KFROW',np.zeros(self.nK))       # revenu de facteurs depuis le RdM
        self.shWL=d.lam_WL/np.where(self.LSo>0,self.LSo,1)[None,:]          # (H,L)
        YK0=self.KSo+self.KFROW
        self.shK=d.lam_K/np.where(YK0>0,YK0,1)[None,:]                      # (INST,K)
        self.shK_H=self.shK[:self.nH,:]; self.shK_F=self.shK[self.nH:self.nH+self.nF,:]
        self.shK_G=self.shK[iG,:]; self.shK_R=self.shK[iR,:]
        ag=d.INST; self.TR={(a,b):d.C('AG',a,'AG',b) for a in ag for b in ag}
        def trR(x): return sum(self.TR[(x,b)] for b in ag)
        def trP(x): return sum(self.TR[(a,x)] for a in ag)
        self.trRecvH=np.array([trR(h) for h in d.H]); self.trPaidH=np.array([trP(h) for h in d.H])
        self.trRecvF=np.array([trR(f) for f in d.F]); self.trPaidF=np.array([trP(f) for f in d.F])
        self.trRecvG=trR('gvt'); self.trPaidG=trP('gvt')
        self.trRecvR=trR('row'); self.trPaidR=trP('row')
        # revenus de base et taux d'imposition directe
        self.YHo=d.lam_WL.sum(1)+self.shK_H@(self.KSo+self.KFROW)+self.trRecvH
        self.ttdh=np.where(self.YHo>0,d.TDHO/np.where(self.YHo>0,self.YHo,1),0.0)
        self.YFo=self.shK_F@(self.KSo+self.KFROW)+self.trRecvF
        self.ttdf=np.where(self.YFo>0,d.TDFO/np.where(self.YFo>0,self.YFo,1),0.0)
        self.SHo=np.array([d.C('OTH','INV','AG',h) for h in d.H])
        YDHo=self.YHo-d.TDHO-self.trPaidH-self.levH
        self.shh=np.where(YDHo>0,self.SHo/np.where(YDHo>0,YDHo,1),0.0)
        # LES (prix d'achat PC0 à la base)
        self.CTHo=d.CO.sum(0)
        self.frisch=np.full(self.nH,self.e['frisch'])
        self.gam=np.where(self.CTHo>0, d.CO/np.where(self.CTHo>0,self.CTHo,1)[None,:],0.0)
        supernum=-self.CTHo/self.frisch
        self.Cmin=CO_q-self.gam*supernum[None,:]/self.PC0[:,None]
        self.cpi_w=d.CO.sum(1)/max(d.CO.sum(),1e-9)              # poids IPC (paniers ménages, base)
        # gvt & investissement
        self.CGo_q=CGO_q; self.VSTKo_q=VSTKO_q; self.INVo_q=INVO_q
        self.gamINV=np.where(d.INVO.sum()>0, d.INVO/d.INVO.sum(),0.0)
        self.SROWo=d.C('OTH','INV','AG','row')                   # solde courant FIXÉ (clôture PEP)
        self.SGo=d.C('OTH','INV','AG','gvt'); self.SFo=sum(d.C('OTH','INV','AG',f) for f in d.F)
        # offres de facteurs (bouclage exact via les emboîtements)
        WC0=ces_price(p.beta_LD,p.B_LD,np.ones((self.nL,self.nJ)),self.sLD); WC0=np.where(p.LDCO>0,WC0,1.0)
        RC0=ces_price(p.beta_KD,p.B_KD,np.ones((self.nK,self.nJ)),self.sKD); RC0=np.where(p.KDCO>0,RC0,1.0)
        fac0=ces_dem(p.beta_VA,p.B_VA,self.va*self.XSTo,np.ones(self.nJ),np.array([WC0,RC0]),self.sVA)
        self.LSo=np.nansum(ces_dem(p.beta_LD,p.B_LD,fac0[0],WC0,np.ones((self.nL,self.nJ)),self.sLD),1)
        self.KSo=np.nansum(ces_dem(p.beta_KD,p.B_KD,fac0[1],RC0,np.ones((self.nK,self.nJ)),self.sKD),1)
        self.KDo=d.KDO.copy(); self.sec_cap=True   # capital SECTORIEL par défaut (PEP-1-t)
        self.x0=np.concatenate([np.ones(self.nI),np.ones(self.nL),np.ones(self.nK),self.XSTo])
        self.x0_sec=np.concatenate([np.ones(self.nI),np.ones(self.nL),np.ones(self.nK*self.nJ),self.XSTo])
        self.LS=self.LSo.copy(); self.KS=self.KSo.copy(); self.KDj=self.KDo.copy()
        self.tic_sim=self.tic.copy()
        self.pwm=np.ones(self.nI); self.pwe=np.ones(self.nI)
        self.tsub_interm=np.zeros((self.nI,self.nJ))
        # --- bouclages macro ---
        self.mcp=False                    # équations pures par défaut ; coins gérés par solve_path (pivots)
        self.closure_lab='plein_emploi'   # 'plein_emploi' | 'chomage' (salaire RÉEL fixe)
        self.closure_inv='epargne'        # 'epargne' | 'exogene'
        self.Wbar=np.ones(self.nL)        # salaire réel de référence (clôture chômage)

    def unpack(self,x):
        n=self.nI
        if self.sec_cap:
            W=x[n:n+self.nL]; R=x[n+self.nL:n+self.nL+self.nK*self.nJ].reshape(self.nK,self.nJ)
            return x[:n],W,R,x[n+self.nL+self.nK*self.nJ:]
        return x[:n],x[n:n+self.nL],x[n+self.nL:n+self.nL+self.nK],x[n+self.nL+self.nK:]
    def residual(self,x):
        """
        Fonction centrale du modèle CGE. Évalue le système d'équations non linéaires.
        Prend un vecteur de variables endogènes (x) et renvoie un vecteur de résidus (erreurs de fermeture).
        Le solveur cherche à annuler ces résidus.
        Contient les blocs :
        - Équations de Prix (composite, domestique, VA)
        - Équations de Production et Demandes de Facteurs
        - Équations de Revenus (institutions, transferts, taxes)
        - Équations de Demande (consommation, investissement)
        - Équilibres des Marchés (biens et services, facteurs)
        """
        d=self.d; p=self.p
        PL,W,R,XST=self.unpack(x)
        PE=self.pwe.copy(); PM=self.pwm.copy()
        PA=ces_price(self.betaM,self.BM,np.array([PL,PM]),self.sM); PA=np.where(self.validQ,PA,PL)
        PMRG=(self.shmg*PA).sum()                                # prix du panier de marge
        PC=(1+self.tic_sim)*PA+self.tmg*PMRG                     # prix d'achat (taxe + marge réelle)
        PT=ces_price(self.betaX,self.BX,np.array([PL,PE]),-self.sX); PT=np.where(self.validX,PT,1.0)
        onesJ=np.ones((1,self.nJ))
        WC=ces_price(p.beta_LD,p.B_LD,W[:,None]*onesJ,self.sLD); WC=np.where(p.LDCO>0,WC,1.0)
        if self.sec_cap:
            RC=ces_price(p.beta_KD,p.B_KD,R,self.sKD); RC=np.where(p.KDCO>0,RC,1.0)
        else:
            RC=ces_price(p.beta_KD,p.B_KD,R[:,None]*onesJ,self.sKD); RC=np.where(p.KDCO>0,RC,1.0)
        PVA=ces_price(p.beta_VA,p.B_VA,np.array([WC,RC]),self.sVA)
        PP=(self.ms*PT[None,:]).sum(1)
        g_zp=PP*(1-self.ttip-self.tspec)-(self.va*PVA+(self.aij*PC[:,None]*(1-self.tsub_interm)).sum(0))
        if getattr(self,'mcp',True):
            # complémentarité (Fischer-Burmeister lissée) : profit<=0  ⊥  XST>=0
            a=-g_zp; b=XST/np.maximum(self.XSTo,1.0)
            r_zp=a+b-np.sqrt(a*a+b*b+1e-14)
        else:
            r_zp=g_zp
        # quantités
        VA=self.va*XST
        fac=ces_dem(p.beta_VA,p.B_VA,VA,PVA,np.array([WC,RC]),self.sVA)
        LD=ces_dem(p.beta_LD,p.B_LD,fac[0],WC,W[:,None]*onesJ,self.sLD)
        Rmat=R if self.sec_cap else R[:,None]*onesJ
        KD=ces_dem(p.beta_KD,p.B_KD,fac[1],RC,Rmat,self.sKD)
        XT=(self.ms*XST[:,None]).sum(0)
        cet=ces_dem(self.betaX,self.BX,XT,PT,np.array([PL,PE]),-self.sX); DS=cet[0]; EX=cet[1]
        # revenus des institutions
        Kinc=((R*self.KDj).sum(1) if self.sec_cap else R*self.KS)+self.KFROW  # + revenu net du RdM (fixe)
        Lemp=LD.sum(1)                                           # emploi effectif
        YHL=self.shWL@(W*Lemp); YHK=self.shK_H@Kinc
        YH=YHL+YHK+self.trRecvH
        YDH=YH-self.ttdh*YH-self.trPaidH-self.levH
        SH=self.shh*YDH; CTH=YDH-SH
        realb=CTH-(PC[:,None]*self.Cmin).sum(0)
        C=self.Cmin+self.gam*realb[None,:]/PC[:,None]
        YF=self.shK_F@Kinc+self.trRecvF
        TDF=(self.ttdf*YF); SF=YF-TDF-self.trPaidF               # épargne des firmes endogène
        # demandes -> recettes fiscales (base = absorption y c. marges)
        DIsum=(self.aij*XST[None,:]).sum(1)
        Qbase=DIsum+C.sum(1)+self.CGo_q+0.0+self.VSTKo_q         # INV ajouté après bouclage épargne
        # investissement : IT = épargnes (SROW FIXÉ, clôture PEP-1-t)
        # (recettes publiques nécessitent Qd -> résolution en deux temps cohérente car tic*INV
        #  entre dans YG ; on itère une fois analytiquement)
        LEV=float((self.tspec*PP*XST).sum()+self.levH.sum()+self.levG)   # prélèvements notionnels
        def gvt_rev(Qd,PAv):
            TICrev=(self.tic_sim*PAv*Qd).sum()
            TIPrev=(self.ttip*PP*XST).sum()+self.sh_lev_gvt*LEV
            TDrev=(self.ttdh*YH).sum()+TDF.sum()
            SUBrev=(self.tsub_interm*self.aij*PC[:,None]*XST[None,:]).sum()
            return TICrev,TIPrev-SUBrev,TDrev
        MRG0=self.shmg*( (self.tmg*Qbase).sum() )
        if self.closure_inv=='exogene':
            INV=self.INVo_q.copy()
            IT_req = (PC*INV).sum() + (PC*self.VSTKo_q).sum()
            # Analytique : point fixe sur adj_sh
            CTH0 = YDH
            realb0 = CTH0 - (PC[:,None]*self.Cmin).sum(0)
            C0 = self.Cmin + self.gam*realb0[None,:]/PC[:,None]
            Qd0 = DIsum + C0.sum(1) + self.CGo_q + INV + self.VSTKo_q
            Qd0 = Qd0 + self.shmg*((self.tmg*Qd0).sum())
            TIC0, TIPrev, TDrev = gvt_rev(Qd0, PA)
            SG0 = TIC0 + TIPrev + TDrev + self.shK_G@Kinc + self.trRecvG - (PC*self.CGo_q).sum() - self.trPaidG
            
            dSH = self.shh * YDH
            dC = - self.gam * dSH[None,:] / np.maximum(PC, 1e-9)[:,None]
            dQd = dC.sum(1) + self.shmg*((self.tmg*dC.sum(1)).sum())
            dTIC = (self.tic_sim * PA * dQd).sum()
            
            dSH_total = dSH.sum()
            num = IT_req - SF.sum() - SG0 - self.SROWo - (1.0-self.sh_lev_gvt)*LEV
            den = dSH_total + dTIC
            adj_sh = num / max(den, 1e-9)
            
            SH = adj_sh * self.shh * YDH
            CTH = YDH - SH
            realb = CTH - (PC[:,None]*self.Cmin).sum(0)
            C = self.Cmin + self.gam*realb[None,:]/PC[:,None]
            Qd = DIsum + C.sum(1) + self.CGo_q + INV + self.VSTKo_q
            Qd = Qd + self.shmg*((self.tmg*Qd).sum())
            TICrev, TIPrev, TDrev = gvt_rev(Qd, PA)
            YG = TICrev + TIPrev + TDrev + self.shK_G@Kinc + self.trRecvG
            SG = YG - (PC*self.CGo_q).sum() - self.trPaidG
            IT = IT_req
            GFCF = IT_req - (PC*self.VSTKo_q).sum()
        else:
            # point fixe analytique sur GFCF (INV=gamINV*GFCF/PC, TICrev linéaire en INV)
            Qd0=Qbase+self.shmg*((self.tmg*Qbase).sum())
            TIC0,TIPrev,TDrev=gvt_rev(Qd0,PA)
            base_inc=TIC0+TIPrev+TDrev+self.shK_G@Kinc+self.trRecvG
            SG0=base_inc-(PC*self.CGo_q).sum()-self.trPaidG
            IT0=SH.sum()+SF.sum()+SG0+self.SROWo+(1.0-self.sh_lev_gvt)*LEV
            # multiplicateur keynésien de l'investissement sur l'épargne publique
            k=(self.tic_sim*PA*self.gamINV/np.maximum(PC, 1e-9)).sum() + (self.shmg*self.tic_sim*PA).sum() * (self.tmg*self.gamINV/np.maximum(PC, 1e-9)).sum()
            VSTKval=(PC*self.VSTKo_q).sum()
            GFCF=(IT0-VSTKval)/max(1.0-k,1e-6)
            INV=self.gamINV*GFCF/PC
            Qd=Qbase+INV+self.shmg*((self.tmg*(Qbase+INV)).sum())
            TICrev,_,_=gvt_rev(Qd,PA)
            YG=TICrev+TIPrev+TDrev+self.shK_G@Kinc+self.trRecvG
            SG=YG-(PC*self.CGo_q).sum()-self.trPaidG
            IT=SH.sum()+SF.sum()+SG+self.SROWo+(1.0-self.sh_lev_gvt)*LEV
        GFCF=IT-(PC*self.VSTKo_q).sum()
        arm=ces_dem(self.betaM,self.BM,Qd,PA,np.array([PL,PM]),self.sM)
        DD=np.where(self.validQ,arm[0],0.0); IM=np.where(self.validQ,arm[1],0.0)
        r_com=np.where(self.validQ,(DS-DD)/np.maximum(self.Qo,1.0),PL-1.0)  # biens sans marché domestique: PL épinglé
        # clôtures facteurs
        cpi=(self.cpi_w*PC).sum()/max((self.cpi_w*self.PC0).sum(),1e-9)
        if self.closure_lab=='chomage':
            r_L=(W-self.Wbar*cpi)/np.maximum(self.Wbar,1e-6)     # salaire RÉEL fixe
        else:
            r_L=(LD.sum(1)-self.LS)/np.maximum(self.LS,1.0)
        if self.sec_cap:
            if getattr(self,'mcp',True):
                # complémentarité : R>=0  ⊥  KD<=KDj (capital sectoriel oisif si rente nulle)
                a=R; b=(self.KDj-KD)/np.maximum(self.KDj,1.0)
                rK=a+b-np.sqrt(a*a+b*b+1e-14)
            else:
                rK=(KD-self.KDj)/np.maximum(self.KDj,1.0)
            r_K=np.where(self.KDo>1e-9,rK,R-1.0).flatten()      # rentes épinglées si K=0
        else:
            r_K=(KD.sum(1)-self.KS)/np.maximum(self.KS,1.0)
        GVTrev=YG
        # loi de Walras : contrainte RdM (équation redondante, doit -> 0 à l'équilibre)
        SROW_imp=(PM*IM).sum()+self.shK_R@Kinc+self.trRecvR-(PE*EX).sum()-self.trPaidR-self.KFROW.sum()
        walras=SROW_imp-self.SROWo
        util=(self.gam*np.log(np.maximum(C-self.Cmin,1e-9))).sum(0)
        self._store=dict(PL=PL,PC=PC,PA=PA,PT=PT,W=W,R=R,XST=XST,VA=VA,LD=LD,KD=KD,DS=DS,
                         EX=EX,IM=IM,C=C,INV=INV,Qd=Qd,YH=YH,YDH=YDH,SH=SH,CTH=CTH,IT=IT,GFCF=GFCF,
                         PVA=PVA,RC=RC,TICrev=TICrev,TIPrev=TIPrev,TDrev=TDrev,GVTrev=GVTrev,
                         YF=YF,SF=SF,SG=SG,SROW=SROW_imp,walras=float(walras),cpi=float(cpi),
                         Lemp=Lemp,util=util,PMRG=float(PMRG),
                         unemp=float(self.LS.sum()-LD.sum()),
                         chomage_pct=float(100*(self.LS.sum()-LD.sum())/self.LS.sum()))
        return np.concatenate([r_zp,r_com,r_L,r_K])
    def _res_log(self,u):
        return self.residual(np.exp(u))
    def solve(self,x0=None,method='lm',tol=1e-9,warn=True,log_vars=None):
        """Résolution robuste : niveaux d'abord sur départ chaud, log (positivité) sinon, replis."""
        from scipy.optimize import root
        warm=x0 is not None
        if x0 is None: x0=(self.x0_sec if self.sec_cap else self.x0).copy()
        x0=np.maximum(np.asarray(x0,float),1e-8)
        opts={'xtol':tol} if method=='lm' else {}
        best=None; bestr=np.inf
        attempts=(['lev','log','base'] if warm else ['log','lev','base'])
        for a in attempts:
            if bestr<=1e-8: break
            if a=='log':
                sol=root(self._res_log,np.log(x0),method=method,options=opts)
                xs=np.exp(sol.x)
            else:
                x0b=x0 if a=='lev' else (self.x0_sec if self.sec_cap else self.x0).copy()
                sol=root(self.residual,x0b,method=method,options=opts); xs=sol.x
            rr=float(np.max(np.abs(self.residual(xs))))
            if rr<bestr: sol.x=xs; best,bestr=sol,rr
        if warn and bestr>1e-6:
            warnings.warn(f"CGE.solve: convergence douteuse (résidu max {bestr:.1e})")
        self.residual(best.x)
        if warn and abs(self._store['walras'])>1e-4*max(abs(self.SROWo),1.0):
            warnings.warn(f"CGE.solve: loi de Walras violée ({self._store['walras']:.2f})")
        return best.x,best
    def solve_path(self,setter,x0=None,step0=0.15,tol=2e-3,max_time=600,verbose=False,method='hybr'):
        """
        Résout le modèle de manière progressive (homotopie) pour assurer la convergence 
        lors de chocs importants.
        Applique le choc par paliers successifs de 0 à 1. Gère également les changements de régime
        (ex: chômage vers plein emploi, contraintes de non-négativité sur l'investissement) 
        grâce à une approche par pivot linéaire local.
        """
        import time as _t
        from scipy.optimize import root
        t0=_t.time()
        if x0 is None:
            setter(self,0.0); x0,_=self.solve(warn=False)
        x=np.asarray(x0,float).copy()
        idle=set()                                   # cellules (k,j) à rente nulle (capital oisif)
        if self.sec_cap:                             # régime initial d'après x0
            _,_,R0,_=self.unpack(x)
            for (k,j) in zip(*np.where((R0<2e-3)&(self.KDo>1e-9))): idle.add((int(k),int(j)))
        nJ,nI,nL=self.nJ,self.nI,self.nL
        def res_piv(xx):
            r=self.residual(xx)
            if idle and self.sec_cap:
                _,_,Rm,_=self.unpack(xx)
                for (k,j) in idle: r[nJ+nI+nL+k*self.nJ+j]=(Rm[k,j]-0.0)*1.0
            return r
        cur=0.0; step=step0; nfail=0
        if verbose:
            with open('convergence_log.csv', 'w') as f:
                f.write('s,residual,walras\n')
        while cur<1.0-1e-12:
            if _t.time()-t0>max_time:
                warnings.warn("solve_path: temps maximal atteint (s=%.3f)"%cur); break
            s=min(cur+step,1.0)
            setter(self,s)
            sol=root(res_piv,x,method='hybr',options={'maxfev':getattr(self,'path_maxiter',20000), 'xtol': 1e-4})
            rr=float(np.max(np.abs(res_piv(sol.x))))
            if rr>tol and method!='lm':  # Fallback to lm if hybr fails
                sol=root(res_piv,x,method='lm',options={'maxiter':getattr(self,'path_maxiter',20000), 'xtol': 1e-4})
                rr=float(np.max(np.abs(res_piv(sol.x))))
            if rr>tol: # Try the robust log solver
                xs, best_sol = self.solve(x0=x, warn=False)
                rr_solve = float(np.max(np.abs(self.residual(xs))))
                if rr_solve < rr:
                    sol = best_sol
                    sol.x = xs
                    rr = rr_solve
            if rr<tol:
                x=sol.x; cur=s; step=min(step*1.6,step0); nfail=0
                if self.sec_cap:                      # mise à jour du régime
                    R=sol.x[nJ+nI+nL:nJ+nI+nL+self.nK*self.nJ].reshape((self.nK,self.nJ))
                    for (k,j) in zip(*np.where((R<2e-3)&(self.KDo>1e-9))): idle.add((int(k),int(j)))
                    for (k,j) in [c for c in idle if self._store['KD'][c[0],c[1]]>self.KDj[c[0],c[1]]*(1+1e-6)]:
                        idle.discard((k,j))           # rente redevient positive : cellule libérée
                if s>=1.0:
                    # Final polish to get exact solution
                    sol_polish = root(res_piv, x, method='lm', options={'xtol': 1e-8})
                    rr_solve = float(np.max(np.abs(self.residual(sol_polish.x))))
                    if rr_solve < 1e-4:  # Accept polish if it converged well
                        x = sol_polish.x
                        rr = rr_solve
                    break
                if verbose: 
                    print("   s=%.3f res=%.1e coins=%d"%(cur,rr,len(idle)))
                    w = float(self._store['walras'])
                    with open('convergence_log.csv', 'a') as f:
                        f.write(f'{cur},{rr},{w}\n')
            else:
                added=False
                if self.sec_cap:                      # pivot spéculatif : rentes écrasées dans l'essai raté
                    _,_,Rf,_=self.unpack(sol.x)
                    for (k,j) in zip(*np.where((Rf<2e-3)&(self.KDo>1e-9))):
                        if (int(k),int(j)) not in idle: idle.add((int(k),int(j))); added=True
                if not added:
                    step/=2
                nfail+=1
                if verbose:
                    w = float(self._store['walras'])
                    with open('convergence_log.csv', 'a') as f:
                        f.write(f'{cur+step},{rr},{w}\n')
                if step<1e-4 or nfail>20:
                    warnings.warn("solve_path: blocage à s=%.4f (res=%.1e)"%(cur,rr)); break
        setter(self,cur)
        rfin=float(np.max(np.abs(self.residual(x))))
        return x,dict(s=cur,res=rfin,idle=sorted(idle),ok=bool(cur>1.0-1e-9 and rfin<1e-3))

    def report(self,x):
        self.residual(x); S=self._store; d=self.d
        Kinc=(S['R']*self.KDj).sum(1) if self.sec_cap else S['R']*self.KS
        exports=float((self.pwe*S['EX']).sum()); imports=float((self.pwm*S['IM']).sum())
        cons=float((S['PC']*S['C'].sum(1)).sum()); inv=float((S['PC']*S['INV']).sum())
        gov=float((S['PC']*self.CGo_q).sum()); vstk=float((S['PC']*self.VSTKo_q).sum())
        return dict(x=x,XST=S['XST'],PL=S['PL'],PC=S['PC'],PVA=S['PVA'],W=S['W'],R=S['R'],
                    VA=S['VA'],LD=S['LD'],KD=S['KD'],EX=S['EX'],IM=S['IM'],C=S['C'],INV=S['INV'],
                    GDP_VA=float(np.nansum(S['VA'])),GFCF=float(S['GFCF']),
                    Conso=cons,Invest=inv,Gov=gov,Export=exports,Import=imports,
                    VSTK=vstk,GDP_dep=cons+inv+gov+vstk+exports-imports,
                    YH=S['YH'],YDH=S['YDH'],SH=S['SH'],CTH=S['CTH'],
                    YHL=self.shWL@(S['W']*S['Lemp']),YHK=self.shK_H@Kinc,
                    YF=S['YF'],SF=S['SF'],SG=float(S['SG']),SROW=float(S['SROW']),
                    walras=S['walras'],cpi=S['cpi'],chomage_pct=S['chomage_pct'],
                    GVTrev=S['GVTrev'],TICrev=S['TICrev'],TIPrev=S['TIPrev'],TDrev=S['TDrev'],
                    util=S['util'])

    def solve_dynamic(self, T=10, delta=0.03, n=0.03, shock=None, verbose=False, sigma_inv=1.0, warm_path=None):
        """Dynamique récursive T périodes (capital mobile ou sectoriel, allocation par rendements)."""
        path=[]; x=(self.x0_sec if self.sec_cap else self.x0).copy()
        LS=self.LSo.copy(); KS=self.KSo.copy(); KDj=self.KDo.copy()
        prevLS,prevKS,prevKD=LS.copy(),KS.copy(),KDj.copy()
        for t in range(T):
            self.LS=LS.copy(); self.KS=KS.copy()
            if self.sec_cap and t>0:
                gj=KDj.sum(0)/np.maximum(self.KDj.sum(0),1e-9)
                nI=self.nI
                x=x.copy(); x[nI+self.nL+self.nK*self.nJ:]*=gj
            self.KDj=KDj.copy()
            self.tic_sim=self.tic.copy()
            if shock is not None: shock(self,t)
            x0t=warm_path[t] if (warm_path is not None and t<len(warm_path)) else x
            if t==0:
                x,sol=self.solve(x0=x0t,warn=False)
            else:
                # homotopie directe sur l'accumulation des facteurs (rapide et robuste)
                LS1,KS1,KD1=self.LS.copy(),self.KS.copy(),self.KDj.copy()
                pLS,pKS,pKD=prevLS.copy(),prevKS.copy(),prevKD.copy()
                def _blend(mm,s):
                    mm.LS=pLS+(LS1-pLS)*s; mm.KS=pKS+(KS1-pKS)*s; mm.KDj=pKD+(KD1-pKD)*s
                x,info=self.solve_path(_blend,x0=x0t,step0=1.0,max_time=120,verbose=verbose)
            prevLS,prevKS,prevKD=self.LS.copy(),self.KS.copy(),self.KDj.copy()
            rep=self.report(x); rep['t']=t; rep['KS']=KS.copy(); rep['LS']=LS.copy(); rep['KDj']=KDj.copy(); rep['x']=x.copy()
            res=float(np.max(np.abs(self.residual(x)))); rep['res']=res
            rep['conv']=bool(res < 1e-5)
            if not rep['conv']: warnings.warn(f"dynamique t={t}: résidu {res:.1e}")
            self.residual(x); Rj=self._store['R']; rep['R']=Rj
            path.append(rep)
            if t == 0:
                # Calcul du multiplicateur d'investissement à l'état stationnaire
                # phi_inv calibre I_0 pour que K croisse au rythme n et compense la sous-évaluation du capital
                PC0 = rep['PC']; Pinv0 = float((self.gamINV * PC0).sum())
                Ireal_0 = rep['GFCF'] / max(Pinv0, 1e-6)
                self.phi_inv = KS.sum() * (delta + n) / max(Ireal_0, 1e-6)
                
            PC = rep['PC']; Pinv = float((self.gamINV * PC).sum())
            Ireal = rep['GFCF'] / max(Pinv, 1e-6)
            Ireal_units = Ireal * self.phi_inv
            
            if self.sec_cap:
                for k in range(self.nK):
                    Rk = Rj[k]; Kk = KDj[k]
                    Ik_tot = Ireal_units * (Kk.sum() / max(KDj.sum(), 1e-9))
                    Rbar = (Rk * Kk).sum() / max(Kk.sum(), 1e-9)
                    w = Kk * np.where(Rbar > 0, (np.maximum(Rk, 1e-6) / Rbar)**sigma_inv, 1.0)
                    w = w / max(w.sum(), 1e-9)
                    KDj[k] = Kk * (1 - delta) + Ik_tot * w
                KS = KDj.sum(1)
            else:
                kshare = KS / max(KS.sum(), 1e-9)
                KS = KS * (1 - delta) + kshare * Ireal_units
            LS = LS * (1 + n)
            if verbose: print(f"    t={t}: PIB(VA)={rep['GDP_VA']/1e3:.0f} Mds  K={KS.sum()/1e3:.0f}  résidu={res:.0e} ({'OK' if rep['conv'] else '~'})")
        return path

def run_demo():
    print("=== MEGC Burkina Faso 2018 - Python (version réparée) ===")
    m=CGE(); r=m.residual(m.x0_sec if m.sec_cap else m.x0)
    print("[0] Résidu année de base: max=%.1e | Walras=%.2e"%(np.max(np.abs(r)),m._store['walras']))
    print("[1] Équilibre statique de référence (BAU):")
    xb,sol=m.solve(); rb=m.report(xb)
    print("    convergence=%s  PIB(VA)=%.0f Mds  Walras=%.2e"%(sol.success,rb['GDP_VA']/1e3,rb['walras']))
    print("[2] Choc fiscal +10 pts taxe agroalimentaire (homotopie):")
    mf=CGE(); food=[i for i,c in enumerate(mf.d.I) if 62<=int(''.join(filter(str.isdigit,c)))<=97]
    tic0=mf.tic.copy()
    def setter(mm,s): mm.tic_sim=tic0.copy(); mm.tic_sim[food]+=0.10*s
    xf,info=mf.solve_path(setter,x0=xb,verbose=True); rf=mf.report(xf)
    print("    ΔRecettes=%.0f Mds  ΔSG=%.0f Mds  Walras=%.2e  (s=%.2f res=%.1e)"%(
        (rf['GVTrev']-rb['GVTrev'])/1e3,(rf['SG']-rb['SG'])/1e3,rf['walras'],info['s'],info['res']))
    print("[3] Dynamique récursive BAU (3 périodes, ~minutes — augmenter T selon besoin):")
    md=CGE(); path=md.solve_dynamic(T=3,verbose=True)
    g=[p['GDP_VA'] for p in path]
    print("    Croissance PIB(VA) cumulée: %.1f%%"%(100*(g[-1]/g[0]-1)))

if __name__=="__main__":
    run_demo()
