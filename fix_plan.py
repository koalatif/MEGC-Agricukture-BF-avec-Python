# Plan pour corriger la clôture exogène et la robustesse

# 1. Correction analytique exacte de l'épargne pour closure_inv == 'exogene'
# Dans cge.py, ligne 186 :
"""
        if self.closure_inv=='exogene':
            INV=self.INVo_q.copy()
            # On calcule l'investissement requis
            IT_req = (PC*INV).sum() + (PC*self.VSTKo_q).sum()
            
            # Pour boucler, il faut SH_req = IT_req - SF - SG - SROWo - LEV
            # Or SG dépend de la consommation C, qui dépend de SH.
            # CTH = YDH - adj_sh * shh * YDH
            # dC / d(adj_sh) = - gam * (shh*YDH).sum() / PC
            # dQd / d(adj_sh) = dC + shmg * (tmg * dC).sum()
            # dTICrev / d(adj_sh) = (tic_sim * PA * dQd).sum()
            # dSG / d(adj_sh) = dTICrev
            
            # Étape 1 : Calculer le point de base (adj_sh = 0)
            CTH0 = YDH
            realb0 = CTH0 - (PC[:,None]*self.Cmin).sum(0)
            C0 = self.Cmin + self.gam*realb0[None,:]/PC[:,None]
            Qd0 = DIsum + C0.sum(1) + self.CGo_q + INV + self.VSTKo_q
            Qd0 = Qd0 + self.shmg*((self.tmg*Qd0).sum())
            TIC0, TIPrev, TDrev = gvt_rev(Qd0, PA)
            SG0 = TIC0 + TIPrev + TDrev + self.shK_G@Kinc + self.trRecvG - (PC*self.CGo_q).sum() - self.trPaidG
            
            # Étape 2 : Calculer la dérivée (dSG / d(adj_sh))
            dSH_total = (self.shh * YDH).sum()
            dC = - self.gam * dSH_total / PC[:,None]
            dQd = dC.sum(1) + self.shmg*((self.tmg*dC.sum(1)).sum())
            dTIC = (self.tic_sim * PA * dQd).sum()
            dSG = dTIC  # car TIPrev et TDrev ne changent pas
            
            # Étape 3 : Résoudre adj_sh
            # SH = adj_sh * dSH_total
            # SG = SG0 + adj_sh * dSG
            # IT_req = SH + SF.sum() + SG + self.SROWo + (1.0-self.sh_lev_gvt)*LEV
            # adj_sh * (dSH_total + dSG) = IT_req - SF.sum() - SG0 - SROWo - LEV
            
            num = IT_req - SF.sum() - SG0 - self.SROWo - (1.0-self.sh_lev_gvt)*LEV
            den = dSH_total + dSG
            adj_sh = num / max(den, 1e-9)
            
            # Appliquer l'ajustement
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
            GFCF = (IT - (PC*self.VSTKo_q).sum()) # pour cohérence
"""

# 2. Pour T3 et T5 :
# Le blocage est dû aux bornes de Fisher-Burmeister.
# Nous allons remplacer la résolution d'homotopie (solve_path) par un appel à solve()
# avec une suite d'essais 'log', 'lev', qui force la positivité des quantités (XST > 0)
# et évite que XST prenne une valeur non nulle quand le profit est très négatif.
