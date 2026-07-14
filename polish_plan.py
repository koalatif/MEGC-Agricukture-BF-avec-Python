# Patching cge.py and validation_finale.py

# In cge.py, solve_path should polish the final solution at s=1.0:
'''
        if s == 1.0 and rr > 1e-8:
            xs, sol_final = self.solve(x0=x, warn=False)
            if float(np.max(np.abs(self.residual(xs)))) < rr:
                x = xs
                rr = float(np.max(np.abs(self.residual(x))))
'''

# In validation_finale.py, T3 uses root directly. I will change it to use m2.solve().
'''
# T3 - Choc fiscal +10 pts agroalimentaire : recyclage budgétaire + Walras
...
x2, _ = m2.solve(x0=xb, warn=False)
r2 = m2.report(x2)
res2 = np.max(np.abs(m2.residual(x2)))
'''

