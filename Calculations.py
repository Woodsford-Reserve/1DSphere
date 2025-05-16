# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:17:24 2025

@author: woods
"""

#from SLD_DD import Solve
from LDQ_DD import Solve
#from Test import Solve

import numpy as np
import matplotlib.pyplot as plt


I_reg = np.array([100])

"""
################
# source problem
################

R = np.array([1.])

bc = {"type":"isotropic","value":0.}

matprops = {"sigt":np.array([1.0]),
            "sigs":np.array([0.0]),
               "q":np.array([1.0])}

sol1 = Solve(R, I_reg, 8, bc, matprops, do_angular=False)
leak1 = sol1.solve()

sol2 = Solve(R, I_reg, 16, bc, matprops, do_angular=False)
leak2 = sol2.solve()

sol3 = Solve(R, I_reg, 32, bc, matprops, do_angular=False)
leak3 = sol3.solve()

sol4 = Solve(R, I_reg, 64, bc, matprops, do_angular=False)
leak4 = sol4.solve()

sol5 = Solve(R, I_reg, 128, bc, matprops, do_angular=False)
leak5 = sol5.solve()

sol6 = Solve(R, I_reg, 256, bc, matprops, do_angular=False)
leak6 = sol6.solve()

sol7 = Solve(R, I_reg, 512, bc, matprops, do_angular=False)
leak7 = sol7.solve()

sol8 = Solve(R, I_reg, 1024, bc, matprops, do_angular=False)
leak8 = sol8.solve()

sol9 = Solve(R, I_reg, 2048, bc, matprops, do_angular=False)
leak9 = sol9.solve()

sol10 = Solve(R, I_reg, 4096, bc, matprops, do_angular=False)
leak10 = sol10.solve()


#####################
print("\nLeakages")
#####################

print(8, leak1)
print(16, leak2)
print(32, leak3)
print(64, leak4)
print(128, leak5)
print(256, leak6)
print(512, leak7)
print(1024, leak8)
print("'Exact':", leak10)


#########################
print("\n3 Point Ratios")
#########################

order = np.linalg.norm(sol1.Phi - sol2.Phi) / np.linalg.norm(sol2.Phi - sol3.Phi)
print(8, order)

order = np.linalg.norm(sol2.Phi - sol3.Phi) / np.linalg.norm(sol3.Phi - sol4.Phi)
print(16, order)

order = np.linalg.norm(sol3.Phi - sol4.Phi) / np.linalg.norm(sol4.Phi - sol5.Phi)
print(32, order)

order = np.linalg.norm(sol4.Phi - sol5.Phi) / np.linalg.norm(sol5.Phi - sol6.Phi)
print(64, order)

order = np.linalg.norm(sol5.Phi - sol6.Phi) / np.linalg.norm(sol6.Phi - sol7.Phi)
print(128, order)

order = np.linalg.norm(sol6.Phi - sol7.Phi) / np.linalg.norm(sol7.Phi - sol8.Phi)
print(256, order)

order = np.linalg.norm(sol7.Phi - sol8.Phi) / np.linalg.norm(sol8.Phi - sol9.Phi)
print(512, order)

order = np.linalg.norm(sol8.Phi - sol9.Phi) / np.linalg.norm(sol9.Phi - sol10.Phi)
print(1024, order)
"""


#######################
# incident flux problem
#######################

R = np.array([1.])

bc = {"type":"isotropic","value":1.}

matprops = {"sigt":np.array([1.0]),
            "sigs":np.array([0.0]),
               "q":np.array([0.0])}

sol1 = Solve(R, I_reg, 8, bc, matprops, do_angular=False)
sol1.solve()

sol2 = Solve(R, I_reg, 16, bc, matprops, do_angular=False)
sol2.solve()

sol3 = Solve(R, I_reg, 32, bc, matprops, do_angular=False)
sol3.solve()

sol4 = Solve(R, I_reg, 64, bc, matprops, do_angular=False)
sol4.solve()

sol5 = Solve(R, I_reg, 128, bc, matprops, do_angular=False)
sol5.solve()

sol6 = Solve(R, I_reg, 256, bc, matprops, do_angular=False)
sol6.solve()

sol7 = Solve(R, I_reg, 512, bc, matprops, do_angular=False)
sol7.solve()

sol8 = Solve(R, I_reg, 1024, bc, matprops, do_angular=False)
sol8.solve()

sol9 = Solve(R, I_reg, 2048, bc, matprops, do_angular=True)
sol9.solve()

Phi = sol9.exact(4096, plot=False, error=True)


#################
print("\nErrors")
#################

error = np.linalg.norm(sol1.Phi - Phi)
print(8, error)

error = np.linalg.norm(sol2.Phi - Phi)
print(16, error)

error = np.linalg.norm(sol3.Phi - Phi)
print(32, error)

error = np.linalg.norm(sol4.Phi - Phi)
print(64, error)

error = np.linalg.norm(sol5.Phi - Phi)
print(128, error)

error = np.linalg.norm(sol6.Phi - Phi)
print(256, error)

error = np.linalg.norm(sol7.Phi - Phi)
print(512, error)

error = np.linalg.norm(sol8.Phi - Phi)
print(1024, error)

error = np.linalg.norm(sol9.Phi - Phi)
print(2048, error)