# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 14:55:26 2025

@author: camde
"""

import numpy as np
import math

N = 1000
phi_0 = np.zeros(N)
phi_1 = np.zeros(N)
p = 0.2

err = 1
tol = 1e-6
it = 0
while err > tol:
    it += 1
    for i in range(N):
        if i == 0:
            phi_1[0] = 1
        elif i == 1:
            phi_1[1] = phi_1[0]/(1-p) - p*phi_0[1]/(1-p)
        else:
            phi_1[i] = phi_1[i-1]/(1-p) - p*phi_0[i]/(1-p)
    if np.sum(phi_1**2) != math.inf:
        err = np.sqrt(np.sum(np.abs(phi_1**2 - phi_0**2)) / np.sum(phi_1**2))
    else:
        err = 1
    phi_0 = np.copy(phi_1)

print("Total # of Iterations:", it)