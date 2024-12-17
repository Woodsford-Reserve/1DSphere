# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:51:46 2024

@author: camde
"""

import numpy as np
import math
import matplotlib.pyplot as plt

'''
0 = All negatives same value
1 = First 2 negative directions are value
2 = gradual decrease (epsilon)
'''


# returns psi(mu)
def getPsi(mu, value, sol):
    psi = 0.0
    if sol == 1:
        psi = 1 - ((1 + mu) / value)
        if psi < 0:
            psi = 0
    if sol == 2:
        if mu < -0.75:
            psi = 1
    if sol == 3:
        psi = value / 2
    if sol == 4:
        norm = abs(1 / (-0.5 * math.log(1 + value) + 0.5 * math.log(value) - 1 / (2 + 2 * value)))
        psi = norm / (1 + value - mu ** 2) - norm / (1 + value)
    return psi

# makes the Gauss S2 quadrature for mu with n directions
def getMu(n):
    # Gauss S2 quadrature
    n2 = int(n / 2)
    w = (1 / n2) * np.ones((n2,2)) 
    mu_half = np.linspace(-1, 1, n2 + 1)
    mu1 = 0.5*(mu_half[:-1] + mu_half[1:])
    mu2 = np.zeros((n2, 2))
    mu2[:,0] = w[:,0]*(-1./np.sqrt(3.)) + mu1[:]
    mu2[:,1] = w[:,1]*(1./np.sqrt(3.)) + mu1[:]
    
    mu2 = mu2.flatten()
    w = w.flatten()
        
    return mu2

def AnalyticalSolution(n, num, siga, mu, value):
    dr = 1 / num / 2
    dmu = 2 / n
    R = np.linspace(dr, 1 - dr, num)
    R0 = 1
    psi = np.zeros((n + 1, num))
    phi = np.zeros(num)
    theta = np.linspace(0,math.pi,(n+1))
    print("Theta1 \t\t\t d \t\t\t x \t\t\t (d^2 - x) / x")
    for i in range(num):
        for j in range(n+1):
            '''
            if i == 0:
                R[i] = 0
            if i == (num - 1):
                R[i] = 1
            '''
            # theta1 = math.acos(mu[j])
            theta1 = theta[j]
            theta2 = math.pi - math.asin(R[i] / R0 * math.sin(theta1))
            theta3 = theta2 - theta1
            y = R0 / R[i] * math.sin(math.pi - theta2)
            x2 = math.sin(theta1)
            d = np.sqrt(R[i] ** 2 + R0 ** 2 - 2 * R[i] * R0 * math.cos(theta3))
            x = R[i] ** 2 + R0 ** 2 - 2 * R[i] * R0 * math.cos(theta3)
            val = (d ** 2 - x) / x
            val2 = (y - x2)
            psi[j][i] = getPsi(math.cos(theta2), value, 3) * np.exp(-1 * siga * d)
            phi[i] += psi[j][i] * dmu
            if i == 0 or i == (num - 1):
                print(theta1, theta2, theta3, d)
        if i == 0:
            print()
            print("Theta1 \t\t\t d \t\t\t x \t\t\t (d^2 - x) / x")
    return phi, R, psi


      
n = 40
num = 1000

mu = getMu(n)

value = .1

phi, R, psi = AnalyticalSolution(n, num, 1, mu, value)

plt.plot(R, phi)
plt.xlabel("r (cm)")
plt.ylabel("Flux")
plt.title("1D Spherical Analytical Solution")