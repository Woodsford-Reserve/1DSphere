# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 20:29:37 2024

@author: woods
"""

import numpy as np
import matplotlib.pyplot as plt
                
        
# quadrature class
class Quad:
    def __init__(self, quad_dict):
        # number of mu-cells
        self.N_dir   = quad_dict["directions"]
        self.N_cells = int(self.N_dir/2)
        
        # mu-cell boundaries
        self.mu_half = np.linspace(-1.,1.,self.N_dir+1)
        
        # quadrature weights
        self.w  = (1./self.N_cells)*np.ones(self.N_dir)
        
        # local Gauss S2 quadrature points
        if quad_dict["quadrature"] == "gauss":
            self.mu = np.zeros(self.N_dir) 
            for n_mu in range(self.N_cells):
                self.mu[2*n_mu]   = self.w[2*n_mu]*(-1./np.sqrt(3.))  + self.mu_half[2*n_mu+1]
                self.mu[2*n_mu+1] = self.w[2*n_mu+1]*(1./np.sqrt(3.)) + self.mu_half[2*n_mu+1]
            
        # midpoint quadrature points
        if quad_dict["quadrature"] == "midpoint":
            self.mu = 0.5*(self.mu_half[:-1]+self.mu_half[1:])
        
        # exact alpha (1-mu^2)
        if quad_dict["alpha"] == "exact":
            self.alpha = 1 - self.mu_half**2
        
        # approximate alpha (1-mu^2)
        if quad_dict["alpha"] == "approximate":
            self.alpha = np.zeros(self.N_dir+1)
            self.alpha[0] = 0
            for n_mu in range(self.N_cells):
                self.alpha[2*n_mu+1] = self.alpha[2*n_mu]   - 2*self.mu[2*n_mu]*self.w[2*n_mu]
                self.alpha[2*n_mu+2] = self.alpha[2*n_mu+1] - 2*self.mu[2*n_mu+1]*self.w[2*n_mu+1] 
                
        # beta
        self.beta = np.zeros(self.N_dir)
        for nd in range(self.N_dir):
            self.beta[nd] = (self.mu[nd] - self.mu_half[nd])/\
                            (self.mu_half[nd+1] - self.mu_half[nd])
        
        
        
# solver class
class Spectral_Radius:
    def __init__(self, quad_dict):
        # initialization
        self.quad = Quad(quad_dict)
        
        
    # solver 
    def eigenvalues(self, sig=1e-8, do_quadratic=True, do_plot=False, do_last_cell=False):
        
        ########################
        ########################
        print("\nStep (Half):")
        ########################
        ########################
        
        # spectral radii
        rho_step_half = np.zeros(self.quad.N_cells)
        
        # loop over cells
        for n_mu in range(self.quad.N_cells):
            # direction quantities
            mu      = self.quad.mu[2*n_mu:2*n_mu+2]
            alpha   = self.quad.alpha[2*n_mu+1:2*n_mu+3]
            mu_m    = self.quad.mu_half[2*n_mu+1]
            mu_half = self.quad.mu_half[2*n_mu+2]
            
            # basis functions
            if do_quadratic and (n_mu == 0):
                B_minus = lambda u: ((u+1)*(u-mu[1]))/((mu[0]+1)*(mu[0]-mu[1]))
                B_plus  = lambda u: ((u+1)*(u-mu[0]))/((mu[1]+1)*(mu[1]-mu[0]))              
            else:
                B_minus = lambda u: (mu[1]-u)/(mu[1]-mu[0])
                B_plus  = lambda u: (u-mu[0])/(mu[1]-mu[0])
            
            # step matrix
            A = np.zeros((2,2))
            A[0,0] = 1
            A[0,1] = 0
            A[1,0] = alpha[1]*B_minus(mu_half) - alpha[0]*B_minus(mu_m)
            A[1,1] = alpha[1]*B_plus(mu_half)  - alpha[0]*B_plus(mu_m)
            
            # Petrov-Galerkin matrix
            B = np.zeros((2,2))
            B[0,0] = B_minus(mu_m)
            B[0,1] = B_plus(mu_m)
            B[1,0] = alpha[1]*B_minus(mu_half) - alpha[0]*B_minus(mu_m)
            B[1,1] = alpha[1]*B_plus(mu_half)  - alpha[0]*B_plus(mu_m)
                       
            # eigenvalues
            C = np.linalg.inv(A + sig*np.eye(2)) @ (A - B)
            eigvals = np.linalg.eigvals(C)
            rho_step_half[n_mu] = np.max(np.abs(eigvals))
            
            # print eigenvalues
            if (n_mu == 0) or (n_mu == int(self.quad.N_cells / 2) - 1) or \
                    (n_mu == int(self.quad.N_cells / 2)) or (n_mu == self.quad.N_cells-2):
                print(n_mu+1, "/", self.quad.N_cells, ":", eigvals)
                
                
        ########################
        ########################
        print("\nStep (Full):")
        ########################
        ########################
        
        # spectral radii
        rho_step_full = np.zeros(self.quad.N_cells)
        
        # loop over cells
        for n_mu in range(self.quad.N_cells-1):
            # direction quantities
            mu      = self.quad.mu[2*n_mu:2*n_mu+2]
            alpha   = self.quad.alpha[2*n_mu+1:2*n_mu+3]
            mu_m    = self.quad.mu_half[2*n_mu+1]
            mu_half = self.quad.mu_half[2*n_mu+2]
            
            # basis functions
            if do_quadratic and (n_mu == 0):
                B_minus = lambda u: ((u+1)*(u-mu[1]))/((mu[0]+1)*(mu[0]-mu[1]))
                B_plus  = lambda u: ((u+1)*(u-mu[0]))/((mu[1]+1)*(mu[1]-mu[0]))              
            else:
                B_minus = lambda u: (mu[1]-u)/(mu[1]-mu[0])
                B_plus  = lambda u: (u-mu[0])/(mu[1]-mu[0])
            
            # step matrix
            A = np.zeros((2,2))
            A[0,0] = 1
            A[0,1] = 0
            A[1,0] = -alpha[0]
            A[1,1] =  alpha[1]
            
            # Petrov-Galerkin matrix
            B = np.zeros((2,2))
            B[0,0] = B_minus(mu_m)
            B[0,1] = B_plus(mu_m)
            B[1,0] = alpha[1]*B_minus(mu_half) - alpha[0]*B_minus(mu_m)
            B[1,1] = alpha[1]*B_plus(mu_half)  - alpha[0]*B_plus(mu_m)
                       
            # eigenvalues
            C = np.linalg.inv(A + sig*np.eye(2)) @ (A - B)
            eigvals = np.linalg.eigvals(C)
            rho_step_full[n_mu] = np.max(np.abs(eigvals))
            
            # print eigenvalues
            if (n_mu == 0) or (n_mu == int(self.quad.N_cells / 2) - 1) or \
                    (n_mu == int(self.quad.N_cells / 2)) or (n_mu == self.quad.N_cells-2):
                print(n_mu+1, "/", self.quad.N_cells, ":", eigvals)
                
        # last cell is not invertible
        rho_step_full[-1] = 1
                
                
        ########################
        ########################
        print("\nDiamond (Half):")
        ########################
        ########################
        
        # spectral radii
        rho_DD_half = np.zeros(self.quad.N_cells)
        
        # loop over cells
        for n_mu in range(self.quad.N_cells):
            # direction quantities
            mu      = self.quad.mu[2*n_mu:2*n_mu+2]
            alpha   = self.quad.alpha[2*n_mu+1:2*n_mu+3]
            mu_m    = self.quad.mu_half[2*n_mu+1]
            mu_half = self.quad.mu_half[2*n_mu+2]
            
            # basis functions
            if do_quadratic and (n_mu == 0):
                B_minus = lambda u: ((u+1)*(u-mu[1]))/((mu[0]+1)*(mu[0]-mu[1]))
                B_plus  = lambda u: ((u+1)*(u-mu[0]))/((mu[1]+1)*(mu[1]-mu[0]))              
            else:
                B_minus = lambda u: (mu[1]-u)/(mu[1]-mu[0])
                B_plus  = lambda u: (u-mu[0])/(mu[1]-mu[0])
            
            # step matrix
            A = np.zeros((2,2))
            A[0,0] = 2
            A[0,1] = 0
            A[1,0] = alpha[1]*B_minus(mu_half) - alpha[0]*B_minus(mu_m)
            A[1,1] = alpha[1]*B_plus(mu_half)  - alpha[0]*B_plus(mu_m)
            
            # Petrov-Galerkin matrix
            B = np.zeros((2,2))
            B[0,0] = B_minus(mu_m)
            B[0,1] = B_plus(mu_m)
            B[1,0] = alpha[1]*B_minus(mu_half) - alpha[0]*B_minus(mu_m)
            B[1,1] = alpha[1]*B_plus(mu_half)  - alpha[0]*B_plus(mu_m)
                       
            # eigenvalues
            C = np.linalg.inv(A + sig*np.eye(2)) @ (A - B)
            eigvals = np.linalg.eigvals(C)
            rho_DD_half[n_mu] = np.max(np.abs(eigvals))
            
            # print eigenvalues
            if (n_mu == 0) or (n_mu == int(self.quad.N_cells / 2) - 1) or \
                    (n_mu == int(self.quad.N_cells / 2)) or (n_mu == self.quad.N_cells-2):
                print(n_mu+1, "/", self.quad.N_cells, ":", eigvals)
                
                
        ########################
        ########################
        print("\nDiamond (Full):")
        ########################
        ########################
        
        # spectral radii
        rho_DD_full = np.zeros(self.quad.N_cells)
        
        # loop over cells
        for n_mu in range(self.quad.N_cells-1):
            # direction quantities
            mu      = self.quad.mu[2*n_mu:2*n_mu+2]
            alpha   = self.quad.alpha[2*n_mu+1:2*n_mu+3]
            mu_m    = self.quad.mu_half[2*n_mu+1]
            mu_half = self.quad.mu_half[2*n_mu+2]
            
            # basis functions
            if do_quadratic and (n_mu == 0):
                B_minus = lambda u: ((u+1)*(u-mu[1]))/((mu[0]+1)*(mu[0]-mu[1]))
                B_plus  = lambda u: ((u+1)*(u-mu[0]))/((mu[1]+1)*(mu[1]-mu[0]))              
            else:
                B_minus = lambda u: (mu[1]-u)/(mu[1]-mu[0])
                B_plus  = lambda u: (u-mu[0])/(mu[1]-mu[0])
            
            # step matrix
            A = np.zeros((2,2))
            A[0,0] = 2
            A[0,1] = 0
            A[1,0] = -2*(alpha[0] + alpha[1])
            A[1,1] = 2*alpha[1]
            
            # Petrov-Galerkin matrix
            B = np.zeros((2,2))
            B[0,0] = B_minus(mu_m)
            B[0,1] = B_plus(mu_m)
            B[1,0] = alpha[1]*B_minus(mu_half) - alpha[0]*B_minus(mu_m)
            B[1,1] = alpha[1]*B_plus(mu_half)  - alpha[0]*B_plus(mu_m)
                       
            # eigenvalues
            C = np.linalg.inv(A + sig*np.eye(2)) @ (A - B)
            eigvals = np.linalg.eigvals(C)
            rho_DD_full[n_mu] = np.max(np.abs(eigvals))
            
            # print eigenvalues
            if (n_mu == 0) or (n_mu == int(self.quad.N_cells / 2) - 1) or \
                    (n_mu == int(self.quad.N_cells / 2)) or (n_mu == self.quad.N_cells-2):
                print(n_mu+1, "/", self.quad.N_cells, ":", eigvals)
                
        # last cell is not invertible
        rho_DD_full[-1] = 1
                
                
        ###################################
        ###################################
        print("\nWeighted Diamond (Half):")
        ###################################
        ###################################
        
        # spectral radii
        rho_WD_half = np.zeros(self.quad.N_cells)
        
        # loop over cells
        for n_mu in range(self.quad.N_cells):
            # direction quantities
            mu      = self.quad.mu[2*n_mu:2*n_mu+2]
            alpha   = self.quad.alpha[2*n_mu+1:2*n_mu+3]
            mu_m    = self.quad.mu_half[2*n_mu+1]
            mu_half = self.quad.mu_half[2*n_mu+2]
            beta    = self.quad.beta[2*n_mu]
            
            # basis functions
            if do_quadratic and (n_mu == 0):
                B_minus = lambda u: ((u+1)*(u-mu[1]))/((mu[0]+1)*(mu[0]-mu[1]))
                B_plus  = lambda u: ((u+1)*(u-mu[0]))/((mu[1]+1)*(mu[1]-mu[0]))              
            else:
                B_minus = lambda u: (mu[1]-u)/(mu[1]-mu[0])
                B_plus  = lambda u: (u-mu[0])/(mu[1]-mu[0])
            
            # step matrix
            A = np.zeros((2,2))
            A[0,0] = 1 / beta
            A[0,1] = 0
            A[1,0] = alpha[1]*B_minus(mu_half) - alpha[0]*B_minus(mu_m)
            A[1,1] = alpha[1]*B_plus(mu_half)  - alpha[0]*B_plus(mu_m)
            
            # Petrov-Galerkin matrix
            B = np.zeros((2,2))
            B[0,0] = B_minus(mu_m)
            B[0,1] = B_plus(mu_m)
            B[1,0] = alpha[1]*B_minus(mu_half) - alpha[0]*B_minus(mu_m)
            B[1,1] = alpha[1]*B_plus(mu_half)  - alpha[0]*B_plus(mu_m)
                       
            # eigenvalues
            C = np.linalg.inv(A + sig*np.eye(2)) @ (A - B)
            eigvals = np.linalg.eigvals(C)
            rho_WD_half[n_mu] = np.max(np.abs(eigvals))
            
            # print eigenvalues
            if (n_mu == 0) or (n_mu == int(self.quad.N_cells / 2) - 1) or \
                    (n_mu == int(self.quad.N_cells / 2)) or (n_mu == self.quad.N_cells-2):
                print(n_mu+1, "/", self.quad.N_cells, ":", eigvals)
                
                
        ###################################
        ###################################
        print("\nWeighted Diamond (Full):")
        ###################################
        ###################################
        
        # spectral radii
        rho_WD_full = np.zeros(self.quad.N_cells)
        
        # loop over cells
        for n_mu in range(self.quad.N_cells-1):
            # direction quantities
            mu      = self.quad.mu[2*n_mu:2*n_mu+2]
            alpha   = self.quad.alpha[2*n_mu+1:2*n_mu+3]
            mu_m    = self.quad.mu_half[2*n_mu+1]
            mu_half = self.quad.mu_half[2*n_mu+2]
            beta    = self.quad.beta[2*n_mu:2*n_mu+2]
            
            # basis functions
            if do_quadratic and (n_mu == 0):
                B_minus = lambda u: ((u+1)*(u-mu[1]))/((mu[0]+1)*(mu[0]-mu[1]))
                B_plus  = lambda u: ((u+1)*(u-mu[0]))/((mu[1]+1)*(mu[1]-mu[0]))              
            else:
                B_minus = lambda u: (mu[1]-u)/(mu[1]-mu[0])
                B_plus  = lambda u: (u-mu[0])/(mu[1]-mu[0])
            
            # step matrix
            A = np.zeros((2,2))
            A[0,0] = 1 / beta[0]
            A[0,1] = 0
            A[1,0] = -(1/beta[0])*alpha[0] - (1-beta[1])/(beta[0]*beta[1])*alpha[1]
            A[1,1] = (1/beta[1])*alpha[1]
            
            # Petrov-Galerkin matrix
            B = np.zeros((2,2))
            B[0,0] = B_minus(mu_m)
            B[0,1] = B_plus(mu_m)
            B[1,0] = alpha[1]*B_minus(mu_half) - alpha[0]*B_minus(mu_m)
            B[1,1] = alpha[1]*B_plus(mu_half)  - alpha[0]*B_plus(mu_m)
                       
            # eigenvalues
            C = np.linalg.inv(A + sig*np.eye(2)) @ (A - B)
            eigvals = np.linalg.eigvals(C)
            rho_WD_full[n_mu] = np.max(np.abs(eigvals))
            
            # print eigenvalues
            if (n_mu == 0) or (n_mu == int(self.quad.N_cells / 2) - 1) or \
                    (n_mu == int(self.quad.N_cells / 2)) or (n_mu == self.quad.N_cells-2):
                print(n_mu+1, "/", self.quad.N_cells, ":", eigvals)
                
        # last cell is not invertible
        rho_WD_full[-1] = 1
                
        # plots
        if do_plot:
            # mu
            mu = np.linspace(-1,1,self.quad.N_cells+1)
            mu = 0.5*(mu[:-1] + mu[1:])
            
            # step
            plt.figure(1)
            plt.semilogy(mu[:-1], rho_step_half[:-1], 'k')
            plt.semilogy(mu[:-1], rho_step_full[:-1], 'r--')
            plt.xlabel('\u03BC')
            plt.ylabel('\u03C1')
            plt.title("Step Spectral Radii ({:n} Angular Cells)".format(self.quad.N_cells))
            plt.legend(["Half","Full"])
            
            # diamond
            plt.figure(2)
            plt.semilogy(mu[:-1], rho_DD_half[:-1], 'k')
            plt.semilogy(mu[:-1], rho_DD_full[:-1], 'r--')
            plt.xlabel('\u03BC')
            plt.ylabel('\u03C1')
            plt.title("Diamond Spectral Radii ({:n} Angular Cells)".format(self.quad.N_cells))
            plt.legend(["Half","Full"])
            
            # weighted diamond
            plt.figure(3)
            plt.semilogy(mu[:-1], rho_WD_half[:-1], 'k')
            plt.semilogy(mu[:-1], rho_WD_full[:-1], 'r--')
            plt.xlabel('\u03BC')
            plt.ylabel('\u03C1')
            plt.title("Weighted Diamond Spectral Radii ({:n} Angular Cells)".format(self.quad.N_cells))
            plt.legend(["Half","Full"])
         
            
         
        # mu = -1
        print()
        ind = np.where(rho_DD_half - rho_step_half < 0.)[0]
        print("\n\nHalf-cell diamond difference outperforms half-cell step\n",
              "in the following cells:", ind+1)
        
        # mu = 1
        ind = np.where(rho_step_full - rho_step_half < 0.)[0]
        print("\nFull-cell step outperforms half-cell step\n",
              "in the following cells:", ind+1)
        
        
        
        # final cell
        if do_last_cell:
            n_mu = self.quad.N_cells-1
            
            # direction quantities
            mu      = self.quad.mu[2*n_mu:2*n_mu+2]
            alpha   = self.quad.alpha[2*n_mu+1:2*n_mu+3]
            mu_m    = self.quad.mu_half[2*n_mu+1]
            mu_half = self.quad.mu_half[2*n_mu+2]
            beta    = self.quad.beta[2*n_mu:2*n_mu+2]
            
            # basis functions
            B_minus = lambda u: (mu[1]-u)/(mu[1]-mu[0])
            B_plus  = lambda u: (u-mu[0])/(mu[1]-mu[0])
            
            # Petrov-Galerkin matrix
            B = np.zeros((2,2))
            B[0,0] = B_minus(mu_m)
            B[0,1] = B_plus(mu_m)
            B[1,0] = alpha[1]*B_minus(mu_half) - alpha[0]*B_minus(mu_m)
            B[1,1] = alpha[1]*B_plus(mu_half)  - alpha[0]*B_plus(mu_m)
            
            # half-cell step matrix
            A = np.zeros((2,2))
            A[0,0] = 1
            A[0,1] = 0
            A[1,0] = alpha[1]*B_minus(mu_half) - alpha[0]*B_minus(mu_m)
            A[1,1] = alpha[1]*B_plus(mu_half)  - alpha[0]*B_plus(mu_m)
            
            # half-cell step preconditioned matrix eigenvalues
            C = np.linalg.inv(A + sig*np.eye(2)) @ (A - B)
            eigvals = np.linalg.eigvals(C)
            print("\n\n\nHalf-cell Step last cell eigenvalues:", eigvals)
            
            # full-cell step matrix
            A = np.zeros((2,2))
            A[0,0] = 1
            A[0,1] = 0
            A[1,0] = -alpha[0]
            A[1,1] =  alpha[1]
            
            # full-cell step preconditioned matrix eigenvalues
            C = np.linalg.inv(A + sig*np.eye(2)) @ (A - B)
            eigvals = np.linalg.eigvals(C)
            print("\nFull-cell Step last cell eigenvalues:", eigvals)
            
            # half-cell diamond
            A = np.zeros((2,2))
            A[0,0] = 2
            A[0,1] = 0
            A[1,0] = alpha[1]*B_minus(mu_half) - alpha[0]*B_minus(mu_m)
            A[1,1] = alpha[1]*B_plus(mu_half)  - alpha[0]*B_plus(mu_m)
            
            # half-cell diamond preconditioned matrix eigenvalues
            C = np.linalg.inv(A + sig*np.eye(2)) @ (A - B)
            eigvals = np.linalg.eigvals(C)
            print("\nHalf-cell Diamond last cell eigenvalues:", eigvals)
            
            # full-cell diamond
            A = np.zeros((2,2))
            A[0,0] = 2
            A[0,1] = 0
            A[1,0] = -2*(alpha[0] + alpha[1])
            A[1,1] = 2*alpha[1]
            
            # full-cell diamond preconditioned matrix eigenvalues
            C = np.linalg.inv(A + sig*np.eye(2)) @ (A - B)
            eigvals = np.linalg.eigvals(C)
            print("\nFull-cell Diamond last cell eigenvalues:", eigvals)
            
            # half-cell weighted diamond
            A = np.zeros((2,2))
            A[0,0] = 1 / beta[0]
            A[0,1] = 0
            A[1,0] = alpha[1]*B_minus(mu_half) - alpha[0]*B_minus(mu_m)
            A[1,1] = alpha[1]*B_plus(mu_half)  - alpha[0]*B_plus(mu_m)
            
            # half-cell weighted diamond preconditioned matrix eigenvalues
            C = np.linalg.inv(A + sig*np.eye(2)) @ (A - B)
            eigvals = np.linalg.eigvals(C)
            print("\nHalf-cell Weighted Diamond last cell eigenvalues:", eigvals)
            
            # full-cell weighted diamond
            A = np.zeros((2,2))
            A[0,0] = 1 / beta[0]
            A[0,1] = 0
            A[1,0] = -(1/beta[0])*alpha[0] - (1-beta[1])/(beta[0]*beta[1])*alpha[1]
            A[1,1] = (1/beta[1])*alpha[1]
            
            # full-cell weighted diamond preconditioned matrix eigenvalues
            C = np.linalg.inv(A + sig*np.eye(2)) @ (A - B)
            eigvals = np.linalg.eigvals(C)
            print("\nFull-cell Weighted Diamond last cell eigenvalues:", eigvals)

    
        
"""
Radius:
-------
The outer radius of each material region is provided as a one-dimensional numpy array;
Below, this numpy array is R

Cells per region:
-----------------
Similarly, the number of cells in each region is provided as a one-dimensional numpy
array; Below, this numpy array is I_reg

Quadrature:
-----------
The quadrature rule specifications are provided as a library, with the number of directions
given as "directions" (this will be an even integer), the quadrature rule given as 
"quadrature" (this will be either "midpoint" for a midpoint rule or "gauss" for local Gauss
S2), and the formula for the alpha coefficients specified by "alpha" (this will be either 
"approximate" for the approximate recursive formula or "exact" for the exact formula (1-mu^2))
"""                                                                           


# quad_dict = {"directions":128,
#              "quadrature":"gauss",
#                   "alpha":"approximate"}

# sol = Spectral_Radius(quad_dict)
# sol.eigenvalues(sig=0., do_quadratic=True, do_plot=True, do_last_cell=True)

n = 20
N, x = np.zeros(n, dtype=int), np.zeros(n)

for i in range(n):
    N[i] = int(8*(2**i))
    
    quad_dict = {"directions":N[i],
                 "quadrature":"gauss",
                      "alpha":"approximate"}
    quad = Quad(quad_dict)
    
    x[i] = quad.alpha[-2] / quad.w[-1]
    
plt.figure(1)
plt.plot(N,x)