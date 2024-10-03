# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 20:29:37 2024

@author: woods
"""

import numpy as np



# mesh class
class Mesh:
    def __init__(self, R, I_reg):
        # number of cells and regions 
        self.I = np.sum(I_reg)
        N_reg = len(I_reg)
        
        # add origin
        self.R = np.insert(R,0,0.)
        
        # cell widths and centers
        self.dr = np.array([])
        for nr in range(N_reg):
            dr_reg = (self.R[nr+1] - self.R[nr])/I_reg[nr]
            self.dr = np.concatenate((self.dr, np.repeat(dr_reg, I_reg[nr])))
        self.r = np.cumsum(self.dr) - self.dr/2
        
        # cell areas and volumes
        self.A = 4*np.pi*(self.r + self.dr/2)**2
        self.A = np.insert(self.A,0,0.)
        self.V = 4*np.pi/3*((self.r + self.dr/2)**3 - (self.r - self.dr/2)**3)
   
                
        
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
    def __init__(self, R, I_reg, quad_dict):
        # initialization
        self.mesh = Mesh(R, I_reg)
        self.quad = Quad(quad_dict)
        
        
    # solver 
    def eigenvalues(self, sigt):
        # eigenvalues
        eigenvalues = np.zeros((self.mesh.I, self.quad.N_cells, 2))
        
        # loop over directions
        for n_mu in range(self.quad.N_cells):
            # direction quantities
            mu      = self.quad.mu[2*n_mu:2*n_mu+2]
            w       = self.quad.w[2*n_mu:2*n_mu+2]
            alpha   = self.quad.alpha[2*n_mu:2*n_mu+3]
            mu_half = self.quad.mu_half[2*n_mu+2]
            beta    = self.quad.beta[2*n_mu:2*n_mu+2]
            
            # negative-mu sweeps
            if (mu[0] < 0):
                # sweep order
                start = self.mesh.I-1
                stop = -1
                inc = -1
                
            # positive-mu sweeps
            if (mu[0] > 0):
                # sweep order
                start = 0
                stop = self.mesh.I
                inc = 1
    
            # basis functions
            if (n_mu == 0):
                B_minus = lambda u: ((u+1)*(u-mu[1]))/((mu[0]+1)*(mu[0]-mu[1]))
                B_plus  = lambda u: ((u+1)*(u-mu[0]))/((mu[1]+1)*(mu[1]-mu[0]))
            else:
                B_minus = lambda u: (mu[1]-u)/(mu[1]-mu[0])
                B_plus  = lambda u: (u-mu[0])/(mu[1]-mu[0])
            
            # sweep
            for iel in range(start, stop, inc):
                # cell properties
                A = self.mesh.A[iel:iel+2] 
                if (mu[0] < 0):
                    A_out = A[0]
                if (mu[0] > 0):
                    A_out = A[1]
                V = self.mesh.V[iel]
                
                # matrices
                A_wd = np.zeros((2,2))
                B_wd = np.zeros((2,2))
                B_pg = np.zeros((2,2))
                
                # weighted diamond
                A_wd[0,0] = 2.*np.abs(mu[0])*A_out
                A_wd[1,1] = 2.*np.abs(mu[1])*A_out
                B_wd[0,0] = alpha[1]*(A[1]-A[0])/(2*beta[0]*w[0])
                B_wd[1,1] = alpha[2]*(A[1]-A[0])/(2*beta[1]*w[1])
                
                # first angular cell
                if (n_mu == 0):
                    # angular cell midpoint
                    mu_1 = 0.5*(mu[0]+mu[1])
                    
                    # Petrov-Galerkin
                    B_pg[0,0] = alpha[1]*(A[1]-A[0])/(2*w[0])*B_minus(mu_1)
                    B_pg[0,1] = alpha[1]*(A[1]-A[0])/(2*w[0])*B_plus(mu_1)
                    B_pg[1,0] = (alpha[2]*B_minus(mu_half) - alpha[1]*B_minus(mu_1))*(A[1]-A[0])/(2*w[1])
                    B_pg[1,1] = (alpha[2]*B_plus(mu_half) - alpha[1]*B_plus(mu_1))*(A[1]-A[0])/(2*w[1])
                         
                # other angular cells
                else:
                    # Petrov-Galerkin
                    B_pg[0,0] = alpha[1]*(A[1]-A[0])/(4*w[0])
                    B_pg[0,1] = alpha[1]*(A[1]-A[0])/(4*w[0])
                    B_pg[1,0] = (alpha[2]*B_minus(mu_half) - 0.5*alpha[1])*(A[1]-A[0])/(2*w[1])
                    B_pg[1,1] = (alpha[2]*B_plus(mu_half) - 0.5*alpha[1])*(A[1]-A[0])/(2*w[1])
                
                # eigenvalues
                matrix = np.linalg.inv(A_wd + B_wd + sigt*V*np.eye(2)) @ (B_wd - B_pg)
                eigenvalues[iel,n_mu,:] = np.abs(np.linalg.eigvals(matrix))
                
        # return spectral radius
        rho = np.max(eigenvalues)
        print("Spetral Radius:", rho)
        return rho

        

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



R = np.array([1.])
I_reg = np.array([40])

bc_dict = {"type":"isotropic","value":0.}

sigt = 1.

quad_dict = {"directions":4,
             "quadrature":"gauss",
                  "alpha":"approximate"}

sol = Spectral_Radius(R, I_reg, quad_dict)
sol.eigenvalues(sigt)