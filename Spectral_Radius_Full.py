# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 20:29:37 2024

@author: woods
"""

import numpy as np
import matplotlib.pyplot as plt



# mesh class
class Mesh:
    def __init__(self, matIDs, R, I_reg):
        # number of cells and regions 
        self.I = np.sum(I_reg)
        N_reg = len(matIDs)
        
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
        
        # material ID
        self.matID = np.array([], dtype=int)
        for nr in range(N_reg):
            self.matID = np.concatenate((self.matID, np.repeat(matIDs[nr],\
                                                               I_reg[nr])))
   
                
        
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
        self.beta = np.zeros(N_dir)
        for nd in range(N_dir):
            self.beta[nd] = (self.mu[nd] - self.mu_half[nd])/\
                            (self.mu_half[nd+1] - self.mu_half[nd])
        
        
        
# solver class
class Spectral_Radius:
    def __init__(self, R, I_reg, quad_dict, matprops, do_quadratic=True):
        
        nmats = len(matprops["sigt"])
        matIDs = np.arange(nmats)
        
        # initialization
        self.mesh = Mesh(matIDs, R, I_reg)
        self.quad = Quad(quad_dict)
        self.matprops = matprops
        
        # do quadratic continuous in first cell
        self.do_quadratic = do_quadratic
        
        # check for void
        self.do_balance = True 
        if np.all(matprops["sigt"] == 0.) and np.all(matprops["sigs"] == 0.) and \
                              np.all(matprops["q"] == 0.):
            self.do_balance = False

        
    # compute spectral radii
    def radius(self, scheme):
        
        # spectral radius
        rho_array = np.zeros((self.quad.N_cells,self.mesh.I))
        
        # loop over directions
        for n_mu in range(self.quad.N_cells):
            
            # direction quantities
            mu      = self.quad.mu[2*n_mu:2*n_mu+2]
            mu_m    = 0.5*(mu[0] + mu[1])
            w       = self.quad.w[2*n_mu:2*n_mu+2]
            alpha   = self.quad.alpha[2*n_mu:2*n_mu+3]
            mu_half = self.quad.mu_half[2*n_mu+2]
            if scheme == "step":
                beta = 1.
            if scheme == "DD":
                beta = 1./2.
            if scheme == "WD":
                beta = self.quad.beta[2*n_mu]
    
            # basis functions
            if self.do_quadratic and (n_mu == 0):
                B_minus = lambda u: ((u+1)*(u-mu[1]))/((mu[0]+1)*(mu[0]-mu[1]))
                B_plus  = lambda u: ((u+1)*(u-mu[0]))/((mu[1]+1)*(mu[1]-mu[0]))
            else:
                B_minus = lambda u: (mu[1]-u)/(mu[1]-mu[0])
                B_plus  = lambda u: (u-mu[0])/(mu[1]-mu[0])
            
            # loop over cells
            for iel in range(self.mesh.I):
                
                # cell properties
                A     = self.mesh.A[iel:iel+2]
                if (mu[0] < 0):
                    A_out = A[0]
                if (mu[0] > 0):
                    A_out = A[1]
                V     = self.mesh.V[iel]
                matID = self.mesh.matID[iel]
                sigt  = self.matprops["sigt"][matID]
                
                # constants
                # gamma
                gamma_minus = 4. * abs(mu[0])*w[0]/alpha[1] * A_out/(A[1]-A[0])
                gamma_plus  = 4. * abs(mu[1])*w[1]/alpha[1] * A_out/(A[1]-A[0])
                
                # delta
                epsilon = 2. * w[0]/alpha[1] * sigt*V/(A[1]-A[0]) 
                
                # beta 
                zeta_minus = 1./beta - B_minus(mu_m)
                zeta_plus  = B_plus(mu_m)
                
                # epsilon
                eta = (alpha[2]*B_plus(mu_half)  - alpha[1]*B_plus(mu_m)) \
                    / (alpha[2]*B_minus(mu_half) - alpha[1]*B_minus(mu_m))
                        
                # zeta
                theta = alpha[1] / (alpha[2]*B_minus(mu_half) - alpha[1]*B_minus(mu_m))
                
                # spectral radius
                rho = abs((zeta_minus*(gamma_plus*theta + eta + epsilon*theta) + zeta_plus) \
                    / ((gamma_minus + 1./beta + epsilon)*(gamma_plus*theta + eta + epsilon*theta)))
                rho_array[n_mu,iel] = rho
        
        # return spectral radii
        return rho_array
   

     

"""
Radius:
-------
The outer radius of each material region is provided as a one-dimensional numpy array;
Below, this numpy array is R

Cells per region:
-----------------
Similarly, the number of cells in each region is provided as a one-dimensional numpy
array; Below, this numpy array is I_reg

Boundary conditions:
--------------------
The boundary conditions are provided as a library, with the type given as "type" (this
will be either "isotropic" or "anisotropic") and the source given as "value" (this will 
either be a float if the "type" is "isotropic" or a one-dimensional numpy array of size
N_dir/2 if the "type" is "anisotropic"); Below, the boundary condition library is bc

Quadrature:
-----------
The quadrature rule specifications are provided as a library, with the number of directions
given as "directions" (this will be an even integer), the quadrature rule given as 
"quadrature" (this will be either "midpoint" for a midpoint rule or "gauss" for local Gauss
S2), and the formula for the alpha coefficients specified by "alpha" (this will be either 
"approximate" for the approximate recursive formula or "exact" for the exact formula (1-mu^2))
                                                                              
Material properties:
--------------------
The material properties are given as a library with the total cross section given as 
"sigt" (this will be a numpy array of the sigt for each material region), the scattering
cross section given as "sigs" (this will be a numpy array of the sigt for each material 
region), and the volumetric sources given as "q" (this will be a numpy array of the sigt 
for each material region); Below, the material properties are given as matprops
"""



R = np.array([1.])
I_reg = np.array([10])
N_dir = 128

quad_dict = {"directions":N_dir,
             "quadrature":"gauss",
                  "alpha":"approximate"}

matprops = {"sigt":np.array([1.0]),
            "sigs":np.array([0.0]),
               "q":np.array([0.0])}

sol = Spectral_Radius(R, I_reg, quad_dict, matprops, do_quadratic=True)

rho_step = sol.radius("step")
rho_DD   = sol.radius("DD")