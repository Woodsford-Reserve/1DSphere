# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:16:48 2024

@author: c.woodsford.9788
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
   
                
            
class Quad:
    def __init__(self, N_dir):
        # gauss-legendre quadrature
        self.N_dir = N_dir
        self.mu,self.w = np.polynomial.legendre.leggauss(N_dir)
        
        # mu-cell boundaries and alpha 
        self.mu_half = np.zeros(N_dir+1)
        self.mu_half[0] = -1.
        self.alpha = np.zeros(N_dir+1)
        for nd in range(N_dir):
            self.mu_half[nd+1] = self.mu_half[nd] + self.w[nd]
            self.alpha[nd+1] = self.alpha[nd] - 2*self.mu[nd]*self.w[nd]
        
        # beta
        self.beta = np.zeros(N_dir)
        for nd in range(N_dir):
            self.beta[nd] = (self.mu[nd] - self.mu_half[nd])/\
                            (self.mu_half[nd+1] - self.mu_half[nd])
    
    
    
class Solve:
    def __init__(self, R, I_reg, N_dir, bc, matprops):
        nmats = len(matprops["sigt"])
        matIDs = np.arange(nmats)
        
        # initialization
        self.mesh = Mesh(matIDs, R, I_reg)
        self.quad = Quad(N_dir)
        self.bc = bc
        self.matprops = matprops
        
        
    # solver 
    def solve(self):
        # boundary flux 
        self.psi_bound = np.zeros(self.quad.N_dir)
        self.psi_bound[:int(self.quad.N_dir/2)] = self.bc["value"]/2         
        
        # initial guess
        Phi_0, Phi_m1 = np.zeros(self.mesh.I), np.zeros(self.mesh.I)
        
        # source iteration
        err, tol = 1, 1e-6
        it = 0
        print("\nIteration :: Error")
        while (err > tol):
            it += 1
            # scalar flux iterate
            Phi_1 = np.zeros(self.mesh.I)
            
            # starting direction sweep
            self.psi_0, psi_mu = self.start(Phi_0)
            
            # sweeps
            for nd in range(self.quad.N_dir):
                psi, psi_mu = self.sweep(nd, psi_mu, Phi_0)
                Phi_1 += psi*self.quad.w[nd]
                
            # calculate error
            if (it == 1):
                err = 1
            else:
                delta = np.sqrt(np.sum(((Phi_1 - Phi_0)/Phi_1)**2))
                rho = np.sum(np.abs(Phi_1 - Phi_0))/np.sum(np.abs(Phi_0 - Phi_m1))
                err = delta/np.abs(1 - rho)
            print(str(it)+" :: "+str(err))
            
            # next iteration
            Phi_m1 = Phi_0
            Phi_0 = Phi_1
            
        # converged solution
        self.Phi = Phi_0
        
        # balance parameter
        bal = self.balance()
        print("Balance: "+str(bal))
            
                
    # starting direction
    def start(self, Phi):
        # starting direction ingoing flux
        mu0, mu1 = self.quad.mu[0], self.quad.mu[1]
        psi_x = self.psi_bound[0]*(mu1 + 1)/(mu1 - mu0) - \
                self.psi_bound[1]*(mu0 + 1)/(mu1 - mu0)
            
        # starting direction sweep
        psi_mu = np.zeros(self.mesh.I) 
        for iel in range(self.mesh.I-1,-1,-1):
            # cell properties
            dr    = self.mesh.dr[iel]
            matID = self.mesh.matID[iel]
            sigt  = self.matprops["sigt"][matID]
            sigs  = self.matprops["sigs"][matID]
            q     = self.matprops["q"][matID]
            
            # calculate cell-average angular flux
            psi_mu[iel] = psi_x + (sigs*Phi[iel] + q)*dr/4
            psi_mu[iel] /= (1 + sigt*dr/2)
            
            # update ingoing flux
            psi_x = 2*psi_mu[iel] - psi_x
            
        # return origin and starting direction angular flux
        psi_0 = psi_x
        return psi_0, psi_mu
        
    
    # sweep function
    def sweep(self, nd, psi_mu, Phi):
        # direction quantities
        mu      = self.quad.mu[nd]
        w       = self.quad.w[nd]
        alpha   = self.quad.alpha[nd:nd+2]
        beta    = self.quad.beta[nd]
        
        # negative-mu sweeps
        if (mu < 0):
            # angular flux at boundary
            psi_x = self.psi_bound[nd]
            # sweep order
            start = self.mesh.I-1
            stop = -1
            inc = -1
            
        # positive-mu sweeps
        if (mu > 0):
            # angular flux at origin
            psi_x = self.psi_0
            # sweep order
            start = 0
            stop = self.mesh.I
            inc = 1
        
        # sweep
        psi = np.zeros(self.mesh.I)
        for iel in range(start, stop, inc):
            # cell properties
            A     = self.mesh.A[iel:iel+2]
            if (mu < 0):
                A_out = A[0]
            if (mu > 0):
                A_out = A[1]
            V     = self.mesh.V[iel]
            matID = self.mesh.matID[iel]
            sigt  = self.matprops["sigt"][matID]
            sigs  = self.matprops["sigs"][matID]
            q     = self.matprops["q"][matID]
            
            # calculate cell-average angular flux
            psi[iel] = (sigs*Phi[iel] + q)/2*V + np.abs(mu)*(A[1] + A[0])*psi_x \
                     + (A[1] - A[0])/(2*w)*(alpha[1]*(1/beta - 1) + alpha[0])*psi_mu[iel]
            psi[iel] /= (2*np.abs(mu)*A_out + alpha[1]*(A[1] - A[0])/(2*beta*w) + sigt*V)
            
            # update ingoing fluxes
            psi_x = 2*psi[iel] - psi_x
            psi_mu[iel] = (psi[iel] - (1-beta)*psi_mu[iel])/beta
        
        # positive-mu sweep
        if (mu > 0):
            # boundary flux
            self.psi_bound[nd] = psi_x
             
        # return angular fluxes
        return psi, psi_mu
    
    
    # calculate balance parameter
    def balance(self):
        # source and absorption
        source, absorp = 0, 0
        for iel in range(self.mesh.I):
            # cell properties
            V     = self.mesh.V[iel]
            matID = self.mesh.matID[iel]
            sigt  = self.matprops["sigt"][matID]
            sigs  = self.matprops["sigs"][matID]
            q     = self.matprops["q"][matID]
            # source and absorption rates
            source += q*V
            absorp += (sigt - sigs)*self.Phi[iel]*V
        # leakage
        leak = np.sum(self.quad.mu*self.psi_bound*self.quad.w)*self.mesh.A[-1]
        # balance parameter
        bal = np.abs(source - (absorp + leak))
        return bal
    
    
    # plot solution
    def plot(self):
        plt.figure(1)
        plt.plot(self.mesh.r, self.Phi, 'r')
        plt.xlabel("r (cm)")
        plt.ylabel("Flux")
        plt.title("1D Spherical Transport Solution (Weighted Diamond-Diamond Difference)")
        
        
    
R = np.array([1.,2.])
I_reg = np.array([40,40])
N_dir = 8 

bc = {"type":"source","value":0.}

matprops = {"sigt":np.array([1.0,1.0]),
            "sigs":np.array([0.5,0.9]),
               "q":np.array([1.0,0.0])}

sol = Solve(R, I_reg, N_dir, bc, matprops)
sol.solve()
sol.plot()