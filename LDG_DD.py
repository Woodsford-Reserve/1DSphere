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
    def __init__(self, N_dir):
        # number of mu-cells
        self.N_dir   = N_dir
        self.N_cells = int(N_dir/2)
        
        # mu-cell boundaries
        self.mu_half = np.linspace(-1.,1.,self.N_cells+1)
        
        # mu-cell midpoints
        mu = 0.5*(self.mu_half[:-1] + self.mu_half[1:])
        
        # local Gauss S2 quadrature
        self.w  = (1./self.N_cells)*np.ones(N_dir)
        self.mu = np.zeros(N_dir) 
        for n_mu in range(self.N_cells):
            self.mu[2*n_mu]   = self.w[2*n_mu]*(-1./np.sqrt(3.))  + mu[n_mu]
            self.mu[2*n_mu+1] = self.w[2*n_mu+1]*(1./np.sqrt(3.)) + mu[n_mu]
        
        # alpha (1-mu^2)
        self.alpha = np.zeros(3*self.N_cells+1)
        for n_mu in range(self.N_cells):
            self.alpha[3*n_mu]   = 1 - self.mu_half[n_mu]**2
            self.alpha[3*n_mu+1] = 1 - self.mu[2*n_mu]**2
            self.alpha[3*n_mu+2] = 1 - self.mu[2*n_mu+1]**2  
        
        
        
# solver class
class Solve:
    def __init__(self, R, I_reg, N_dir, bc, matprops, do_angular=False):
        nmats = len(matprops["sigt"])
        matIDs = np.arange(nmats)
        
        # initialization
        self.mesh = Mesh(matIDs, R, I_reg)
        self.quad = Quad(N_dir)
        self.bc = bc
        self.matprops = matprops
        
        # plot angular fluxes
        self.do_angular = do_angular
        
        # check for void
        self.do_balance = True 
        if np.all(matprops["sigt"] == 0.) and np.all(matprops["sigs"] == 0.) and \
                              np.all(matprops["q"] == 0.):
            self.do_balance = False
        
        
    # solver 
    def solve(self):
        # angular fluxes
        if self.do_angular == True:
            self.psi = np.zeros((self.quad.N_dir,self.mesh.I))
        
        # isotropic flux boundary condition
        self.psi_bound = np.zeros(self.quad.N_dir)
        if (self.bc["type"] == "isotropic"):
            self.psi_bound[:int(self.quad.N_dir/2)] = self.bc["value"]/2.
        # anisotropic flux boundary condition
        if (self.bc["type"] == "anisotropic"):
            self.psi_bound[:int(self.quad.N_dir/2)] = self.bc["value"][:]
        
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
            for n_mu in range(self.quad.N_cells):
                Phi_1, psi_mu = self.sweep(n_mu, psi_mu, Phi_0, Phi_1)
                
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
        if self.do_balance:
            bal = self.balance()
            print("Balance: "+str(bal))
    

    # starting direction
    def start(self, Phi):
        # starting direction ingoing flux
        if (self.quad.N_dir == 2):
            psi_x = self.psi_bound[0]
        else:
            mu0, mu1 = self.quad.mu[0], self.quad.mu[1]
            psi_x = self.psi_bound[0]*(mu1 + 1)/(mu1 - mu0) - \
                    self.psi_bound[1]*(mu0 + 1)/(mu1 - mu0)
            if (psi_x < 0):
                psi_x = 0
            
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
    def sweep(self, n_mu, psi_mu, Phi_0, Phi_1):
        # direction quantities
        mu      = self.quad.mu[2*n_mu:2*n_mu+2]
        w       = self.quad.w[2*n_mu:2*n_mu+2]
        alpha   = self.quad.alpha[3*n_mu:3*n_mu+4]
        mu_half = self.quad.mu_half[n_mu:n_mu+2]
        
        # basis functions
        if (n_mu == 0):
            B_S     = lambda u: ((u-mu[0])*(u-mu[1]))/((-1-mu[0])*(-1-mu[1]))
            B_minus = lambda u: ((u+1)*(u-mu[1]))/((mu[0]+1)*(mu[0]-mu[1]))
            B_plus  = lambda u: ((u+1)*(u-mu[0]))/((mu[1]+1)*(mu[1]-mu[0]))
        else:
            B_minus = lambda u: (mu[1]-u)/(mu[1]-mu[0])
            B_plus  = lambda u: (u-mu[0])/(mu[1]-mu[0])
        
        # negative-mu sweeps
        if (mu[0] < 0):
            # angular flux at boundary
            psi_x = np.copy(self.psi_bound[2*n_mu:2*n_mu+2])
            # sweep order
            start = self.mesh.I-1
            stop = -1
            inc = -1
            
        # positive-mu sweeps
        if (mu[0] > 0):
            # angular flux at origin
            psi_x = self.psi_0*np.ones(2)
            # sweep order
            start = 0
            stop = self.mesh.I
            inc = 1
            
        # sweep
        for iel in range(start, stop, inc):
            # cell properties
            A     = self.mesh.A[iel:iel+2] 
            if (mu[0] < 0):
                A_out = A[0]
            if (mu[0] > 0):
                A_out = A[1]
            V     = self.mesh.V[iel]
            matID = self.mesh.matID[iel]
            sigt  = self.matprops["sigt"][matID]
            sigs  = self.matprops["sigs"][matID]
            q     = self.matprops["q"][matID]
            
            # first angular cell
            if (n_mu == 0):
                # coefficients
                a00 = -2*mu[0]*A[0]*w[0] + alpha[3]*(A[1]-A[0])/2.*B_minus(mu_half[1]) + sigt*V*w[0]
                a01 = -2*mu[1]*A[0]*w[1] + alpha[3]*(A[1]-A[0])/2.*B_plus(mu_half[1]) + sigt*V*w[1]
                a10 = -2*mu[0]**2*A[0]*w[0] + (mu_half[1]*alpha[3]*B_minus(mu_half[1]) - alpha[1]*w[0]) \
                      *(A[1]-A[0])/2. + mu[0]*sigt*V*w[0]
                a11 = -2*mu[1]**2*A[0]*w[1] + (mu_half[1]*alpha[3]*B_plus(mu_half[1]) - alpha[2]*w[1]) \
                      *(A[1]-A[0])/2. + mu[1]*sigt*V*w[1]
                    
                # source terms
                b0 = (sigs*Phi_0[iel]+q)/2.*(w[0]+w[1])*V - alpha[3]*(A[1]-A[0])/2.*B_S(mu_half[1])*psi_mu[iel] \
                     - (A[1]+A[0])*(mu[0]*psi_x[0]*w[0] + mu[1]*psi_x[1]*w[1])
                b1 = (sigs*Phi_0[iel]+q)/2.*(mu[0]*w[0]+mu[1]*w[1])*V - mu_half[1]*alpha[3]*(A[1]-A[0])/2. \
                     *B_S(mu_half[1])*psi_mu[iel] - (A[1]+A[0])*(mu[0]**2*psi_x[0]*w[0] + mu[1]**2*psi_x[1]*w[1])
            
            # other angular cells
            else:
                # coefficients
                a00 = 2*np.abs(mu[0])*A_out*w[0] + alpha[3]*B_minus(mu_half[1])*(A[1]-A[0])/2. + sigt*V*w[0]
                a01 = 2*np.abs(mu[1])*A_out*w[1] + alpha[3]*B_plus(mu_half[1])*(A[1]-A[0])/2. + sigt*V*w[1]
                a10 = 2*np.abs(mu[0])*mu[0]*A_out*w[0] + (mu_half[1]*alpha[3]*B_minus(mu_half[1]) - alpha[1]*w[0]) \
                      *(A[1]-A[0])/2. + mu[0]*sigt*V*w[0]
                a11 = 2*np.abs(mu[1])*mu[1]*A_out*w[1] + (mu_half[1]*alpha[3]*B_plus(mu_half[1]) - alpha[2]*w[1]) \
                      *(A[1]-A[0])/2. + mu[1]*sigt*V*w[1]
                    
                # source terms
                b0 = (sigs*Phi_0[iel]+q)/2.*(w[0]+w[1])*V + alpha[0]*(A[1]-A[0])/2.*psi_mu[iel] \
                     + (A[1]+A[0])*(np.abs(mu[0])*psi_x[0]*w[0] + np.abs(mu[1])*psi_x[1]*w[1])
                b1 = (sigs*Phi_0[iel]+q)/2.*(mu[0]*w[0]+mu[1]*w[1])*V + mu_half[0]*alpha[0]*(A[1] \
                     - A[0])/2.*psi_mu[iel] + (A[1]+A[0])*(np.abs(mu[0])*mu[0]*psi_x[0]*w[0] \
                     + np.abs(mu[1])*mu[1]*psi_x[1]*w[1]) 
                                                           
            # calculate Gauss point angular fluxes
            psi_minus = (a11*b0 - a01*b1)/(a00*a11 - a10*a01)
            psi_plus  = (a00*b1 - a10*b0)/(a00*a11 - a10*a01)
            
            # update angular flux
            if self.do_angular == True:
                self.psi[2*n_mu,iel]   = psi_minus
                self.psi[2*n_mu+1,iel] = psi_plus
            
            # add flux contribution
            Phi_1[iel] += (psi_minus*w[0] + psi_plus*w[1])
            
            # update ingoing fluxes
            psi_x[0] = 2*psi_minus - psi_x[0]
            psi_x[1] = 2*psi_plus  - psi_x[1]
            if (n_mu == 0):
                psi_mu[iel] = psi_mu[iel]*B_S(mu_half[1]) + psi_minus*B_minus(mu_half[1]) \
                                           + psi_plus*B_plus(mu_half[1])
            else:
                psi_mu[iel] = psi_minus*B_minus(mu_half[1]) + psi_plus*B_plus(mu_half[1])
                          
        # positive-mu sweep
        if (mu[0] > 0):
            # boundary flux
            self.psi_bound[2*n_mu:2*n_mu+2] = psi_x[:]
            
        # return angular fluxes
        return Phi_1, psi_mu
    
    
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
        bal = np.abs(source - (absorp + leak)) / source
        return bal
    
    
    # plot solution
    def plot(self):
        plt.figure(1)
        plt.plot(self.mesh.r, self.Phi, 'b:')
        plt.xlabel("r (cm)")
        plt.ylabel("Flux")
        plt.title("1D Spherical Transport Solution (Linear Discontinous-Diamond Difference)")
        
        
    # plot angular fluxes
    def angular(self):
        if self.do_angular == True:
            for nd in range(self.quad.N_dir):
                plt.figure(nd+2)
                plt.plot(self.mesh.r, self.psi[nd,:], 'b:')
                plt.xlabel("r (cm)")
                plt.ylabel("Angular Flux")
        
        
    
R = np.array([1.])
I_reg = np.array([1000])
N_dir = 8

bc = {"type":"isotropic","value":0.}

matprops = {"sigt":np.array([1.0]),
            "sigs":np.array([0.0]),
               "q":np.array([1.0])}

sol = Solve(R, I_reg, N_dir, bc, matprops, True)
sol.solve()
sol.plot()
sol.angular()


# sol = Solve(R, I_reg, 128, bc, matprops)
# sol.solve()
# Phi_1 = sol.Phi

# sol = Solve(R, I_reg, 256, bc, matprops)
# sol.solve()
# Phi_2 = sol.Phi

# sol = Solve(R, I_reg, 512, bc, matprops)
# sol.solve()
# Phi_3 = sol.Phi

# ratio = np.sqrt(np.sum((Phi_1 - Phi_2)**2)) \
#        / np.sqrt(np.sum((Phi_2 - Phi_3)**2))
# print("\nRatio:",ratio)
# print("Order Accuracy:",np.log(ratio)/np.log(2))