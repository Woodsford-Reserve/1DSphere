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
        
        
        
# solver class
class Solve:
    def __init__(self, R, I_reg, quad_dict, bc_dict, matprops, do_angular=False):
        nmats = len(matprops["sigt"])
        matIDs = np.arange(nmats)
        
        # initialization
        self.mesh = Mesh(matIDs, R, I_reg)
        self.quad = Quad(quad_dict)
        self.bc = bc_dict
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
        alpha   = self.quad.alpha[2*n_mu:2*n_mu+3]
        mu_half = self.quad.mu_half[2*n_mu+2]

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
                # angular cell midpoint
                mu_1 = 0.5*(mu[0]+mu[1])
                
                # coefficients
                a00 = -2*mu[0]*A[0] + alpha[1]*(A[1]-A[0])/(2*w[0])*B_minus(mu_1) + sigt*V
                a01 = alpha[1]*(A[1]-A[0])/(2*w[0])*B_plus(mu_1)
                a10 = (alpha[2]*B_minus(mu_half) - alpha[1]*B_minus(mu_1))*(A[1]-A[0])/(2*w[1])
                a11 = -2*mu[1]*A[0] + (alpha[2]*B_plus(mu_half) - alpha[1]*B_plus(mu_1)) \
                      *(A[1]-A[0])/(2*w[1]) + sigt*V
                    
                # source terms
                b0 = (sigs*Phi_0[iel]+q)/2*V - mu[0]*(A[1]+A[0])*psi_x[0] \
                      - alpha[1]*(A[1]-A[0])/(2*w[0])*B_S(mu_1)*psi_mu[iel]
                b1 = (sigs*Phi_0[iel]+q)/2*V - mu[1]*(A[1]+A[0])*psi_x[1] \
                      - (A[1]-A[0])/(2*w[1])*(alpha[2]*B_S(mu_half) - alpha[1]*B_S(mu_1))*psi_mu[iel]
                
            # other angular cells
            else:
                # coefficients
                a00 = 2*np.abs(mu[0])*A_out + alpha[1]*(A[1]-A[0])/(4*w[0]) + sigt*V
                a01 = alpha[1]*(A[1]-A[0])/(4*w[0])
                a10 = (alpha[2]*B_minus(mu_half) - 0.5*alpha[1])*(A[1]-A[0])/(2*w[1])
                a11 = 2*np.abs(mu[1])*A_out + (alpha[2]*B_plus(mu_half) - 0.5*alpha[1])\
                      *(A[1]-A[0])/(2*w[1]) + sigt*V
                    
                # source terms
                b0 = (sigs*Phi_0[iel]+q)/2*V + np.abs(mu[0])*(A[1]+A[0])*psi_x[0] \
                     + alpha[0]*(A[1]-A[0])/(2*w[0])*psi_mu[iel]
                b1 = (sigs*Phi_0[iel]+q)/2*V + np.abs(mu[1])*(A[1]+A[0])*psi_x[1]
            
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
                psi_mu[iel] = psi_mu[iel]*B_S(mu_half) + psi_minus*B_minus(mu_half) \
                                           + psi_plus*B_plus(mu_half)
            else:
                psi_mu[iel] = psi_minus*B_minus(mu_half) + psi_plus*B_plus(mu_half)
                          
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
        plt.plot(self.mesh.r, self.Phi, 'k--')
        plt.xlabel("r (cm)")
        plt.ylabel("Flux")
        plt.title("1D Spherical Transport Solution (Linear Discontinous-Diamond Difference)")
        
        
    # plot angular fluxes
    def angular(self):
        if self.do_angular == True:
            for nd in range(self.quad.N_dir):
                plt.figure(nd+2)
                plt.plot(self.mesh.r, self.psi[nd,:], 'k--')
                plt.xlabel("r (cm)")
                plt.ylabel("Angular Flux")
        
        
    
R = np.array([1.])
I_reg = np.array([1000])

bc_dict = {"type":"isotropic","value":0.}

matprops = {"sigt":np.array([0.0]),
            "sigs":np.array([0.0]),
               "q":np.array([1.0])}

quad_dict = {"directions":8,
              "quadrature":"gauss",
                  "alpha":"approximate"}

sol = Solve(R, I_reg, quad_dict, bc_dict, matprops, True)
sol.solve()
sol.plot()
sol.angular()


# quad_dict = {"directions":64,
#               "quadrature":"midpoint",
#                   "alpha":"approximate"}
# sol = Solve(R, I_reg, quad_dict, bc_dict, matprops)
# sol.solve()
# Phi_1 = sol.Phi

# quad_dict = {"directions":128,
#               "quadrature":"midpoint",
#                   "alpha":"approximate"}
# sol = Solve(R, I_reg, quad_dict, bc_dict, matprops)
# sol.solve()
# Phi_2 = sol.Phi

# quad_dict = {"directions":256,
#               "quadrature":"midpoint",
#                   "alpha":"approximate"}
# sol = Solve(R, I_reg, quad_dict, bc_dict, matprops)
# sol.solve()
# Phi_3 = sol.Phi

# ratio = np.sqrt(np.sum((Phi_1 - Phi_2)**2)) \
#         / np.sqrt(np.sum((Phi_2 - Phi_3)**2))
# print("\nRatio:",ratio)
# print("Order Accuracy:",np.log(ratio)/np.log(2))