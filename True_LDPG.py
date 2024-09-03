# -*- coding: utf-8 -*-
"""
Created on Sat May 11 11:38:28 2024

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
        
        # boundary flux
        self.psi_bound = np.zeros(self.quad.N_dir)
        self.psi_bound[:int(self.quad.N_dir/2)] = self.bc["value"]/2.
        
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
        mu0, mu1 = self.quad.mu[0], self.quad.mu[1]
        psi_x = self.psi_bound[0]*(mu1 + 1)/(mu1 - mu0) \
              - self.psi_bound[1]*(mu0 + 1)/(mu1 - mu0)
            
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
        mu_half = self.quad.mu_half[2*n_mu:2*n_mu+3]

        # basis functions
        if (n_mu == 0):
            B_S     = lambda u: ((u-mu[0])*(u-mu[1]))/((-1-mu[0])*(-1-mu[1]))
            B_minus = lambda u: ((u+1)*(u-mu[1]))/((mu[0]+1)*(mu[0]-mu[1]))
            B_plus  = lambda u: ((u+1)*(u-mu[0]))/((mu[1]+1)*(mu[1]-mu[0]))
        else:
            B_minus = lambda u: (mu[1]-u)/(mu[1]-mu[0])
            B_plus  = lambda u: (u-mu[0])/(mu[1]-mu[0])
        
        # basis function zeroth moments
        mm = self.integrate(mu_half[0], mu_half[1], lambda u: (mu[1]-u)/(mu[1]-mu[0])) / w[0]
        mp = self.integrate(mu_half[0], mu_half[1], lambda u: (u-mu[0])/(mu[1]-mu[0])) / w[0]
        pm = self.integrate(mu_half[1], mu_half[2], lambda u: (mu[1]-u)/(mu[1]-mu[0])) / w[1]
        pp = self.integrate(mu_half[1], mu_half[2], lambda u: (u-mu[0])/(mu[1]-mu[0])) / w[1]
        
        # basis function first moments
        mu_mm = self.integrate(mu_half[0], mu_half[1], lambda u: u*(mu[1]-u)/(mu[1]-mu[0])) / w[0]
        mu_mp = self.integrate(mu_half[0], mu_half[1], lambda u: u*(u-mu[0])/(mu[1]-mu[0])) / w[0]
        mu_pm = self.integrate(mu_half[1], mu_half[2], lambda u: u*(mu[1]-u)/(mu[1]-mu[0])) / w[1]
        mu_pp = self.integrate(mu_half[1], mu_half[2], lambda u: u*(u-mu[0])/(mu[1]-mu[0])) / w[1]
        
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
                a00 = -2*mu_mm*A[0] + alpha[1]*(A[1]-A[0])/(2*w[0])*B_minus(mu_half[1]) + mm*sigt*V
                a01 = -2*mu_mp*A[0] + alpha[1]*(A[1]-A[0])/(2*w[0])*B_plus(mu_half[1]) + mp*sigt*V
                a10 = -2*mu_pm*A[0] + (alpha[2]*B_minus(mu_half[2]) - alpha[1]*B_minus(mu_half[1])) \
                                             *(A[1]-A[0])/(2*w[1]) + pm*sigt*V
                a11 = -2*mu_pp*A[0] + (alpha[2]*B_plus(mu_half[2]) - alpha[1]*B_plus(mu_half[1])) \
                                             *(A[1]-A[0])/(2*w[1]) + pp*sigt*V
                                      
                # source terms
                b0 = (sigs*Phi_0[iel]+q)/2*V - (A[1]+A[0])*(mu_mm*psi_x[0]+mu_mp*psi_x[1]) \
                         - alpha[1]*(A[1]-A[0])/(2*w[0])*B_S(mu_half[1])*psi_mu[iel]
                b1 = (sigs*Phi_0[iel]+q)/2*V - (A[1]+A[0])*(mu_pm*psi_x[0]+mu_pp*psi_x[1]) \
                     - (alpha[2]*B_S(mu_half[2]) - alpha[1]*B_S(mu_half[1]))*(A[1]-A[0])/(2*w[1]) \
                                                *psi_mu[iel]
                
            # other angular cells
            else:
                # coefficients
                a00 = 2*np.abs(mu_mm)*A_out + alpha[1]*(A[1]-A[0])/(4*w[0]) + mm*sigt*V
                a01 = 2*np.abs(mu_mp)*A_out + alpha[1]*(A[1]-A[0])/(4*w[0]) + mp*sigt*V
                a10 = 2*np.abs(mu_pm)*A_out + (alpha[2]*B_minus(mu_half[2]) - 0.5*alpha[1]) \
                                      *(A[1]-A[0])/(2*w[1]) + pm*sigt*V
                a11 = 2*np.abs(mu_pp)*A_out + (alpha[2]*B_plus(mu_half[2]) - 0.5*alpha[1]) \
                                      *(A[1]-A[0])/(2*w[1]) + pp*sigt*V
                                      
                # source terms
                b0 = (sigs*Phi_0[iel]+q)/2*V + (A[1]+A[0])*(np.abs(mu_mm)*psi_x[0]+np.abs(mu_mp)*psi_x[1]) \
                                      + alpha[0]*(A[1]-A[0])/(2*w[0])*psi_mu[iel]
                b1 = (sigs*Phi_0[iel]+q)/2*V + (A[1]+A[0])*(np.abs(mu_pm)*psi_x[0]+np.abs(mu_pp)*psi_x[1])
            
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
            if n_mu == 0:
                psi_mu[iel] = psi_mu[iel]*B_S(mu_half[2]) + psi_minus*B_minus(mu_half[2]) \
                                          + psi_plus*B_plus(mu_half[2]) 
            else:
                psi_mu[iel] = psi_minus*B_minus(mu_half[2]) + psi_plus*B_plus(mu_half[2])
        
        # positive-mu sweep
        if (mu[0] > 0):
            # boundary flux
            self.psi_bound[2*n_mu:2*n_mu+2] = psi_x[:]
            
        # return angular fluxes
        return Phi_1, psi_mu
    
    
    # integration function
    def integrate(self, a, b, f):
         # quadrature points
         x1 = (b-a)/2.*(-1./np.sqrt(3.)) + (b+a)/2.
         x2 = (b-a)/2.*( 1./np.sqrt(3.)) + (b+a)/2.
         w1 = w2 = (b-a)/2.
         # return integrals
         integral = f(x1)*w1 + f(x2)*w2
         return integral
    
    
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
        leak = 0
        for i in range(int(self.quad.N_dir/2),int(self.quad.N_dir)):
            leak += self.quad.mu[i] * self.psi_bound[i] * self.quad.w[i]
        leak *= self.mesh.A[-1]
        # bsource
        bsource = 0
        for i in range(int(self.quad.N_dir/2)):
            bsource += np.abs(self.quad.mu[i])*self.psi_bound[i]*self.quad.w[i]
        bsource *= self.mesh.A[-1]
        # balance parameter
        bal = np.abs(source + bsource - (absorp + leak)) / (source + bsource)
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

bc_dict = {"type":"source","value":0.}

matprops = {"sigt":np.array([1.0]),
            "sigs":np.array([0.0]),
               "q":np.array([1.0])}

quad_dict = {"directions":8,
             "quadrature":"gauss",
                  "alpha":"exact"}

sol = Solve(R, I_reg, quad_dict, bc_dict, matprops, True)
sol.solve()
sol.plot()
sol.angular()
