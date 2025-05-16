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
            bal, leak = self.balance()
            print("Balance: "+str(bal))
            
            # return leakage
            return leak
    

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
        psi_mu     = np.zeros(self.mesh.I) 
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
        dmu     = mu[1] - mu[0]
        w       = self.quad.w[2*n_mu:2*n_mu+2]
        alpha   = self.quad.alpha[3*n_mu:3*n_mu+4]
        mu_half = self.quad.mu_half[n_mu:n_mu+2]
        
        # basis functions
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
            
            # coefficients
            a00 = 2*np.abs(mu[0])*A_out + (A[1]-A[0])/2.*(alpha[3]/w[0]*B_minus(mu_half[1])**2 \
                                             + alpha[1]/dmu) + sigt*V
            a01 = (A[1]-A[0])/2.*(alpha[3]/w[0]*B_minus(mu_half[1])*B_plus(mu_half[1]) + alpha[2]/dmu)
            a10 = (A[1]-A[0])/2.*(alpha[3]/w[0]*B_minus(mu_half[1])*B_plus(mu_half[1]) - alpha[1]/dmu)
            a11 = 2*np.abs(mu[1])*A_out + (A[1]-A[0])/2.*(alpha[3]/w[0]*B_plus(mu_half[1])**2 \
                                             - alpha[2]/dmu) + sigt*V
                
            # source terms
            b0 = (sigs*Phi_0[iel]+q)/2.*V + np.abs(mu[0])*(A[1]+A[0])*psi_x[0] \
                 + (A[1]-A[0])/(2.*w[0])*alpha[0]*B_minus(mu_half[0])*psi_mu[iel]
            b1 = (sigs*Phi_0[iel]+q)/2.*V + np.abs(mu[1])*(A[1]+A[0])*psi_x[1] \
                 + (A[1]-A[0])/(2.*w[0])*alpha[0]*B_plus(mu_half[0])*psi_mu[iel]
                                                           
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
        return bal, leak
    
    
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
        
    
    # exact solution function
    def exact(self, N_dir, plot=True, error=False):
        # quadrature set
        mu_,w_ = np.polynomial.legendre.leggauss(N_dir)
        
        # exact solution
        Phi_exact = np.zeros(self.mesh.I)
        
        if plot == True:
            Phi_plus       = np.zeros(self.mesh.I)
            Phi_plus_exact = np.zeros(self.mesh.I)
            
            Phi_minus       = np.zeros(self.mesh.I)
            Phi_minus_exact = np.zeros(self.mesh.I)
        
        # loop over cells
        for iel in range(self.mesh.I):
            # cell properties
            r  = self.mesh.r[iel]
            r0 = self.mesh.R[-1]
            siga = self.matprops["sigt"][0]
            
            # loop over directions
            for n in range(N_dir):
                # direction properties
                mu = mu_[n]
                w  = w_[n]
                theta1 = np.pi - np.arccos(mu)
                
                # distance
                d = np.sqrt(r**2 + r0**2 - 2.*r*r0*np.cos(np.pi - \
                            np.arcsin(r/r0*np.sin(theta1)) - theta1))
                    
                # quadrature integration
                psi = self.bc["value"]/2.*np.exp(-siga*d)
                Phi_exact[iel] += psi*w
                
                if plot == True:
                    if mu < 0.:
                        Phi_minus[iel] += self.psi[n,iel]*self.quad.w[n]
                        Phi_plus_exact[iel] += psi*w
                    else:
                        Phi_plus[iel] += self.psi[n,iel]*self.quad.w[n]
                        Phi_minus_exact[iel] += psi*w
                    
        # plot error
        if plot == True:
            plt.figure()
            Phi_err = np.abs(Phi_exact - self.Phi) / Phi_exact
            plt.plot(self.mesh.r, Phi_err, 'k')
            
            plt.figure()
            Phi_err = np.abs(Phi_plus_exact - Phi_plus) / Phi_plus_exact
            plt.plot(self.mesh.r, Phi_err)
            
            plt.figure()
            Phi_err = np.abs(Phi_minus_exact - Phi_minus) / Phi_minus_exact
            plt.plot(self.mesh.r, Phi_err)
              
        # return exact solution
        if error == True:
            return Phi_exact


"""
Radius:
-------
The outer radius of each material region is provided as a one-dimensional numpy array;
Below, this numpy array is R

Cells per region:
-----------------
Similarly, the number of cells in each region is provided as a one-dimensional numpy
array; Below, this numpy array is I_reg

Number of directions:
---------------------
The number of quadrature directions is given as an even integer; Below, the number
of directions is N_dir

Boundary conditions:
--------------------
The boundary conditions are provided as a library, with the type given as "type" (this
will be either "isotropic" or "anisotropic") and the source given as "value" (this will 
either be a float if the "type" is "isotropic" or a one-dimensional numpy array of size
N_dir/2 if the "type" is "anisotropic"); Below, the boundary condition library is bc
                                                                              
Material properties:
--------------------
The material properties are given as a library with the total cross section given as 
"sigt" (this will be a numpy array of the sigt for each material region), the scattering
cross section given as "sigs" (this will be a numpy array of the sigt for each material 
region), and the volumetric sources given as "q" (this will be a numpy array of the sigt 
for each material region); Below, the material properties are given as matprops
"""

        

plot, error = False, False
exact, leak = False, False

if plot == True:
    R = np.array([1.])
    I_reg = np.array([1000])
    N_dir = 32
    
    bc = {"type":"isotropic","value":1.}
    
    matprops = {"sigt":np.array([1.0]),
                "sigs":np.array([0.0]),
                   "q":np.array([0.0])}
    
    sol = Solve(R, I_reg, N_dir, bc, matprops, do_angular=True)
    sol.solve()
    sol.plot()
    #sol.angular()
    
    if exact == True:
        sol.exact(N_dir)

if error == True:
    if exact == False:
        R = np.array([1.])
        I_reg = np.array([1000])
        N_dir = 16
        
        bc = {"type":"isotropic","value":0.}
        
        matprops = {"sigt":np.array([1.0]),
                    "sigs":np.array([0.0]),
                       "q":np.array([1.0])}
        
        sol1 = Solve(R, I_reg, N_dir, bc, matprops, do_angular=False)
        sol1.solve()
        
        sol2 = Solve(R, I_reg, 2*N_dir, bc, matprops, do_angular=False)
        sol2.solve()
        
        sol3 = Solve(R, I_reg, 4*N_dir, bc, matprops, do_angular=False)
        sol3.solve()
        
        order = np.linalg.norm(sol1.Phi - sol2.Phi) / np.linalg.norm(sol2.Phi - sol3.Phi)
        print()
        print(order)
        
    if exact == True:
        if leak == False:
            R = np.array([1.])
            I_reg = np.array([10000])
            N_dir = 128
            
            bc = {"type":"isotropic","value":1.}
            
            matprops = {"sigt":np.array([1.0]),
                        "sigs":np.array([0.0]),
                           "q":np.array([0.0])}
            
            sol1 = Solve(R, I_reg, N_dir, bc, matprops, do_angular=False)
            sol1.solve()
            
            sol2 = Solve(R, I_reg, 2*N_dir, bc, matprops, do_angular=True)
            sol2.solve()
            
            Phi = sol2.exact(1024, plot=False, error=True)
            
            order = np.linalg.norm(sol1.Phi - Phi) / np.linalg.norm(sol2.Phi - Phi)
            print()
            print(order)
            
        if leak == True:
            R = np.array([1.])
            I_reg = np.array([10000])
            N_dir = 128
            
            bc = {"type":"isotropic","value":0.}
            
            matprops = {"sigt":np.array([1.0]),
                        "sigs":np.array([0.0]),
                           "q":np.array([1.0])}
            
            sol1 = Solve(R, I_reg, N_dir, bc, matprops, do_angular=False)
            leak1 = sol1.solve()
            
            sol2 = Solve(R, I_reg, 2*N_dir, bc, matprops, do_angular=False)
            leak2 = sol2.solve()
            
            sol3 = Solve(R, I_reg, 1024, bc, matprops)
            leak = sol3.solve()
            
            order = np.abs(leak1 - leak) / np.abs(leak2 - leak)
            print()
            print(order)