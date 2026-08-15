# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 20:29:37 2024

@author: woods
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import scipy
from io import StringIO

# mesh class
class Mesh:
    def __init__(self, matIDs, md1):
        # number of cells and regions 
        R = md1["R"]
        I_reg = md1["I_reg"]
        
        self.I = int(np.sum(I_reg))
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
        N_cells = self.N_cells
        
        # mu-cell boundaries
        self.mu_half = np.linspace(-1.,1.,self.N_dir+1)
        
        # mu-cell midpoints
        mu_half = np.linspace(-1.,1.,N_cells+1)
        mu = 0.5*(mu_half[:-1] + mu_half[1:])
        
        # quadrature weights
        self.w  = (1./self.N_cells)*np.ones(self.N_dir)
        
        # local Gauss S2 quadrature points
        if quad_dict["quadrature"] == "gauss":
            self.mu = np.zeros(self.N_dir)
            for n_mu in range(self.N_cells):
                self.mu[2*n_mu]   = self.w[2*n_mu]*(-1./np.sqrt(3.))  + self.mu_half[2*n_mu+1]
                self.mu[2*n_mu+1] = self.w[2*n_mu+1]*(1./np.sqrt(3.)) + self.mu_half[2*n_mu+1]
        
        # local Gauss S(N/2 + 1) quadrature
        temp_mu, temp_w = np.polynomial.legendre.leggauss(N_cells + 1)
        mu2 = np.zeros((N_cells, N_cells + 1))
        w2 = (1/N_cells)*temp_w
        for i in range(N_cells + 1):
            mu2[:,i] = (1/N_cells)*temp_mu[i] + mu[:]
        
        self.mu2 = mu2
        self.w2 = w2
        
            
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
    
        

class Scatter:
    def __init__(self, sd1, quad, sig_s):
        self.scatter_type = sd1["scatter_type"]
        self.scatter_order = sd1["scatter_order"]
        
        # creates sig_l matrix for a given input string
        # sig_l = I
        if sd1["sig_l"] == "delta":
            sig_l = np.ones(self.scatter_order + 1) * sig_s[0]
        
        # Henyey-Greenstein
        # sig_l[i] = sig_s * mu_bar ^ i
        # sig_s(1 - mu_bar) = 1
        elif sd1["sig_l"] == "HG":
            sig_l = np.zeros(self.scatter_order + 1)
            sig_l[0] = sig_s[0]
            mu_bar = 1 - (1/sig_l[0])
            print(mu_bar)
            for i in range(1, self.scatter_order + 1):
                sig_l[i] = sig_l[0] * mu_bar ** i

        # Fokker-Planck
        # sig_L = 0
        # sig_0 - sig_l = 1/2 * l * (l + 1)
        elif sd1["sig_l"] == "FP":
            sig_l = np.zeros(self.scatter_order + 1)
            sig_l[0] = 0.5*self.scatter_order*(self.scatter_order + 1)
            for i in range(1,self.scatter_order + 1):
                sig_l[i] = sig_l[0] - 0.5*i*(i + 1)
        
        # turns the 1-D array into a 2-D array
        self.sig_l = np.diag(sig_l)
            
        
        if self.scatter_type != "default":
            self.D = np.zeros((self.scatter_order + 1, quad.N_dir))
            self.M = np.zeros((quad.N_dir, self.scatter_order + 1))
        if self.scatter_type == "Sn":
            for i in range(quad.N_dir):
                for j in range(self.scatter_order + 1):
                    self.D[j][i] = scipy.special.eval_legendre(j, quad.mu[i])*quad.w[i] 
                    self.M[i][j] = (2*j + 1)/2 * scipy.special.eval_legendre(j, quad.mu[i])
            # sanity check
            # print("MD=I:",np.allclose(self.M @ self.D, np.eye(quad.N_dir)))
                    
        
        
# solver class
class Solve:
    def __init__(self, mesh_dict, quad_dict, bc_dict, matprops, quadratic, sweep_type, scatter_dict, do_angular=False):
        nmats = len(matprops["sigt"])
        matIDs = np.arange(nmats)
        
        # initialization
        self.mesh = Mesh(matIDs, mesh_dict)
        self.quad = Quad(quad_dict)
        self.bc = bc_dict
        self.matprops = matprops
        self.quadratic = quadratic
        self.sweep_type = sweep_type
        self.scatter = Scatter(scatter_dict, self.quad, matprops["sigs"])
        
        if sd1["sig_l"] == "FP":
            self.xs_fix()
    
        
        # plot angular fluxes
        self.do_angular = do_angular
        self.spatial_it = np.zeros((self.mesh.I, self.quad.N_cells))
        
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
            self.psi_01 = np.zeros((self.quad.N_dir,self.mesh.I))
        
        # isotropic flux boundary condition
        self.psi_bound = np.zeros(self.quad.N_dir)
        if (self.bc["type"] == "isotropic0"):
            self.psi_bound[:int(self.quad.N_dir/2)] = self.bc["value"]/2.
        # anisotropic flux boundary condition
        type1 = self.bc["type"][:-1]
        if (type1 == "anisotropic"):
            '''
            self.psi_bound[0] = 1 / (self.quad.w[0] * 2)
            self.psi_bound[1] = self.psi_bound[0]
            '''
            for i in range(int(self.quad.N_dir / 2)):
                self.psi_bound[i] = getPsi(self.quad.mu[i],self.bc)
                if self.psi_bound[i] < 0:
                    self.psi_bound[i] = 0.
        if self.bc["type"] == "anisotropic5":
            self.psi_bound[0] = 1.
            self.psi_bound[1:] = 0.
        
        
        # Normalizes to a Unit Net Current
        sum1 = 0
        for i in range(int(self.quad.N_dir / 2)):
            sum1 += self.psi_bound[i] * self.quad.w[i] * abs(self.quad.mu[i])
        for i in range(int(self.quad.N_dir / 2)):
            self.psi_bound[i] = 1 * self.psi_bound[i] / sum1
        self.current_norm = sum1
        
        
        # initial guess
        Phi_0, Phi_m1 = np.zeros(self.mesh.I), np.zeros(self.mesh.I)
        
        # source iteration
        err, tol = 1, 1e-6
        it = 0
        start_time = time.time()
        while (err > tol):
            it += 1
            # scalar flux iterate
            Phi_1 = np.zeros(self.mesh.I)
            
            # starting direction sweep
            self.psi_0, psi_mu = self.start(Phi_0)
            
            # sweeps
            if self.sweep_type == "direct":
                if it == 1:
                    print("Direct Solution")
                    print("\nIteration :: Error")
                for n_mu in range(self.quad.N_cells):
                    Phi_1, psi_mu = self.direct_sweep(n_mu, psi_mu, Phi_0, Phi_1, it)
            elif self.sweep_type == "cellwise":
                if it == 1:
                    print("Cellwise Solution")
                    print("\nIteration :: Error")
                for n_mu in range(self.quad.N_cells):
                    Phi_1, psi_mu = self.cellwise_sweep(n_mu, psi_mu, Phi_0, Phi_1, it)
            elif self.sweep_type == "angular":
                if it == 1:
                    print("Angular Solution")
                    print("\nIteration :: Error")
                for n_mu in range(self.quad.N_cells):
                    Phi_1, psi_mu = self.angular_sweep(n_mu, psi_mu, Phi_0, Phi_1, it)
            else:
                print(self.sweep_type)
            
            if self.scatter.scatter_type == "GQ2" and it == 1:
                self.scatter.M = scipy.linalg.pinv(self.scatter.D)
                # sanity check
                # print("MD=I:",np.allclose(self.scatter.M @ self.scatter.D, np.eye(self.quad.N_dir)))
            
            # creates the S vectors for each spatial cell using previous results
            if self.scatter.scatter_type !="default":
                S = np.zeros((self.mesh.I,self.quad.N_dir))
                psi = self.psi.T
                for i in range(self.mesh.I):
                    matID = self.mesh.matID[i]
                    sigs  = self.matprops["sigs"][matID]
                    if self.scatter.scatter_order != 0:
                        S[i][:] = self.scatter.M @ self.scatter.sig_l @ self.scatter.D @ psi[i][:]
                    else:
                        S[i][:] = self.scatter.M * self.scatter.sig_l @ self.scatter.D @ psi[i][:]
                self.scatter.S = S
            
            '''
            # pulls scalar flux
            if self.scatter.scatter_type !="default":
                phi = np.zeros(self.mesh.I)
                for i in range(self.mesh.I):
                    for j in range(self.quad.N_dir):
                        phi[i] += self.psi[j][i] * self.quad.w[j]
                print(phi)
            '''
                    
                
            # calculate error
            if (it == 1):
                err = 1
                rho = 1
            else:
                delta = np.sqrt(np.sum((Phi_1 - Phi_0)**2) / np.sum(Phi_1**2))
                rho = np.sum(np.abs(Phi_1 - Phi_0))/np.sum(np.abs(Phi_0 - Phi_m1))
                err = delta/np.abs(1 - rho)
            if np.all(matprops["sigs"] == 0) and self.scatter.scatter_type == "default":
                err = 0
            print(str(it)+"\t\t  :: "+str(err))
            
            # next iteration
            Phi_m1 = Phi_0
            Phi_0 = Phi_1
        
        end_time = time.time()
        self.calc_time = end_time - start_time
        # converged solution
        self.Phi = Phi_0
        
        # balance parameter
        if self.do_balance:
            bal = self.balance()
            self.bal = bal
            print("\nBalance: "+str(bal))
    

    # starting direction
    def start(self, Phi):
        # starting direction ingoing flux
        if (self.quad.N_dir == 2):
            psi_x = self.psi_bound[0]
        else:
            mu0, mu1 = self.quad.mu[0], self.quad.mu[1]
            # psi_x is the psi value for mu = -1 (starting direction flux)
            psi_x = self.psi_bound[0]*(mu1 + 1)/(mu1 - mu0) - \
                    self.psi_bound[1]*(mu0 + 1)/(mu1 - mu0)
            # psi_x = getPsi(-1, self.bc)
            
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
            if self.scatter.scatter_type == "default":
                psi_mu[iel] = psi_x + (sigs*Phi[iel] + q)*dr/4
                psi_mu[iel] /= (1 + sigt*dr/2)
            else:
                if np.all(Phi) == 0:
                    self.scatter.S = np.zeros((self.mesh.I,self.quad.N_dir))
                if np.all(self.psi.T[iel]) != 0:
                    psi_mu[iel] = psi_x + (self.scatter.S[iel][0] / self.psi[0][iel] * psi_x + q/2)*dr/2
                    psi_mu[iel] /= (1 + sigt*dr/2)
                else:
                    psi_mu[iel] = psi_x + (q/2)*dr/2
                    psi_mu[iel] /= (1 + sigt*dr/2)
            
            # update ingoing flux
            psi_x = 2*psi_mu[iel] - psi_x
            
        # return origin and starting direction angular flux
        psi_0 = psi_x
        return psi_0, psi_mu  

        
    # sweep function
    def direct_sweep(self, n_mu, psi_mu, Phi_0, Phi_1, it):
        quadratic = self.quadratic
        # direction quantities
        mu      = self.quad.mu[2*n_mu:2*n_mu+2]
        w       = self.quad.w[2*n_mu:2*n_mu+2]
        alpha   = self.quad.alpha[2*n_mu:2*n_mu+3]
        mu_half = self.quad.mu_half[2*n_mu+2]

        # basis functions
        if (n_mu == 0) and quadratic:
            B_S     = lambda u: ((u-mu[0])*(u-mu[1]))/((-1-mu[0])*(-1-mu[1]))
            B_minus = lambda u: ((u+1)*(u-mu[1]))/((mu[0]+1)*(mu[0]-mu[1]))
            B_plus  = lambda u: ((u+1)*(u-mu[0]))/((mu[1]+1)*(mu[1]-mu[0]))
        else:
            B_minus = lambda u: (mu[1]-u)/(mu[1]-mu[0])
            B_plus  = lambda u: (u-mu[0])/(mu[1]-mu[0])
        
        B_minus_linear = lambda u: (mu[1]-u)/(mu[1]-mu[0])
        B_plus_linear  = lambda u: (u-mu[0])/(mu[1]-mu[0])
        
        # creates D matrix for scattering
        if self.scatter.scatter_type == 'GQ2' and it == 1:
            D = np.zeros((self.scatter.scatter_order + 1, 2))
            # sum over L
            for i in range(self.scatter.scatter_order + 1):
                # sum over k in interval m
                for j in range(len(self.quad.w2)):
                    D[i][0] += B_minus_linear(self.quad.mu2[n_mu][j]) * scipy.special.eval_legendre(i,self.quad.mu2[n_mu][j]) * self.quad.w2[j]
                    D[i][1] += B_plus_linear( self.quad.mu2[n_mu][j]) * scipy.special.eval_legendre(i,self.quad.mu2[n_mu][j]) * self.quad.w2[j]
            for i in range(self.scatter.scatter_order + 1):
                self.scatter.D[i][2*n_mu] = D[i][0]
                self.scatter.D[i][2*n_mu+1] = D[i][1]
        
        
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
            
            # gets S vector for scatter source
            if self.scatter.scatter_type == "default":
                S = (sigs*Phi_0[iel] / 2)*np.ones(self.quad.N_dir)
            elif it == 1:
                S = np.zeros(self.quad.N_dir)
            else:
                S = self.scatter.S[iel]

            # first angular cell
            if (n_mu == 0) and quadratic:
                # angular cell midpoint
                mu_1 = 0.5*(mu[0]+mu[1])
                
                # coefficients
                a00 = -2*mu[0]*A[0] + alpha[1]*(A[1]-A[0])/(2*w[0])*B_minus(mu_1) + sigt*V
                a01 = alpha[1]*(A[1]-A[0])/(2*w[0])*B_plus(mu_1)
                a10 = (alpha[2]*B_minus(mu_half) - alpha[1]*B_minus(mu_1))*(A[1]-A[0])/(2*w[1])
                a11 = -2*mu[1]*A[0] + (alpha[2]*B_plus(mu_half) - alpha[1]*B_plus(mu_1)) \
                      *(A[1]-A[0])/(2*w[1]) + sigt*V
                    
                # source terms
                b0 = (S[2*n_mu]+q/2)*V - mu[0]*(A[1]+A[0])*psi_x[0] \
                      - alpha[1]*(A[1]-A[0])/(2*w[0])*B_S(mu_1)*psi_mu[iel]
                b1 = (S[2*n_mu+1]+q/2)*V - mu[1]*(A[1]+A[0])*psi_x[1] \
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
                b0 = (S[2*n_mu]+q/2)*V + np.abs(mu[0])*(A[1]+A[0])*psi_x[0] \
                     + alpha[0]*(A[1]-A[0])/(2*w[0])*psi_mu[iel]
                b1 = (S[2*n_mu+1]+q/2)*V + np.abs(mu[1])*(A[1]+A[0])*psi_x[1]
            
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
            if (n_mu == 0) and quadratic:
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
    
    def cellwise_sweep(self, n_mu, psi_mu, Phi_0, Phi_1, it):
        quadratic = self.quadratic
        # direction quantities
        mu      = self.quad.mu[2*n_mu:2*n_mu+2]
        w       = self.quad.w[2*n_mu:2*n_mu+2]
        alpha   = self.quad.alpha[2*n_mu:2*n_mu+3]
        mu_half = self.quad.mu_half[2*n_mu+2] 
        
        # basis functions
        if (n_mu == 0) and quadratic:
            B_S     = lambda u: ((u-mu[0])*(u-mu[1]))/((-1-mu[0])*(-1-mu[1]))
            B_minus = lambda u: ((u+1)*(u-mu[1]))/((mu[0]+1)*(mu[0]-mu[1]))
            B_plus  = lambda u: ((u+1)*(u-mu[0]))/((mu[1]+1)*(mu[1]-mu[0]))
        else:
            B_minus = lambda u: (mu[1]-u)/(mu[1]-mu[0])
            B_plus  = lambda u: (u-mu[0])/(mu[1]-mu[0])
            B_S     = lambda u: 0*u
        
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
            
        for iel in range(start, stop, inc):
            
            # cell properties
            A     = self.mesh.A[iel:iel+2] 
            dA = A[1] - A[0]
            if (mu[0] < 0):
                A_out = A[0]
            if (mu[0] > 0):
                A_out = A[1]
            V     = self.mesh.V[iel]
            matID = self.mesh.matID[iel]
            sigt  = self.matprops["sigt"][matID]
            sigs  = self.matprops["sigs"][matID]
            q     = self.matprops["q"][matID]
            
            spatial_it = 0
            spatial_it_mat = self.spatial_it
            err = 1
            tol = 1e-6
            # angular cell midpoint
            mu_1 = 0.5*(mu[0]+mu[1])
            
            psi_minus0 = psi_x[0]
            psi_plus0 = psi_x[1]
            if self.do_angular and iel != start:
                psi_minus0 = self.psi[2*n_mu,iel - inc]
                psi_plus0 = self.psi[2*n_mu+1,iel - inc]
            #  psi_minus0, psi_plus0 = 0,0
            while err > tol:
                spatial_it += 1
            
                psi_minus = (sigs*Phi_0[iel]+q)/2*V + np.abs(mu[0])*(A[1]+A[0])*psi_x[0] \
                    + dA/(2*w[0])*(alpha[1]*((1-B_minus(mu_1))*psi_minus0 - B_plus(mu_1)*psi_plus0 \
                    - B_S(mu_1)*psi_mu[iel]) + alpha[0]*psi_mu[iel])
                psi_minus /= 2*np.abs(mu[0])*A_out + alpha[1]*dA/(2*w[0]) + sigt*V
                
                psi_plus = (sigs*Phi_0[iel]+q)/2*V + np.abs(mu[1])*(A[1]+A[0])*psi_x[1] \
                        -1*dA/(2*w[1])*((alpha[2]*B_minus(mu_half) - alpha[1]*B_minus(mu_1))*psi_minus \
                                +(alpha[2]*B_S(mu_half) - alpha[1]*B_S(mu_1))*psi_mu[iel])
                psi_plus /= 2*np.abs(mu[1])*A_out + dA/(2*w[1])*(alpha[2]*B_plus(mu_half) - alpha[1]*B_plus(mu_1)) + sigt*V

                # update angular flux
                if self.do_angular == True:
                    self.psi[2*n_mu,iel]   = psi_minus
                    self.psi[2*n_mu+1,iel] = psi_plus
                
                if np.all(self.matprops['q'] == 0) and np.all(self.psi_bound[:self.quad.N_cells] == 0):
                    err = psi_plus + psi_minus
                else:
                    err = np.sqrt((psi_minus0 - psi_minus)**2 + (psi_plus0 - psi_plus)**2) / np.sqrt(psi_plus**2 + psi_minus**2)
                
                psi_minus0 = psi_minus
                psi_plus0 = psi_plus
                
            if (n_mu == 0) and quadratic:
                psi_mu[iel] = psi_mu[iel]*B_S(mu_half) + psi_minus*B_minus(mu_half) \
                        + psi_plus*B_plus(mu_half)
            else:
                psi_mu[iel] = psi_minus*B_minus(mu_half) + psi_plus*B_plus(mu_half) 
            
            # add flux contribution
            Phi_1[iel] += (psi_minus*w[0] + psi_plus*w[1])
        
            
            # update ingoing fluxes
            psi_x[0] = 2*psi_minus - psi_x[0]
            psi_x[1] = 2*psi_plus  - psi_x[1]
            # err = tol * 0.9
            spatial_it_mat[iel, n_mu] += spatial_it   
        

        self.spatial_it = spatial_it_mat
        # positive-mu sweep
        if (mu[0] > 0):
            # boundary flux
            self.psi_bound[2*n_mu:2*n_mu+2] = psi_x[:]
            
        # return angular fluxes
        return Phi_1, psi_mu
    
    def angular_sweep(self, n_mu, psi_mu, Phi_0, Phi_1, it):
        err = 1
        tol = 1e-6
        
        
        quadratic = self.quadratic
        # direction quantities
        mu      = self.quad.mu[2*n_mu:2*n_mu+2]
        w       = self.quad.w[2*n_mu:2*n_mu+2]
        alpha   = self.quad.alpha[2*n_mu:2*n_mu+3]
        mu_half = self.quad.mu_half[2*n_mu+2]
        
        # basis functions
        if (n_mu == 0) and quadratic:
            B_S     = lambda u: ((u-mu[0])*(u-mu[1]))/((-1-mu[0])*(-1-mu[1]))
            B_minus = lambda u: ((u+1)*(u-mu[1]))/((mu[0]+1)*(mu[0]-mu[1]))
            B_plus  = lambda u: ((u+1)*(u-mu[0]))/((mu[1]+1)*(mu[1]-mu[0]))
        else:
            B_minus = lambda u: (mu[1]-u)/(mu[1]-mu[0])
            B_plus  = lambda u: (u-mu[0])/(mu[1]-mu[0])
            B_S     = lambda u: 0*u
        
        
        
        spatial_it = 0   
        # sweep
        while err > tol:
            spatial_it += 1
            
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
            
            for iel in range(start, stop, inc):
                # cell properties
                A     = self.mesh.A[iel:iel+2] 
                dA = A[1] - A[0]
                if (mu[0] < 0):
                    A_out = A[0]
                if (mu[0] > 0):
                    A_out = A[1]
                V     = self.mesh.V[iel]
                matID = self.mesh.matID[iel]
                sigt  = self.matprops["sigt"][matID]
                sigs  = self.matprops["sigs"][matID]
                q     = self.matprops["q"][matID]
                


                if spatial_it == 1:
                    psi_minus0 = psi_x[0]
                    psi_plus0 = psi_x[1]
                else:
                    psi_minus0 = self.psi[2*n_mu,iel]
                    psi_plus0 = self.psi[2*n_mu+1,iel]
                
                # mu m (mu at cell middle)
                mu_1 = 0.5*(mu[0]+mu[1])
                
                # Solving for psi_minus and psi_plus
                psi_minus = (sigs*Phi_0[iel]+q)/2*V + np.abs(mu[0])*(A[1]+A[0])*psi_x[0] \
                    + dA/(2*w[0])*(alpha[1]*((1-B_minus(mu_1))*psi_minus0 - B_plus(mu_1)*psi_plus0 \
                    - B_S(mu_1)*psi_mu[iel]) + alpha[0]*psi_mu[iel])
                psi_minus /= 2*np.abs(mu[0])*A_out + alpha[1]*dA/(2*w[0]) + sigt*V
                
                psi_plus = (sigs*Phi_0[iel]+q)/2*V + np.abs(mu[1])*(A[1]+A[0])*psi_x[1] \
                        - dA/(2*w[1])*((alpha[2]*B_minus(mu_half) - alpha[1]*B_minus(mu_1))*psi_minus \
                                +(alpha[2]*B_S(mu_half) - alpha[1]*B_S(mu_1))*psi_mu[iel])
                psi_plus /= 2*np.abs(mu[1])*A_out + dA/(2*w[1])*(alpha[2]*B_plus(mu_half) - alpha[1]*B_plus(mu_1)) + sigt*V
                
                # update ingoing fluxes
                psi_x[0] = 2*psi_minus - psi_x[0]
                psi_x[1] = 2*psi_plus  - psi_x[1]
                      
                
                
                # update angular flux
                if self.do_angular == True:
                    self.psi_01[2*n_mu,iel]   = self.psi[2*n_mu,iel]
                    self.psi_01[2*n_mu+1,iel] = self.psi[2*n_mu+1,iel]
                    self.psi[2*n_mu,iel]   = psi_minus
                    self.psi[2*n_mu+1,iel] = psi_plus
                
                
            psi = np.copy(self.psi)
            psi_01 = np.copy(self.psi_01)

            err = np.sqrt(np.sum((self.Apsi[2*n_mu,:] - psi[2*n_mu,:])**2 + (self.Apsi[2*n_mu+1,:] - psi[2*n_mu+1,:]) ** 2) \
                    /(np.sum(self.Apsi[2*n_mu,:]**2 + self.Apsi[2*n_mu+1,:]**2)))
            err = np.max(np.sqrt((psi_01[2*n_mu,:] - psi[2*n_mu,:])**2 + (psi_01[2*n_mu+1,:] - psi[2*n_mu+1,:]) ** 2) \
                    /(np.sqrt(psi[2*n_mu,:]**2 + psi[2*n_mu+1,:]**2)))
            
            
            
            
            
        self.spatial_it[it - 1, n_mu] = spatial_it   
        
        
        # add flux contribution
        Phi_1[:] += (psi[2*n_mu,:]*w[0] + psi[2*n_mu+1,:]*w[1])
        
        
        if (n_mu == 0) and quadratic:
            psi_mu[:] = psi_mu[:]*B_S(mu_half) + psi[2*n_mu,:]*B_minus(mu_half) \
                        + psi[2*n_mu+1,:]*B_plus(mu_half)
        else:
            psi_mu[:] = psi[2*n_mu,:]*B_minus(mu_half) + psi[2*n_mu+1,:]*B_plus(mu_half) 
        
        
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
        self.leak = leak
        # bsource
        bsource = 0
        for i in range(int(self.quad.N_dir/2)):
            bsource += np.abs(self.quad.mu[i])*self.psi_bound[i]*self.quad.w[i]
        bsource *= self.mesh.A[-1]
        # balance parameter
        bal = np.abs(source + bsource - (absorp + leak)) / (source + bsource)
        print()
        print("source:",source,"bsource:",bsource)
        print("absorp:",absorp,"leak:",leak)
        
        return bal
    
    def AnalyticalSolve(self, n, local):
        N_cells = int(n / 2)
        
        # local mu values (Same as in the solution)
        if local:
            # mu-cell boundaries
            mu_half = np.linspace(-1.,1.,int(N_cells+1))
        
            # mu-cell midpoints
            mu = 0.5*(mu_half[:-1] + mu_half[1:])
        
            # local Gauss S2 quadrature
            w  = (1./N_cells)*np.ones(n)
            mu2 = np.zeros(n) 
            for n_mu in range(N_cells):
               mu2[2*n_mu]   = w[2*n_mu]*(-1./np.sqrt(3.))  + mu[n_mu]
               mu2[2*n_mu+1] = w[2*n_mu+1]*(1./np.sqrt(3.)) + mu[n_mu]
        # global mu values
        else:
            mu2, w = np.polynomial.legendre.leggauss(n)
        Aphi = np.zeros(self.mesh.I)
        Apsi = np.zeros((n,self.mesh.I))
        siga = self.matprops["sigt"][0] - self.matprops["sigs"][0]
        # Only works for pure absorber with a boundary source
        if self.matprops["sigs"][0] == 0 and self.matprops["q"][0] == 0:
            for i in range(self.mesh.I):
                for j in range(int(n)):
                    theta1 = (math.acos((mu2[j])))
                    theta2 = (math.pi - math.asin(self.mesh.r[i] / self.mesh.R[1] * (math.sin(theta1))))
                    d = (math.sqrt(self.mesh.r[i] ** 2 + self.mesh.R[1] ** 2 - 2 * self.mesh.r[i] * self.mesh.R[1] * math.cos(theta2 - theta1)))
                    Apsi[j][i] = (getPsi(math.cos(theta2),self.bc) * math.exp(-1 * siga * d))
                    Aphi[i] += Apsi[j][i] * w[j]
        # test form given theta and R
        test = False
        if test:
            theta1 = math.acos(-0.25)
            r = 1/2
            theta2 = (math.pi - math.asin(r / self.mesh.R[1] * (math.sin(theta1))))
            d = (math.sqrt(r ** 2 + self.mesh.R[1] ** 2 - 2 * r * self.mesh.R[1] * math.cos(theta2 - theta1)))
            Apsi_0 = (getPsi(math.cos(theta2),self.bc) * math.exp(-1 * siga * d))
            print(Apsi_0)
        self.Apsi = Apsi
        self.Aphi = Aphi
    
    # Calculates a global L2 error as described in the paper
    def globalL2Error(self):
        err = np.zeros((self.quad.N_dir,self.mesh.I))
        for i in range(self.mesh.I):
            for m in range(self.quad.N_dir):
                err[m][i] = (self.Apsi[m][i] - self.psi[m][i]) ** 2 * self.quad.w[m]
        self.globalErr = np.sqrt(np.sum(err) / self.mesh.I)
    
    # Calculates a relative L2 error as described in the paper
    def relL2Error(self):
        err = np.zeros(self.mesh.I)
        for i in range(self.mesh.I):
            num = 0
            denom = 0
            for m in range(self.quad.N_dir):
                num += (self.Apsi[m][i] - self.psi[m][i]) ** 2 * self.quad.w[m]
                denom += (self.Apsi[m][i]) ** 2 * self.quad.w[m]
            err[i] = np.sqrt(num / denom)
        self.rel_err = err
        
    # Calculates a relative L2 error as described in the paper for the negative mus
    def relL2Error_inc(self):
        err = np.zeros(self.mesh.I)
        for i in range(self.mesh.I):
            num = 0
            denom = 0
            for m in range(int(self.quad.N_dir / 2)):
                num += (self.Apsi[m][i] - self.psi[m][i]) ** 2 * self.quad.w[m]
                denom += (self.Apsi[m][i]) ** 2 * self.quad.w[m]
            err[i] = np.sqrt(num / denom)
        self.rel_err_inc = err
    
    # Calculates a relative L2 error as described in the paper for the positive mus
    def relL2Error_out(self):
        err = np.zeros(self.mesh.I)
        for i in range(self.mesh.I):
            num = 0
            denom = 0
            for m in range(int(self.quad.N_dir / 2), self.quad.N_dir):
                num += (self.Apsi[m][i] - self.psi[m][i]) ** 2 * self.quad.w[m]
                denom += (self.Apsi[m][i]) ** 2 * self.quad.w[m]
            err[i] = np.sqrt(num / denom)
        self.rel_err_out = err
    
    
    # plot solution
    def plot(self, name="sol"):
        plt.figure(1)
        plt.plot(self.mesh.r, self.Phi, label=name)
        # plt.plot(self.mesh.r, self.Aphi)
        plt.xlabel("r (cm)")
        plt.ylabel("Flux")
        plt.title("1D Spherical Transport Solution (PG Linear Discontinous-Diamond Difference)")
        plt.legend()
    
        
    def plotErr(self):
        err = np.zeros_like(self.Phi)
        L21 = np.sqrt(np.sum((self.Phi - self.Aphi) ** 2))
        L22 = np.sqrt(np.sum(self.Aphi ** 2))
        for i in range(len(self.Phi)):
            err[i] = 100 * (self.Aphi[i] - self.Phi[i]) / self.Aphi[i]
        plt.figure(2)
        plt.plot(self.mesh.r, err)
        plt.xlabel("r (cm)")
        plt.ylabel("Error (%)")
        plt.title("PG Spherical Transport Error")
        return L21 , L22
   
    # plot angular fluxes
    def angular(self):
        if self.do_angular == True:
            for nd in range(self.quad.N_dir):
                plt.figure(nd+2)
                plt.plot(self.mesh.r, self.psi[nd,:], 'k--')
                plt.xlabel("r (cm)")
                plt.ylabel("Angular Flux")
    
    def xs_fix(self):
        matprops = self.matprops
        if sd1["sig_l"] == "FP":
            self.matprops["sigs"] = self.scatter.sig_l[0][0] * np.ones_like(matprops["sigs"])
            self.matprops["sigt"] = matprops["sigt"] + self.scatter.sig_l[0][0] * np.ones_like(matprops["sigs"])
            


def getPsi(mu, bc):
    psi = 0.0
    boundType = bc["type"]
    value = bc["value"]
    type1 = boundType[:-1]
    if boundType == "isotropic0":
        return 2.
    type2 = int(boundType[-1:])
    if type2 == 1:
        norm = abs(1 / (-2 + 3/2 * value))
        psi = (1 - ((1 + mu) / value)) * norm
    if type2 == 2:
        if mu < -0.75:
            psi = 1
    if type2 == 3:
        psi = 1
    if type2 == 4:
        norm = abs(1 / (-0.5 * math.log(1 + value) + 0.5 * math.log(value) - 1 / (2 + 2 * value)))
        psi = norm / (1 + value - mu ** 2) - norm / (1 + value)
    if psi < 0:
        psi = 0
    return psi  
        

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

'''
Original Input Values
R = np.array([1.])
I_reg = np.array([40])

bc_dict = {"type":"isotropic","value":0.}

matprops = {"sigt":np.array([1.0]),
            "sigs":np.array([0.0]),
               "q":np.array([1.0])}

quad_dict = {"directions":8,
             "quadrature":"gauss",
                  "alpha":"approximate"}
'''

def inputVals():
    working = True
    input_name = "input.csv"
    if input_name != 'input.csv':
        print('Please do not change the input file name from input.csv.')
        working = False
        
    if working:
        file1 = open(input_name, 'r')
        data = np.genfromtxt(file1, delimiter = ',', dtype=str)
        file1.close()
        
        try:
            N_mats = int(data[0][1])
            R = np.zeros(N_mats)
            I_reg = np.zeros(N_mats)
            N_dir = int(data[3][1])
            bc = {str(data[4][0]):str(data[4][1]),
                  str(data[5][0]):float(data[5][1])}
            sigt = np.zeros(N_mats)
            sigs = np.zeros(N_mats)
            q = np.zeros(N_mats)
            quadrature = str(data[9][1])
            alpha = str(data[10][1])
            quadratic = str(data[11][1])
            if quadratic == "T":
                quadratic = True
            elif quadratic == "F":
                quadratic = False
            else:
                print("Please only use T or F for quadratic")
                working = False
            sweep_type = str(data[12][1])
            scatter_type = str(data[13][1])
            scatter_order = str(data[14][1])
            scatter_order = int(scatter_order)
            sig_l = data[15][1]
            
            for i in range(N_mats):
                R[i] = float(data[1][i + 1])
                I_reg[i] = int(data[2][i + 1])
                sigt[i] = float(data[6][i + 1])
                sigs[i] = float(data[7][i + 1])
                q[i] = float(data[8][i + 1])
            matprops = {str(data[6][0]):sigt,
                        str(data[7][0]):sigs,
                        str(data[8][0]):q}
            md1 = {"R":R,
                   "I_reg":I_reg}
            qd1 = {"directions":N_dir,
                         "quadrature":quadrature,
                         "alpha":alpha}
            sd1 = {"scatter_type":scatter_type,
                   "scatter_order":scatter_order,
                   "sig_l":sig_l}
        except(ValueError):
            print("Make sure all values are the correct type and filled in.")
            working = False
            return 0, 0, 0, 0, 0, 0, working, 0, 0, 0, 0, 0
        else:
            if N_dir % 2 == 0:
                return md1, qd1, bc, matprops, input_name, working, quadratic, sweep_type, sd1
            else:
                return 1, 0, 0, 0, 0, 0, False, 0, 0, 0, 0, 0


def output(solved, name = "output\\output"):
    with open( name + "_phi.csv", "wb") as a:
        np.savetxt(a, solved.Phi, delimiter=",")
    if solved.do_angular:
        with open(name +"_psi.csv", "wb") as a:
            np.savetxt(a, np.transpose(solved.psi), delimiter=",")

def output_rel_err(solved, name):
    with open(name + "_rel_err.csv", "wb") as a:
        np.savetxt(a, solved.rel_err, delimiter=",")

def output_rel_err_inc(solved, name):
    with open(name + "_rel_err_inc.csv", "wb") as a:
        np.savetxt(a, solved.rel_err_inc, delimiter=",")

def output_rel_err_out(solved, name):
    with open(name + "_rel_err_out.csv", "wb") as a:
        np.savetxt(a, solved.rel_err_out, delimiter=",")

def balance_test():
    md1, qd1, bc, matprops, name, working, quadratic, sweep_type, sd1= inputVals()
    N_dir = qd1["directions"]
    GQ2_bal = np.zeros(N_dir)
    Sn_bal = np.zeros(N_dir)
    tol = np.zeros(N_dir)
    for i in range(N_dir):
        sd1["scatter_order"] = i
        sd1["sig_l"] = np.ones(i + 1)
        sd1["scatter_type"] = "GQ2"
        GQ2_sol = Solve(md1, qd1, bc, matprops, quadratic, sweep_type, sd1, True)
        GQ2_sol.solve()
        sd1["scatter_type"] = "Sn"
        Sn_sol = Solve(md1, qd1, bc, matprops, quadratic, sweep_type, sd1, True)
        Sn_sol.solve()
        GQ2_bal[i] = GQ2_sol.bal
        Sn_bal[i] = Sn_sol.bal
        tol[i] = 1e-6
    x = np.linspace(0,N_dir - 1,N_dir)
    plt.figure(2)
    plt.semilogy(x,GQ2_bal, label="GQ2")
    plt.semilogy(x,Sn_bal, label="Sn")
    plt.semilogy(x,tol, label="tol")
    plt.title("Balance as a Function of Scatter Order")
    plt.xlabel("Scatter Order (L)")
    plt.ylabel("Balance")
    plt.legend()


def scatter_absorp_test():
    md1, qd1, bc, matprops, name, working, quadratic, sweep_type, sd1= inputVals()
    scatter_sol = Solve(md1, qd1, bc, matprops, quadratic, sweep_type, sd1, True)
    scatter_sol.solve()
    matprops["sigt"] = np.array(0.5*matprops["sigt"])
    matprops["sigs"] = np.zeros(1)
    absorp_sol = Solve(md1, qd1, bc, matprops, quadratic, sweep_type, sd1, True)
    absorp_sol.solve()
    scatter_sol.plot("scatter")
    absorp_sol.plot("absorp")
    return scatter_sol, absorp_sol

def plot_input(name, solved):
    with open(name, "r") as a:
        input_phi = np.genfromtxt(a, delimiter=",")
    plt.figure()
    plt.plot(solved.mesh.r, input_phi, label="WD Sn")
    plt.plot(solved.mesh.r, solved.Phi, label="PG GQ2")
    plt.xlabel("R")
    plt.ylabel("Flux")
    plt.title("WD Sn vs PG GQ2")
    plt.legend()
    plt.show()
        

   



md1, qd1, bc, matprops, name, working, quadratic, sweep_type, sd1= inputVals()


if working:
    
    sol8 = Solve(md1, qd1, bc, matprops, quadratic, sweep_type, sd1, True)
    sol8.AnalyticalSolve(qd1["directions"], local = True)
    sol8.solve()
    sol8.plot()
    # scatter_sol, absorp_sol = scatter_absorp_test()
    # balance_test()
    # phi = plot_input("output\\output_phi.csv", sol8)
    # output(sol8, "output//S64_LDPQ_HighlyAnisotropic")
    sol8.globalL2Error()
    sol8.relL2Error()
    sol8.relL2Error_inc()
    sol8.relL2Error_out()
    # output(sol8)
    name2 = "output\\S64_LDP"
    output_rel_err(sol8, name2)
    output_rel_err_inc(sol8, name2)
    output_rel_err_out(sol8, name2)
    
    '''
    sol8.globalL2Error()
    sol8.relL2Error()
    sol8.relL2Error_inc()
    sol8.relL2Error_out()
    output(sol8)
    output_rel_err(sol8, "output\\S8_LDP")
    output_rel_err_inc(sol8, "output\\S8_LDP")
    output_rel_err_out(sol8, "output\\S8_LDP")
    print("Global Error:", sol8.globalErr)
    print(sol8.calc_time, "seconds")
    if sweep_type == "cellwise":
        print(np.average(sol8.spatial_it))
    if sweep_type == "angular":
        print(np.average(sol8.spatial_it[0]))
        '''

elif md1["R"] == 1:
    print("Please have N_dir be an even integer.")

plt.show()