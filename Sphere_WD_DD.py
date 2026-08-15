# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:16:48 2024

@author: c.woodsford.9788
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy

# mesh class
class Mesh:
    def __init__(self, matIDs, R, I_reg):
        # number of cells and regions 
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
    def __init__(self, N_dir, globe):
        # number of directions
        self.N_dir = N_dir
        N_cells = int(N_dir/2)
        
        # sets up global gauss points for global gauss Sn solution
        if globe:
            self.mu, self.w = np.polynomial.legendre.leggauss(N_dir)
            mu_half = np.zeros(N_dir + 1)
            mu_half[0] = -1
            for i in range(1, N_dir+1):
                mu_half[i] = mu_half[i-1] + self.w[i-1]
            self.mu_half = mu_half
        
        # sets up N/2 local S_2 Quadratures and S_N/2 + 1 Quadratures
        else:
            # mu-cell midpoints
            mu_half = np.linspace(-1.,1.,N_cells+1)
            mu = 0.5*(mu_half[:-1] + mu_half[1:])
        
            # local Gauss S2 quadrature
            self.w = (1./N_cells)*np.ones((N_cells,2))
            self.mu = np.zeros((N_cells,2))
            self.mu[:,0] = self.w[:,0]*(-1./np.sqrt(3.)) + mu[:]
            self.mu[:,1] = self.w[:,1]*(1./np.sqrt(3.)) + mu[:]
        
            # local Gauss S(N/2 + 1) quadrature
            temp_mu, temp_w = np.polynomial.legendre.leggauss(N_cells + 1)
            mu2 = np.zeros((N_cells, N_cells + 1))
            w2 = (1/N_cells)*temp_w
            for i in range(N_cells + 1):
                mu2[:,i] = (1/N_cells)*temp_mu[i] + mu[:]
            
            # S_N/2+1 quadrature set
            self.mu2 = mu2
            self.w2 = w2

            # S_2 quadrature set
            self.mu = self.mu.flatten()
            self.w = self.w.flatten()
        
            # mu-cell boundaries and alpha
            self.mu_half = np.linspace(-1.,1.,self.N_dir+1)
        self.alpha = np.zeros(self.N_dir+1)
        for nd in range(N_dir):
            self.alpha[nd+1] = self.alpha[nd] - 2*self.mu[nd]*self.w[nd]
        
        # beta
        self.beta = np.zeros(N_dir)
        for nd in range(N_dir):
            self.beta[nd] = (self.mu[nd] - self.mu_half[nd])/\
                            (self.mu_half[nd+1] - self.mu_half[nd])

# scatter class
class Scatter:
    def __init__(self, sd1, quad, matprops):
        # input values for scattering
        self.scatter_type = sd1["scatter_type"]
        self.scatter_order = sd1["scatter_order"]
        self.sig_l_type = sd1["sig_l"]
        
        sig_s = matprops["sigs"]
        sig_t = matprops["sigt"]
        
        # creates sig_l matrix for a given input string
        # sig_l = I
        if sd1["sig_l"] == "delta":
            sig_l = np.ones(self.scatter_order + 1) * sig_s[0]
        
        # Henyey-Greenstein
        # sig_l[i] = sig_s * mu_bar ^ i
        # sig_s(1 - mu_bar) = 1
        elif sd1["sig_l"][0:2] == "HG":
            sig_l = np.zeros(self.scatter_order + 1)
            sig_l[0] = sig_s[0]
            mu_bar = 1 - (1/sig_l[0])
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
        
        # Fokker-Planck 2
        # sig_L = -sig_0
        # sig_0 - sig_l = 1/2 * l * (l + 1)
        elif sd1["sig_l"] == "FP2":
            sig_l = np.zeros(self.scatter_order + 1)
            sig_l[0] = 0.25*self.scatter_order*(self.scatter_order + 1)
            for i in range(1,self.scatter_order + 1):
                sig_l[i] = sig_l[0] - 0.5*i*(i + 1)
        
        # Truncated Fokker-Planck
        # sig_trunc = 0
        # sig_0 - sig_l = 1/2 * l * (l + 1)
        elif sd1["sig_l"][0:3] == "FPT":
            trunc = int(sd1["sig_l"][3:])
            sig_l = np.zeros(self.scatter_order + 1)
            sig_l[0] = 0.5*trunc*(trunc + 1)
            for i in range(1,trunc + 1):
                sig_l[i] = sig_l[0] - 0.5*i*(i + 1)
        
        # Isotropic
        # sig_0 = sig_s
        # sig_l = 0 for l != 0
        elif sd1["sig_l"] == "isotropic":
            sig_l = np.zeros(self.scatter_order + 1)
            sig_l[0] = sig_s[0]
        
        # turns the 1-D array into a 2-D array
        self.sig_l = np.diag(sig_l)
        '''
        # test truncate cross sectrion
        for l in range(self.scatter_order + 1):
            if l >= 6:
                self.sig_l[l][l] = 0
        '''
        # creates D & M matrices
        if self.scatter_type != "default":
            self.D = np.zeros((self.scatter_order + 1, quad.N_dir))
            self.M = np.zeros((quad.N_dir, self.scatter_order + 1))
        # forms D & M matrices for Sn
        if self.scatter_type == "Sn":
            P_mat = np.zeros((self.scatter_order+1,quad.N_dir))
            for i in range(quad.N_dir):
                for j in range(self.scatter_order + 1):
                    self.D[j][i] = scipy.special.eval_legendre(j, quad.mu[i])*quad.w[i] 
                    self.M[i][j] = (2*j + 1)/2 * scipy.special.eval_legendre(j, quad.mu[i])
                    #P_mat[i][j] = scipy.special.eval_legendre(i, quad.mu[j])
            #u, s, v = np.linalg.svd(P_mat)
            #print("SVD Cond #:",abs(np.max(s)/np.min(s)))
        
        # forms D & M matrices for GQ2
        elif self.scatter_type == "GQ2":
            # loops over angular cells for D matrix construction
            for n_mu in range(int(quad.N_dir/2)):
                mu      = quad.mu[2*n_mu:2*n_mu+2]
                B_minus_linear = lambda u: (mu[1]-u)/(mu[1]-mu[0])
                B_plus_linear  = lambda u: (u-mu[0])/(mu[1]-mu[0])
                
                # creates D matrix for scattering
                if self.scatter_type == 'GQ2':
                    D = np.zeros((self.scatter_order + 1, 2))
                    # sum over L
                    for i in range(self.scatter_order + 1):
                        # sum over k in interval m
                        for j in range(len(quad.w2)):
                            D[i][0] += B_minus_linear(quad.mu2[n_mu][j]) * scipy.special.eval_legendre(i,quad.mu2[n_mu][j]) * quad.w2[j]
                            D[i][1] += B_plus_linear( quad.mu2[n_mu][j]) * scipy.special.eval_legendre(i,quad.mu2[n_mu][j]) * quad.w2[j]
                    for i in range(self.scatter_order + 1):
                        self.D[i][2*n_mu] = D[i][0]
                        self.D[i][2*n_mu+1] = D[i][1]
            self.M = scipy.linalg.solve(self.D, np.eye(quad.N_dir))
        
        elif self.scatter_type == "GQ3":
            P_mat = np.zeros((self.scatter_order+1,quad.N_dir))
            for i in range(self.scatter_order+1):
                for j in range(quad.N_dir):
                    P_mat[i][j] = scipy.special.eval_legendre(i, quad.mu[j])
            print("Legendre Polynomials Cond #:",np.linalg.cond(P_mat))
            u, s, v = np.linalg.svd(P_mat)
            print("SVD Cond #:",abs(np.max(s)/np.min(s)))
            Q = weighted_Gram_Schmidt(P_mat, quad.w)
            Q_saved = np.copy(Q)
            for l in range(self.scatter_order+1):
                val = inner_prod(Q[l], Q[l], quad.w)
                Q[l] = np.copy(Q[l])/np.sqrt(val)
                Q[l] = np.copy(Q[l])*np.sqrt(2/(2*l+1))
            '''
            # Used to check normalization
            for l in range(self.scatter_order+1):
                for l2 in range(self.scatter_order+1):
                    if l == l2:
                        print(inner_prod(Q[l], Q[l2], quad.w))
                        print(2/(2*l+1))
            '''
            
            D = np.zeros((self.scatter_order+1,quad.N_dir))
            M = np.zeros((quad.N_dir,self.scatter_order+1))
            for i in range(quad.N_dir):
                for j in range(self.scatter_order+1):
                    D[j][i] = Q[j][i] * quad.w[i]
                    M[i][j] = (2*j+1)/2 * Q[j][i]
            self.M = M
            self.D = D
        
        
        
        # sanity check
        if self.scatter_type != "default":
            self.MD = self.M @ self.D
            print("MD=I:",np.allclose(self.MD, np.eye(quad.N_dir)))
            # gives sig_t
            if sd1["sig_l"][0:2] == "FP":
                sigt = (sig_t[0] - sig_s[0] + self.sig_l[0][0])
            else:
                sigt = sig_t[0]
            MΣD = self.M @ self.sig_l @ self.D
            print("Max eig:", np.max(np.linalg.eig(MΣD)[0])/sigt)
            print("Min eig:", np.min(np.linalg.eig(MΣD)[0])/sigt)
            self.MΣD = MΣD
            print("D cond #:",np.linalg.cond(self.D))
            # print("Test:",np.linalg.svd(MΣD/self.matprops["sigt"][0], compute_uv=False)[-1])
            print("All eigs:",np.linalg.eig(MΣD)[0]/sigt)

        # gives corrected scattering ratio
        if (self.sig_l_type[0:2] != "FP"):
            self.c = sig_s[0] / sig_t[0]
        elif self.sig_l_type[0:2] == "FP":
            self.c = self.sig_l[0][0] / (sig_t[0] - sig_s[0] + self.sig_l[0][0])
        else:
            self.c = 0
        
    # increases truncation of FPT during a run
    # testing to see if this will converge
    def increase_trunc(self,L):
        sig_l = np.zeros(self.scatter_order + 1)
        sig_l[0] = 0.5*L*(L + 1)
        for i in range(1,L + 1):
            sig_l[i] = sig_l[0] - 0.5*i*(i + 1)
        
        # turns the 1-D array into a 2-D array
        self.sig_l = np.diag(sig_l)
       
# solver class    
class Solve:
    def __init__(self, R, I_reg, N_dir, bc, matprops, scatter_dict, do_angular=False, globe=False):
        nmats = len(matprops["sigt"])
        matIDs = np.arange(nmats)
        
        # initialization
        self.mesh = Mesh(matIDs, R, I_reg)
        self.quad = Quad(N_dir, globe)
        self.bc = bc
        self.matprops = matprops
        self.scatter = Scatter(scatter_dict, self.quad, matprops)
        
        # fixes cross section values for Fokker-Planck formulations
        if (sd1["sig_l"][0:2] == "FP"):
            self.xs_fix()
        
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
        if (self.bc["type"] == "isotropic0"):
            self.psi_bound[:int(self.quad.N_dir/2)] = self.bc["value"]/2.
        # anisotropic flux boundary condition
        type1 = self.bc["type"][:-1]
        if (type1 == "anisotropic"):
            for i in range(int(self.quad.N_dir / 2)):
                self.psi_bound[i] = getPsi(self.quad.mu[i],self.bc)
                if self.psi_bound[i] < 0:
                    self.psi_bound[i] = 0
        
        if self.bc["type"] == "anisotropic5":
            self.psi_bound[0] = 1.
            self.psi_bound[1:] = 0.
        if self.bc["type"] == "anisotropic6":
            for i in range(int(N_dir/2)):
                mu_bar = -0.25
                self.psi_bound[i] = 1/2*(1-mu_bar**2)/(1+mu_bar**2-2*mu_bar*self.quad.mu[i])**(3/2)
        
        # Normalizes to a Unit Net Current
        sum1 = 0
        for i in range(int(self.quad.N_dir / 2)):
            sum1 += self.psi_bound[i] * self.quad.w[i] * abs(self.quad.mu[i])
        for i in range(int(self.quad.N_dir / 2)):
            if sum1 != 0:
                self.psi_bound[i] = self.psi_bound[i] / sum1
        self.current_norm = sum1
            
        
        # initial guess
        Phi_0, Phi_m1 = np.zeros(self.mesh.I), np.zeros(self.mesh.I)
        
        # source iteration
        err, tol = 1, 1e-4
        it = 0
        
        print("\nIteration :: Error")
        while (err > tol) and it < 25000:
            # creates the S vectors for each spatial cell using previous results
            if self.scatter.scatter_type !="default":
                S = np.zeros((self.mesh.I,self.quad.N_dir))
                phi_moments = np.zeros((self.mesh.I,self.scatter.scatter_order+1))
                scatter_moments = np.zeros((self.mesh.I,self.scatter.scatter_order+1))
                psi_temp = np.copy(self.psi)
                psi = psi_temp.T
                for i in range(self.mesh.I):
                    phi_moments[i] = self.scatter.D @ psi[i]
                    scatter_moments[i] = self.scatter.sig_l @ phi_moments[i]
                    # S[i] = scipy.linalg.lu_solve((lu,piv), scatter_moments[i])
                    S[i][:] = self.scatter.M @ scatter_moments[i]
                self.scatter.phi_moments = phi_moments
                self.scatter.scatter_moments = scatter_moments
                self.scatter.S = S
            else:
                self.scatter.S = np.zeros((self.mesh.I,self.quad.N_dir))
            
            
            # scalar flux iterate
            Phi_1 = np.zeros(self.mesh.I)
                        
            # starting direction sweep
            self.psi_0, psi_mu = self.start(Phi_0)
            self.psi_mu = np.copy(psi_mu)
            
            # sweeps
            for nd in range(self.quad.N_dir):
                Phi_1, psi_mu = self.sweep(nd, psi_mu, Phi_0, Phi_1, it)
            it += 1             
            
            # calculate error
            if (it == 1):
                err = 1
            elif self.scatter.scatter_type == "default":
                delta = np.sqrt(np.sum((Phi_1 - Phi_0)**2) / np.sum(Phi_1**2))
                rho = np.sum(np.abs(Phi_1 - Phi_0))/np.sum(np.abs(Phi_0 - Phi_m1))
                err = delta/np.abs(1 - rho)
            else:
                delta = np.max(abs((Phi_1 - Phi_0) / (Phi_1+1e-300)))
                rho = self.scatter.c
                err = delta/abs(1 - rho)
            print(str(it)+"\t\t  :: "+str(err))
            
            # next iteration
            Phi_m1 = Phi_0
            Phi_0 = Phi_1
            
            
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
        mu0, mu1 = self.quad.mu[0], self.quad.mu[1]
        psi_x = self.psi_bound[0]*(mu1 + 1)/(mu1 - mu0) - \
                self.psi_bound[1]*(mu0 + 1)/(mu1 - mu0)

        # psi_x = getPsi(-1, self.bc)
            
        # starting direction sweep
        psi_mu = np.zeros(self.mesh.I) 
        psi_mu2 = np.zeros(self.mesh.I) 
        for iel in range(self.mesh.I-1,-1,-1):
            # cell properties
            dr    = self.mesh.dr[iel]
            matID = self.mesh.matID[iel]
            sigt  = self.matprops["sigt"][matID]
            sigs  = self.matprops["sigs"][matID]
            q     = self.matprops["q"][matID]
            
            psi_temp = np.copy(self.psi)
            
            # calculate cell-average angular flux
            if self.scatter.scatter_type == "default":
                psi_mu[iel] = psi_x + (sigs*Phi[iel] + q)*dr/4
                psi_mu[iel] /= (1 + sigt*dr/2)
            else:
                interp1 = self.scatter.S[iel][0]
                interp2 = self.scatter.S[iel][1]
                interp_S0 = (-1-self.quad.mu[1])/(self.quad.mu[0]-self.quad.mu[1])*(interp1-interp2)+interp2
                if interp_S0 < 0:
                    interp_S0 = 0
                

                psi_mu[iel] = psi_x + (interp_S0 + q/2)*dr/2
                psi_mu[iel] /= (1 + sigt*dr/2)
            
            
            # update ingoing flux
            psi_x = 2*psi_mu[iel] - psi_x
        # return origin and starting direction angular flux
        psi_0 = psi_x
        return psi_0, psi_mu 
    
    # sweep function
    def sweep(self, nd, psi_mu, Phi_0, Phi_1, it):
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
            
            # gets S vector for scatter source
            if self.scatter.scatter_type == "default":
                S = (sigs*Phi_0[iel] / 2)*np.ones(self.quad.N_dir)
            else:
                S = self.scatter.S[iel]
            
            # calculate cell-average angular flux
            psi = (S[nd] + q/2)*V + np.abs(mu)*(A[1] + A[0])*psi_x \
                + (A[1] - A[0])/(2*w)*(alpha[1]*(1/beta - 1) + alpha[0])*psi_mu[iel]
            psi /= (2*np.abs(mu)*A_out + alpha[1]*(A[1] - A[0])/(2*beta*w) + sigt*V)
            
            # update angular flux
            if self.do_angular == True:
                self.psi[nd,iel]   = psi
            
            # add flux contribution
            Phi_1[iel] += psi*w
            
            # update ingoing fluxes
            psi_x = 2*psi - psi_x
            psi_mu[iel] = (psi - (1-beta)*psi_mu[iel])/beta
        
        # positive-mu sweep
        if (mu > 0):
            # boundary flux
            self.psi_bound[nd] = psi_x
        # print(sigt, sigs)     
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
        print("Source:", source, "Bsource:", bsource)
        print("Absorp:", absorp, "Leak:", leak)
        return bal

    def AnalyticalSolve(self, n, local):
        N_cells = int(n / 2)
        
        # mu-cell boundaries
        if local:
            mu_half = np.linspace(-1.,1.,int(N_cells+1))
        
            # mu-cell midpoints
            mu = 0.5*(mu_half[:-1] + mu_half[1:])
        
            # local Gauss S2 quadrature
            w  = (1./N_cells)*np.ones(n)
            mu2 = np.zeros(n) 
            for n_mu in range(N_cells):
               mu2[2*n_mu]   = w[2*n_mu]*(-1./np.sqrt(3.))  + mu[n_mu]
               mu2[2*n_mu+1] = w[2*n_mu+1]*(1./np.sqrt(3.)) + mu[n_mu]
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
        self.Apsi = Apsi
        self.Aphi = Aphi
    
    # Calculates a global L2 error as described in the paper
    def globalL2Error(self):
        err = np.zeros((self.quad.N_dir,self.mesh.I))
        for i in range(self.mesh.I):
            for m in range(self.quad.N_dir):
                err[m][i] = (self.Apsi[m][i] - self.psi[m][i]) ** 2 * self.quad.w[m]
        self.err = err
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
    def plot(self, name="sol", line="-", col = "r"):
        plt.figure(1)
        plt.plot(self.mesh.r, self.Phi, label=name, ls=line, c=col)
        # plt.plot(self.mesh.r, self.Aphi)
        # plt.ticklabel_format(axis='y', style='plain', useOffset=False)
        plt.xlabel("r (cm)")
        plt.ylabel("Flux")
        plt.legend()
        #plt.title("1D Spherical Transport Solution (Weighted Diamond-Diamond Difference)")
    
    
    def plotErr(self):
        err = np.zeros_like(self.Phi)
        L21 = np.sqrt(np.sum((self.Phi - self.Aphi) ** 2))
        L22 = np.sqrt(np.sum(self.Aphi ** 2))
        for i in range(len(self.Phi)):
            err[i] = 100 * (self.Aphi[i] - self.Phi[i]) / self.Aphi[i]
        self.err = err
        plt.figure(2)
        plt.plot(self.mesh.r, self.err)
        plt.xlabel("r (cm)")
        plt.ylabel("Error (%)")
        plt.title("WD Spherical Transport Error")
        return L21 , L22
 
        
    # plot angular fluxes
    def angular(self):
        if self.do_angular == True:
            for nd in range(self.quad.N_dir):
                plt.figure(nd+2)
                plt.plot(self.mesh.r, self.psi[nd,:], 'r')
                plt.xlabel("r (cm)")
                plt.ylabel("Angular Flux")
    
    def xs_fix(self):
        matprops = self.matprops
        self.matprops["sigt"] = matprops["sigt"] - matprops["sigs"] + self.scatter.sig_l[0][0] * np.ones_like(matprops["sigs"])
        self.matprops["sigs"] = self.scatter.sig_l[0][0] * np.ones_like(matprops["sigs"])
        

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

'''       
  Original Input Values  
R = np.array([1.])
I_reg = np.array([40])
N_dir = 8 

bc = {"type":"isotropic","value":0.}

matprops = {"sigt":np.array([1.0]),
            "sigs":np.array([0.0]),
               "q":np.array([1.0])}
'''

def inputVals():
    working = True
    name = "input.csv"
    if name != 'input.csv':
        print('Please do not change the input file name from input.csv.')
        working = False
        
    if working:
        file1 = open(name, 'r')
        data = np.genfromtxt(file1, delimiter = ',', dtype=str)
        file1.close()
        
        try:
            R = np.zeros(np.size(data[1]) - 1)
            I_reg = np.zeros(np.size(data[1]) - 1)
            N_dir = int(data[3][1])
            bc = {str(data[4][0]):str(data[4][1]),
                  str(data[5][0]):float(data[5][1])}
            sigt = np.zeros(np.size(data[1]) - 1)
            sigs = np.zeros(np.size(data[1]) - 1)
            q = np.zeros(np.size(data[1]) - 1)
            for i in range(np.size(data[1]) - 1):
                R[i] = float(data[1][i + 1])
                I_reg[i] = int(data[2][i + 1])
                sigt[i] = float(data[6][i + 1])
                sigs[i] = float(data[7][i + 1])
                q[i] = float(data[8][i + 1])
            matprops = {str(data[6][0]):sigt,
                        str(data[7][0]):sigs,
                        str(data[8][0]):q}
            sweep_type = str(data[12][1])
            scatter_type = str(data[13][1])
            scatter_order = str(data[14][1])
            scatter_order = int(scatter_order)
            sig_l = data[15][1]
            
            md1 = {"R":R,
                   "I_reg":I_reg}
            sd1 = {"scatter_type":scatter_type,
                   "scatter_order":scatter_order,
                   "sig_l":sig_l}
        except(ValueError):
            print("Make sure all values are the correct type and filled in.")
            working = False
            return 0, 0, 0, 0, 0, 0, working, 0
        else:
            if N_dir % 2 == 0:
                return R, I_reg, N_dir, bc, matprops, name, working, sd1
            else:
                return 1, 0, 0, 0, 0, 0, working, 0


def output(solved, name = "output\\output"):
    with open( name + "_phi.csv", "wb") as a:
        np.savetxt(a, solved.Phi, delimiter=",")
    '''
    if solved.do_angular:
        with open(name +"_psi.csv", "wb") as a:
            np.savetxt(a, np.transpose(solved.psi), delimiter=",")
    '''

def output_Asol(solved):
    with open("output\\output_Aphi.csv", "wb") as a:
        np.savetxt(a, solved.Aphi, delimiter=",")
    if solved.do_angular:
        with open("output\\output_Apsi.csv", "wb") as a:
            np.savetxt(a, np.transpose(solved.Apsi), delimiter=",")

def output_rel_err(solved, name):
    with open(name + "_rel_err.csv", "wb") as a:
        np.savetxt(a, solved.rel_err, delimiter=",")

def output_rel_err_inc(solved, name):
    with open(name + "_rel_err_inc.csv", "wb") as a:
        np.savetxt(a, solved.rel_err_inc, delimiter=",")

def output_rel_err_out(solved, name):
    with open(name + "_rel_err_out.csv", "wb") as a:
        np.savetxt(a, solved.rel_err_out, delimiter=",")

def scatter_absorp_test():
    R, I_reg, N_dir, bc, matprops, name, working, sd1 = inputVals()
    scatter_sol = Solve(R, I_reg, N_dir, bc, matprops, sd1, do_angular=True, globe=False)
    scatter_sol.solve()
    matprops2 = {"sigt":[1.],
                "sigs":[0.],
                "q":[0.]}
    absorp_sol = Solve(R, I_reg, N_dir, bc, matprops2, sd1, do_angular=True, globe=False)
    absorp_sol.solve()
    scatter_sol.plot("scatter")
    absorp_sol.plot("absorp", (0, (3, 4, 3, 4)), "k")
    return scatter_sol, absorp_sol

def output_flux_moments(solved, name="output\\output"):
    flux_moment = np.zeros((solved.mesh.I,solved.quad.N_dir))
    psi = solved.psi.T
    for iel in range(solved.mesh.I):
        flux_moment[iel] = solved.scatter.D @ psi[iel]
    with open( name + "_flux_moments.csv", "wb") as a:
        np.savetxt(a, flux_moment, delimiter=",")

# Orthonormailzation for GQ3
def inner_prod(u, v, w):
    return np.sum(u*v*w)

def weighted_Gram_Schmidt(V, w):
    U = []
    for v in V:
        u = np.copy(v)
        for prev_u in U:
            proj = np.dot(v, prev_u) / np.dot(prev_u, prev_u) * prev_u
            u = u - proj
        U.append(u)
    U = np.array(U)
    return U

R, I_reg, N_dir, bc, matprops, name, working, sd1 = inputVals()

L2 = False
everything = False

if working:
    glob = True
    sol_GGSn = Solve(R, I_reg, N_dir, bc, matprops, sd1, do_angular=True, globe=glob)
    sol_GGSn.solve()
    # sol8.AnalyticalSolve(N_dir, local = True)
    sol_GGSn.plot()
    if glob:
        temp_name = "_GG_"
    else:
        temp_name = "_LG_"
    name2 = "output//aniso3_" + sd1["sig_l"] + "_S" + str(N_dir) +"P" + str(sd1["scatter_order"]) + temp_name + sd1["scatter_type"]
    # output_flux_moments(sol_GGSn, name2)
    output(sol_GGSn, name2)
    # output(sol8, "output//WD_S8_HG")
    # scat, absorp = scatter_absorp_test()
    '''
    sol8.globalL2Error()
    sol8.relL2Error()
    sol8.relL2Error_inc()
    sol8.relL2Error_out()
    # output(sol8)
    
    name2 = "output\\S64_WD"
    output_rel_err(sol8, name2)
    output_rel_err_inc(sol8, name2)
    output_rel_err_out(sol8, name2)
    
    
    sol8.AnalyticalSolve(N_dir, local = True)
    sol8.plot()
    sol8.globalL2Error()
    sol8.relL2Error()
    sol8.relL2Error_inc()
    sol8.relL2Error_out()
    output(sol8)
    output_rel_err(sol8)
    output_rel_err_inc(sol8)
    output_rel_err_out(sol8)
    print("Global Error:", sol8.globalErr)
    '''

elif R == 1:
    print("Please have N_dir be an even integer.")
    
plt.show()