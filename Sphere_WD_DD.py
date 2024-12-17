# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:16:48 2024

@author: c.woodsford.9788
"""

import numpy as np
import math
import matplotlib.pyplot as plt



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
    def __init__(self, N_dir):
        # number of directions
        self.N_dir = N_dir
        N_cells = int(N_dir/2)
        
        # mu-cell midpoints
        mu_half = np.linspace(-1.,1.,N_cells+1)
        mu = 0.5*(mu_half[:-1] + mu_half[1:])
        
        # local Gauss S2 quadrature
        self.w = (1./N_cells)*np.ones((N_cells,2))
        self.mu = np.zeros((N_cells,2))
        self.mu[:,0] = self.w[:,0]*(-1./np.sqrt(3.)) + mu[:]
        self.mu[:,1] = self.w[:,1]*(1./np.sqrt(3.)) + mu[:]
        
        # quadrature set
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
                    self.psi_bound[i] = 0
            '''
        sum1 = 0
        for i in range(int(self.quad.N_dir / 2)):
            sum1 += self.psi_bound[i] * self.quad.w[i] * self.quad.mu[i]
        for i in range(int(self.quad.N_dir / 2)):
            self.psi_bound[i] = -1 * self.psi_bound[i] / sum1
            '''
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
                Phi_1, psi_mu = self.sweep(nd, psi_mu, Phi_0, Phi_1)
                
            # calculate error
            if (it == 1):
                err = 1
            else:
                delta = np.sqrt(np.sum(((Phi_1 - Phi_0)/Phi_1)**2))
                rho = np.sum(np.abs(Phi_1 - Phi_0))/np.sum(np.abs(Phi_0 - Phi_m1))
                err = delta/np.abs(1 - rho)
            print(str(it)+"\t\t  :: "+str(err))
            
            # next iteration
            Phi_m1 = Phi_0
            Phi_0 = Phi_1
            
        # converged solution
        self.Phi = Phi_0
        
        # balance parameter
        if self.do_balance:
            bal = self.balance()
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
    def sweep(self, nd, psi_mu, Phi_0, Phi_1):
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
            
            # calculate cell-average angular flux
            psi = (sigs*Phi_0[iel] + q)/2*V + np.abs(mu)*(A[1] + A[0])*psi_x \
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
        return bal

    def AnalyticalSolve(self, n):
        
        N_cells = int(n / 2)
        
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
        
        Aphi = np.zeros(self.mesh.I)
        Apsi = np.zeros((n,self.mesh.I))
        siga = self.matprops["sigt"][0] - self.matprops["sigs"][0]
        if self.matprops["sigs"][0] == 0 and self.matprops["q"][0] == 0:
            for i in range(self.mesh.I):
                for j in range(int(n)):
                    theta1 = math.acos(mu2[j])
                    theta2 = math.pi - math.asin(self.mesh.r[i] / self.mesh.R[1] * math.sin(theta1))
                    d = math.sqrt(self.mesh.r[i] ** 2 + self.mesh.R[1] ** 2 - 2 * self.mesh.r[i] * self.mesh.R[1] * math.cos(theta2 - theta1))
                    Apsi[j][i] = getPsi(math.cos(theta2),self.bc) * math.exp(-1 * siga * d)
                    Aphi[i] += Apsi[j][i] * 2 / n
        self.Apsi = Apsi
        self.Aphi = Aphi
    
    
    # plot solution
    def plot(self):
        plt.figure(1)
        plt.plot(self.mesh.r, self.Phi, 'r')
        plt.plot(self.mesh.r, self.Aphi)
        plt.xlabel("r (cm)")
        plt.ylabel("Flux")
        plt.title("1D Spherical Transport Solution (Weighted Diamond-Diamond Difference)")
    
    
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
        

def getPsi(mu, bc):
    psi = 0.0
    boundType = bc["type"]
    value = bc["value"]
    type1 = boundType[:-1]
    if boundType == "isotropic0":
        return value / 2
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
        except(ValueError):
            print("Make sure all values are the correct type and filled in.")
            working = False
            return 0, 0, 0, 0, 0, 0, working
        else:
            if N_dir % 2 == 0:
                return R, I_reg, N_dir, bc, matprops, name, working
            else:
                return 1, 0, 0, 0, 0, 0, working


def output(solved):
    with open("output_phi.csv", "wb") as a:
        np.savetxt(a, solved.Phi, delimiter=",")
    if solved.do_angular:
        with open("output_psi.csv", "wb") as a:
            np.savetxt(a, np.transpose(solved.psi), delimiter=",")

def L2norm(sol, sol2, sol4):
    L2norm1 = np.sqrt(np.sum((sol.Phi - sol2.Phi) ** 2))
    L2norm2 = np.sqrt(np.sum((sol2.Phi - sol4.Phi) ** 2))
    print(L2norm1 / L2norm2)
    return L2norm1 , L2norm2

def L2norm2(sol, sol2):
    L2norm1 = np.sqrt(np.sum((sol.err) ** 2))
    L2norm2 = np.sqrt(np.sum((sol2.err) ** 2))
    print(L2norm1 / L2norm2)
    return L2norm1 , L2norm2

R, I_reg, N_dir, bc, matprops, name, working = inputVals()

L2 = True

if working:
    sol = Solve(R, I_reg, N_dir, bc, matprops, False)
    sol.solve()
    if L2:
        sol2 = Solve(R, I_reg, 2 * N_dir, bc, matprops, False)
        sol4 = Solve(R, I_reg, 4 * N_dir, bc, matprops, False)
        sol2.solve()
        sol4.solve()
        sol2.AnalyticalSolve(1000)
        sol4.AnalyticalSolve(1000)
        r10, r11 = sol2.plotErr()
        r20, r21 = sol4.plotErr()
        
        num, denom = L2norm(sol, sol2, sol4)
    
    sol.AnalyticalSolve(1000)

    sol.plot()
    r00, r01 = sol.plotErr()
    print(r00 / r10)
    print(r10 / r20)
    
    sol.angular()
    output(sol)
elif R == 1:
    print("Please have N_dir be an even integer.")