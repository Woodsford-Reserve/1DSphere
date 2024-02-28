# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:16:48 2024

@author: c.woodsford.9788

Branch Version 1.3.1
Changes:
    Attempt 1 at Solving Flux using local system
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
        self.R = np.insert(R, 0, 0.)

        # cell widths and centers
        self.dr = np.array([])
        for nr in range(N_reg):
            dr_reg = (self.R[nr + 1] - self.R[nr]) / I_reg[nr]
            self.dr = np.concatenate((self.dr, np.repeat(dr_reg, I_reg[nr])))
        self.r = np.cumsum(self.dr) - self.dr / 2

        # cell areas and volumes
        self.A = 4 * np.pi * (self.r + self.dr / 2) ** 2
        self.A = np.insert(self.A, 0, 0.)
        self.V = 4 * np.pi / 3 * ((self.r + self.dr / 2) ** 3 - (self.r - self.dr / 2) ** 3)

        # material ID
        self.matID = np.array([], dtype=int)
        for nr in range(N_reg):
            self.matID = np.concatenate((self.matID, np.repeat(matIDs[nr], \
                                                               I_reg[nr])))


class Quad:
    def __init__(self, N_dir, Toggle=False, angularWidths=[]):
        # gauss-legendre quadrature

        # Toggles usage of Local Quadrature instead of GL Scheme
        if Toggle == True:
            if N_dir % 4 != 0:
                raise ValueError("No. of dir. is not doubly divisible by 2!")
            else:
                self.N_dir = N_dir

            # Creates Number of Cells, Weight Value, and Cell Boundaries
            cellNum = int(N_dir / 2)
            if len(angularWidths) == 0:
                width = 2 / cellNum
                cellBounds = [-1]
                for i in range(cellNum):
                    cellBounds.append(cellBounds[-1] + width)
            else:
                posAngularCells = angularWidths.copy()
                posAngularCells.reverse()
                angularWidths.extend(posAngularCells)
                cellBounds = [-1]
                for i in range(len(angularWidths)):
                    cellBounds.append(cellBounds[-1] + angularWidths[i])

            # Creates List to contain all collocation points
            muList = []
            for i in range(len(cellBounds) - 1):
                width = cellBounds[i + 1] - cellBounds[i]
                muMinus = (1 - 1 / np.sqrt(3)) * width / 2 + cellBounds[i]
                muPlus = (1 + 1 / np.sqrt(3)) * width / 2 + cellBounds[i]
                muList.append([muMinus, muPlus])
            self.w = []
            for i in range(len(cellBounds) - 1):
                weight = (cellBounds[i + 1] - cellBounds[i]) / 2
                self.w.append(weight)
                self.w.append(weight)
            # convert list of list to an array, then flatten along rows.
            self.mu = np.array(muList).flatten('C')

        # Revert back to global quadrature routine
        else:
            if N_dir % 2 != 0:
                raise ValueError("No. of dir. is not even!")
            else:
                self.N_dir = N_dir
            self.mu, self.w = np.polynomial.legendre.leggauss(N_dir)

        # mu-cell boundaries and alpha
        self.mu_half = np.zeros(N_dir + 1)
        self.mu_half[0] = -1.
        self.alpha = np.zeros(N_dir + 1)
        for nd in range(N_dir):
            self.mu_half[nd + 1] = self.mu_half[nd] + self.w[nd]
            self.alpha[nd + 1] = self.alpha[nd] - 2 * self.mu[nd] * self.w[nd]

        # beta
        self.beta = np.zeros(N_dir)
        for nd in range(N_dir):
            self.beta[nd] = (self.mu[nd] - self.mu_half[nd]) / \
                            (self.mu_half[nd + 1] - self.mu_half[nd])

class Solve:
    def __init__(self, R, I_reg, N_dir, bc, matprops, localQuad=False, angCellWidths=[]):
        nmats = len(matprops["sigt"])
        matIDs = np.arange(nmats)

        # initialization
        self.mesh = Mesh(matIDs, R, I_reg)
        self.quad = Quad(N_dir, localQuad, angCellWidths)
        self.bc = bc
        self.matprops = matprops

    # solver
    def solve(self):
        # boundary flux
        self.psi_bound = np.zeros(self.quad.N_dir)
        self.psi_bound[:int(self.quad.N_dir / 2)] = self.bc["value"] / 2

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
                Phi_1 += psi * self.quad.w[nd]

            # calculate error
            if (it == 1):
                err = 1
            else:
                delta = np.sqrt(np.sum(((Phi_1 - Phi_0) / Phi_1) ** 2))
                rho = np.sum(np.abs(Phi_1 - Phi_0)) / np.sum(np.abs(Phi_0 - Phi_m1))
                err = delta / np.abs(1 - rho)
            print(str(it) + " :: " + str(err))

            # next iteration
            Phi_m1 = Phi_0
            Phi_0 = Phi_1

        # converged solution
        self.Phi = Phi_0

        # balance parameter
        bal = self.balance()
        print("Balance: " + str(bal))

    # starting direction
    def start(self, Phi):
        # starting direction ingoing flux
        mu0, mu1 = self.quad.mu[0], self.quad.mu[1]
        psi_x = self.psi_bound[0] * (mu1 + 1) / (mu1 - mu0) - \
                self.psi_bound[1] * (mu0 + 1) / (mu1 - mu0)

        # starting direction sweep
        psi_mu = np.zeros(self.mesh.I)
        for iel in range(self.mesh.I - 1, -1, -1):
            # cell properties
            dr = self.mesh.dr[iel]
            matID = self.mesh.matID[iel]
            sigt = self.matprops["sigt"][matID]
            sigs = self.matprops["sigs"][matID]
            q = self.matprops["q"][matID]

            # calculate cell-average angular flux
            psi_mu[iel] = psi_x + (sigs * Phi[iel] + q) * dr / 4
            psi_mu[iel] /= (1 + sigt * dr / 2)

            # update ingoing flux
            psi_x = 2 * psi_mu[iel] - psi_x

        # return origin and starting direction angular flux
        psi_0 = psi_x
        return psi_0, psi_mu

    # sweep function
    def sweep(self, nd, psi_mu, Phi):
        # direction quantities
        mu = self.quad.mu[nd]
        w = self.quad.w[nd]
        alpha = self.quad.alpha[nd:nd + 2]
        beta = self.quad.beta[nd]

        # negative-mu sweeps
        if (mu < 0):
            # angular flux at boundary
            psi_x = self.psi_bound[nd]
            # sweep order
            start = self.mesh.I - 1
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
            A = self.mesh.A[iel:iel + 2]
            if (mu < 0):
                A_out = A[0]
            if (mu > 0):
                A_out = A[1]
            V = self.mesh.V[iel]
            matID = self.mesh.matID[iel]
            sigt = self.matprops["sigt"][matID]
            sigs = self.matprops["sigs"][matID]
            q = self.matprops["q"][matID]

            # calculate cell-average angular flux
            psi[iel] = (sigs * Phi[iel] + q) / 2 * V + np.abs(mu) * (A[1] + A[0]) * psi_x \
                       + (A[1] - A[0]) / (2 * w) * (alpha[1] * (1 / beta - 1) + alpha[0]) * psi_mu[iel]
            psi[iel] /= (2 * np.abs(mu) * A_out + alpha[1] * (A[1] - A[0]) / (2 * beta * w) + sigt * V)

            # update ingoing fluxes
            psi_x = 2 * psi[iel] - psi_x
            psi_mu[iel] = (psi[iel] - (1 - beta) * psi_mu[iel]) / beta

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
            V = self.mesh.V[iel]
            matID = self.mesh.matID[iel]
            sigt = self.matprops["sigt"][matID]
            sigs = self.matprops["sigs"][matID]
            q = self.matprops["q"][matID]
            # source and absorption rates
            source += q * V
            absorp += (sigt - sigs) * self.Phi[iel] * V
        # leakage
        leak = np.sum(self.quad.mu * self.psi_bound * self.quad.w) * self.mesh.A[-1]
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

def LocalGQSolve(N_dir, angCellWidths):
    # Utilzing different solve operation for local quadrature sets in angle (1 region for now)
    # Spatial Mesh Creator - Stores Cell Centers
    MaxRadius = 1
    CellNumber = 40
    rCellLeft = []
    rCellCenter = []
    rCellRight = []
    cellWidths = []
    width = MaxRadius / CellNumber
    centerOffset = width / 2
    for i in range(CellNumber):
        rCellLeft.append(i * width)
        rCellCenter.append(centerOffset + i * width)
        rCellRight.append((i + 1) * width)
        cellWidths.append(width)

    # Material Properties
    sigT = 1.0

    # Solving Slab Geometry
    psiBound = 0
    psiIncoming = psiBound
    Q = 1
    totalSpatialCells = len(rCellCenter)
    psiSpatialCells = []
    psiSpatialLeft = []
    psiSpatialRight = []
    for i in range(totalSpatialCells):
        # Psi Calculations
        psiSpatialRight.insert(0, psiIncoming)
        psiCenter = Q * cellWidths[totalSpatialCells - 1 - i] - 2 * psiIncoming
        psiCenter /= sigT * cellWidths[totalSpatialCells - 1 - i] - 2
        psiSpatialCells.insert(0, psiCenter)
        psiOut = (2 * psiCenter - psiIncoming)
        psiSpatialLeft.insert(0, psiOut)
        psiIncoming = psiOut

    # Area and Volume Elements
    V = []
    ALeft = []
    ARight = []
    deltaA = []
    for i in range(totalSpatialCells):
        V.append(4 * np.pi / 3 * (rCellRight[i] ** 3 - rCellLeft[i] ** 3))
        ALeft.append(4 * np.pi * rCellLeft[i] ** 2)
        ARight.append(4 * np.pi * rCellRight[i] ** 2)
        deltaA.append(4 * np.pi * (rCellRight[i] ** 2 - rCellLeft[i] ** 2))

    # Extracts Quadrature Information from process above
    QuadCharacteristics = Quad(N_dir, True, angCellWidths)
    # Even Indicies refer to mu-, Odd Indicies refer to mu+
    # Pairs of indicies form a cell
    muList = QuadCharacteristics.mu
    muWeights = QuadCharacteristics.w
    alphaList = QuadCharacteristics.alpha
    betaList = QuadCharacteristics.beta

    # Begin routine for angular cells
    psi = []
    for i in range(totalSpatialCells):
        psiTemp = []
        # mu < 0
        # psiIncoming Cancels out since associated alpha is 0.
        psiIncoming = 0
        psiAngularMinus = []
        psiAngularPlus = []
        psiAngular = []
        for j in range(int(len(muList) / 4)):
            CoefOne = 8 * muWeights[2 * j] * muList[2 * j] * ARight[i]
            CoefOne += deltaA[i] * (alphaList[2 * j] + alphaList[2 * j + 1]) / 2
            CoefOne += 4 * muWeights[2 * j] * sigT * V[i]

            CoefTwo = deltaA[i] * (alphaList[2 * j] + alphaList[2 * j + 1]) / 2

            CoefThr = -4 * muWeights[2 * j] * muList[2 * j] * ARight[i] * psiSpatialLeft[i]
            CoefThr += -4 * muWeights[2 * j] * muList[2 * j] * ALeft[i] * psiSpatialLeft[i]
            CoefThr += -2 * deltaA[i] * alphaList[2 * j] * psiIncoming - Q * V[i]

            CoefFor = 2 * deltaA[i] * alphaList[2 * j + 1] * betaList[2 * j]
            CoefFor += deltaA[i] * (alphaList[2 * j] + alphaList[2 * j + 1]) / 2

            CoefFiv = 8 * muWeights[2 * j + 1] * muList[2 * j + 1] * ARight[i]
            CoefFiv += 2 * deltaA[i] * alphaList[2 * j + 1] * betaList[2 * j + 1]
            CoefFiv += -1 * deltaA[i] * (alphaList[2 * j] + alphaList[2 * j + 1]) / 2
            CoefFiv += 4 * muWeights[2 * j + 1] * sigT * V[i]

            CoefSix = -4 * ARight[i] * muWeights[2 * j + 1] * muList[2 * j + 1] * psiSpatialRight[i]
            CoefSix += -4 * ALeft[i] * muWeights[2 * j + 1] * muList[2 * j + 1] * psiSpatialRight[i]
            CoefSix += -1 * Q * V[i]

            psiPlus = (CoefOne * CoefSix - CoefThr * CoefFor) / (CoefTwo * CoefFor - CoefOne * CoefFiv)
            psiMinus = (CoefTwo * CoefSix - CoefThr * CoefFiv) / (CoefTwo * CoefFor - CoefOne * CoefFiv)
            psiCenter = (psiPlus + psiMinus) / 2
            psiAngularPlus.append(psiPlus)
            psiAngularMinus.append(psiMinus)
            psiAngular.append(psiCenter)
            psiIncoming = 2 * psiCenter - psiIncoming
        psiTemp.append(psiAngular)

        # mu > 0
        # psiIncoming Cancels out since associated alpha is 0.
        psiIncoming = 0
        psiAngularMinus = []
        psiAngularPlus = []
        psiAngular = []
        for j in range(int(len(muList) / 4)):
            minusIndex = int(len(muList) - 2 - 2 * j)
            plusIndex = int(len(muList) - 1 - 2 * j)

            CoefOne = 8 * muWeights[minusIndex] * muList[minusIndex] * ARight[i]
            CoefOne += deltaA[i] * (alphaList[minusIndex] + alphaList[plusIndex]) / 2
            CoefOne += -1 * deltaA[i] * alphaList[minusIndex] * betaList[minusIndex]
            CoefOne += 4 * muWeights[minusIndex] * sigT * V[i]

            CoefTwo = deltaA[i] * (alphaList[2 * j] + alphaList[2 * j + 1]) / 2
            CoefTwo += deltaA[i] * alphaList[minusIndex] * betaList[plusIndex]

            CoefThr = -4 * muWeights[minusIndex] * muList[minusIndex] * ARight[i] * psiSpatialLeft[i]
            CoefThr += -4 * muWeights[minusIndex] * muList[minusIndex] * ALeft[i] * psiSpatialLeft[i]
            CoefThr += -1 * Q * V[i]

            CoefFor = -1 * deltaA[i] * (alphaList[2 * j] + alphaList[2 * j + 1]) / 2

            CoefFiv = 8 * muWeights[plusIndex] * muList[plusIndex] * ARight[i]
            CoefFiv += -1 * deltaA[i] * (alphaList[2 * j] + alphaList[2 * j + 1]) / 2
            CoefFiv += 4 * muWeights[plusIndex] * sigT * V[i]

            CoefSix = -4 * ARight[i] * muWeights[plusIndex] * muList[plusIndex] * psiSpatialRight[i]
            CoefSix += -4 * ALeft[i] * muWeights[plusIndex] * muList[plusIndex] * psiSpatialRight[i]
            CoefSix += deltaA[i] * alphaList[plusIndex] + psiIncoming
            CoefSix += -1 * Q * V[i]

            psiPlus = (CoefOne * CoefSix - CoefThr * CoefFor) / (CoefTwo * CoefFor - CoefOne * CoefFiv)
            psiMinus = (CoefTwo * CoefSix - CoefThr * CoefFiv) / (CoefTwo * CoefFor - CoefOne * CoefFiv)
            psiCenter = (psiPlus + psiMinus) / 2
            psiAngularPlus.append(psiPlus)
            psiAngularMinus.append(psiMinus)
            psiAngular.append(psiCenter)
            psiIncoming = 2 * psiCenter - psiIncoming
        psiTemp.append(psiAngular)
        psi.append(psiTemp)
    print(psi)
R = np.array([1., 2.])
I_reg = np.array([40, 40])
N_dir = 8
# localAngularCellLengths should have length equal to 1/4 * N_dir and should sum to 1
# Intervals start from -1, and should end at 0 (It will be mirrored for 0 to 1). Leave empty for uniform cells.
localAngularCellLengths = [0.75, 0.25]

bc = {"type": "source", "value": 0.}

matprops = {"sigt": np.array([1.0, 1.0]),
            "sigs": np.array([0.5, 0.9]),
            "q": np.array([1.0, 0.0])}


#sol = Solve(R, I_reg, N_dir, bc, matprops)
#sol.solve()
#sol.plot()

LocalGQSolve(N_dir, localAngularCellLengths)
#sol = Solve(R, I_reg, N_dir, bc, matprops, True, localAngularCellLengths)
#sol.solve()
#sol.plot()
