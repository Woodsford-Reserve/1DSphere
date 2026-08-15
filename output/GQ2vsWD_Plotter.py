# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 16:42:36 2025

@author: camde
"""

import numpy as np
import matplotlib.pyplot as plt


def inputValues(name):
    file1 = open(name, 'r')
    data = np.genfromtxt(file1, delimiter = ',', dtype=float)
    file1.close()
    return data
    
def plotHG(n, N_dir):
    name = "HG_S" + str(N_dir) + "_"
    GG_Sn = inputValues(name+"GG_Sn_phi.csv")
    LG_GQ2 = inputValues(name+"LG_GQ2_phi.csv")
    LDGQ_GQ2 = inputValues(name+"LDGQ_GQ2_phi.csv")
    LDGQ_Sn = inputValues(name+"LDGQ_Sn_phi.csv")

    mat_length = len(GG_Sn)
    x = np.linspace(1 / (2*mat_length), 1 - (1 / (2*mat_length)), mat_length)
    plt.figure(n)
    plt.plot(x, GG_Sn, label="GG_Sn", ls="--")
    plt.plot(x, LG_GQ2, label="WD_GQ2", ls="-")
    plt.plot(x, LDGQ_GQ2, label="LDGQ_GQ2", ls="-.")
    plt.plot(x, LDGQ_Sn, label="LDGQ_Sn", ls=":")
    plt.xlabel("R")
    plt.ylabel("Flux")
    plt.legend()
    # plt.title("Henyey-Greenstein S"+str(N_dir))
    
    n += 1
    return n

def plotFP(n, N_dir):
    name = "FP_S" + str(N_dir) + "_"
    
    GG_Sn = inputValues(name+"GG_Sn_phi.csv")
    LG_GQ2 = inputValues(name+"LG_GQ2_phi.csv")
    LDGQ_GQ2 = inputValues(name+"LDGQ_GQ2_phi.csv")
    LDGQ_Sn = inputValues(name+"LDGQ_Sn_phi.csv")

    mat_length = len(GG_Sn)
    x = np.linspace(1 / (2*mat_length), 1 - (1 / (2*mat_length)), mat_length)
    plt.figure(n)
    plt.plot(x, GG_Sn, label="GG_Sn", ls="--")
    plt.plot(x, LG_GQ2, label="WD_GQ2", ls="-")
    plt.plot(x, LDGQ_GQ2, label="LDGQ_GQ2", ls="-.")
    plt.plot(x, LDGQ_Sn, label="LDGQ_Sn", ls=":")
    plt.xlabel("R")
    plt.ylabel("Flux")
    plt.legend()
    # plt.title("Fokker-Planck S"+str(N_dir))
    
    n += 1
    return n

def plotFP2(n, N_dir):
    name = "FP2_S" + str(N_dir) + "_"
    GG_Sn = inputValues(name+"GG_Sn_phi.csv")
    LG_GQ2 = inputValues(name+"LG_GQ2_phi.csv")
    LDGQ_GQ2 = inputValues(name+"LDGQ_GQ2_phi.csv")
    LDGQ_Sn = inputValues(name+"LDGQ_Sn_phi.csv")

    mat_length = len(GG_Sn)
    x = np.linspace(1 / (2*mat_length), 1 - (1 / (2*mat_length)), mat_length)
    plt.figure(n)
    plt.plot(x, GG_Sn, label="GG_Sn", ls="--")
    plt.plot(x, LG_GQ2, label="WD_GQ2", ls="-")
    plt.plot(x, LDGQ_GQ2, label="LDGQ_GQ2", ls="-.")
    plt.plot(x, LDGQ_Sn, label="LDGQ_Sn", ls=":")
    plt.xlabel("R")
    plt.ylabel("Flux")
    plt.legend()
    # .title("Fokker-Planck 2 S"+str(N_dir))
    
    n += 1
    return n

def plotGGfromMoments(n):
    N_dir = 4
    name = "FP_S" + str(N_dir) +"_GG_Sn_flux_moments.csv"
    GG_S4 = inputValues(name).T
    N_dir *= 2
    name = "FP_S" + str(N_dir) +"_GG_Sn_flux_moments.csv"
    GG_S8 = inputValues(name).T
    N_dir *= 2
    name = "FP_S" + str(N_dir) +"_GG_Sn_flux_moments.csv"
    GG_S16 = inputValues(name).T
    N_dir *= 2
    name = "FP_S" + str(N_dir) +"_GG_Sn_flux_moments.csv"
    GG_S32 = inputValues(name).T
    S4_flux = GG_S4[0]
    S8_flux = GG_S8[0]
    S16_flux = GG_S16[0]
    S32_flux = GG_S32[0]
    
    s4 = np.copy(S4_flux)
    s8 = np.copy(S8_flux)
    s16 = np.copy(S16_flux)
    s32 = np.copy(S32_flux)
    
    s4_norm = (s32-s4)**2 / s32**2
    s8_norm = (s32-s8)**2 / s32**2
    s16_norm = (s32-s16)**2 / s32**2
    
    mat_length = len(S8_flux)
    x = np.linspace(1 / (2*mat_length), 1 - (1 / (2*mat_length)), mat_length)
    plt.figure(n)
    '''
    plt.plot(x, S4_flux, label="S4", ls=":")
    plt.plot(x, S8_flux, label="S8", ls="--")
    plt.plot(x, S16_flux, label="S16", ls="-")
    plt.plot(x, S32_flux, label="S32", ls="-.")
    '''
    plt.semilogy(x, s4_norm, label="S4", ls="-.")
    plt.semilogy(x, s8_norm, label="S8", ls="--")
    plt.semilogy(x, s16_norm, label="S16", ls="-")
    
    plt.xlabel("R")
    plt.ylabel("Flux")
    plt.legend()
    plt.title("Tesing Different Quadratures")

def plotS16(n, sig):
    N_dir = 16
    name = sig + "_S" + str(N_dir) +"_LDGQ_GQ2_phi.csv"
    S16 = inputValues(name)
    name = sig + "T3_S" + str(N_dir) +"_LDGQ_GQ2_phi.csv"
    S16P3 = inputValues(name)
    name = sig + "T7_S" + str(N_dir) +"_LDGQ_GQ2_phi.csv"
    S16P7 = inputValues(name)
    name = sig + "T3_S" + str(N_dir) +"_LDGQ_Sn_phi.csv"
    S16Sn = inputValues(name)
    if sig == "HG":
        name = sig + "_S" + str(N_dir) +"_LDGQ_Sn_phi.csv"
        S16Sn15 = inputValues(name)
    
    
    s16 = np.copy(S16)
    s16p3 = np.copy(S16P3)
    s16p7 = np.copy(S16P7)
    s16sn = np.copy(S16Sn)
    if sig == "HG":
        s16sn15 = np.copy(S16Sn15)
        sn15_norm = (s16-s16sn15)**2 / s16**2
    
    
    p3_norm = (s16-s16p3)**2 / s16**2
    p7_norm = (s16-s16p7)**2 / s16**2
    sn_norm = (s16-s16sn)**2 / s16**2
    
    mat_length = len(s16)
    x = np.linspace(1 / (2*mat_length), 1 - (1 / (2*mat_length)), mat_length)
    plt.figure(n)
    
    plt.semilogy(x, p3_norm, label="P3", ls="-")
    plt.semilogy(x, p7_norm, label="P7", ls=":")
    plt.semilogy(x, sn_norm, label="P3 Sn", ls="-.")
    if sig == "HG":
        plt.semilogy(x, sn15_norm, label="P15 Sn", ls="--")
    
    
    plt.xlabel("R")
    plt.ylabel("Flux Error")
    plt.legend()
    title = "Tesing Different XS Expansions for " + sig + " S16 LDGQ GQ2"
    plt.title(title)
    
    n+=1
    plt.figure(n)
    
    plt.plot(x, S16, label="S16 P15", ls=":")
    plt.plot(x, S16P3, label="S16 P3", ls="--")
    plt.plot(x, S16P7, label="S16 P7", ls="-")
    plt.plot(x, S16Sn, label="S16 P3 Sn", ls="-.")
    if sig == "HG":
        plt.plot(x, S16Sn15, label="P15 Sn", ls=":")
    plt.xlabel("R")
    plt.ylabel("Flux")
    plt.legend()
    plt.title(title)
    n += 1
    return n
    


def plotGG(n):
    N_dir = 4
    name = "FP2_S" + str(N_dir) +"_GG_Sn_phi.csv"
    GG_S4 = inputValues(name)
    N_dir *= 2
    name = "FP2_S" + str(N_dir) +"_GG_Sn_phi.csv"
    GG_S8 = inputValues(name)
    N_dir *= 2
    name = "FP2_S" + str(N_dir) +"_GG_Sn_phi.csv"
    GG_S16 = inputValues(name)
    N_dir *= 2
    name = "FP2_S" + str(N_dir) +"_GG_Sn_phi.csv"
    GG_S32 = inputValues(name)
    N_dir *= 2
    name = "FP2_S" + str(N_dir) +"_GG_Sn_phi.csv"
    GG_S64 = inputValues(name)
    name = "FPT7_S" + str(N_dir) +"_GG_Sn_phi.csv"
    P7 = inputValues(name)
    name = "FPT15_S" + str(N_dir) +"_GG_Sn_phi.csv"
    P15 = inputValues(name)
    name = "FPT31_S" + str(N_dir) +"_GG_Sn_phi.csv"
    P31 = inputValues(name)
    
    s4 = np.copy(GG_S4)
    s8 = np.copy(GG_S8)
    s16 = np.copy(GG_S16)
    s32 = np.copy(GG_S32)
    s64 = np.copy(GG_S64)
    p7 = np.copy(P7)
    p15 = np.copy(P15)
    p31 = np.copy(P31)
    
    s4_norm = (s64-s4)**2 / s64**2
    s8_norm = (s64-s8)**2 / s64**2
    s16_norm = (s64-s16)**2 / s64**2
    s32_norm = (s64-s32)**2 / s64**2
    p7_norm = (s64-p7)**2 / s64**2
    p15_norm = (s64-p15)**2 / s64**2
    p31_norm = (s64-p31)**2 / s64**2
    
    mat_length = len(s4)
    x = np.linspace(1 / (2*mat_length), 1 - (1 / (2*mat_length)), mat_length)
    plt.figure(n)
    
    '''
    plt.plot(x, GG_S4, label="S4", ls=":")
    plt.plot(x, GG_S8, label="S8", ls="--")
    plt.plot(x, GG_S16, label="S16", ls="-")
    plt.plot(x, GG_S32, label="S32", ls="-.")
    
    plt.plot(x, GG_S64, label="S64", ls="-")
    plt.plot(x, GG_S64_P7, label="S64_P7", ls="--")
    '''
    plt.semilogy(x, s4_norm, label="S4", ls="-.")
    plt.semilogy(x, s8_norm, label="S8", ls="--")
    plt.semilogy(x, s16_norm, label="S16", ls="-")
    plt.semilogy(x, s32_norm, label="S32", ls=":")
    '''
    plt.semilogy(x, p7_norm, label="P7", ls="-")
    plt.semilogy(x, p15_norm, label="P15", ls=":")
    plt.semilogy(x, p31_norm, label="P31", ls="--")
    '''
    plt.xlabel("R")
    plt.ylabel("Flux")
    plt.legend()
    plt.title("Testing Different Quadratures")
    # plt.semilogy(x, p15_norm, label="P15", ls=":")
    # plt.semilogy(x, p31_norm, label="P31", ls="--")
    n += 1
    plt.figure(n)
    plt.plot(x, GG_S64, label="S64", ls="-")
    plt.plot(x, P31, label="P15", ls=":")
    plt.plot(x, P31, label="P31", ls="-.")
    
    plt.xlabel("R")
    plt.ylabel("Flux")
    plt.legend()
    plt.title("Testing Different Quadratures")
    return n



def plotGQ3(n, sig_l):
    N_dir = np.array([8,16,32,64])
    N_mom_GQ3 = np.array([7,15,31,31])
    N_mom_GG = np.array([7,15,31,63])
    mat_length = 1000
    x = np.linspace(1 / (2*mat_length), 1 - (1 / (2*mat_length)), mat_length)
    line = np.array(["-",":","-.","--"])
    plt.figure(n)
    for i in range(len(N_dir)):
        GG_leg = "GG Sn S"+str(N_dir[i])+"P"+str(N_mom_GG[i])
        GG_name  = "aniso3_"+sig_l+"_S"+str(N_dir[i])+"P"+str(N_mom_GG[i])+"_GG_Sn_phi.csv"
        GG_data = np.copy(inputValues(GG_name))
        plt.plot(x, GG_data, label=GG_leg, ls=line[i])
    plt.xlabel("R")
    plt.ylabel("Flux")
    plt.legend()
    title = "GG Sn "+sig_l+" Comparison"
    # plt.title(title)
    n += 1
    plt.figure(n)
    for i in range(len(N_dir)):
        if i == 0:
            plt.plot(x, GG_data, label=GG_leg, ls=line[-1])
        GQ3_leg = "GQ3 S"+str(N_dir[i])+"P"+str(N_mom_GQ3[i])
        GQ3_name = "aniso3_"+sig_l+"_S"+str(N_dir[i])+"P"+str(N_mom_GQ3[i])+"_LDGQ_GQ3_phi.csv"
        GQ3_data = np.copy(inputValues(GQ3_name))    
        plt.plot(x, GQ3_data, label=GQ3_leg, ls=line[i])   
        
    plt.xlabel("R")
    plt.ylabel("Flux")
    plt.legend()
    title = "GQ3 "+sig_l+" Comparison"
    # plt.title(title)
    n += 1
    plt.figure(n)
    N_dir = [4,8,16]
    N_mom_Sn = [3,7,15]
    for i in range(len(N_dir)):
        if i == 0:
            GQ3_leg = "GQ3 S64P31"
            GQ3_name = sig_l+"_S64P31_LDGQ_GQ3_phi.csv"
            GQ3_data = np.copy(inputValues(GQ3_name))
            plt.plot(x, GQ3_data, label=GQ3_leg, ls=line[-1])
        leg = "Sn S"+str(N_dir[i])+"P"+str(N_mom_Sn[i])
        name = "LDG_"+sig_l+"_S"+str(N_dir[i])+"P"+str(N_mom_Sn[i])+"_LDGQ_Sn_phi.csv"
        data = np.copy(inputValues(name))    
        plt.plot(x, data, label=leg, ls=line[i])   
    
    plt.xlabel("R")
    plt.ylabel("Flux")
    plt.legend()
    n += 1
    return n

def plot_aniso(n):
    N_dir = np.array([8,16])
    mat_length = 1000
    x = np.linspace(1 / (2*mat_length), 1 - (1 / (2*mat_length)), mat_length)
    line = np.array(["-",":","-.","--"])
    sig_l = "isotropic"
    for i in range(len(N_dir)):
        GG_leg = "GG Sn S"+str(N_dir[i])+"P"+str(0)
        GG_name  = "aniso3_"+sig_l+"_S"+str(N_dir[i])+"P"+str(0)+"_GG_Sn_phi.csv"
        GG_data = np.copy(inputValues(GG_name))
        GQ3_leg = "GQ3 S"+str(N_dir[i])+"P"+str(0)
        GQ3_name = "aniso3_"+sig_l+"_S"+str(N_dir[i])+"P"+str(0)+"_LDGQ_GQ3_phi.csv"
        GQ3_data = np.copy(inputValues(GQ3_name))
        LG_leg = "LG Sn S"+str(N_dir[i])+"P"+str(0)
        LG_name  = "aniso3_"+sig_l+"_S"+str(N_dir[i])+"P"+str(0)+"_LG_Sn_phi.csv"
        LG_data = np.copy(inputValues(LG_name))
        analytical_leg = "Analytical Solution"
        analytical_name = "aniso3_analytical_phi.csv"
        analytical_data = np.copy(inputValues(analytical_name))
        plt.figure(n)
        plt.plot(x, GG_data, label=GG_leg, ls=line[0])
        plt.plot(x, GQ3_data, label=GQ3_leg, ls=line[1])
        plt.plot(x, LG_data, label=LG_leg, ls=line[2])
        plt.plot(x, analytical_data, label=analytical_leg, ls=line[3])
        plt.xlabel("R")
        plt.ylabel("Flux")
        plt.legend()
        n += 1
    return n

n = 0



# n = plotFP(n, 8)
# n = plotFP2(n, 8)

# n = plotHG(n, 8)
# n = plotHG(n, 16)

# n = plotFP(n, 16)
# n = plotFP2(n, 16)

# n = plotGGfromMoments(n)
# n = plotGG(n)

# n = plotS16(n, "FP")
# n = plotS16(n, "HG")

n = plotGQ3(n, "FP2")
#n = plotGQ3(n, "HG75")
#n = plotGQ3(n, "isotropic")
#n = plotGQ3(n, "HG95")
#n = plot_aniso(n)

plt.show()