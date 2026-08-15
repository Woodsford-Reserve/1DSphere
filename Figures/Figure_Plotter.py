# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 16:42:36 2025

@author: camde
"""

import numpy as np
import matplotlib.pyplot as plt

def inputVals(name):
    file1 = open(name, 'r')
    data = np.genfromtxt(file1, delimiter = ',', dtype=str)
    data_names = data[0]
    true_data = data[1:].transpose().astype(float)
    file1.close()
    
    return data_names, true_data


def plotData(name, x_label, y_label, n):
    data_name, data = inputVals(name)
    plt.figure(n)
    plot_types = ["-", "--", "-.", ":", (0, (6,3,3,3)), (0,(3,3,1,3))]
    for i in range(1,len(data[:,0])):
        plt.plot(data[0], data[i], ls=plot_types[i-1], label=data_name[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    n += 1
    return n

def plotLogData(name, x_label, y_label, n):
    data_name, data = inputVals(name)
    plt.figure(n)
    plot_types = ["-", "--", "-.", ":", (0, (6,3,3,3)), (0,(3,3,1,3))]
    for i in range(1,len(data[:,0])):
        plt.semilogy(data[0], data[i], linestyle=plot_types[i-1], label=data_name[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    n += 1
    return n



    
def plotHG(n):
    return 0

n = 0
n = plotData("Fig_1.csv", "R", "Flux", n)
n = plotData("Fig_2.csv", "R", "Flux", n)
n = plotLogData("Fig_3a.csv", "R", "Relative Error", n)
n = plotLogData("Fig_3b.csv", "R", "Relative Error", n)
n = plotLogData("Fig_4a.csv", "R", "Relative Error", n)
n = plotLogData("Fig_4b.csv", "R", "Relative Error", n)
n = plotLogData("Fig_5a.csv", "R", "Relative Error", n)
n = plotLogData("Fig_5b.csv", "R", "Relative Error", n)
n = plotLogData("Fig_6a.csv", "R", "Relative Error", n)
n = plotLogData("Fig_6b.csv", "R", "Relative Error", n)
n = plotLogData("Fig_6c.csv", "R", "Relative Error", n)
n = plotLogData("Fig_6d.csv", "R", "Relative Error", n)
n = plotData("Fig_7.csv", "R", "Flux", n)
n = plotData("Fig_8.csv", "R", "Flux", n)
n = plotData("Fig_9.csv", "R", "Flux", n)
n = plotData("Fig_10.csv", "R", "Flux", n)




plt.show()