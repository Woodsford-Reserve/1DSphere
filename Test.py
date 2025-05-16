# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 12:35:00 2025

@author: woods
"""

import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import lpmv, factorial



# generate symmetric 1D quadrature
def symmetric1d(N):
    
    # quadrature set on subinterval
    phi = np.zeros(2)
    
    # points (reference interval)
    beta = np.sqrt(0.5*(1. - N/np.pi * np.sin(np.pi/N)))
    
    # points (real subinterval)
    phi[0] = -np.arcsin(beta) + np.pi/(2*N)
    phi[1] =  np.arcsin(beta) + np.pi/(2*N)
    
    # complete quadrature set
    points  = np.zeros((N,2))
    weights = np.pi/(2*N) * np.ones((N,2))
    
    # loop over subintervals
    for n in range(N):
        points[n,0] = phi[0] + np.pi * n/N
        points[n,1] = phi[1] + np.pi * n/N
    
    # return quadrature set
    return points, weights



# perfrom 1D integral
def integrate1d(integrand, points, weights, N_azimu):
    
    # true (local) integral value
    a = np.pi / N_azimu
    integral_true = quad(integrand, 0., a)[0]
    
    # quadrature integral value
    integral_quad = 0.
    for k in range(2):
        integral_quad += integrand(points[0,k])*weights[0,k]
            
    # print integral values
    print("{:f} |" .format(integral_true).rjust(11),
          "{:f} ||".format(integral_quad).rjust(12),
          "{:e}"   .format(abs(integral_true-integral_quad)).rjust(12))
    
    
    # true (global) integral value
    integral_true = quad(integrand, 0., np.pi)[0]
    
    # quadrature integral value
    integral_quad = 0.
    for n in range(N_azimu):
        for k in range(2):
            integral_quad += integrand(points[n,k])*weights[n,k]
            
    # print integral values
    print("{:f} |" .format(integral_true).rjust(11),
          "{:f} ||".format(integral_quad).rjust(12),
          "{:e}"   .format(abs(integral_true-integral_quad)).rjust(12))
    
    

# 1D integrals
def integrals1d(N_azimu):
    
    # integrands
    integrands = [#constant
                  ["constant", lambda phi: 1.],
                  [],
                  
                  # linear
                  ["cos", lambda phi: np.cos(phi)],
                  ["sin", lambda phi: np.sin(phi)],
                  [],
                  
                  # quadratic
                  ["cos^2",   lambda phi: np.cos(phi)**2                 ],
                  ["cos*sin", lambda phi: np.cos(phi)    * np.sin(phi)   ],
                  ["sin^2",   lambda phi:                  np.sin(phi)**2],
                  [],
                  
                  # cubic
                  ["cos^3",     lambda phi: np.cos(phi)**3                 ],
                  ["cos^2*sin", lambda phi: np.cos(phi)**2 * np.sin(phi)   ],
                  ["cos*sin^2", lambda phi: np.cos(phi)    * np.sin(phi)**2],
                  ["sin^3",     lambda phi:                  np.sin(phi)**3],
                  [],
                  
                  # quartic
                  ["cos^4",       lambda phi: np.cos(phi)**4                 ],
                  ["cos^3*sin",   lambda phi: np.cos(phi)**3 * np.sin(phi)   ],
                  ["cos^2*sin^2", lambda phi: np.cos(phi)**2 * np.sin(phi)**2],
                  ["cos*sin^3",   lambda phi: np.cos(phi)    * np.sin(phi)**3],
                  ["sin^4",       lambda phi:                  np.sin(phi)**4],
                  [],
                  
                  # quintic
                  ["cos^5",       lambda phi: np.cos(phi)**5                 ],
                  ["cos^4*sin",   lambda phi: np.cos(phi)**4 * np.sin(phi)   ],
                  ["cos^3*sin^2", lambda phi: np.cos(phi)**3 * np.sin(phi)**2],
                  ["cos^2*sin^3", lambda phi: np.cos(phi)**2 * np.sin(phi)**3],
                  ["cos*sin^4",   lambda phi: np.cos(phi)    * np.sin(phi)**4],
                  ["sin^5",       lambda phi:                  np.sin(phi)**5],
                  [],
                  
                  # sextic
                  ["cos^6",       lambda phi: np.cos(phi)**6                 ],
                  ["cos^5*sin",   lambda phi: np.cos(phi)**5 * np.sin(phi)   ],
                  ["cos^4*sin^2", lambda phi: np.cos(phi)**4 * np.sin(phi)**2],
                  ["cos^3*sin^3", lambda phi: np.cos(phi)**3 * np.sin(phi)**3],
                  ["cos^2*sin^4", lambda phi: np.cos(phi)**2 * np.sin(phi)**4],
                  ["cos*sin^5",   lambda phi: np.cos(phi)    * np.sin(phi)**5],
                  ["sin^6",       lambda phi:                  np.sin(phi)**6],
                  [],
                  
                  # septic
                  ["cos^7",       lambda phi: np.cos(phi)**7                 ],
                  ["cos^6*sin",   lambda phi: np.cos(phi)**6 * np.sin(phi)   ],
                  ["cos^5*sin^2", lambda phi: np.cos(phi)**5 * np.sin(phi)**2],
                  ["cos^4*sin^3", lambda phi: np.cos(phi)**4 * np.sin(phi)**3],
                  ["cos^3*sin^4", lambda phi: np.cos(phi)**3 * np.sin(phi)**4],
                  ["cos^2*sin^5", lambda phi: np.cos(phi)**2 * np.sin(phi)**5],
                  ["cos*sin^6",   lambda phi: np.cos(phi)    * np.sin(phi)**6],
                  ["sin^7",       lambda phi:                  np.sin(phi)**7]]


    # generate quadrature set
    points, weights = symmetric1d(N_azimu)

    # perform 1D integrals
    print("\n######################################")
    print("############ Symmetric 1D ############")
    print("######################################")
    for integrand in integrands:
        if integrand == []:
            print("\n--------------------------------------")
            continue
        print("\n" + integrand[0])
        integrate1d(integrand[1], points, weights, N_azimu)
    print("\n")
   
        

# integration accuracy 
def accuracy(orders):
    
    # compute integration accuracy
    print("\n#############################################################")
    print("################### Integration Accuracy ####################")
    print("#############################################################")
    
    # errors
    # linear
    cos_errors = np.zeros(len(orders))
    sin_errors = np.zeros(len(orders))
    
    # cubic
    cos3_errors     = np.zeros(len(orders))
    cos2_sin_errors = np.zeros(len(orders))
    cos_sin2_errors = np.zeros(len(orders))
    sin3_errors     = np.zeros(len(orders))
    
    
    # linear
    for i,order in enumerate(orders):

        # generate quadrature
        points, weights = symmetric1d(order) 
        
        # errors
        cos_err = 0.
        sin_err = 0.
        
        # loop over subintervals
        for n in range(order):
            
            # bounds of integration
            a = np.pi * n/order
            b = np.pi * (n+1)/order
            
            
            # cosine
            # true integral value
            integral_true = quad(lambda phi: np.cos(phi), a, b)[0]
            
            # quadrature integral value
            integral_quad = np.cos(points[n,0])*weights[n,0] \
                          + np.cos(points[n,1])*weights[n,1]
                          
            # cosine integral error
            cos_err += abs(integral_true - integral_quad)
            
            
            # sine
            # true integral value
            integral_true = quad(lambda phi: np.sin(phi), a, b)[0]
            
            # quadrature integral value
            integral_quad = np.sin(points[n,0])*weights[n,0] \
                          + np.sin(points[n,1])*weights[n,1]
                          
            # sine integral error
            sin_err += abs(integral_true - integral_quad)
            
        
        # L2 error norm
        cos_errors[i] = cos_err
        sin_errors[i] = cos_err
      
        
    # output erros
    print("\nLinear")
    print("cos".rjust(18), "sin".rjust(31), "\n",
                         " {:e}  |".format(cos_errors[0]), 
          "    ---     ||  {:e}  |".format(sin_errors[0]), 
          "    ---")
    for i in range(1,len(orders)):
        print("  {:e}  |" .format(cos_errors[i]),
               " {:f}  ||".format(cos_errors[i-1]/cos_errors[i]),
               " {:e}  |" .format(sin_errors[i]),
               " {:f}"    .format(sin_errors[i-1]/sin_errors[i]))
        
        
    # cubic
    for i,order in enumerate(orders):

        # generate quadrature
        points, weights = symmetric1d(order) 
        
        # errors
        cos3_err     = 0.
        cos2_sin_err = 0.
        cos_sin2_err = 0.
        sin3_err     = 0.
        
        # loop over subintervals
        for n in range(order):
            
            # bounds of integration
            a = np.pi * n/order
            b = np.pi * (n+1)/order
            
            
            # cos^3
            # true integral value
            integral_true = quad(lambda phi: np.cos(phi)**3, a, b)[0]
            
            # quadrature integral value
            integral_quad = np.cos(points[n,0])**3*weights[n,0] \
                          + np.cos(points[n,1])**3*weights[n,1]
                          
            # cosine integral error
            cos3_err += abs(integral_true - integral_quad)
            
            
            # cos^2 * sin
            # true integral value
            integral_true = quad(lambda phi: np.cos(phi)**2*np.sin(phi), a, b)[0]
            
            # quadrature integral value
            integral_quad = np.cos(points[n,0])**2*np.sin(points[n,0])*weights[n,0] \
                          + np.cos(points[n,1])**2*np.sin(points[n,1])*weights[n,1]
                          
            # sine integral error
            cos2_sin_err += abs(integral_true - integral_quad)
            
            
            # cos * sin^2
            # true integral value
            integral_true = quad(lambda phi: np.cos(phi)*np.sin(phi)**2, a, b)[0]
            
            # quadrature integral value
            integral_quad = np.cos(points[n,0])*np.sin(points[n,0])**2*weights[n,0] \
                          + np.cos(points[n,1])*np.sin(points[n,1])**2*weights[n,1]
                          
            # sine integral error
            cos_sin2_err += abs(integral_true - integral_quad)
            
            
            # sin^3
            # true integral value
            integral_true = quad(lambda phi: np.sin(phi)**3, a, b)[0]
            
            # quadrature integral value
            integral_quad = np.sin(points[n,0])**3*weights[n,0] \
                          + np.sin(points[n,1])**3*weights[n,1]
                          
            # cosine integral error
            sin3_err += abs(integral_true - integral_quad)
            
        
        # L2 error norm
        cos3_errors[i]     = cos3_err
        cos2_sin_errors[i] = cos2_sin_err
        cos_sin2_errors[i] = cos_sin2_err
        sin3_errors[i]     = sin3_err
      
        
    # output erros
    print("\n\nCubic")
    print("cos^3".rjust(19), "cos^2*sin".rjust(33), "\n",
                         " {:e}  |".format(cos3_errors[0]), 
          "    ---     ||  {:e}  |".format(cos2_sin_errors[0]), 
          "    ---")
    for i in range(1,len(orders)):
        print("  {:e}  |" .format(cos3_errors[i]),
               " {:f}  ||".format(cos3_errors[i-1]/cos3_errors[i]),
               " {:e}  |" .format(cos2_sin_errors[i]),
               " {:f}"    .format(cos2_sin_errors[i-1]/cos2_sin_errors[i]))

    print("\n" + "cos*sin^2".rjust(21), "sin^3".rjust(29), "\n",
                         " {:e}  |".format(cos_sin2_errors[0]), 
          "    ---     ||  {:e}  |".format(sin3_errors[0]), 
          "    ---")
    for i in range(1,len(orders)):
        print("  {:e}  |" .format(cos_sin2_errors[i]),
               " {:f}  ||".format(cos_sin2_errors[i-1]/cos_sin2_errors[i]),
               " {:e}  |" .format(sin3_errors[i]),
               " {:f}"    .format(sin3_errors[i-1]/sin3_errors[i]))
    print("\n")


    
# generate symmetric 2D quadrature
def symmetric2D(N):
    
    N_polar = N
    N_azimu = int(N/2)
    
    # generate 1D Gauss polar quadrature
    pol_pts = np.polynomial.legendre.leggauss(N_polar)[0]
    pol_wts = np.polynomial.legendre.leggauss(N_polar)[1]
    
    # generate 1D symmetric azimuthal quadrature
    az_pts, az_wts = symmetric1d(N_azimu)
    
    # product quadrature set
    prod_quad = np.zeros((N_polar,N_azimu,2,3))
    
    # loop over polar levels
    for m in range(N_polar):
        
        # loop over azimuthal cells
        for n in range(N_azimu):
            
            # loop over local Gauss points
            for k in range(2):
                
                # points
                prod_quad[m,n,k,0] = pol_pts[m]
                prod_quad[m,n,k,1] = az_pts[n,k]
                
                # weights
                prod_quad[m,n,k,2] = pol_wts[m]*az_wts[n,k]

    # return product quadrature
    return prod_quad



# perform 2D integral
def integrate2d(integrand, prod_quad, N):
    
    N_polar = N
    N_azimu = int(N/2)
    
    # true (global) integral value
    integral_true = dblquad(integrand, 0., np.pi, -1., 1.)[0]
    
    # quadrature integral value
    integral_quad = 0.
    for i in range(N_polar):
        for j in range(N_azimu):
            for k in range(2):
                integral_quad += integrand(prod_quad[i,j,k,0],
                                           prod_quad[i,j,k,1])*prod_quad[i,k,k,2]
                
    # integral error
    error = abs(integral_true-integral_quad)

    # integral values as string
    string = "{:f} |" .format(integral_true).rjust(12) + \
             "{:f} ||".format(integral_quad).rjust(13) + \
             "{:e}"   .format(error).rjust(13)
             
    # return string
    return string, error
    
    
    
# 2D integrals (harmonics)
def integrals2d(N, tol=1e-8):
    
    print("\n################################################")
    print("################# Symmetric 2D #################")
    print("################################################")
    
    # generate product quadrature
    prod_quad = symmetric2D(N)
    
    # loop over polar orders
    for l in range(2*N-1):
        
        # loop over azimuthal orders
        print()
        for m in range(l+1):
            
            # reject harmonics
            if ((l >= N) and (m >= N or m <= l-N)):
                continue
            
            # generate harmonic
            integrand = lambda xi, phi: np.sqrt((2. - 1.*(m == 0)) * factorial(l-m) \
                                                                   / factorial(l+m)) \
                                              * lpmv(l,m,xi) * np.cos(m*phi)
            
            # integrate harmonic
            string, error = integrate2d(integrand, prod_quad, N)
            
            # print integral
            print("{:n}".format(l).rjust(3) + "{:n} ||".format(m).rjust(6) + string)

    print("\n")



# quadrature order
N = 8    
    
# 1D quadrature
# integrals in first subinterval
print()
integrals1d(N)

# integration accuracy
orders = [4, 8, 16, 32, 64, 128, 256, 512, 1024] 
accuracy(orders)

# 2D quadrature
# 2D global integrals
integrals2d(N, tol=1.)