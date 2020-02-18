from __future__ import division, print_function

import os
import numpy as np
from numpy.polynomial.polynomial import polyval
import math
from math import pi

# FUNCTIONS TO USE

def R_sum(u, N_tau, N_x, param, phi_fun):
    '''
    Computes the right-side integral as a triple-Riemann sum with N_tau,
    N_x and unknown gamma.
    '''
    
    T = param['T']
    tau_sum = 0
    for j in range(N_tau):
        tau = 2*T*j / N_tau
        tau_sum += phase_sum(u, tau, N_x, param, phi_fun)
    
    return tau_sum / N_tau


def R_sum_gauss(u, N_tau, N_x, param, sigma):
    '''
    Computes the right-side integral as a Riemann sum with N_tau,
    N_x and unknown sigma, using a Gaussian distribution for the asymptotic
    phase differences.
    '''
    
    T = param['T']
    tau_sum = 0
    for j in range(N_tau):
        tau = 2*T*j / N_tau
        tau_sum += phase_gauss(u, tau, N_x**2, param, sigma)
    
    return tau_sum / N_tau


# Distance function
def dists(F_fun, G_fun, interval, steps=20):
    '''
    Given two functions F_fun, G_fun: R -> R^N, returns the supremum difference
    along each co-ordinate between F_fun, G_fun over all t in the closed interval, 
    computed with uniform steps, and the respective timepoint at which it occurs.
    '''
    
    # Time array
    t_arr = np.linspace(interval[0], interval[1], num=steps)
    
    # Dimension
    N = F_fun(interval[0]).size
    # Construct full difference array
    diff_arr = np.zeros((t_arr.size, N))
    
    # Calculate difference at each time point
    for i in range(t_arr.size):
        t = t_arr[i]
        diff_t = np.abs(F_fun(t) - G_fun(t))
        diff_arr[i] = diff_t
    
    # Compute the maximum distances and respective timepoints
    dist = np.amax(diff_arr, axis=0)
    dist_k = np.argmax(diff_arr, axis=0)
    t_dist = t_arr[dist_k]
        
    return t_dist, dist


# Fixed-point equation
def Omega_root(phi1, tau0, param):
    '''
    Given an N by N dimensional array of initial delay values, provides the
    global frequency and phase offset function with phi_1 = 0. Here,
    phi1 is an N-1 array.
    '''
    
    # Define function
    gain = param['gain']
    g = param['g']
    w0 = param['w0']
    
    N = tau0.shape[0]
    Omega = phi1[0]
    phi = np.concatenate((np.array([0]), phi1[1:]))
    Delta = (phi[:,None] - phi).T
    
    sin0 = np.sin(-Omega*np.maximum(tau0 + gain*Delta, np.zeros((N,N))) + Delta)
    
    return (Omega - w0) - (g/N)*np.sum(sin0, axis=1)


# SUPPLEMENTARY FUNCTIONS

def phase_sum(u, tau, N, param, phi_fun):
    '''
    Computes the right-side integral as a double-Riemann sum with N steps,
    at delay tau, and prediction phi function phi(x)
    '''
    
    w0 = param['omega0']
    g = param['g']
    a = param['a']
    T = param['T']
    gain = param['gain']
    
    x_arr = phi_fun(np.arange(N)/N)
    N_diffs = (x_arr[:,None] - x_arr).T
    z0 = np.zeros(N_diffs.shape)
    N_arr = np.sin(-u*np.maximum(tau + gain*N_diffs, z0) + N_diffs)
    
    return np.sum(N_arr) / N**2


def phase_gauss(u, tau, N, param, sigma):
    '''
    Computes the right-side integral as a double-Riemann sum with N steps,
    at delay tau, and a Gaussian distribution of differences at mean 0 and
    variance sigma^2.
    '''
    
    w0 = param['omega0']
    g = param['g']
    a = param['a']
    T = param['T']
    gain = param['gain']
    
    z0 = np.zeros(N)
    N_diffs = -pi + 2*pi*np.arange(N) / N
    gauss = ((np.sqrt(2*pi)*sigma)**-1)*np.exp(-N_diffs**2 / (2*sigma**2))
    
    N_arr = np.sin(-u*np.maximum(tau + gain*N_diffs, z0) + N_diffs)*gauss
    
    return 2*pi*np.sum(N_arr) / N


if __name__ == '__main__':
    z = np.random.random(size=(3,4))
    Y = polyval(z, [2,3,4])
    
    