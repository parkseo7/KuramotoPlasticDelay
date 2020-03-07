from __future__ import division, print_function

import os
import numpy as np
from numpy import linalg
from numpy.polynomial.polynomial import polyval
from scipy import misc
import math
from math import pi

# BRUTE FORCE COMPUTATION OF EIGENVALUES

# N-LIMIT ASYMPTOTIC INTEGRANDS

def sine_fold(delta2, L=pi, steps=50):
    '''
    Approximates the integral for the folded sine integral in the decomposition
    of the Omega equation, where tauE = 0
    '''
    
    Delta = np.linspace(0, L, num=steps)
    gauss = np.exp(-Delta**2/(2*delta2))
    
    S = (2*pi*delta2)**(-1/2)*np.sin(Delta)*gauss
    
    return L*np.sum(S) / steps


def sine_fold_gain(Omega, delta2, tau0, gain, L=pi, steps=50):
    '''
    Approximates the integral for the folded sine integral in the decomposition
    of the Omega equation, where tauE > 0.
    '''
    
    Delta = np.linspace(0, L, num=steps)
    gauss = np.exp(-Delta**2/(2*delta2))
    sine = np.sin(-Omega*tau0 + (1-Omega*gain)*Delta)
    
    S = (2*pi*delta2)**(-1/2)*sine*gauss
    
    return L*np.sum(S) / steps


def sine_fold_poly(Omega, delta2, tau0, gain, deg=2):
    '''
    Returns the terms of the sine Gaussian expanded polynomial up
    to degree deg.
    '''
    
    a = 1 - Omega*gain
    terms = np.zeros(deg+1)
    
    # Fill in odd powers
    for k in range(deg):
            numer = (-1)**k*a**(2*k+1)*(2*delta2)**(k+1/2)*misc.factorial(k)
            denom = misc.factorial(2*k+1)*2*np.sqrt(pi)

            terms[k] = numer / denom
            
    return terms


# MATRICES

def IM(z, g, gain, Omega, tauE, Delta):
    '''
    Given complex number z, with parameter set param, returns the complex 
    eigenvalue matrix, reduced to N dimensions. 
    '''
    
    # Check if z = -1:
    if z == -1:
        return np.zeros((2,2))
    
    # SETUP
    N = Delta.shape[0]
    
    # Define matrix entries
    cos_M = np.cos(-Omega*tauE + Delta)
    exp_z = g*cos_M*np.exp(-z*tauE) / N
    
    scos_M = np.sum(cos_M, axis=1)
    M_i = (g/N)*(1 - Omega*gain / (z + 1))*np.diag(scos_M)
    
    M_ij = (g/N)*(Omega*gain / (z + 1))*cos_M
    
    eM = z*np.eye(N) + M_i - exp_z + M_ij 
    
    return eM


def M(z, g, gain, Omega, tauE, Delta):
    '''
    Given complex number z, with parameter set param, computes the complex 
    determinant of eigenvalue matrix in the top left quadrant. param is a
    dictionary that includes parameters N, Omega, phi array.
    '''
    
    # SETUP
    N = Delta.shape[0]
    
    # Define each matrix
    M_1 = M1(z, g, Omega, tauE, Delta)
    M_2 = (g*Omega/N)*M2(Omega, tauE, Delta)
    M_3 = gain*M3(N)
    M_4 = (z+1)*np.eye(N**2)
    
    # Concatenate the matrices appropriately
    M_top = np.concatenate((M_1, M_2), axis=1)
    M_bot = np.concatenate((M_3, M_4), axis=1)
    
    M = np.concatenate((M_top, M_bot), axis=0)
    
    return M


def M1(z, g, Omega, tauE, Delta):
    '''
    Given an N by N tauE equilibrium matrix, and an N-dim phi array, returns
    the M1 matrix (top left), which is an N by N^2 matrix relevant to the
    eigenvalue matrix.
    '''
    
    cos_M = np.cos(-Omega*tauE + Delta)
    M1_2 = cos_M*np.exp(-z*tauE)
    M1_1_arr = np.sum(cos_M, axis=1)
    M1_1 = np.diag(M1_1_arr)
    
    N = M1_1_arr.size
    M1_0 = np.eye(N)
    
    return z*M1_0 - (g/N)*(M1_2 - M1_1)


def M2(Omega, tauE, Delta):
    '''
    Given an N by N tauE equilibrium matrix, and an N-dim phi array, returns
    the M2 matrix (top right), which is an N by N^2 matrix relevant to the
    eigenvalue matrix, while missing the g*Omega / N coeff.
    '''
    
    # Construct the cosine matrix.
    cos_M = np.cos(-Omega*tauE + Delta)
    
    # Define M2
    N = cos_M.shape[0]
    M2_mat = np.zeros((N, N**2))
    
    # Fill M2 with appropriate values using cos_M
    for i in range(N):
        M2_mat[i][i*N:(i+1)*N] = cos_M[i]
    
    return -1*M2_mat


def M3(N):
    '''
    Defines the M3 matrix (bottom left), which is an N^2 by N matrix whose
    ijth row is 1 on the jth index, -1 on the ith index, and 0 elsewhere,
    1 <= i,j <= N. This matrix is missing the gain coefficient.
    '''
    
    M3_mat = np.zeros((N**2, N))
    
    # Fill in entries
    for i in range(N):
        for j in range(N):
            M3_mat[i*N + j, j] = 1
            M3_mat[i*N + j, i] -= 1
    
    return -1*M3_mat

    
if __name__ == '__main__':
    # N = 900
    # a = np.random.random(size=(N,N)) + 1j*np.random.random(size=(N,N))
    # d = linalg.slogdet(a)
    
    # X = (2+1j)*np.eye(2)
    # d = linalg.slogdet(X)
    
    # slogdet computes the log norm of the complex determinant.
    # First complex value is the normalized determinant (sign)
    delta2 = 1**2
    s = sine_fold(delta2)