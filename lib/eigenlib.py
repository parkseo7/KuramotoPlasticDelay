from __future__ import division, print_function

import os
import numpy as np
from numpy import linalg
from numpy.polynomial.polynomial import polyval
from scipy import misc
from scipy import special
from scipy import stats
import math
from math import pi

# BRUTE FORCE COMPUTATION OF EIGENVALUES

# N-LIMIT ASYMPTOTIC INTEGRANDS

def sine_fold(t, delta2, L=pi, steps=50):
    '''
    Approximates the integral for the folded sine integral in the decomposition
    of the Omega equation, where tauE = 0
    '''
    
    Delta = np.linspace(0, L, num=steps)
    gauss = np.exp(-Delta**2/(2*delta2))
    
    S = (2*pi*delta2)**(-1/2)*np.sin(t*Delta)*gauss
    
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


# Eigenvalues in N-limit

def cos_int(z, Omega, delta2, tau0, gain, L=pi, steps=50):
    '''
    Compute the integrand on the right-side of the eigenvalue equation.
    '''
    
    Delta = np.linspace(0, L, num=steps)
    gauss = np.exp(-Delta**2/(2*delta2))
    cosine = np.cos(-Omega*tau0 + (1-Omega*gain)*Delta)
    
    S = (2*pi*delta2)**(-1/2)*cosine*(np.exp(-z*(tau0 + gain*Delta)/gain) - 1)*gauss
    
    return L*np.sum(S) / steps


def sin_pow_ints(n, Omega, delta2, tau0, gain, L=pi, steps=50):
    '''
    Returns an array of sine Gaussian power terms, obtained by taking the power
    series of the exponential term.
    '''
    
    pass


def eig_coeffs(M, Omega, delta2, tau0, gain):
    '''
    Returns the coefficient of all powers of the eigenvalue term lambda,
    in the Gaussian asymptotic expansion of C(x)e^-bx * gauss up to the Mth
    degree.
    '''
    
    coeffs = np.zeros(M+1)
    
    for m in range(1,M+1):
        cos_term = cos_coeff(m, M, Omega, delta2, tau0, gain)
        sin_term = sin_coeff(m, M, Omega, delta2, tau0, gain)
        coeffs[m] = cos_term + sin_term
        
    return coeffs


def cos_coeff(n, M, Omega, delta2, tau0, gain):
    '''
    Returns the coefficient of the nth power of the eigenvalue term lambda,
    in the Gaussian asymptotic expansion of cos(ax)e^-bx * gauss up to the Mth
    degree.
    '''
    
    if n > M:
        return 0
    
    # Define A,B
    A = 1 - Omega*gain
    B = 1 # gain
    n_fac = misc.factorial(n)
    
    # Summation
    coeff = 0
    k = 0
    K = (M-n)/2
    while k <= K:
        numer = (-1)**(k+n)*A**(2*k)*B**n*gauss_moment(delta2, 2*k+n)
        denom = misc.factorial(2*k)*n_fac
        
        coeff += (numer/denom)
        k += 1
        
    return np.sin(-Omega*tau0)*coeff


def sin_coeff(n, M, Omega, delta2, tau0, gain):
    '''
    Returns the coefficient of the nth power of the eigenvalue term lambda,
    in the Gaussian asymptotic expansion of sin(ax)e^-bx * gauss up to the Mth
    degree.
    '''
    
    if n > M:
        return 0
    
    # Define A,B
    A = 1 - Omega*gain
    B = 1 # gain
    n_fac = misc.factorial(n)
    
    # Summation
    coeff = 0
    k = 0
    K = (M-n-1)/2
    while k <= K:
        numer = (-1)**(k+n)*A**(2*k+1)*B**n*gauss_moment(delta2, 2*k+n+1)
        denom = misc.factorial(2*k+1)*n_fac
        
        coeff += (numer/denom)
        k += 1
        
    return np.cos(-Omega*tau0)*coeff

        
def gauss_moment(delta2, m):
    '''
    Returns the nth moment of the absolute Gaussian random variable 
    (divided by 2) with variance delta2.
    '''
    
    delta = np.sqrt(delta2)
    
    if m % 2 == 0:
        b = 1/2
    else:
        b = 1/np.sqrt(2*pi)
    
    return b*delta**m*misc.factorial2(m-1)


# MODIFIED EIGENVALUE INTEGRAL EXPANSION

def Omega_infty_asy(Omega, delta2, param, deg_sin=50):
    '''
    The asymptotically expanded integral for Omega. To be used as a root equation
    for N-limit synchronization point (Omega, delta2).
    '''
    
    # Parameters
    w0 = param['omega0']
    g = param['g']
    tau0 = param['tau0']
    gain = param['gain']
    
    # Other values
    A = 1 - gain*Omega
    
    term1 = np.sin(-Omega*tau0)*np.exp(-(A**2*delta2)/2)/2
    term2 = trig_power_gauss2(A, 0, delta2, deg=deg_sin)
    term2 *= np.cos(-Omega*tau0) / np.sqrt(2*pi)
    term3 = trig_power_gauss2(1, 0, delta2, deg=deg_sin)
    term3 *= 1 / np.sqrt(2*pi)
    
    return w0 + g * (term1 + term2 - term3)


def eig_infty_scale(z, Omega, delta2, param, deg=2, R=1, L=pi, steps=100):
    '''
    Returns the right-side of the fixed-point scaled eigenvalue equation
    (by R*gain), with the power series of the exponential term up to degree deg.
    '''
    
    # Parameters
    w0 = param['omega0']
    g = param['g']
    tau0 = param['tau0']
    gain = param['gain']
    
    I_0 = R * gain * g * cos_fold_gain(Omega, delta2, tau0, gain, L=L, steps=steps)
    
    coeffs = np.zeros(deg+1)
    for n in range(deg+1):
        coeffs[n] = eig_infty_coeff(n, Omega, delta2, param, R=R, L=L, steps=steps)
    
    z_pow = powers(z, deg)
    
    taus = tau0 / (R*gain)
    RS = np.sum(coeffs * z_pow)*np.exp(-z*taus) - I_0
    
    return RS


def eig_infty_poly(deg, Omega, delta2, param, R=1, L=pi, steps=100):
    '''
    Returns an array of coefficients of lambda**n of the roots of the expanded
    polynomial for eigenvalues, with taus = 0. To be used with np.root.
    '''    
    
    if delta2 < 1e-10:
        return np.array([1,0])
    
    else:
        (coeffs, taus, I_0) = eig_infty_terms(deg, Omega, delta2, param, R=R, L=L, steps=steps)
        
        poly_coeffs = coeffs.copy()
        poly_coeffs[0] += -I_0
        poly_coeffs[1] += -1
        
        return np.flip(poly_coeffs,0)
    
    
def eig_infty_terms(deg, Omega, delta2, param, R=1, L=pi, steps=100):
    '''
    Returns an array of coefficients of lambda**n of the asymptotically expanded integral 
    for lambda. To be used as a root equation for N-limit eigenvalues at (Omega, delta2).
    Here delta2 > 0.
    '''
    
    # Parameters
    w0 = param['omega0']
    g = param['g']
    tau0 = param['tau0']
    gain = param['gain']
    
    I_0 = R*gain*g * cos_fold_gain(Omega, delta2, tau0, gain, L=L, steps=steps)
    
    coeffs = np.zeros(deg+1)
    for n in range(deg+1):
        coeffs[n] = eig_infty_coeff(n, Omega, delta2, param, R=R, L=L, steps=steps)
    
    taus = tau0 / (R*gain)
    
    return (coeffs, taus, I_0)


def eig_infty_coeff(n, Omega, delta2, param, R=1, L=pi, steps=100):
    '''
    Returns the coefficient of lambda**n of the asymptotically expanded integral 
    for lambda. To be used as a root equation for N-limit eigenvalues at (Omega, delta2).
    Here delta2 > 0.
    '''
    
    # Parameters
    w0 = param['omega0']
    g = param['g']
    tau0 = param['tau0']
    gain = param['gain']
    
    # Change of variables
    A = 1 - gain*Omega
    
    cos_term = np.cos(-Omega*tau0) * cos_xN_gauss(A, n, delta2, L=L, steps=steps)
    sin_term = np.sin(-Omega*tau0) * sin_xN_gauss(A, n, delta2, L=L, steps=steps)
    
    coeff = R*gain*g*(cos_term - sin_term) / ((-R)**n * misc.factorial(n))
    
    return coeff


def cos_fold_gain(Omega, delta2, tau0, gain, L=pi, steps=50):
    '''
    Approximates the integral for the folded sine integral in the decomposition
    of the Omega equation, where tauE > 0.
    '''
    
    Delta = np.linspace(0, L, num=steps)
    gauss = np.exp(-Delta**2/(2*delta2))
    cosine = np.cos(-Omega*tau0 + (1-Omega*gain)*Delta)
    
    S = (2*pi*delta2)**(-1/2)*cosine*gauss
    
    return L*np.sum(S) / steps


# COSINE AND SINE POWER GAUSSIAN TERMS

def cos_xN_gauss(t, N, delta2, L=pi, steps=50):
    '''
    Returns an approximate integral of cos(tx)*x^N rho(x) from 0 to pi.
    '''
    
    Delta = np.linspace(0, L, num=steps)
    gauss = np.exp(-Delta**2/(2*delta2))
    
    S = (2*pi*delta2)**(-1/2)*np.cos(t*Delta)*Delta**N*gauss
    
    return L*np.sum(S) / steps


def sin_xN_gauss(t, N, delta2, L=pi, steps=50):
    '''
    Returns an approximate integral of sin(tx)*x^N rho(x) from 0 to L.
    '''
    
    Delta = np.linspace(0, L, num=steps)
    gauss = np.exp(-Delta**2/(2*delta2))
    
    S = (2*pi*delta2)**(-1/2)*np.sin(t*Delta)*Delta**N*gauss
    
    return L*np.sum(S) / steps


# COSINE AND SINE POWER GAUSSIAN TERMS (ANALYTIC EXPANSION)

def trig_power_gauss1(t, n, delta2):
    '''
    Returns the analytic term for the integral cos(tx)*x^(2k)*rho(x) or
    sin(tx)*x^(2k+1)*rho(x), where n = 2k or n = 2k+1, from 0 to L.
    '''
    
    # Use hermite polynomials
    H_n = special.hermite(n)
    
    # Convert to probabilist:
    H_e = lambda x: 2**(-n/2) * H_n(x/np.sqrt(2))
    
    # Return the expression
    dchar_dt = lambda y: H_e(y) * np.exp(-y**2/2) / 2
    
    delta = np.sqrt(delta2)
    sign = (-1)**(n - math.ceil(n/2))
    return sign * delta**n * dchar_dt(np.sqrt(delta2)*t)


def trig_power_gauss2(t, n, delta2, deg=2):
    '''
    Returns the analytic power series for the integral cos(tx)*x^(2k)*rho(x) or
    sin(tx)*x^(2k+1)*rho(x), where n = 2k or n = 2k+1, from 0 to L, up to degree
    deg.
    '''
    
    S = 0
    delta = np.sqrt(delta2)
    k = math.ceil((n-1)/2)
    while (2*k+1-n) <= deg:
        term = (-1)**k * t**(2*k+1-n) * delta**(2*k+1) / misc.factorial2(2*k+1)
        term *= deriv_fac(2*k+1,n)
        S += term
        k += 1
    
    sign = (-1)**(math.floor(n/2))
    return sign * (np.sqrt(2*pi)**-1) * S


# POWER SERIES
def sin_gauss_asy_series(t, delta2, N):
    '''
    Returns the asympototic expansion of the sine folded Gaussian with variance
    delta2, by taking the power series of sine up to degree N.
    '''
    
    S = 0
    delta = np.sqrt(delta2)
    for k in range(N+1):
        S += (-1)**k * (t*delta)**(2*k+1) / misc.factorial2(2*k+1)
    
    return (np.sqrt(2*pi))**-1 * S


def exp_power_series(z, N):
    '''
    Returns the power series at z, summing up to the Nth degree.
    '''
    
    S = 0
    for k in range(N+1):
        S += z**k / misc.factorial(k)
    
    return S


def powers(x, M):
    '''
    Returns an array of all powers of x up to degree M.
    '''
    
    return np.array([x**n for n in range(M+1)])


def deriv_fac(n, k):
    '''
    Returns the coefficient n! / (n-k)!, as the result of differentiating an
    power k times.
    '''
    
    M = 1
    for i in range(k):
        M *= (n-i)
    
    return M


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
    t = 10
    n = 3
    delta2 = 0.1**2
    deg = 3
    S = trig_power_gauss2(t, n, delta2, deg=2)
