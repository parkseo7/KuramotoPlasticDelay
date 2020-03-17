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


# 2D ANALYSIS
def Omega2D(Omega, param):
    '''
    Determines the fixed-point equations for Omega, Delta, given initial guesses
    Omega0, Delta0.
    '''
    
    # Parameters
    g = param['g']/2
    w0 = param['omega0']
    gain = param['gain']
    tau0 = param['tau0']
    
    # Here Delta = Delta_21
    Delta_fun = lambda u: np.arcsin((w0 - u)/g)
    
    # Fixed-point equation for Omega:
    Omega_fun = lambda u: u - w0 - g*np.sin(-u*tau0 + (1 - gain*u)*Delta_fun(u))
    
    return Omega_fun(Omega), Delta_fun(Omega)


def Omega2D_root(u, tau0, param):
    '''
    Returns the 2D system of equations for Omega, Delta_21, to be used with
    an optimization function in 2-dimensions.
    '''
    
    # Define function
    gain = param['gain']
    g = param['g']/2
    w0 = param['omega0']
    
    Omega, Delta = u
    pmDelta = Delta*np.array([1,-1])
    sin0 = np.sin(-Omega*np.maximum(tau0 + gain*pmDelta, np.zeros(2)) + pmDelta)
    
    f = Omega - w0 - g*sin0
    return f


def eig2D_det(z, Omega, Delta, param):
    '''
    Returns the 2x2 determinant complex eigenvalue criterion, in the form of
    an exponential polynomial. Here, Omega, Delta are (one of the) solutions
    to the fixed-point equation given by Omega2D. We assume that Delta > 0.
    '''
    
    # Parameters
    g = param['g']/2
    w0 = param['omega0']
    gain = param['gain']
    tau0 = param['tau0']
    
    # Defined parameters
    k = Omega*gain
    C_12 = g*np.cos(-Omega*tau0 + (1 - k)*Delta)
    C_21 = g*np.cos(Delta)
    
    # Polynomials
    P = (z*(z+1) + C_12*(z+1-k))*(z+C_21) + C_12*C_21*k
    Q = -C_12*C_21*(z+1)
    E = np.exp(-z*(tau0 + gain*Delta))
    
    # return np.array([P, Q, E, P + Q*E, P + Q])
    return P + Q*E


def eig2D_cubic(Omega, Delta, param):
    '''
    Returns the coefficients of the quartic polynomial P(z) + Q(z) from the
    exponential polynomial in our eigenvalue equation.
    '''
    
    # Parameters
    g = param['g']/2
    w0 = param['omega0']
    gain = param['gain']
    tau0 = param['tau0']
    
    # Defined parameters
    k = Omega*gain
    C_12 = g*np.cos(-Omega*tau0 + (1 - k)*Delta)
    C_21 = g*np.cos(Delta)
    C = C_12 + C_21
    C_2 = C_12*C_21
    
    b_3 = 1
    b_2 = 1 + C_12 + C_21
    b_1 = C_12*(1-k) + C_21
    b_0 = 0
    
    return np.array([b_3, b_2, b_1, b_0])


# N-LIMIT ANALYSIS

def Omega_infty(u, delta2, param, L=pi, steps=50):
    '''
    Computes the right-side integral as a Riemann sum with N steps,
    at delay tau, and a Gaussian distribution of differences at mean 0 and
    variance sigma^2.
    '''
    
    w0 = param['omega0']
    g = param['g']
    gain = param['gain']
    tau0 = param['tau0']
    
    N = steps
    z0 = np.zeros(N)
    Delta = -L + 2*L*np.arange(N) / N
    
    if delta2 == 0:
        sin_arr = np.sin(-u*tau0 + z0)
    else:
        gauss = ((np.sqrt(2*pi*delta2))**-1)*np.exp(-Delta**2 / (2*delta2))
        sin_arr = np.sin(-u*np.maximum(tau0 + gain*Delta, z0) + Delta)*gauss
    
    return w0 + g*2*L*np.sum(sin_arr) / N


def Omega_infty_double(u, delta2, param, L=pi, steps=50):
    '''
    Computes the right-side integral as a double-Riemann sum with N steps,
    at delay tau, and a Gaussian distribution of differences at mean 0 and
    variance sigma^2.
    '''
    
    w0 = param['omega0']
    g = param['g']
    gain = param['gain']
    tau0 = param['tau0']
    
    N = steps
    z0 = np.zeros((N,N))
    Delta = -L + 2*L*np.arange(N) / N
    
    if delta2 == 0:
        sin_arr = np.sin(-u*tau0 + z0)
        integral = (2*L/N) * np.sum(sin_arr)
        
    else:
        Delta_diff = (Delta[:,None] - Delta).T
        sin_diff = np.sin(-u*np.maximum(tau0 + gain*Delta, z0) + Delta)
        gauss = ((np.sqrt(2*pi*delta2))**-1)*np.exp(-Delta**2 / (2*delta2))
        gauss_pair = np.matmul(gauss[:,None], np.array([gauss]))
        sin_arr = sin_diff * gauss_pair
        integral = (2*L/N)**2 * np.sum(sin_arr)
    
    return w0 + g * integral


def eigN_limit(z, Omega, delta2, tau0, param, steps=50, cap=10):
    '''
    Returns the (signed) error of the eigenvalue N-limit equation at eigenvalue
    z. To be used to generate an error heatmap. If tauE > cap, let tauE = cap.
    '''
    
    # Parameters
    g = param['g']
    gain = param['gain']
    k = Omega*gain
    
    Delta = np.linspace(-pi, pi, num=steps)
    zeros_N = np.zeros(Delta.size)
    ones_N = np.ones(Delta.size)
    tauE = np.maximum(tau0 + gain*Delta, zeros_N)
    tauE = np.minimum(tauE, cap*ones_N)
    
    C = g*np.cos(-Omega*tauE + Delta)
    H = np.maximum(Delta, zeros_N)
    
    gauss = (2*pi*delta2)**(-1/2)*np.exp(-Delta**2/(2*delta2))
    
    term1 = z*(z+1)
    term2 = np.sum(C*(z+1-k*H)*gauss)*(2*pi/zeros_N.size)
    term3 = np.sum(C*(k*H - (z+1)*np.exp(-z*tauE))*gauss)*(2*pi/zeros_N.size)
    
    return term1 + term2 + term3


def eigN_limit2(z, Omega, delta2, tau0, param, steps=50, std=2):
    '''
    Simplified version of the eigenvalue equation.
    '''
    
    # Parameters
    g = param['g']
    gain = param['gain']
    k = Omega*gain
    
    delta = np.sqrt(delta2)
    Delta = np.linspace(-std*delta, std*delta, num=steps)
    zeros_N = np.zeros(Delta.size)
    tauE = np.maximum(tau0 + gain*Delta, zeros_N)
    
    C = g*np.cos(-Omega*tauE + Delta)
    
    gauss = (2*pi*delta2)**(-1/2)*np.exp(-Delta**2/(2*delta2))
    
    return np.sum(C*(np.exp(-z*tauE) - 1)*gauss)*2*std*delta/zeros_N.size - z
    

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


def quadratic_roots(coeffs):
    '''
    Returns the two branch roots of the quadratic using the coefficients.
    '''
    
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    
    r1 = (-b + np.sqrt(b**2 - 4*a*c, dtype='complex64'))/(2*a)
    r2 = (-b - np.sqrt(b**2 - 4*a*c, dtype='complex64'))/(2*a)
    
    return (r1, r2)


def abs_diff(poly_array):
    '''
    Given an array of two polynomial terms, returns the squared absolute 
    difference.
    '''
    
    P = poly_array[0]
    Q = poly_array[1]
    
    return np.abs(P)**2 - np.abs(Q)**2


def sign_log(x):
    '''
    Given any float x, returns the sign log scale of x.
    '''
    
    return np.sign(x)*np.log(1 + np.abs(x))


def sign_log_inv(y):
    '''
    Given any float y, returns the inverse of the sign log scale.
    '''
    
    if y > 0:
        x = np.exp(y) - 1
    elif y < 0:
        x = 1 - np.exp(-y)
    else:
        x = 0
    
    return x


# UNUSED
def eig2D_poly(z, Omega, Delta, tau0, param):
    '''
    Returns the 2x2 determinant complex eigenvalue criterion, in the form of
    an exponential polynomial. Here, Omega, Delta are (one of the) solutions
    to the fixed-point equation given by Omega2D. We assume that Delta > 0.
    '''
    
    # Parameters
    g = param['g']/2
    w0 = param['omega0']
    gain = param['gain']
    
    # Defined parameters
    k = Omega*gain
    C_12 = g*np.cos(-Omega*tau0[0] + (1 - k)*Delta)
    C_21 = g*np.cos(Delta)
    
    # Polynomials
    P = (z*(z+1) + C_12*(z+1-k))*(z*(z+1) + C_21*(z+1-k)) + C_12*C_21*k*(z+1-k)
    Q = -C_12*C_21*(z+1)*(z+1-k)
    E = np.exp(-z*(tau0[0] + gain*Delta))
    
    return np.array([P, Q, E, P + Q*E, P + Q])


def eig2D_quartic(Omega, Delta, tau0, param):
    '''
    Returns the coefficients of the quartic polynomial P(z) - Q(z) from the
    exponential polynomial in our eigenvalue equation.
    '''
    
    # Parameters
    g = param['g']/2
    w0 = param['omega0']
    gain = param['gain']
    
    # Defined parameters
    k = Omega*gain
    C_12 = g*np.cos(-Omega*tau0[0] + (1 - k)*Delta)
    C_21 = g*np.cos(Delta)
    C = C_12 + C_21
    C_2 = C_12*C_21
    
    # Coefficients
    b_4 = 1
    b_3 = 2 + C
    b_2 = 1 + C*(2-k)
    b_1 = C*(1-k) + 2*C_2*(1-k) + C_2*k - C_2*(2-k)
    b_0 = C_2*(1-k)**2 + C_2*k*(1-k) - C_2*(1-k)
    
    return np.array([b_4, b_3, b_2, b_1, b_0])


def invar_err(Omega, U, N_x, param):
    '''
    Evaluate the (approximate) integral giving the error of the invariance
    criterion. That is, if invar_err = 0, then Omega, U is a plausible
    synchronization state.
    '''
    
    # Parameters
    tau0 = param['T']
    gain = param['gain']
    
    # Function to integrate (x = theta)
    f = lambda x: np.abs(np.sin(-Omega*tau0 + (1-Omega*gain)*(U - x)) + np.sin(x))
    
    # Compute Riemann sum
    x_arr = np.linspace(0,U, num=N_x)
    err = np.sum(f(x_arr))*U/N_x
    
    return err


def invar_err2(Omega, L, dist_fun, N, param):
    '''
    Sanity check.
    '''
    
    # Parameters
    tau0 = param['T']
    gain = param['gain']
    w0 = param['omega0']
    g = param['g']
    
    phi = np.linspace(0,U)/N
    phi_diffs = (phi[:,None] - phi).T
    z0 = np.zeros((N,N))
    
    sin_arr = np.sin(-Omega*np.maximum(tau0 + gain*phi_diffs, z0) + phi_diffs)    
    den_arr = dist_fun(phi)
    
    # Summation
    err = Omega - w0 - g*np.sum(sin_arr, axis=1)/N
    
    return err

 
def invar_err3(Omega, phi, param):
    '''
    Given asymptotic solutions, computes the invariance error.
    '''
    
    # Parameters
    tau0 = param['T']
    gain = param['gain']
    w0 = param['omega0']
    g = param['g']
    
    N = phi.size
    phi_diffs = (phi[:,None] - phi).T
    z0 = np.zeros(phi_diffs.shape)
    sin_arr = np.sin(-Omega*np.maximum(tau0 + gain*phi_diffs, z0) + phi_diffs)    
    
    # Summation
    err = Omega - w0 - g*np.sum(sin_arr, axis=1)/N
    
    return err


def quartic_roots(coeffs):
    '''
    Given an array of coefficients (in increasing order of degrees) up to
    order 4, returns the quartic roots corresponding to the polynomial with
    the inputted coefficients.
    '''
    
    # Coefficients
    a,b,c,d,e = coeffs

    # Define p,q
    p = (8*a*c - 3*b**2) / (8*a**2)
    q = (b**3 - 4*a*b*c + 8*a**2*d) / (8*a**3)
    
    D_0 = c**2 - 3*b*d + 12*a*e
    D_1 = 2*c**3 - 9*b*c*d + 27*b**2*e + 27*a*d**2 - 72*a*c*e
    
    Q = (np.sqrt(D_1 + (D_1**2 - 4*D_0**3), dtype='complex64')/2)**(1/3)
    S = 0.5*(-2*p/3 + (Q + D_0/Q)/(3*a))**0.5
    
    # Roots
    T_1 = -b/(4*a)
    T_2 = 0.5*np.sqrt(-4*S**2 - 2*p + q/S, dtype='complex64')
    
    x1 = T_1 - S - T_2
    x2 = T_1 - S + T_2
    x3 = T_1 + S - T_2
    x4 = T_1 + S + T_2
    
    return np.array([x1,x2,x3,x4])


if __name__ == '__main__':
    # Parameters
    g = 1.5
    gain = 30
    Omega = 0.6
    delta2 = 0.4**2
    steps=50
    tau0 = 0.1
    k = Omega*gain
    
    Delta = np.linspace(-pi, pi, num=steps)
    zeros_N = np.zeros(Delta.size)
    tauE = np.maximum(tau0 + gain*Delta, zeros_N)
    C = g*np.cos(-Omega*tauE + Delta)
    H = np.maximum(Delta, zeros_N)
    
    gauss = (2*pi*delta2)**(-1/2)*np.exp(-Delta**2/(2*delta2))
    
    z = -1+1j
    term1 = z*(z+1)
    term2 = np.sum(C*(z+1-k*H)*gauss)*(2*pi/zeros_N.size)
    term3 = np.sum(C*(k*H - (z+1)*np.exp(-z*tauE))*gauss)*(2*pi/zeros_N.size)
    
    