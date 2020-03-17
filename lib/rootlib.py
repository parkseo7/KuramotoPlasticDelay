from __future__ import division, print_function

import os
import numpy as np

from scipy import optimize
import matplotlib.pyplot as plt
import math
from math import pi


# ROOT FINDING LIBRARY


# Single-root
def find_root(func, x0, a, b, steps=20):
    '''
    Finds a root of func with initial guess x0. Utilizes the optimize.root 
    function. If multiple roots are found, returns the closest root to x0.
    '''
    
    all_roots = root_on_interval(func, a, b, steps=steps)
    if all_roots.size == 0:
        return (x0, False)
    else:
        root = all_roots[np.argmin(np.abs(all_roots - x0))]
    
    if root.size > 1:
        root = root[0]
    
    return (root, True)


def root_on_interval(func, a, b, steps=20):
    '''
    Given a 1-D function func, finds all roots of the function on the interval 
    [a,b]. Steps is the partition number N in which the array of func is
    computed over a,b. 
    '''
    
    # Compute all arrays
    x_array = np.linspace(a,b, num=steps)
    
    N0 = x_array.size
    f_array = np.array([func(x_array[k]) for k in range(N0)])
    
    # Shifted f_array:
    f_array1 = np.array([f_array[j+1] for j in range(N0-1)])
    
    # Initial zero check:
    zero_array = x_array[f_array == 0]
    
    # Check for all sign changes:
    sign_ind_array = np.argwhere(f_array[:-1]*f_array1 < 0)
    
    # Implement bisect method:
    root_array = np.array([optimize.bisect(func, x_array[l], x_array[l+1]) 
                           for l in sign_ind_array])
    
    all_root_array = np.concatenate((zero_array, root_array))
    
    return all_root_array


def root_on_array(f_array):
    '''
    Given an array f_arr, returns all indices i such that f_arr[i] and f_arr[i+1]
    are different signs.
    '''
    
    N = f_array.size
    f_array1 = f_array[:N-1]
    f_array2 = f_array[1:]
    
    ind_array = np.argwhere(f_array1*f_array2 < 0)
    
    return ind_array