from __future__ import division, print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi

# Processing all asymptotic behaviour of solutions, using an MAT file containing all information


# SUPPLEMENTARY FUNCTIONS

def weight_avg(X, Y, asy):
    '''
    Given a 1-array X, computes the weighted average of Y along each column, 
    using the last asy percentage of Y.
    '''
    
    # Starting index
    ind_s = int(X.size*(1 - asy))

    # Difference array for X
    X_asy1 = X[ind_s:-1]
    X_asy2 = X[ind_s+1:]
    
    X_diff = X_asy2 - X_asy1
    X_diff_rep = np.tile(X_diff, (Y.shape[1],1)).T
    
    # Average value along all asy indices
    Y_asy = Y[ind_s:-1]
    
    L_sum = np.sum(Y_asy*X_diff_rep, axis=0)
    L_avg = L_sum / (X[-1] - X[ind_s])
    
    return L_avg


def mod_pi(X):
    '''
    Given an array of values X, returns a modded version of X, such that all
    elements of X are in interval [-pi, pi].
    '''
    
    X_mod = np.mod(X, 2*pi)
    
    # Change all negatives:
    is_pi = (X_mod < pi)
    is_2pi = (X_mod > pi)
    
    X_mod2 = -(X_mod - pi)
    Y = X_mod*is_pi + X_mod2*is_2pi
    
    return Y


if __name__ == '__main__':
    pass

