# KuramotoDelay
Edited version of the Kuramoto Delay Network model simulations

# Numerical Methods for Kuramoto Delay Oscillators

The code to generate all plots in the manuscript 'Synchronization and resilience in the Kuramoto white matter network model with adaptive state-dependent delays'. All code is done in Python using the following libraries:

* Numpy
* Scipy
* Matplotlib
* tqdm (for progress bar)

All scripts are done with Jupyter Notebook

# Libraries

Personal modules in the Libraries folder contains all relevant functions that provide the numerical methods. These are imported by the Jupyter Notebook scripts. The following is a quick guide describing what each module contains.

* comp_pack: Contains functions to process the numerical arrays into the arrays that are plotted.
* eig_pack: Contains methods to generate the meshes for the global frequency and stability eigenvalues (for Figure 2).
* fun_pack: Contains the relevant DDE and Omega, eigenvalue functions.
* num_pack: Contains the numerical method to generate the solution arrays.
* kura_pack: Contains parameter structures for generating the numerical solutions for the DDE.
* plot_pack: Contains basic plot configurations to be called into the plot scripts.

# Figures

Each Jupyter Notebook file contains the exact script that was used to generate each plot in the manuscript. Cells in the scripts are able to generate and export the array files used, which are stored in the 'data' folder. Other cells in the scripts import and process the array files to plot the relevant graphs. We provide a quick guide in how to reproduce each figure:

## Figure 2

The script for generating Figure 2 can be found in the notebook file ```eig_script.ipynb```. The script utilizes the data set \data\Fig2_dataset. 

## Figure 3-6 (generating arrays)

Computation of the solution arrays is done in the notebook file ```comp_script.ipynb```. All relevant arrays are exported as .gz files.

## Figure 3, 5 plots

These figures use a single solution instance for each plot, and are done in the notebook file ```fig_script_single.ipynb```. Edit the 'dataset_name' to the folder containing the relevant arrays. The folders containing the arrays used to generate the figures are found in \data\Fig3_dataset and \data\Fig5_dataset.

## Figure 4, 6 plots

These figures use a many solution instances with varying parameter values for each plot, and are done in the notebook file ```fig_script_multiple.ipynb```. Edit the 'dataset_name' to the folder containing the relevant arrays. The folders containing the arrays used to generate the figures are found in \data\Fig4_dataset and \data\Fig6_dataset.

## UPDATES:
Some updates:
 - Numerics: Fully utilizing the MATLAB DDESD script on our N by N^2 dimensional system, with N = 30-40, tf = 30-60, which is a known 4th order method with strong refinement on a test case.
 - Updated linearization on the Kuramoto network model, using classic DDE analysis. There is an SDE component however, and the IVP must be addressed (not necessarily solved).
 - Scripts containing the following:
   - asy_script: Individual asymptotic analysis of numerical simulation. That is, determining whether the phases converge asymptotically to a common linear function with phase offsets. Furthermore, outputs a line of best fit for asymptotic phases offsets and the numerical global frequency.
   - eigen_script: A heatmap of approximate eigenvalue locations using the determinant of a large eigenvalue matrix, based on our linearized system.
   - predict_script: Uses the line of best fits and Gaussian approximation of phase differences to find the theoretical global frequency under the fixed-point equation. Plots a graph under increasing gain.
