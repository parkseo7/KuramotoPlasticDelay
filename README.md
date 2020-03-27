# KuramotoDelay
Edited version of the Kuramoto Delay Network model simulations

# Numerical Methods for Kuramoto Delay Oscillators

The code to generate all plots in the manuscript 'Synchronization and resilience in the Kuramoto white matter network model with adaptive state-dependent delays'. All numerical arrays are generation using Matlab scripts and functions, primarily applying the `ddesd` function (outputs .mat files). All numerical processing is done using Python's `numpy` and `scipy` libraries. All graphing is done using `matplotlib`. All output scripts are done using Jupyter Notebook (.pynb files). The `tqdm` library is optional as a progress bar.

# Figures

The Matlab scripts are able to generate and export numerical arrays as .mat files. Each Jupyter Notebook file contains the exact script that was used to generate each plot in the manuscript. Cells in the scripts are able to generate and export the array files used, which are stored in the 'data' folder. Other cells in the scripts import and process the array files to plot the relevant graphs. We provide a quick guide in how to reproduce each figure as shown in the manuscript:

## Figure 2

The script for generating Figure 2 in the manuscript can be found in the notebook file `fig2D_analysis.ipynb`. The script does not require a .mat file.

## Figure 3

The script for generating Figure 3 in the manuscript can be found in the notebook file `fig3_numerics.ipynb`. The script requires multiple .mat files generated using the Matlab scripts `script2D.m` or `script2D_iter.m` (if one wants to compute several trials at once).

## Figure 4

The script for generating Figure 4 in the manuscript can be found in the notebook file `figND_analysis.ipynb`. The script does not require a .mat file.

## Figure 5

The script for generating Figure 5 in the manuscript can be found in the notebook file `figND_numerics.ipynb`. The script requires multiple .mat files generated using the Matlab scripts `scriptND.m` or `scriptND_iter.m` (if one wants to compute several trials at once).

## Figure 6

The script for generating Figure 6 in the manuscript can be found in the notebook file `figINJ_single.ipynb`. The script requires a two .mat files ('sol_p.mat' for plasticity, and 'sol_np.mat' for non-plasticity), generated using the Matlab script `script_inj.m`. To toggle plasticity, set the `gain` parameter to either 0 or any positive value.

## Figure 7

The script for generating Figure 6 in the manuscript can be found in the notebook file `figINJ_multiple.ipynb`. The script requires a two groups of .mat files, generated using the Matlab script `script_inj.m` or `script_inj_iter.m` (not ideal for large dimensions due to computation time). To toggle plasticity, set the `gain` parameter to either 0 or any positive value.

# UPDATES:

**The following major updates were made from the first submission:**
 - Numerics: All generation of numerical solutions are now done through Matlab's `ddesd` function.
 - Updated linearization on the Kuramoto network model. History functions are modified using `IVPhistory.m`.
 - All scripts are displayed using Jupyter notebook, importing python libraries containing various functions for the task.
