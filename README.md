# KuramotoDelay
Edited version of the Kuramoto Delay Network model simulations

Some updates:
 - Numerics: Fully utilizing the MATLAB DDESD script on our N by N^2 dimensional system, with N = 30-40, tf = 30-60, which is a known 4th order method with strong refinement on a test case.
 - Updated linearization on the Kuramoto network model, using classic DDE analysis. There is an SDE component however, and the IVP must be addressed (not necessarily solved).
 - Scripts containing the following:
  - asy_script: Individual asymptotic analysis of numerical simulation. That is, determining whether the phases converge asymptotically to a common linear function with phase offsets. Furthermore, outputs a line of best fit for asymptotic phases offsets and the numerical global frequency.
  - eigen_script: A heatmap of approximate eigenvalue locations using the determinant of a large eigenvalue matrix, based on our linearized system.
  - predict_script: Uses the line of best fits and Gaussian approximation of phase differences to find the theoretical global frequency under the fixed-point equation. Plots a graph under increasing gain.
