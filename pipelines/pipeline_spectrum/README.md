# Spectral Analysis Pipeline (pipeline_spectrum)

This pipeline computes and visualizes the eigenvalues and eigenfunctions of the non-local Neumann dispersal operator using the Galerkin method. It adheres to strict modularity and numerical stability standards.

## Architecture

The system consists of independent modules:
*   manager.py: The main CLI entry point.
*   kernel.py: Defines dispersal kernels (Gaussian, Laplace, Quartic).
*   galerkin.py: Constructs the Galerkin matrix with robust Mass Matrix Preconditioning (Cholesky, Lowdin).
*   discretization.py: Numerical integration quadratures (Trapezoidal, Simpson, Clenshaw-Curtis).
*   plotting.py: Visualization utilities for spectra and eigenfunctions.
*   validator.py: A comprehensive benchmarking tool to evaluate the RMSE and Condition Number across various configurations.

## Usage Examples

Run all commands from within the pipeline_spectrum directory using the root virtual environment.

### 1. Sweep Experiment (Eigenvalues vs Domain Size)

Generates a continuous spectrum band diagram showing eigenvalues beta as a function of the domain size L.

> ..\..\.venv\Scripts\python.exe manager.py sweep -b legendre -q clenshaw-curtis -p cholesky -K gaussian --L_min 0.2 --L_max 10.0 --num_L 20 -n 200 -k 2000 --output ./output/

### 2. Single Experiment (Eigenfunction Gallery)

Computes the spectrum for a single domain size L=5.0 and generates a gallery of the top 6 corresponding eigenfunctions.

> ..\..\.venv\Scripts\python.exe manager.py single -b legendre -q clenshaw-curtis -p cholesky -K laplace -n 150 -k 1500 --output ./output/

### 3. Run the Validator

Generates the convergence report, convergence plots, and error boxplots.

> ..\..\.venv\Scripts\python.exe validator.py
