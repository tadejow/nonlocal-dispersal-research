# Non-local Dispersal Spectrum Pipeline (`pipeline_spectrum`)

This module provides a robust, ISO-compliant numerical pipeline for computing the discrete spectrum (eigenvalues and eigenfunctions) of non-local dispersal operators on bounded domains $[-L, L]$.

The operator solved here is:
$$ \mathcal{L}u = \int_{-L}^L J(x-y)u(y)dy - b(x)u(x) $$
where $b(x) = \int_{-L}^L J(x-y)dy$.

## Architecture

The codebase is split into modular components adhering to clean software design principles:

- `manager.py` - The CLI orchestrator providing a clean interface to configure and run experiments.
- `kernel.py` - Defines continuous dispersal kernels (e.g., standard `GaussianKernel`).
- `discretization.py` - Implements numerical quadratures (Trapezoidal, Simpson's, and Clenshaw-Curtis) for 1D and 2D integrals.
- `galerkin.py` - Assembles the operator matrix using Galerkin projection with different bases (Laplacian eigenfunctions, Legendre polynomials, Canonical grid basis).
- `plotting.py` - Handles generation of publication-ready visualizations ($\beta \times L$ sweeps and eigenfunction galleries).

## Requirements

Ensure your virtual environment has the following packages installed:
```bash
pip install numpy scipy matplotlib
```

## Usage

The pipeline is driven via the `manager.py` CLI script, which supports two main modes: `single` and `sweep`.

### 1. Single Domain Experiment (`single`)
Computes the spectrum for a fixed domain size $L$ and generates a 3x2 gallery of the first 6 principal eigenfunctions.

**Example:**
```bash
python manager.py single -b legendre -q simpson -L 5.0 -n 40
```
This runs the solver on $[-5, 5]$ using 40 Legendre polynomials and Simpson's quadrature rule. 

### 2. Domain Size Sweep (`sweep`)
Computes the principal eigenvalues across a range of domain sizes $L$ and generates a $\beta$ vs $L$ convergence plot.

**Example:**
```bash
python manager.py sweep -b laplacian -q clenshaw-curtis --L_min 0.5 --L_max 8.0 --num_L 20
```
This sweeps $L \in [0.5, 8.0]$ in 20 steps, plotting the convergence of the top 6 eigenvalues.

## CLI Arguments Reference

Use `python manager.py --help` for full details. Common flags include:
- `mode` : Required. Either `single` or `sweep`.
- `-b, --basis` : Galerkin basis. Options: `laplacian` (default), `legendre`, `canonical`.
- `-q, --quadrature` : Integration rule. Options: `trapezoidal`, `simpson` (default), `clenshaw-curtis`.
- `-n, --N_basis` : Number of basis functions to use in projection (default: 30).
- `-k, --N_quad` : Number of quadrature nodes (grid resolution) used for integrals (default: 500).
- `-L` : Domain half-width for single runs (default: 5.0).
- `-o, --output` : Output directory for plots (default: `output/`).
