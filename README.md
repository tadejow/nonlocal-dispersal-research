# Non-Local Dispersal Research

This repository contains the codebase and mathematical manuscripts for the research project on non-local dispersal operators and their spectral properties.

## Repository Structure

The project is divided into two primary domains in accordance with standard software and academic architectures:

*   **pipelines/**: Contains Python-based numerical simulation pipelines used to empirically investigate the properties of non-local dispersal operators and reaction-diffusion models.
*   **papers/**: Contains the LaTeX manuscripts, mathematical proofs, and associated figures for publication.

## Associated Publications

Our numerical simulations support the following research publications located in the `papers/` directory:

1. **Spectrum of Symmetrical Non-Local Operators** (`papers/spectrum/symmetrical/`)
   *   **Focus**: Proofs of the existence of the principal eigenvalue, spectral gap, and continuous spectrum boundaries for integral operators.
   *   **Simulations**: Supported by `pipelines/pipeline_spectrum/`, which empirically validates the Galerkin method, Legendre bounds, and eigenfunction convergence.
2. **Klausmeier Reaction-Diffusion Model** (`papers/klausmeier/`)
   *   **Focus**: Vegetation pattern formation (Klausmeier model) under non-local effects. Includes asymptotic analysis and critical threshold approximations.
   *   **Simulations**: Supported by `pipelines/pipeline_klausmeier/`, demonstrating kernel impacts (Gaussian, Laplace, Quartic) on pattern emergence and critical patch size.
3. **Single Population Dynamics** (`papers/single_population/`)
   *   **Focus**: Foundational equations and analysis for single-species non-local dispersal dynamics.

## Quick Start (Pipelines)

To run the numerical simulations, a Python virtual environment with `numpy`, `scipy`, and `matplotlib` is required. The environment is already configured in `.venv/`.

```powershell
# Activate the virtual environment
.venv\Scripts\Activate.ps1
```

For specific execution examples, please refer to the `README.md` files located in each respective subdirectory.
