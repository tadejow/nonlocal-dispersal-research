# Numerical Pipelines

This directory hosts isolated, modular pipelines for conducting numerical experiments on non-local operators. Each pipeline is designed to be self-contained and operates via a standardized Command Line Interface (CLI).

## Available Pipelines

*   **pipeline_spectrum/**: Focuses on the spectral analysis of the non-local Neumann operator [u](x) = \int J(x-y)(u(y)-u(x))dy$. Features Galerkin approximations, various basis functions, and numerical preconditioning.
*   **pipeline_klausmeier/**: Implements the extended Klausmeier model for vegetation patterns using non-local dispersal kernels.

Please navigate to individual pipeline directories to read detailed instructions on usage and available CLI commands.
