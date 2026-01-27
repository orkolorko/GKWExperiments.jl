# GKWExperiments

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://orkolorko.github.io/GKWExperiments.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://orkolorko.github.io/GKWExperiments.jl/dev/)
[![Build Status](https://github.com/orkolorko/GKWExperiments.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/orkolorko/GKWExperiments.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/orkolorko/GKWExperiments.jl/graph/badge.svg?token=wZBJIXHZH1)](https://codecov.io/gh/orkolorko/GKWExperiments.jl)

`GKWExperiments.jl` collects numerical helpers for studying the Gauss–Kuzmin–
Wirsing (GKW) transfer operator using [ArbNumerics.jl](https://github.com/JeffreySarnoff/ArbNumerics.jl)
and [BallArithmetic.jl](https://github.com/orkolorko/BallArithmetic.jl).
It provides:

- Routines for evaluating Hurwitz and Dirichlet zeta functions
- Galerkin discretisations of the transfer operator
- **Rigorous eigenvalue certification** via resolvent bounds and spectral projectors
- **Spectral expansion** of L^n · 1 with the Gauss problem (convergence to invariant measure)

## Certification Scripts

The `scripts/` directory contains ready-to-run certification examples:

### Basic Certification (`scripts/certify_eigenvalues.jl`)

Certifies eigenvalues and plots L^k · 1 iterates:

```bash
julia --project scripts/certify_eigenvalues.jl
```

Generates `scripts/Lk_iterates.png` showing convergence to the invariant density.

### High-Precision Spectral Expansion (`scripts/high_precision_spectral.jl`)

Computes rigorous spectral expansion L^n · 1 = Σᵢ λᵢⁿ Πᵢ(1) with:
- Certified eigenvalues (λ₁ ≈ 1, λ₂ ≈ -0.3037, ...)
- Spectral projectors for non-normal operators
- Tail error bounds by compactness
- The Gauss problem: ∫₀ˣ Lⁿ·1 dt → log₂(1+x)

```bash
# Serial mode
julia --project scripts/high_precision_spectral.jl

# Distributed mode (4 workers)
julia -p 4 --project scripts/high_precision_spectral.jl
```

The distributed mode uses BallArithmetic's `DistributedExt` extension for parallel
resolvent certification.

## Documentation

The package documentation is built with [Documenter.jl](https://documenter.juliadocs.org/).
To preview it locally run:

```julia
import Pkg
Pkg.activate("docs")
Pkg.instantiate()
include("docs/make.jl")
```

Documenter writes the generated HTML to `docs/build`.

