```@meta
CurrentModule = GKWExperiments
```

# GKWExperiments.jl

`GKWExperiments.jl` provides ArbNumerics-based tools for experimenting with the
transfer operator that appears in the study of the Gauss–Kuzmin–Wirsing
constant.  The package exposes high-level helpers for precomputing Hurwitz zeta
tables and for assembling Galerkin discretisations of the operator.

Key features:
- Hurwitz and Dirichlet zeta function evaluation
- Galerkin discretisations of the GKW transfer operator
- **Rigorous eigenvalue certification** via resolvent bounds
- **Spectral projectors** for non-normal operators
- **Spectral expansion** L^n · 1 = Σᵢ λᵢⁿ Πᵢ(1)
- **The Gauss problem**: convergence of ∫₀ˣ Lⁿ·1 dt to log₂(1+x)

## Getting started

```julia
julia> using GKWExperiments, ArbNumerics

julia> ArbNumerics.setprecision(128)
128

julia> s = ArbComplex(1//2);

julia> Z, ws = zeta_shift_table_on_circle(s; K=4, N=64);

julia> size(Z), length(ws)
((5, 64), 64)
```

All exported functions are documented in the API reference below.  Docstrings
are available at the REPL through Julia's help mode as usual, e.g. `?mid`.

## Further reading

- [Discretization](discretization.md): how the matrix $A_K$ is assembled and how $\varepsilon_K$ is bounded.
- [Certification Pipeline](certification_pipeline.md): detailed walkthrough of the two-script resolvent/NK pipeline.

## Certification Scripts

The `scripts/` directory contains ready-to-run examples for eigenvalue certification
and spectral analysis.

### Basic Certification

`scripts/certify_eigenvalues.jl` certifies eigenvalues and generates plots:

```bash
julia --project scripts/certify_eigenvalues.jl
```

This produces:
- Certification log with eigenvalue bounds
- Projection of the constant function 1 onto each eigenspace
- Plot of L^k · 1 iterates converging to the invariant density

### High-Precision Spectral Expansion

`scripts/high_precision_spectral.jl` computes the rigorous spectral expansion:

```math
L^n \cdot 1 = \sum_{i=1}^{m} \lambda_i^n \Pi_i(1) + O(|\lambda_{m+1}|^n)
```

where Πᵢ is the spectral projector onto the i-th eigenspace.

```bash
# Serial mode
julia --project scripts/high_precision_spectral.jl

# Distributed mode (parallel resolvent certification)
julia -p 4 --project scripts/high_precision_spectral.jl
```

Key outputs:
- Certified eigenvalues: λ₁ ≈ 1, λ₂ ≈ -0.3037 (Wirsing constant), ...
- Spectral projector coefficients Πᵢ(1)
- Tail error bounds using compactness
- The Gauss problem: G_n(x) = ∫₀ˣ Lⁿ·1 dt → log₂(1+x)

### Distributed Computing

BallArithmetic.jl provides a `DistributedExt` extension for parallel resolvent
certification. When `Distributed` is loaded, `run_certification` accepts workers:

```julia
using Distributed
addprocs(4)
@everywhere using BallArithmetic, BallArithmetic.CertifScripts

# Use distributed certification
cert_data = run_certification(A, circle, workers())
```

## API Reference

```@index
```

```@autodocs
Modules = [
    GKWExperiments,
    GKWExperiments.GKWDiscretization,
    GKWExperiments.ArbZeta,
    GKWExperiments.Constants,
    GKWExperiments.Polynomials,
    GKWExperiments.EigenspaceCertification,
    GKWExperiments.InfiniteDimensionalLift,
]
```
