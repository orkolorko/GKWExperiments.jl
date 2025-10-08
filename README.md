# GKWExperiments

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://orkolorko.github.io/GKWExperiments.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://orkolorko.github.io/GKWExperiments.jl/dev/)
[![Build Status](https://github.com/orkolorko/GKWExperiments.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/orkolorko/GKWExperiments.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/orkolorko/GKWExperiments.jl/graph/badge.svg?token=wZBJIXHZH1)](https://codecov.io/gh/orkolorko/GKWExperiments.jl)

`GKWExperiments.jl` collects numerical helpers for studying the Gauss–Kuzmin–
Wirsing (GKW) transfer operator using [ArbNumerics.jl](https://github.com/JeffreySarnoff/ArbNumerics.jl).
It provides routines for evaluating Hurwitz and Dirichlet zeta functions as
well as utilities that build Galerkin discretisations of the transfer
operator.

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

