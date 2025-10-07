```@meta
CurrentModule = GKWExperiments
```

# GKWExperiments.jl

`GKWExperiments.jl` provides ArbNumerics-based tools for experimenting with the
transfer operator that appears in the study of the Gauss–Kuzmin–Wirsing
constant.  The package exposes high-level helpers for precomputing Hurwitz zeta
tables and for assembling Galerkin discretisations of the operator.

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

```@index
```

```@autodocs
Modules = [GKWExperiments, GKWExperiments.GKWDiscretization, GKWExperiments.ArbZeta]
```
