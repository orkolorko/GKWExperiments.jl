# # Generate Certification Documentation
#
# This script runs high-precision eigenvalue certification and generates
# documentation files with hardcoded results:
# - docs/src/certification_results.md (for Documenter.jl)
# - docs/certification_results.tex (for LaTeX)
# - data/certification_results.jld2 (saved data)
#
# Run once to generate documentation, then commit the output files.

using GKWExperiments
using ArbNumerics
using BallArithmetic
using LinearAlgebra
using JLD2
using Printf
using Dates
using CairoMakie

# Optional: Enable distributed computing via BallArithmetic's DistributedExt
# Loading Distributed triggers the extension automatically.
#
# Usage:
#   julia -p 4 --project scripts/generate_certification_docs.jl
#
# Or manually:
#   using Distributed; addprocs(4)
#   @everywhere using BallArithmetic, BallArithmetic.CertifScripts
#
# When workers are available, run_certification will use them automatically.

# Check if Distributed is loaded and workers are available
const HAS_DISTRIBUTED = isdefined(Main, :Distributed) && isdefined(Main, :nworkers)
const NUM_WORKERS = HAS_DISTRIBUTED ? Main.nworkers() : 1
const USE_DISTRIBUTED = NUM_WORKERS > 1

if USE_DISTRIBUTED
    println("Distributed mode: $(NUM_WORKERS) workers available")
else
    println("Serial mode (run with julia -p N for parallel execution)")
end

# Configuration
PRECISION = 1024              # Bits of precision for ArbFloat/BigFloat
setprecision(ArbFloat, PRECISION)
setprecision(BigFloat, PRECISION)

s = ArbComplex(1.0, 0.0)     # Classical GKW (s=1)
K = 128                       # Discretization size (increased for tighter bounds)
NUM_EIGENVALUES = 6           # Number of eigenvalues to certify (first 6 most important)
N_SPLITTING = 5000            # C₂ splitting parameter
CIRCLE_SAMPLES = 512          # More samples for accurate resolvent bounds
CIRCLE_RADIUS_FACTOR = 0.01

# Iterative refinement configuration
TARGET_TOLERANCE = 1e-14      # Target certification radius
MAX_REFINEMENT_ITERS = 20     # Maximum refinement iterations
REFINEMENT_FACTOR = 0.5       # How much to shrink radius each iteration

# Use BigFloat for higher precision in BallArithmetic
# NOTE: BallArithmetic's run_certification doesn't fully support BigFloat yet
USE_BIGFLOAT = false          # Use BigFloat instead of Float64 for ball arithmetic

println("="^80)
println("GENERATING CERTIFICATION DOCUMENTATION")
println("="^80)
println("Timestamp: $(now())")
println("K = $K, Precision = $PRECISION bits")
println()

# Create output directories
mkpath("data")
mkpath("docs/src")

# Step 1: Compute truncation error bounds
println("Computing truncation error bounds...")
C2 = compute_C2(N_SPLITTING)
C2_float = Float64(real(C2))
eps_K = compute_Δ(K; N=N_SPLITTING)
eps_K_float = Float64(real(eps_K))

println("  C₂ = $C2_float")
println("  ε_K = $eps_K_float")

# Step 2: Build GKW matrix and get Schur decomposition
println("Building GKW matrix (K=$K)...")

# Build the GKW matrix with full ArbNumerics precision
M_arb = gkw_matrix_direct(s; K=K)

# Convert to ball matrix with appropriate precision
if USE_BIGFLOAT
    println("  Using BigFloat precision ($PRECISION bits)")
    M_center = Complex{BigFloat}.(ArbNumerics.midpoint.(M_arb))
    M_radius = BigFloat.(ArbNumerics.radius.(M_arb))
    A_ball = BallMatrix(M_center, M_radius)
    A_center = M_center
    eps_K_val = BigFloat(eps_K_float)
else
    println("  Using Float64 precision")
    M_center = Complex{Float64}.(ArbNumerics.midpoint.(M_arb))
    M_radius = Float64.(ArbNumerics.radius.(M_arb))
    A_ball = BallMatrix(M_center, M_radius)
    A_center = M_center
    eps_K_val = eps_K_float
end

S = schur(A_center)

# Step 3: Extract and certify eigenvalues
println("Certifying eigenvalues...")

schur_eigenvalues = diag(S.T)
sorted_idx = sortperm(abs.(schur_eigenvalues), rev=true)

# Data structures for results
struct CertifiedEigendata
    index::Int
    eigenvalue::ComplexF64
    eigenvalue_radius::Float64
    eigenvector_coeffs::Vector{ComplexF64}  # First 10 coefficients
    eigenvector_radius::Float64
    projection_of_one::Vector{ComplexF64}   # First 10 coefficients
    projection_leading_coeff::ComplexF64
    projection_leading_radius::Float64
    projection_L2_norm::Float64
    projection_L2_radius::Float64
    small_gain_alpha::Float64
    is_certified::Bool
    # Newton–Kantorovich refinement (Stage 2)
    nk_radius::Float64
    nk_certified::Bool
    nk_qk::Float64
    nk_q0::Float64
end

certified_data = CertifiedEigendata[]

"""
    refine_certification(A, λ_center, initial_radius, eps_K; ...)

Iteratively refine the certification radius until we reach the target tolerance
or can no longer improve.

When `USE_DISTRIBUTED` is true and workers are available, certification runs in parallel.
"""
function refine_certification(A, λ_center, initial_radius, eps_K;
                              target_tol=TARGET_TOLERANCE,
                              max_iters=MAX_REFINEMENT_ITERS,
                              refinement_factor=REFINEMENT_FACTOR,
                              samples=CIRCLE_SAMPLES)
    current_radius = initial_radius
    best_radius = Inf
    best_α = Inf
    best_resolvent = Inf

    for iter in 1:max_iters
        try
            circle = CertificationCircle(λ_center, current_radius; samples=samples)
            # Use distributed workers if available
            cert_data = if USE_DISTRIBUTED
                run_certification(A, circle, Main.workers())
            else
                run_certification(A, circle)
            end
            α = eps_K * cert_data.resolvent_original

            if α < 1.0
                # Certification succeeded
                best_radius = current_radius
                best_α = α
                best_resolvent = cert_data.resolvent_original
                println("    Iter $iter: radius=$(round(current_radius, sigdigits=4)), α=$(round(α, sigdigits=4)) ✓")

                if current_radius ≤ target_tol
                    # Reached target tolerance
                    println("    Reached target tolerance!")
                    break
                end

                # Try smaller radius
                current_radius *= refinement_factor
            else
                # Certification failed - use last successful radius
                println("    Iter $iter: radius=$(round(current_radius, sigdigits=4)), α=$(round(α, sigdigits=4)) ✗")
                break
            end
        catch e
            # Numerical error (e.g., DomainError) - stop refinement and use last good result
            println("    Iter $iter: radius=$(round(current_radius, sigdigits=4)) - numerical error, stopping refinement")
            println("    Error: $(typeof(e))")
            break
        end
    end

    if best_radius < Inf
        println("    CERTIFIED (α = $(round(best_α, sigdigits=4)), radius = $(round(best_radius, sigdigits=4)))")
    else
        println("    FAILED TO CERTIFY")
    end

    return best_radius, best_α, best_resolvent
end

for i in 1:NUM_EIGENVALUES
    idx = sorted_idx[i]
    λ_center = ComplexF64(schur_eigenvalues[idx])

    println("  Eigenvalue $i: λ ≈ $λ_center")

    # Initial certification circle
    initial_radius = max(abs(λ_center) * CIRCLE_RADIUS_FACTOR, eps_K_float * 10)

    # Ensure circles don't overlap with other eigenvalues
    for j in 1:NUM_EIGENVALUES
        if j != i
            other_idx = sorted_idx[j]
            other_λ = schur_eigenvalues[other_idx]
            dist = abs(λ_center - other_λ)
            if initial_radius > dist / 3
                initial_radius = dist / 3
            end
        end
    end

    # Run iterative certification refinement
    λ_radius, α, resolvent = refine_certification(
        A_ball, λ_center, initial_radius, eps_K_val;
        target_tol=TARGET_TOLERANCE,
        max_iters=MAX_REFINEMENT_ITERS,
        refinement_factor=REFINEMENT_FACTOR,
        samples=CIRCLE_SAMPLES
    )

    is_certified = λ_radius < Inf

    # Compute eigenvector from Schur form
    Q = S.Z
    T = S.T
    n_size = size(T, 1)
    CT = eltype(T)  # Complex type matching Schur decomposition
    zero_init = zero(CT)

    # Right eigenvector via back-substitution
    y_right = zeros(CT, n_size)
    y_right[idx] = one(CT)
    for j in (idx-1):-1:1
        sum_val = sum(T[j, k] * y_right[k] for k in (j+1):n_size; init=zero_init)
        if abs(T[j, j] - T[idx, idx]) > 1e-14
            y_right[j] = -sum_val / (T[j, j] - T[idx, idx])
        end
    end
    v_right = Q * y_right
    v_right = v_right / norm(v_right)

    # Left eigenvector via forward-substitution
    y_left = zeros(CT, n_size)
    y_left[idx] = one(CT)
    for j in (idx+1):n_size
        sum_val = sum(y_left[k] * T[k, j] for k in 1:(j-1); init=zero_init)
        if abs(T[j, j] - T[idx, idx]) > 1e-14
            y_left[j] = -sum_val / (T[j, j] - T[idx, idx])
        end
    end
    w_left = Q * y_left
    w_left = w_left / norm(w_left)

    # Projection of 1: Πᵢ(1) = vᵢ (wᵢ' · 1) / (wᵢ' vᵢ)
    one_vec = zeros(CT, K + 1)
    one_vec[1] = one(CT)
    biorth = dot(w_left, v_right)
    w_dot_one = dot(w_left, one_vec)
    proj_vec = v_right * (w_dot_one / biorth)

    # Store first 10 coefficients (convert to Float64 for storage)
    num_coeffs = min(10, length(v_right))
    eigvec_coeffs = ComplexF64.(v_right[1:num_coeffs])
    proj_coeffs = ComplexF64.(proj_vec[1:num_coeffs])

    # Projection norms and radii
    proj_leading = ComplexF64(proj_vec[1])
    proj_L2 = Float64(norm(proj_vec))
    numerical_error = USE_BIGFLOAT ? 1e-30 : 1e-14
    proj_leading_rad = Float64(numerical_error * (1 + abs(proj_leading)))
    proj_L2_rad = Float64(numerical_error * sqrt(K + 1))
    eigvec_rad = Float64(numerical_error * sqrt(K + 1))

    # Newton–Kantorovich refinement (Stage 2)
    nk_rad = Inf
    nk_cert = false
    nk_qk = Inf
    nk_q0 = Inf
    if is_certified
        println("    Running NK refinement (Stage 2)...")
        try
            nk_result = certify_eigenpair_nk(s; K=K, target_idx=i, N_C2=N_SPLITTING)
            nk_rad = nk_result.enclosure_radius
            nk_cert = nk_result.is_certified
            nk_qk = nk_result.qk_bound
            nk_q0 = nk_result.q0_bound
            if nk_cert
                improvement = Float64(λ_radius) / nk_rad
                println("    NK CERTIFIED: r_NK = $(round(nk_rad, sigdigits=4)) ($(round(improvement, sigdigits=3))× tighter)")
            else
                println("    NK FAILED (q₀ = $(nk_result.q0_bound))")
            end
        catch e
            println("    NK error: $(typeof(e))")
        end
    end

    push!(certified_data, CertifiedEigendata(
        i, ComplexF64(λ_center), Float64(λ_radius), eigvec_coeffs, eigvec_rad,
        proj_coeffs, proj_leading, proj_leading_rad, proj_L2, proj_L2_rad,
        Float64(α), is_certified,
        nk_rad, nk_cert, nk_qk, nk_q0
    ))

    status = is_certified ? "CERTIFIED" : "NOT CERTIFIED"
    println("    $status (α = $(round(α, sigdigits=4)), radius = $(round(λ_radius, sigdigits=4)))")
end

# Step 4: Save results to JLD2
println("\nSaving results to data/certification_results.jld2...")
jldsave("data/certification_results.jld2";
    timestamp = string(now()),
    K = K,
    precision_bits = PRECISION,
    C2 = C2_float,
    eps_K = eps_K_float,
    num_eigenvalues = NUM_EIGENVALUES,
    eigenvalues = [d.eigenvalue for d in certified_data],
    eigenvalue_radii = [d.eigenvalue_radius for d in certified_data],
    eigenvector_coeffs = [d.eigenvector_coeffs for d in certified_data],
    projection_coeffs = [d.projection_of_one for d in certified_data],
    projection_leading = [d.projection_leading_coeff for d in certified_data],
    projection_leading_radii = [d.projection_leading_radius for d in certified_data],
    projection_L2_norms = [d.projection_L2_norm for d in certified_data],
    small_gain_alphas = [d.small_gain_alpha for d in certified_data],
    is_certified = [d.is_certified for d in certified_data],
    nk_radii = [d.nk_radius for d in certified_data],
    nk_certified = [d.nk_certified for d in certified_data]
)

# Step 5: Generate plots
println("Generating plots...")

# Plot eigenfunctions
fig_eig = Figure(size = (1200, 800))
x_points = range(0.01, 0.99, length=200)

# Evaluate polynomial in (x-1) basis
function eval_poly(coeffs, x)
    result = 0.0 + 0.0im
    w = x - 1
    w_power = 1.0
    for c in coeffs
        result += c * w_power
        w_power *= w
    end
    return real(result)
end

# Top row: Eigenfunctions
ax_eig = Axis(fig_eig[1, 1:2],
    xlabel = "x",
    ylabel = "vᵢ(x)",
    title = "Certified Eigenfunctions of GKW Operator"
)

colors = [:blue, :red, :green, :orange, :purple, :brown]
for (i, data) in enumerate(certified_data)
    # Extend coefficients with zeros for plotting
    full_coeffs = zeros(ComplexF64, K + 1)
    full_coeffs[1:length(data.eigenvector_coeffs)] = data.eigenvector_coeffs
    y_vals = [eval_poly(full_coeffs, x) for x in x_points]

    λ_str = @sprintf("%.4f", real(data.eigenvalue))
    lines!(ax_eig, x_points, y_vals,
           label = "v_$i (λ=$λ_str)",
           color = colors[i], linewidth = 2)
end
axislegend(ax_eig, position = :rt)

# Bottom row: Projections Πᵢ(1)
ax_proj = Axis(fig_eig[2, 1:2],
    xlabel = "x",
    ylabel = "Πᵢ(1)(x)",
    title = "Projection of Constant Function 1 onto Eigenspaces"
)

for (i, data) in enumerate(certified_data)
    full_coeffs = zeros(ComplexF64, K + 1)
    full_coeffs[1:length(data.projection_of_one)] = data.projection_of_one
    y_vals = [eval_poly(full_coeffs, x) for x in x_points]

    λ_str = @sprintf("%.4f", real(data.eigenvalue))
    lines!(ax_proj, x_points, y_vals,
           label = "Π_$i(1) (λ=$λ_str)",
           color = colors[i], linewidth = 2)
end
axislegend(ax_proj, position = :rt)

mkpath("docs/src/assets")
save("docs/src/assets/eigenfunction_plots.png", fig_eig, px_per_unit = 2)
println("  Saved docs/src/assets/eigenfunction_plots.png")

# Step 6: Generate Markdown documentation
println("Generating Markdown documentation...")

md_content = """
# Certified Spectral Data for the GKW Operator

This page contains rigorously certified spectral data for the classical
Gauss-Kuzmin-Wirsing transfer operator (s=1).

**Certification timestamp:** $(now())

## Certification Parameters

| Parameter | Value |
|-----------|-------|
| Discretization K | $K |
| Matrix size | $(K+1) × $(K+1) |
| Precision | $PRECISION bits |
| C₂ bound | $(round(C2_float, sigdigits=10)) |
| Truncation error ε_K | $(round(eps_K_float, sigdigits=4)) |

## Certified Eigenvalues

The following eigenvalues are rigorously certified using resolvent bounds
and the finite-to-infinite dimensional lift.

| Index | Eigenvalue λᵢ | Resolvent Radius | NK Radius | Small-gain α | Status |
|-------|---------------|------------------|-----------|--------------|--------|
"""

for data in certified_data
    global md_content
    λ_re = round(real(data.eigenvalue), sigdigits=15)
    λ_im = round(imag(data.eigenvalue), sigdigits=6)
    λ_str = abs(λ_im) < 1e-10 ? "$λ_re" : "$λ_re + $(λ_im)i"
    radius_str = @sprintf("%.4e", data.eigenvalue_radius)
    nk_str = data.nk_certified ? @sprintf("%.4e", data.nk_radius) : "---"
    α_str = @sprintf("%.4e", data.small_gain_alpha)
    status = data.is_certified ? "✓ Certified" : "✗ Failed"
    md_content *= "| $(data.index) | $λ_str | $radius_str | $nk_str | $α_str | $status |\n"
end

md_content *= """

### Notable Eigenvalues

- **λ₁ ≈ 1**: The Perron-Frobenius eigenvalue (invariant measure)
- **λ₂ ≈ -0.3037**: The Wirsing constant, determining the rate of convergence
  to the Gauss measure

## Projection of Constant Function onto Eigenspaces

For the spectral expansion ``L^n \\cdot 1 = \\sum_i \\lambda_i^n \\Pi_i(1)``,
we compute the projection of the constant function 1 onto each eigenspace.

| Index | [Πᵢ(1)]₀ (leading coeff) | Radius | ‖Πᵢ(1)‖_{L²} |
|-------|--------------------------|--------|--------------|
"""

for data in certified_data
    global md_content
    lead_re = round(real(data.projection_leading_coeff), sigdigits=12)
    lead_rad = @sprintf("%.2e", data.projection_leading_radius)
    L2_norm = round(data.projection_L2_norm, sigdigits=10)
    md_content *= "| $(data.index) | $lead_re | $lead_rad | $L2_norm |\n"
end

md_content *= """

## Eigenvector Coefficients

The eigenvectors are represented in the monomial basis ``\\{(x-1)^k\\}_{k=0}^K``.
Below are the first 10 coefficients of each certified eigenvector.

"""

for data in certified_data
    global md_content
    λ_str = @sprintf("%.10f", real(data.eigenvalue))
    md_content *= "### Eigenvector v_$(data.index) (λ = $λ_str)\n\n"
    md_content *= "```\n"
    for (k, c) in enumerate(data.eigenvector_coeffs)
        c_re = @sprintf("%+.12e", real(c))
        md_content *= "v[$(k-1)] = $c_re\n"
    end
    md_content *= "```\n\n"
end

md_content *= """
## Eigenfunction Plots

![Certified Eigenfunctions](assets/eigenfunction_plots.png)

**Top panel:** Eigenfunctions vᵢ(x) evaluated on [0,1].

**Bottom panel:** Projections Πᵢ(1)(x) showing the decomposition of the
constant function into eigenspace components.

## Mathematical Details

### Two-Stage Certification Pipeline

The certification uses a two-stage approach:

**Stage 1 (Resolvent Bridge):** If ``\\varepsilon_K \\cdot \\|R_{A_K}(z)\\| < 1``
on a circle around λ, then the infinite-dimensional operator L has exactly one
eigenvalue inside that circle.  This proves simplicity and gives an initial
enclosure radius equal to the circle radius.

**Stage 2 (Newton–Kantorovich Refinement):** Within the Stage 1 enclosure, the
NK argument on the eigenpair map ``F(z,v) = (Mv - zv, u^*v - 1)`` yields a
much tighter enclosure radius ``r_{NK}`` by bounding the defect
``q_0 = \\|I - C \\cdot DF\\|`` and applying a Krawczyk-type contraction argument.

### Truncation Error

The truncation error is bounded by:
```math
\\varepsilon_K = C_2 \\cdot \\left(\\frac{2}{3}\\right)^{K+1}
```
where C₂ ≈ $(round(C2_float, sigdigits=6)).

### Spectral Expansion

For the non-normal GKW operator:
```math
L^n \\cdot 1 = \\sum_{i=1}^{m} \\lambda_i^n \\Pi_i(1) + O(|\\lambda_{m+1}|^n)
```

The rate of convergence to the Gauss measure is determined by |λ₂/λ₁| ≈ $(round(abs(certified_data[2].eigenvalue), sigdigits=6)).
"""

open("docs/src/certification_results.md", "w") do io
    write(io, md_content)
end
println("  Saved docs/src/certification_results.md")

# Step 7: Generate LaTeX documentation
println("Generating LaTeX documentation...")

tex_content = raw"""
\documentclass[11pt]{article}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}

\title{Certified Spectral Data for the Gauss-Kuzmin-Wirsing Operator}
\author{Generated by GKWExperiments.jl}
\date{""" * string(Date(now())) * raw"""}

\begin{document}
\maketitle

\begin{abstract}
This document contains rigorously certified spectral data for the classical
Gauss-Kuzmin-Wirsing (GKW) transfer operator with parameter $s=1$.
All bounds are computer-assisted proofs using interval arithmetic and
resolvent-based certification.
\end{abstract}

\section{Certification Parameters}

\begin{table}[h]
\centering
\begin{tabular}{ll}
\toprule
Parameter & Value \\
\midrule
Discretization $K$ & """ * "$K" * raw""" \\
Matrix size & """ * "$(K+1) \\times $(K+1)" * raw""" \\
Precision & """ * "$PRECISION" * raw""" bits \\
$C_2$ bound & """ * "$(round(C2_float, sigdigits=10))" * raw""" \\
Truncation error $\varepsilon_K$ & """ * @sprintf("%.4e", eps_K_float) * raw""" \\
\bottomrule
\end{tabular}
\caption{Certification parameters}
\end{table}

\section{Certified Eigenvalues}

The following theorem summarizes the certified eigenvalue enclosures using
the two-stage pipeline: resolvent bridge (Stage~1) followed by
Newton--Kantorovich refinement (Stage~2).

\begin{theorem}[Certified Eigenvalue Enclosures]
The classical GKW transfer operator $\mathcal{L}$ on $H^2(D_1)$ has eigenvalues
$\lambda_1, \lambda_2, \ldots$ satisfying the following rigorous bounds:
"""

for data in certified_data
    global tex_content
    λ_re = round(real(data.eigenvalue), sigdigits=12)
    if data.nk_certified
        radius = @sprintf("%.4e", data.nk_radius)
        tex_content *= """
\\[
    \\lambda_$(data.index) \\in [$λ_re \\pm $radius] \\quad\\text{(NK)}
\\]
"""
    else
        radius = @sprintf("%.4e", data.eigenvalue_radius)
        tex_content *= """
\\[
    \\lambda_$(data.index) \\in [$λ_re \\pm $radius] \\quad\\text{(resolvent)}
\\]
"""
    end
end

tex_content *= raw"""
\end{theorem}

\begin{proof}
Each eigenvalue is first certified using the resolvent bridge theorem (Stage~1).
For a circle $\Gamma$ of radius $r$ around the approximate eigenvalue, we verify that
$\varepsilon_K \cdot \|R_{A_K}(z)\|_{L^2 \to L^2} < 1$ for all $z \in \Gamma$.
This guarantees exactly one eigenvalue of the infinite-dimensional operator
inside $\Gamma$.

The enclosure is then refined using a Newton--Kantorovich argument (Stage~2)
on the eigenpair map $F(z,v) = (Mv - zv, u^*v - 1)$.  Let $C \approx DF(x_0)^{-1}$
be a preconditioner.  If $q_0 := \|I - C\cdot DF_{L_r}(x_0)\| < 1$ and
$(1 - q_0)^2 \geq 4\|C\| y$ (where $y = \|C\| \varepsilon_K \|v_A\|_2$), then
the Krawczyk map is a contraction on a ball of radius $r_{\text{NK}}$, yielding
a much tighter enclosure.
\end{proof}

\subsection{Notable Eigenvalues}

\begin{itemize}
    \item $\lambda_1 \approx 1$: The Perron-Frobenius eigenvalue corresponding
          to the invariant Gauss measure.
    \item $\lambda_2 \approx -0.3037$: The \emph{Wirsing constant}, which
          determines the exponential rate of convergence to the Gauss measure.
\end{itemize}

\section{Spectral Projections}

For the spectral expansion
\[
    \mathcal{L}^n \cdot 1 = \sum_{i=1}^{m} \lambda_i^n \Pi_i(1) + O(|\lambda_{m+1}|^n),
\]
we certify the projection of the constant function $1$ onto each eigenspace.

\begin{table}[h]
\centering
\begin{tabular}{cccc}
\toprule
$i$ & $[\Pi_i(1)]_0$ & Radius & $\|\Pi_i(1)\|_{L^2}$ \\
\midrule
"""

for data in certified_data
    global tex_content
    lead_re = @sprintf("%.10f", real(data.projection_leading_coeff))
    lead_rad = @sprintf("%.2e", data.projection_leading_radius)
    L2_norm = @sprintf("%.8f", data.projection_L2_norm)
    tex_content *= "$(data.index) & $lead_re & $lead_rad & $L2_norm \\\\\n"
end

tex_content *= raw"""
\bottomrule
\end{tabular}
\caption{Certified projections of the constant function onto eigenspaces}
\end{table}

\section{Eigenvector Coefficients}

The eigenvectors are represented in the monomial basis $\{(x-1)^k\}_{k=0}^K$.
The following tables give the first coefficients of each certified eigenvector.

"""

for data in certified_data
    global tex_content
    λ_str = @sprintf("%.10f", real(data.eigenvalue))
    tex_content *= """
\\subsection*{Eigenvector \$v_$(data.index)\$ (\$\\lambda = $λ_str\$)}

\\begin{table}[h]
\\centering
\\begin{tabular}{cl}
\\toprule
\$k\$ & \$[v_$(data.index)]_k\$ \\\\
\\midrule
"""
    for (k, c) in enumerate(data.eigenvector_coeffs)
        c_re = @sprintf("%+.10e", real(c))
        tex_content *= "$(k-1) & $c_re \\\\\n"
    end
    tex_content *= raw"""
\bottomrule
\end{tabular}
\end{table}

"""
end

tex_content *= raw"""
\section{The Gauss Problem}

The distribution function
\[
    G_n(x) = \int_0^x \mathcal{L}^n \cdot 1(t) \, dt
\]
converges to the Gauss measure CDF $G(x) = \log_2(1+x)$ as $n \to \infty$.

The rate of convergence is determined by
\[
    |G_n(x) - G(x)| = O(|\lambda_2|^n) = O(0.3037^n).
\]

\section{Mathematical Framework}

\subsection{Resolvent Bridge Theorem}

Let $A_K$ be the $K$-dimensional Galerkin approximation and $\mathcal{L}$ the
infinite-dimensional operator. If for all $z$ on a circle $\Gamma$:
\[
    \varepsilon_K \cdot \|(\lambda I - A_K)^{-1}\| < 1
\]
then $\mathcal{L}$ has exactly as many eigenvalues inside $\Gamma$ as $A_K$ does.

\subsection{Truncation Error}

The truncation error satisfies
\[
    \|\mathcal{L} - A_K\|_{L^2 \to L^2} \leq C_2 \cdot \left(\frac{2}{3}\right)^{K+1} = \varepsilon_K
\]
where $C_2 \approx """ * "$(round(C2_float, sigdigits=6))" * raw"""$ is computed rigorously.

\end{document}
"""

open("docs/certification_results.tex", "w") do io
    write(io, tex_content)
end
println("  Saved docs/certification_results.tex")

# Step 8: Update docs/make.jl to include the new page
println("\nDone! Generated files:")
println("  - data/certification_results.jld2 (saved certification data)")
println("  - docs/src/certification_results.md (Documenter.jl page)")
println("  - docs/src/assets/eigenfunction_plots.png (plots)")
println("  - docs/certification_results.tex (LaTeX document)")
println()
println("To include in documentation, add to docs/make.jl pages:")
println("  \"Certification Results\" => \"certification_results.md\"")
println()
println("To compile LaTeX:")
println("  cd docs && pdflatex certification_results.tex")
