# # Certified Spectral Data for GKW Operator — LaTeX Generation Script
#
# This script produces `data/certified_spectral_data.tex` for inclusion in Gauss.tex.
# It computes:
#   - High-precision eigenvalue enclosures (K=128)
#   - Eigenvector coefficients (first 20 per eigenvector) with rigorous radii
#   - Projection of 1 onto each eigenspace with rigorous radii
#   - Comparison table: direct resolvent vs polynomial deflation certification
#
# ## Rigor
#
# Eigenvalue intervals and spectral projectors come from BallArithmetic's rigorous
# block Schur decomposition (Miyajima VBD).  Projections of 1 are computed via
# Ball-arithmetic matrix–vector products, so their radii are rigorous.
# Eigenvector coefficients are the columns of the rigorous Schur Q matrix.
#
# ## Memoization
#
# All intermediate results are saved to `data/checkpoint.jld2`.
# Re-running the script loads existing data and only recomputes what is missing.
# Delete the checkpoint file to force a full recomputation.

using GKWExperiments
using ArbNumerics
using BallArithmetic
using LinearAlgebra
using JLD2
using Printf
using Dates

# ============================================================================
# Configuration
# ============================================================================

const PRECISION = 512   # ArbFloat precision in bits
const K = 128            # Discretization size
const N_SPLITTING = 5000 # C₂ splitting parameter
const CIRCLE_SAMPLES = 256
const CIRCLE_RADIUS = 1e-14  # fixed small radius for tight enclosures
const NUM_EIGENVECTOR_COEFFS = 20
const CHECKPOINT_PATH = joinpath(@__DIR__, "..", "data", "checkpoint.jld2")
const OUTPUT_PATH = joinpath(@__DIR__, "..", "data", "certified_spectral_data.tex")
const MAX_EIGENVALUES = 20  # try to certify up to this many

setprecision(ArbFloat, PRECISION)
setprecision(BigFloat, PRECISION)

# ============================================================================
# Custom Resolvent Evaluation via Back-Substitution
# ============================================================================
#
# The standard run_certification uses svdbox to compute σ_min(T - zI).
# For circles of radius r < 1e-13, the Float64 SVD cannot resolve σ_min
# and fails with "Ball contains zero."
#
# Instead, we compute the resolvent norm via:
#   1. Float64 back-substitution: X ≈ (T - zI)⁻¹
#   2. Ball-arithmetic residual: R = I - (T - zI) × X
#   3. Rigorous bound: ‖(T-zI)⁻¹‖ ≤ ‖X‖ / (1 - ‖R‖)  (Krawczyk)
#
# This works for any r > 0 as long as the back-substitution is stable,
# which is guaranteed for triangular matrices with no zero diagonal entries.

"""
Compute a rigorous upper bound on ‖(T_mid - z·I)⁻¹‖₂ where T_mid is an
upper triangular Schur form (Float64) and z is a sample point.

Uses Float64 back-substitution + Ball-arithmetic residual verification
(Krawczyk-type bound).
"""
function resolvent_norm_triangular(T_mid::AbstractMatrix{<:Number}, z::Number)
    n = size(T_mid, 1)
    CT = ComplexF64

    # Form M = T - zI
    M = Matrix{CT}(T_mid)
    for i in 1:n
        M[i, i] -= CT(z)
    end

    # Float64 back-substitution (exact up to rounding)
    X = inv(UpperTriangular(M))

    # Residual in Ball arithmetic: R = I - M × X
    # BallMatrix with zero radii tracks floating-point rounding rigorously
    bM = BallMatrix(M)
    bX = BallMatrix(X)
    bMX = bM * bX
    # R = I - M*X: extract residual
    R_center = Matrix{CT}(I, n, n) - BallArithmetic.mid(bMX)
    R_radius = BallArithmetic.rad(bMX)  # rad(I) = 0, so rad(R) = rad(MX)
    bR = BallMatrix(R_center, R_radius)

    # Rigorous bounds
    R_norm = collatz_upper_bound_L2_opnorm(bR)
    X_norm = collatz_upper_bound_L2_opnorm(bX)

    if R_norm >= 1.0
        return Inf
    end

    # ‖M⁻¹‖ ≤ ‖X‖ / (1 - ‖R‖)
    return setrounding(Float64, RoundUp) do
        X_norm / (1.0 - R_norm)
    end
end

"""
Certify an eigenvalue enclosure using the Schur-form resolvent bound.

Given:
- Schur form T (Float64 upper triangular) with Z'·mid(A)·Z = T
- BallMatrix A with ‖L - A‖ ≤ ε_K
- Circle of radius r around approximate eigenvalue λ̂

Computes:
  M_r = max_{z on γ} ‖(zI - A_K)⁻¹‖
via the resolvent on the Schur form with Schur-error correction, then checks
  α = ε_K × M_r < 1.

Returns (is_certified, alpha, resolvent_bound).
"""
function certify_eigenvalue_schur(A::BallMatrix, S_schur, λ_center::ComplexF64,
                                   circle_radius::Float64, eps_K::Float64;
                                   num_samples::Int = 256)
    T_mid = S_schur.T
    Z_mid = S_schur.Z
    n = size(T_mid, 1)

    # Schur error bounds (computed once, reusable)
    bZ = BallMatrix(Z_mid)
    bT = BallMatrix(T_mid)
    errF = svd_bound_L2_opnorm(bZ' * bZ - BallMatrix(Matrix{ComplexF64}(I, n, n)))
    errT = svd_bound_L2_opnorm(bZ * bT * bZ' - A)
    sigma_Z = svdbox(bZ)
    norm_Z_up = Float64(BallArithmetic.mid(sigma_Z[1])) + Float64(BallArithmetic.rad(sigma_Z[1]))
    min_sig_lo = max(Float64(BallArithmetic.mid(sigma_Z[end])) - Float64(BallArithmetic.rad(sigma_Z[end])), 0.0)
    min_sig_lo > 0 || error("Schur factor has non-positive smallest singular value")
    norm_Z_inv_up = setrounding(Float64, RoundUp) do; 1.0 / min_sig_lo; end

    errF_up = Float64(BallArithmetic.mid(errF)) + Float64(BallArithmetic.rad(errF))
    errT_up = Float64(BallArithmetic.mid(errT)) + Float64(BallArithmetic.rad(errT))
    norm_Z_sup = max(norm_Z_up - 1.0, 0.0)
    norm_Z_inv_sup = max(norm_Z_inv_up - 1.0, 0.0)
    ε_schur = max(errF_up, errT_up, norm_Z_sup, norm_Z_inv_sup)
    @info "  Schur error ε = $ε_schur"

    # Evaluate resolvent at uniformly spaced points on the circle
    max_resolvent = 0.0
    for j in 0:num_samples-1
        θ = 2π * j / num_samples
        z = λ_center + circle_radius * exp(im * θ)
        res_norm = resolvent_norm_triangular(T_mid, z)
        max_resolvent = max(max_resolvent, res_norm)
    end

    # Apply Schur-error correction (from bound_res_original formula)
    # ‖(zI-A)⁻¹‖ ≤ 2(1+ε²)·l2pseudo / (1 - 2ε(1+ε²)·l2pseudo)
    # where l2pseudo is the max resolvent norm on the Schur form
    factor = 1.0 + ε_schur^2
    numerator = setrounding(Float64, RoundUp) do
        2.0 * factor * max_resolvent
    end
    denominator = setrounding(Float64, RoundDown) do
        1.0 - 2.0 * ε_schur * factor * max_resolvent
    end

    if denominator <= 0
        return (false, Inf, Inf)
    end

    resolvent_bound = setrounding(Float64, RoundUp) do
        numerator / denominator
    end

    alpha = setrounding(Float64, RoundUp) do
        eps_K * resolvent_bound
    end

    return (alpha < 1.0, alpha, resolvent_bound)
end

# ============================================================================
# Checkpoint / Memoization Utilities
# ============================================================================

"""Load checkpoint data, or return empty Dict if none exists."""
function load_checkpoint()
    if isfile(CHECKPOINT_PATH)
        @info "Loading checkpoint from $CHECKPOINT_PATH"
        return load(CHECKPOINT_PATH)
    else
        @info "No checkpoint found, starting fresh"
        return Dict{String,Any}()
    end
end

"""Save checkpoint data."""
function save_checkpoint(data::Dict)
    mkpath(dirname(CHECKPOINT_PATH))
    jldopen(CHECKPOINT_PATH, "w") do f
        for (k, v) in data
            f[k] = v
        end
    end
    @info "Checkpoint saved to $CHECKPOINT_PATH"
end

"""Check if a key exists in checkpoint."""
has_checkpoint(cp, key) = haskey(cp, key)

# ============================================================================
# LaTeX Formatting Helpers
# ============================================================================

"""Format a Float64 in scientific notation for LaTeX."""
function latex_sci(x::Real)
    if !isfinite(x)
        return "\\infty"
    end
    if x == 0.0
        return "0"
    end
    e = floor(Int, log10(abs(x)))
    m = x / 10.0^e
    if e == 0
        return @sprintf("%.4f", x)
    else
        return @sprintf("%.4f \\times 10^{%d}", m, e)
    end
end

"""Format a complex number for LaTeX."""
function format_eigenvalue(z::Number)
    re = real(z)
    im_part = imag(z)
    if abs(im_part) < 1e-50
        return @sprintf("%.15f", re)
    else
        return @sprintf("%.15f %+.15fi", re, im_part)
    end
end

"""Format a complex number in scientific notation for LaTeX."""
function format_complex_sci(z::Number)
    re = real(z)
    im_part = imag(z)
    if abs(im_part) < 1e-50
        return latex_sci(re)
    else
        return "$(latex_sci(re)) $(im_part >= 0 ? "+" : "-") $(latex_sci(abs(im_part)))i"
    end
end

# ============================================================================
# Data Structures
# ============================================================================

struct CertifiedEigenvalue
    index::Int
    center::ComplexF64
    vbd_radius::Float64             # rigorous radius from VBD
    direct_radius::Float64
    direct_alpha::Float64
    direct_certified::Bool
    deflation_radius::Float64
    deflation_alpha::Float64
    deflation_certified::Bool
    nk_radius::Float64
    nk_certified::Bool
    best_radius::Float64
    best_method::String
end

struct CertifiedEigenvector
    index::Int
    coefficients::Vector{ComplexF64}  # centers from rigorous Q matrix
    radii::Vector{Float64}           # rigorous radii from BallMatrix Q
end

struct CertifiedProjection
    index::Int
    coefficients::Vector{ComplexF64}  # centers from rigorous P·e₁
    radii::Vector{Float64}           # rigorous radii from BallVector
    leading_coeff::ComplexF64
    leading_radius::Float64
    l2_norm::Float64
    l2_norm_radius::Float64
end

# ============================================================================
# Phase 1: Setup and Constants
# ============================================================================

println("=" ^ 80)
println("CERTIFIED SPECTRAL DATA GENERATION FOR GKW OPERATOR")
println("=" ^ 80)
println("Date: $(now())")
println("Precision: $PRECISION bits ($(round(Int, PRECISION * log10(2))) decimal digits)")
println("K = $K, N_splitting = $N_SPLITTING")
println()

cp = load_checkpoint()

# Compute C₂ and ε_K
if has_checkpoint(cp, "C2_float") && has_checkpoint(cp, "eps_K_float")
    C2_float = cp["C2_float"]
    eps_K_float = cp["eps_K_float"]
    @info "Loaded C₂ = $C2_float, ε_K = $eps_K_float from checkpoint"
else
    @info "Computing truncation error bounds..."
    C2 = compute_C2(N_SPLITTING)
    C2_float = Float64(real(C2))
    eps_K = compute_Δ(K; N = N_SPLITTING)
    eps_K_float = Float64(real(eps_K))
    cp["C2_float"] = C2_float
    cp["eps_K_float"] = eps_K_float
    save_checkpoint(cp)
end

println("C₂ = $C2_float")
println("ε_K = $eps_K_float")
println()

# ============================================================================
# Phase 2: Build GKW Matrix & Rigorous Block Schur (VBD)
# ============================================================================

s = ArbComplex(1.0, 0.0)  # Classical GKW

# Build BallMatrix (checkpointed)
if has_checkpoint(cp, "gkw_matrix_center") && has_checkpoint(cp, "gkw_matrix_radius")
    @info "Loading GKW matrix from checkpoint..."
    A = BallMatrix(cp["gkw_matrix_center"], cp["gkw_matrix_radius"])
else
    @info "Building GKW matrix with K=$K, precision=$PRECISION bits..."
    M_arb = gkw_matrix_direct(s; K = K)
    A = arb_to_ball_matrix(M_arb)
    cp["gkw_matrix_center"] = BallArithmetic.mid(A)
    cp["gkw_matrix_radius"] = BallArithmetic.rad(A)
    save_checkpoint(cp)
end

# Rigorous block Schur decomposition via VBD
# This gives rigorous eigenvalue intervals, rigorous Q matrix, and clusters.
@info "Computing rigorous block Schur decomposition (VBD)..."
finite_result = certify_gkw_eigenspaces(s; K = K)

vbd = finite_result.block_schur.vbd_result
clusters = finite_result.block_schur.clusters
num_clusters = length(clusters)

println("VBD found $num_clusters eigenvalue clusters")
println("Residual ‖A - QTQ'‖ = $(finite_result.block_schur.residual_norm)")
println("Orthogonality ‖Q'Q - I‖ = $(finite_result.block_schur.orthogonality_defect)")
println()

# Extract eigenvalue centers and VBD radii, sorted by magnitude
cluster_eigenvalues_vbd = Vector{Tuple{Int,ComplexF64,Float64}}()  # (cluster_idx, center, vbd_radius)
for (ci, cluster) in enumerate(clusters)
    λ_ball = vbd.cluster_intervals[cluster[1]]
    λ_center = ComplexF64(BallArithmetic.mid(λ_ball))
    λ_radius = Float64(BallArithmetic.rad(λ_ball))
    push!(cluster_eigenvalues_vbd, (ci, λ_center, λ_radius))
end
sort!(cluster_eigenvalues_vbd; by = x -> -abs(x[2]))

# Get approximate eigenvalues from non-rigorous Schur for targeting
A_center = BallArithmetic.mid(A)
S_approx = schur(A_center)
schur_eigenvalues = diag(S_approx.T)
sorted_schur_idx = sortperm(abs.(schur_eigenvalues), rev = true)

# Use Schur eigenvalues as targets (VBD may not resolve all clusters at large K).
# Map each Schur eigenvalue to the closest VBD cluster for metadata.
eigenvalue_targets = Vector{@NamedTuple{center::ComplexF64, vbd_cluster::Int, vbd_radius::Float64}}()
for rank in 1:min(MAX_EIGENVALUES, length(sorted_schur_idx))
    λ_approx = ComplexF64(schur_eigenvalues[sorted_schur_idx[rank]])
    # Find closest VBD cluster
    best_ci = 1
    best_dist = Inf
    for (ci, λv, _) in cluster_eigenvalues_vbd
        d = abs(λ_approx - λv)
        if d < best_dist
            best_dist = d
            best_ci = ci
        end
    end
    vbd_rad = cluster_eigenvalues_vbd[findfirst(x -> x[1] == best_ci, cluster_eigenvalues_vbd)][3]
    push!(eigenvalue_targets, (center = λ_approx, vbd_cluster = best_ci, vbd_radius = vbd_rad))
end

println("Top eigenvalues (approximate Schur centers):")
for (rank, tgt) in enumerate(eigenvalue_targets[1:min(10, end)])
    println("  λ_$rank ≈ $(round(real(tgt.center), sigdigits=15))  [VBD cluster $(tgt.vbd_cluster), radius = $(latex_sci(tgt.vbd_radius))]")
end
println()

# ============================================================================
# Phase 3: Certify Eigenvalues (Direct + Deflation)
# ============================================================================

# Load previously certified eigenvalues from checkpoint, or start fresh
certified_eigenvalues_data = if has_checkpoint(cp, "certified_eigenvalues")
    cp["certified_eigenvalues"]::Vector{CertifiedEigenvalue}
else
    CertifiedEigenvalue[]
end

certified_eigenvectors_data = if has_checkpoint(cp, "certified_eigenvectors")
    cp["certified_eigenvectors"]::Vector{CertifiedEigenvector}
else
    CertifiedEigenvector[]
end

certified_projections_data = if has_checkpoint(cp, "certified_projections")
    cp["certified_projections"]::Vector{CertifiedProjection}
else
    CertifiedProjection[]
end

already_certified = Set(e.index for e in certified_eigenvalues_data)
eigenvalues_to_process = length(eigenvalue_targets)

# Accumulate certified eigenvalue centers for deflation
certified_centers = ComplexF64[e.center for e in certified_eigenvalues_data]

# Rigorous Q matrix from VBD (if available)
Q_ball = finite_result.block_schur.Q

for rank in 1:eigenvalues_to_process
    rank in already_certified && continue

    tgt = eigenvalue_targets[rank]
    λ_center = tgt.center
    vbd_radius = tgt.vbd_radius
    ci = tgt.vbd_cluster

    println("-" ^ 80)
    @info "Certifying eigenvalue $rank: λ ≈ $(round(real(λ_center), sigdigits=15))"

    # Fixed small circle radius for tight enclosures.
    # With ε_K ≈ 2e-22 and resolvent norm ≈ 1/r on a circle of radius r,
    # α = ε_K / r ≈ 2e-22 / 1e-14 = 2e-8 ≪ 1, so certification succeeds.
    circle_radius = CIRCLE_RADIUS

    # --- Method 1: Direct resolvent certification (back-substitution) ---
    @info "  Direct resolvent certification (back-substitution, r=$circle_radius)..."
    direct_certified, direct_alpha, direct_resolvent = certify_eigenvalue_schur(
        A, S_approx, λ_center, circle_radius, eps_K_float;
        num_samples = CIRCLE_SAMPLES)
    direct_radius = direct_certified ? circle_radius : Inf

    if direct_certified
        @info "  Direct: CERTIFIED (α = $(round(direct_alpha, sigdigits=4)), radius = $circle_radius)"
    else
        @info "  Direct: FAILED (α = $(round(direct_alpha, sigdigits=4)))"
    end

    # --- Method 2: Polynomial deflation (for rank ≥ 2) ---
    deflation_radius = Inf
    deflation_alpha = Inf
    deflation_certified = false

    if rank >= 2 && !isempty(certified_centers)
        @info "  Deflation certification (deflating $(length(certified_centers)) eigenvalues)..."
        try
            defl_result = certify_eigenvalue_deflation(
                A, real(λ_center), real.(certified_centers);
                K = K, N = N_SPLITTING,
                image_circle_radius = 0.5,
                image_circle_samples = CIRCLE_SAMPLES,
                method = :direct,
                backmap_order = 1,
                use_tight_bridge = true
            )
            deflation_alpha = defl_result.small_gain_factor
            deflation_certified = defl_result.is_certified
            deflation_radius = defl_result.eigenvalue_radius

            if deflation_certified
                @info "  Deflation: CERTIFIED (α = $(round(deflation_alpha, sigdigits=4)), radius = $(round(deflation_radius, sigdigits=6)))"
            else
                @info "  Deflation: FAILED (α = $(round(deflation_alpha, sigdigits=4)))"
            end
        catch e
            @warn "  Deflation failed with error: $e"
        end
    end

    # --- Method 3: Newton–Kantorovich (independent of resolvent methods) ---
    # NK is self-contained: it builds its own Jacobian and preconditioner,
    # so it can succeed even when the resolvent/deflation methods fail.
    nk_radius = Inf
    nk_certified = false

    @info "  NK certification..."
    try
        nk_result = certify_eigenpair_nk(s; K=K, target_idx=rank, N_C2=N_SPLITTING)
        nk_radius = nk_result.enclosure_radius
        nk_certified = nk_result.is_certified

        if nk_certified
            @info "  NK: CERTIFIED (r_NK = $(round(nk_radius, sigdigits=6)))"
        else
            @info "  NK: FAILED (q₀ = $(round(nk_result.q0_bound, sigdigits=4)))"
        end
    catch e
        @warn "  NK failed with error: $e"
    end

    # --- Choose best method ---
    best_radius = min(direct_radius, deflation_radius, nk_radius)
    best_method = if best_radius == nk_radius && nk_certified
        "NK"
    elseif best_radius == direct_radius && direct_certified
        "direct"
    elseif deflation_certified
        "deflation"
    else
        "none"
    end

    if best_method == "none"
        @warn "  Eigenvalue $rank: NOT CERTIFIED by any method. Stopping."
        break
    end

    # Record eigenvalue result
    push!(certified_eigenvalues_data, CertifiedEigenvalue(
        rank, λ_center, vbd_radius,
        direct_radius, direct_alpha, direct_certified,
        deflation_radius, deflation_alpha, deflation_certified,
        nk_radius, nk_certified,
        best_radius, best_method
    ))
    push!(certified_centers, λ_center)

    # --- Eigenvector coefficients from rigorous Schur Q (BallMatrix) ---
    # Use the Schur column index (sorted_schur_idx) for this eigenvalue.
    # The Q matrix columns are in the Schur ordering, not cluster ordering.
    n_coeffs = min(NUM_EIGENVECTOR_COEFFS, size(Q_ball, 1))
    q_col_idx = sorted_schur_idx[rank]
    v_coeffs = ComplexF64[ComplexF64(BallArithmetic.mid(Q_ball[j, q_col_idx])) for j in 1:n_coeffs]
    v_radii = Float64[Float64(BallArithmetic.rad(Q_ball[j, q_col_idx])) for j in 1:n_coeffs]
    push!(certified_eigenvectors_data, CertifiedEigenvector(rank, v_coeffs, v_radii))

    # --- Projection of 1 from rigorous VBD projector (BallVector) ---
    # Only available when VBD resolved this cluster separately
    if ci <= length(finite_result.projections_of_one) && length(clusters[ci]) == 1
        proj_ball = finite_result.projections_of_one[ci]
        n_proj = min(NUM_EIGENVECTOR_COEFFS, length(proj_ball))
        proj_coeffs = ComplexF64[ComplexF64(BallArithmetic.mid(proj_ball[j])) for j in 1:n_proj]
        proj_radii = Float64[Float64(BallArithmetic.rad(proj_ball[j])) for j in 1:n_proj]

        lead_ball = finite_result.projection_coefficients[ci]
        lead_coeff = ComplexF64(BallArithmetic.mid(lead_ball))
        lead_radius = Float64(BallArithmetic.rad(lead_ball))

        # L² norm with rigorous radius propagation
        l2_norm_sq = sum(abs2(BallArithmetic.mid(proj_ball[j])) for j in 1:length(proj_ball))
        l2_norm = sqrt(l2_norm_sq)
        l2_norm_radius_sq = sum(BallArithmetic.rad(proj_ball[j])^2 for j in 1:length(proj_ball))
        l2_norm_radius = sqrt(l2_norm_radius_sq)

        push!(certified_projections_data, CertifiedProjection(
            rank, proj_coeffs, proj_radii, lead_coeff, lead_radius, l2_norm, l2_norm_radius
        ))
    end

    # Save checkpoint after each eigenvalue
    cp["certified_eigenvalues"] = certified_eigenvalues_data
    cp["certified_eigenvectors"] = certified_eigenvectors_data
    cp["certified_projections"] = certified_projections_data
    save_checkpoint(cp)

    @info "  Best: $best_method (radius = $(round(best_radius, sigdigits=6)))"
end

num_certified = length(certified_eigenvalues_data)
println()
println("=" ^ 80)
println("Certified $num_certified eigenvalues total")
println("=" ^ 80)
println()

# ============================================================================
# Phase 4: Generate LaTeX Output
# ============================================================================

@info "Generating LaTeX output → $OUTPUT_PATH"
mkpath(dirname(OUTPUT_PATH))

open(OUTPUT_PATH, "w") do io
    # -- Preamble --
    println(io, "% Computer-Assisted Certification Results for GKW Transfer Operator")
    println(io, "% Generated by scripts/generate_certified_latex.jl on $(now())")
    println(io, "% Precision: $PRECISION bits, K=$K, N_splitting=$N_SPLITTING")
    println(io, "%")
    println(io, "% All eigenvalue bounds, eigenvector coefficients, and projections")
    println(io, "% use BallArithmetic's rigorous block Schur decomposition (Miyajima VBD)")
    println(io, "% with interval-arithmetic spectral projectors.  Radii are rigorous.")
    println(io, "%")
    println(io, "% Include in Gauss.tex via: \\input{certified_spectral_data}")
    println(io, "")
    println(io, "\\section*{Computer-Assisted Results}")
    println(io, "")

    # -- Table 1: Parameters --
    println(io, "\\begin{table}[htbp]")
    println(io, "\\centering")
    println(io, "\\caption{Certification parameters for the GKW transfer operator (\$s=1\$).}")
    println(io, "\\label{tab:cert-params}")
    println(io, "\\begin{tabular}{ll}")
    println(io, "\\toprule")
    println(io, "Parameter & Value \\\\")
    println(io, "\\midrule")
    println(io, "Discretization size \$K\$ & $K \\\\")
    println(io, "Matrix size & $(K + 1) \\times $(K + 1) \\\\")
    println(io, "Arithmetic precision & $PRECISION bits \\\\")
    println(io, "Operator norm bound \$C_2\$ & $(latex_sci(C2_float)) \\\\")
    println(io, "Truncation error \$\\varepsilon_K\$ & $(latex_sci(eps_K_float)) \\\\")
    println(io, "\$C_2\$ splitting parameter \$N\$ & $N_SPLITTING \\\\")
    println(io, "Resolvent circle samples & $CIRCLE_SAMPLES \\\\")
    println(io, "VBD residual \$\\|A - QTQ'\\|\$ & $(latex_sci(finite_result.block_schur.residual_norm)) \\\\")
    println(io, "VBD orthogonality \$\\|Q'Q - I\\|\$ & $(latex_sci(finite_result.block_schur.orthogonality_defect)) \\\\")
    println(io, "Eigenvalues certified & $num_certified \\\\")
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")
    println(io, "")

    # -- Table 2: Certified eigenvalues --
    println(io, "\\begin{table}[htbp]")
    println(io, "\\centering")
    println(io, "\\caption{Certified eigenvalue enclosures for the GKW operator (\$s=1\$, \$K=$K\$).}")
    println(io, "\\label{tab:certified-eigenvalues}")
    println(io, "\\begin{tabular}{clllll}")
    println(io, "\\toprule")
    println(io, "\$i\$ & Center \$\\hat\\lambda_i\$ & VBD radius & Resolvent radius & NK radius & Method \\\\")
    println(io, "\\midrule")
    for e in certified_eigenvalues_data
        center_str = format_eigenvalue(e.center)
        vbd_str = latex_sci(e.vbd_radius)
        resolvent_str = e.direct_certified ? latex_sci(e.direct_radius) : "---"
        nk_str = e.nk_certified ? latex_sci(e.nk_radius) : "---"
        println(io, "$(e.index) & \$$(center_str)\$ & \$$(vbd_str)\$ & \$$(resolvent_str)\$ & \$$(nk_str)\$ & $(e.best_method) \\\\")
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")
    println(io, "")

    # -- Table 3: Method comparison --
    println(io, "\\begin{table}[htbp]")
    println(io, "\\centering")
    println(io, "\\caption{Comparison of certification methods: direct resolvent, polynomial deflation, and Newton--Kantorovich.}")
    println(io, "\\label{tab:method-comparison}")
    println(io, "\\begin{tabular}{cllllll}")
    println(io, "\\toprule")
    println(io, "\$i\$ & Direct \$\\alpha\$ & Direct radius & Deflation \$\\alpha\$ & Deflation radius & NK radius & Best \\\\")
    println(io, "\\midrule")
    for e in certified_eigenvalues_data
        d_alpha = e.direct_certified ? latex_sci(e.direct_alpha) : "\\text{fail}"
        d_rad = e.direct_certified ? latex_sci(e.direct_radius) : "---"
        defl_alpha = e.deflation_certified ? latex_sci(e.deflation_alpha) : (e.index >= 2 ? "\\text{fail}" : "---")
        defl_rad = e.deflation_certified ? latex_sci(e.deflation_radius) : "---"
        nk_rad = e.nk_certified ? latex_sci(e.nk_radius) : "---"
        println(io, "$(e.index) & \$$(d_alpha)\$ & \$$(d_rad)\$ & \$$(defl_alpha)\$ & \$$(defl_rad)\$ & \$$(nk_rad)\$ & $(e.best_method) \\\\")
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")
    println(io, "")

    # -- Tables for eigenvector coefficients (from rigorous Schur Q) --
    for ev in certified_eigenvectors_data
        println(io, "\\begin{table}[htbp]")
        println(io, "\\centering")
        println(io, "\\caption{Schur vector coefficients for \$\\lambda_{$(ev.index)}\$ (first $(length(ev.coefficients)) terms, rigorous radii from VBD).}")
        println(io, "\\label{tab:eigvec-$(ev.index)}")
        println(io, "\\begin{tabular}{cll}")
        println(io, "\\toprule")
        println(io, "\$k\$ & Coefficient \$[Q e_i]_k\$ & Radius \\\\")
        println(io, "\\midrule")
        for (k, (c, r)) in enumerate(zip(ev.coefficients, ev.radii))
            coeff_str = format_complex_sci(c)
            rad_str = latex_sci(r)
            println(io, "$(k - 1) & \$$(coeff_str)\$ & \$$(rad_str)\$ \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}")
        println(io, "")
    end

    # -- Table: Projection of 1 onto each eigenspace (from rigorous VBD projector) --
    println(io, "\\begin{table}[htbp]")
    println(io, "\\centering")
    println(io, "\\caption{Projection of the constant function \$1\$ onto each eigenspace: \$\\Pi_i(1) = P_i \\cdot e_1\$ (rigorous radii from VBD).}")
    println(io, "\\label{tab:projections}")
    println(io, "\\begin{tabular}{cllll}")
    println(io, "\\toprule")
    println(io, "\$i\$ & Leading coeff \$[\\Pi_i(1)]_0\$ & Radius & \$\\|\\Pi_i(1)\\|_{H^2}\$ & Radius \\\\")
    println(io, "\\midrule")
    for p in certified_projections_data
        lc_str = format_complex_sci(p.leading_coeff)
        lc_rad = latex_sci(p.leading_radius)
        norm_str = @sprintf("%.12f", p.l2_norm)
        norm_rad = latex_sci(p.l2_norm_radius)
        println(io, "$(p.index) & \$$(lc_str)\$ & \$$(lc_rad)\$ & \$$(norm_str)\$ & \$$(norm_rad)\$ \\\\")
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")
    println(io, "")

    # -- Theorem environment --
    println(io, "\\begin{theorem}[Certified Eigenvalue Enclosures]")
    println(io, "\\label{thm:certified-enclosures}")
    println(io, "Let \$\\mathcal{L}_1 \\colon H^2(\\mathbb{D}_1) \\to H^2(\\mathbb{D}_1)\$ be the GKW transfer operator at \$s=1\$.")
    println(io, "Using a Galerkin discretization of size \$K=$(K)\$ with \$$(PRECISION)\$-bit arithmetic,")
    println(io, "BallArithmetic's rigorous block Schur decomposition (Miyajima VBD),")
    println(io, "and Newton--Kantorovich refinement on the eigenpair map,")
    println(io, "the following eigenvalue enclosures are rigorously certified:")
    println(io, "\\begin{align*}")
    for (j, e) in enumerate(certified_eigenvalues_data)
        center_str = format_eigenvalue(e.center)
        radius_str = latex_sci(e.best_radius)
        method_tag = e.best_method == "NK" ? "\\text{NK}" : "\\text{$(e.best_method)}"
        sep = j < num_certified ? " \\\\" : ""
        println(io, "  \\lambda_{$(e.index)} &\\in [$(center_str) \\pm $(radius_str)] \\quad ($(method_tag))$(sep)")
    end
    println(io, "\\end{align*}")
    println(io, "The truncation error is \$\\varepsilon_K = $(latex_sci(eps_K_float))\$")
    println(io, "and the operator norm bound is \$C_2 = $(latex_sci(C2_float))\$.")
    println(io, "Eigenvector coefficients and spectral projections carry rigorous radii")
    println(io, "from the verified block Schur decomposition.")
    println(io, "\\end{theorem}")
    println(io, "")

end  # close file

@info "LaTeX output written to $OUTPUT_PATH"
println()
println("Done. Include in Gauss.tex via:")
println("  \\input{certified_spectral_data}")
