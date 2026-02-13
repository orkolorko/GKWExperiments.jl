#!/usr/bin/env julia
# Full Certification Pipeline for 50 GKW Eigenvalues
#
# Three-stage pipeline:
#   Stage 1: Adaptive resolvent at small K (proves spectral completeness)
#            + tail bound on separating circle
#   Stage 2: NK certification at K_HIGH (tight eigenpair enclosures)
#   Stage 3: Combine with BigFloat spectral results (v_j, P_j, ℓ_j(1))
#
# The adaptive approach for Stage 1:
#   Start with K=K_START (e.g., 64), attempt resolvent certification.
#   For eigenvalues that fail (α = ε_K·‖R‖ ≥ 1), double K and retry.
#   Continue until all eigenvalues are certified or K exceeds K_HIGH.
#
# Designed to run on ibis (32 cores, 125 GB RAM).
#
# Usage:
#   julia --project --startup-file=no scripts/full_certification_50eigs.jl

using GKWExperiments
using ArbNumerics
using BallArithmetic
using LinearAlgebra
using Printf
using Serialization
using Dates

# Access internal helper for rigorous Arb → Float64 conversion
const _arb_to_float64_upper = GKWExperiments.NewtonKantorovichCertification._arb_to_float64_upper

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

const PRECISION       = 512           # ArbNumerics precision (bits)
const K_START         = 64            # Starting K for adaptive resolvent
const K_MAX_RESOLVENT = 256           # Maximum K for resolvent certification
const K_HIGH          = 512           # K for NK certification (Stage 2)
const P_HIGH          = 1024          # BigFloat precision for K_HIGH data
const NUM_EIGS        = 50            # Number of eigenvalues to certify
const N_SPLITTING     = 5000          # C₂ splitting parameter
const CIRCLE_SAMPLES  = 256           # Samples on resolvent circles
const CIRCLE_RADIUS_FACTOR = 0.01     # Base circle radius = factor × |λ|

const CACHE_DIR = "data"
mkpath(CACHE_DIR)

setprecision(ArbFloat, PRECISION)
setprecision(BigFloat, P_HIGH)

s = ArbComplex(1.0, 0.0)  # Classical GKW (s=1)

println("="^80)
println("FULL CERTIFICATION PIPELINE FOR 50 GKW EIGENVALUES")
println("="^80)
println("Timestamp: $(now())")
println()
println("Configuration:")
println("  K_START (resolvent) = $K_START")
println("  K_MAX (resolvent)   = $K_MAX_RESOLVENT")
println("  K_HIGH (NK)         = $K_HIGH")
println("  NUM_EIGS            = $NUM_EIGS")
println("  N_SPLITTING         = $N_SPLITTING")
println("  CIRCLE_SAMPLES      = $CIRCLE_SAMPLES")
println()

# ═══════════════════════════════════════════════════════════════════════════
# Phase 0: Precompute Constants
# ═══════════════════════════════════════════════════════════════════════════

println("="^80)
println("PHASE 0: CONSTANTS")
println("="^80)
println()

@info "Computing C₂ bound (N=$N_SPLITTING)..."
C2_arb = compute_C2(N_SPLITTING)
C2_float = _arb_to_float64_upper(C2_arb)
@info "  C₂ ≤ $C2_float"

# Precompute ε_K for all K values we might use
K_values = Int[]
K_cur = K_START
while K_cur ≤ K_MAX_RESOLVENT
    push!(K_values, K_cur)
    K_cur *= 2
end
push!(K_values, K_HIGH)
K_values = sort(unique(K_values))

eps_K_table = Dict{Int,Float64}()
for K_val in K_values
    @info "Computing ε_K for K=$K_val..."
    eps_arb = compute_Δ(K_val; N=N_SPLITTING)
    eps_K_table[K_val] = _arb_to_float64_upper(eps_arb)
    @printf("  ε_{%d} = %.6e\n", K_val, eps_K_table[K_val])
end
println()

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Adaptive Resolvent Certification (Stage 1)
# ═══════════════════════════════════════════════════════════════════════════

println("="^80)
println("PHASE 1: ADAPTIVE RESOLVENT CERTIFICATION")
println("="^80)
println()

# Result storage for each eigenvalue
struct ResolventResult
    eigenvalue_index::Int
    lambda_center::ComplexF64
    K_certified::Int          # K at which certification succeeded
    circle_radius::Float64
    resolvent_Ak::Float64     # ‖R_{A_K}(z)‖ on the circle
    alpha::Float64            # ε_K · resolvent_Ak
    eps_K::Float64
    M_inf::Float64            # ‖R_{L_r}(z)‖ ≤ resolvent_Ak / (1-α)
    is_certified::Bool
end

resolvent_results = Vector{ResolventResult}(undef, NUM_EIGS)
uncertified = collect(1:NUM_EIGS)  # indices still to certify

# We need eigenvalue locations; use a moderate-K Schur for initial estimates
@info "Getting eigenvalue locations from K=$K_MAX_RESOLVENT Schur..."
t0 = time()
cache_low = joinpath(CACHE_DIR, "ball_matrix_K$(K_MAX_RESOLVENT).jls")
if isfile(cache_low)
    A_loc = deserialize(cache_low)
    @info "  Loaded from cache."
else
    M_arb_loc = gkw_matrix_direct_fast(s; K=K_MAX_RESOLVENT, threaded=true)
    A_loc = arb_to_ball_matrix(M_arb_loc)
    serialize(cache_low, A_loc)
    @info "  Built and cached."
end
@info "  Done in $(round(time()-t0, digits=1))s"

A_center_loc = BallArithmetic.mid(A_loc)
S_loc = schur(A_center_loc)
λ_all = diag(S_loc.T)
sorted_idx_global = sortperm(abs.(λ_all), rev=true)

# Eigenvalue centers for the first NUM_EIGS
lambda_centers = [ComplexF64(λ_all[sorted_idx_global[i]]) for i in 1:NUM_EIGS]

# Compute gap-based circle radii
circle_radii = zeros(Float64, NUM_EIGS)
for i in 1:NUM_EIGS
    # Base radius
    circle_radii[i] = max(abs(lambda_centers[i]) * CIRCLE_RADIUS_FACTOR, 1e-16)
    # Avoid overlap with other eigenvalues
    for j in 1:min(NUM_EIGS + 5, length(λ_all))
        if j != i
            other_λ = if j ≤ NUM_EIGS
                lambda_centers[j]
            else
                ComplexF64(λ_all[sorted_idx_global[j]])
            end
            dist = abs(lambda_centers[i] - other_λ)
            circle_radii[i] = min(circle_radii[i], dist / 3)
        end
    end
end

println("Eigenvalue locations and circle radii:")
println("-"^70)
for i in 1:min(10, NUM_EIGS)
    @printf("  j=%2d: λ̂ = %+.10e, r_circle = %.4e\n",
            i, real(lambda_centers[i]), circle_radii[i])
end
if NUM_EIGS > 10
    println("  ...")
    for i in (NUM_EIGS-2):NUM_EIGS
        @printf("  j=%2d: λ̂ = %+.10e, r_circle = %.4e\n",
                i, real(lambda_centers[i]), circle_radii[i])
    end
end
println()

# Initialize uncertified results with placeholders
for i in 1:NUM_EIGS
    resolvent_results[i] = ResolventResult(
        i, lambda_centers[i], 0, circle_radii[i], Inf, Inf, Inf, Inf, false)
end

# Adaptive loop over K levels
K_levels = sort([k for k in K_values if k ≤ K_MAX_RESOLVENT])

for K_level in K_levels
    if isempty(uncertified)
        break
    end

    println("-"^80)
    @info "Trying resolvent certification at K=$K_level for $(length(uncertified)) remaining eigenvalues..."

    # Build/load BallMatrix at this K
    cache_k = joinpath(CACHE_DIR, "ball_matrix_K$(K_level).jls")
    local A_ball_k
    if isfile(cache_k)
        A_ball_k = deserialize(cache_k)
        @info "  Loaded BallMatrix from cache."
    else
        @info "  Building GKW matrix at K=$K_level..."
        t0 = time()
        M_arb_k = gkw_matrix_direct_fast(s; K=K_level, threaded=true)
        A_ball_k = arb_to_ball_matrix(M_arb_k)
        serialize(cache_k, A_ball_k)
        @info "  Built in $(round(time()-t0, digits=1))s"
    end

    eps_K = eps_K_table[K_level]

    # Eigenvalue locations from this matrix's Schur
    A_center_k = BallArithmetic.mid(A_ball_k)
    S_k = schur(A_center_k)
    λ_k = diag(S_k.T)
    sorted_idx_k = sortperm(abs.(λ_k), rev=true)

    # Try each uncertified eigenvalue
    newly_certified = Int[]
    for i in uncertified
        # Use eigenvalue from this K's Schur decomposition
        # (more accurate for this matrix size)
        if i ≤ length(λ_k)
            λ_center_k = ComplexF64(λ_k[sorted_idx_k[i]])
        else
            λ_center_k = lambda_centers[i]
        end

        # Circle radius (recompute for this K's eigenvalues)
        r_circle = circle_radii[i]

        circle = CertificationCircle(λ_center_k, r_circle; samples=CIRCLE_SAMPLES)
        t1 = time()

        local cert_data
        try
            cert_data = run_certification(A_ball_k, circle)
        catch e
            @warn "  j=$i: resolvent certification threw error: $(typeof(e))"
            continue
        end
        dt = time() - t1

        resolvent_Ak = cert_data.resolvent_original

        # Small-gain check
        alpha = setrounding(Float64, RoundUp) do
            eps_K * resolvent_Ak
        end

        if alpha < 1.0
            denom = setrounding(Float64, RoundDown) do
                1.0 - alpha
            end
            M_inf = setrounding(Float64, RoundUp) do
                resolvent_Ak / denom
            end

            resolvent_results[i] = ResolventResult(
                i, λ_center_k, K_level, r_circle,
                resolvent_Ak, alpha, eps_K, M_inf, true)
            push!(newly_certified, i)

            @printf("  j=%2d: CERTIFIED at K=%d, α=%.4e, ‖R‖=%.4f, M_∞=%.4f [%.1fs]\n",
                    i, K_level, alpha, resolvent_Ak, M_inf, dt)
        else
            @printf("  j=%2d: FAILED at K=%d, α=%.4e, ‖R‖=%.4f [%.1fs]\n",
                    i, K_level, alpha, resolvent_Ak, dt)
        end
    end

    # Remove newly certified from uncertified list
    filter!(i -> !(i in newly_certified), uncertified)
    @info "  Certified $(length(newly_certified)) eigenvalues at K=$K_level, $(length(uncertified)) remaining."
end

# Summary
num_resolvent_certified = count(r -> r.is_certified, resolvent_results)
println()
println("="^80)
println("PHASE 1 SUMMARY")
println("="^80)
println("  Resolvent certified: $num_resolvent_certified / $NUM_EIGS")
if !isempty(uncertified)
    println("  UNCERTIFIED eigenvalues: $uncertified")
    println("  These need higher K or deflation.")
end
println()

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1b: Tail Bound (separating circle)
# ═══════════════════════════════════════════════════════════════════════════

println("="^80)
println("PHASE 1b: TAIL BOUND")
println("="^80)
println()

# Use the K=K_MAX_RESOLVENT matrix for the tail circle
# The circle separates |λ_{NUM_EIGS}| from |λ_{NUM_EIGS+1}|
λ_N   = lambda_centers[NUM_EIGS]
λ_Np1 = if NUM_EIGS + 1 ≤ length(λ_all)
    ComplexF64(λ_all[sorted_idx_global[NUM_EIGS + 1]])
else
    ComplexF64(0.0)
end
abs_λ_N   = abs(λ_N)
abs_λ_Np1 = abs(λ_Np1)
ρ_tail = (abs_λ_N + abs_λ_Np1) / 2.0

@printf("  |λ_%d| = %.6e,  |λ_%d| = %.6e\n", NUM_EIGS, abs_λ_N, NUM_EIGS+1, abs_λ_Np1)
@printf("  Tail circle radius ρ = %.6e\n", ρ_tail)
@printf("  Gap to λ_%d: %.6e,  gap to λ_%d: %.6e\n",
        NUM_EIGS, abs_λ_N - ρ_tail, NUM_EIGS+1, ρ_tail - abs_λ_Np1)
println()

# Adaptive: try increasing K until tail resolvent certifies
tail_certified = false
local tail_resolvent_Ak, tail_M_inf, tail_alpha, tail_K
for K_level in K_levels
    cache_k = joinpath(CACHE_DIR, "ball_matrix_K$(K_level).jls")
    A_ball_k = deserialize(cache_k)
    eps_K = eps_K_table[K_level]

    @info "Tail bound: trying K=$K_level..."
    circle_tail = CertificationCircle(ComplexF64(0.0), ρ_tail; samples=CIRCLE_SAMPLES)
    t0 = time()
    cert_tail = run_certification(A_ball_k, circle_tail)
    dt = time() - t0

    tail_resolvent_Ak = cert_tail.resolvent_original
    tail_alpha = setrounding(Float64, RoundUp) do
        eps_K * tail_resolvent_Ak
    end

    if tail_alpha < 1.0
        denom = setrounding(Float64, RoundDown) do
            1.0 - tail_alpha
        end
        tail_M_inf = setrounding(Float64, RoundUp) do
            tail_resolvent_Ak / denom
        end
        tail_K = K_level
        tail_certified = true

        @printf("  CERTIFIED at K=%d: α=%.4e, ‖R‖=%.4f, M_∞=%.4f [%.1fs]\n",
                K_level, tail_alpha, tail_resolvent_Ak, tail_M_inf, dt)
        break
    else
        @printf("  FAILED at K=%d: α=%.4e, ‖R‖=%.4f [%.1fs]\n",
                K_level, tail_alpha, tail_resolvent_Ak, dt)
    end
end

if tail_certified
    @printf("\n  Tail bound: ‖R_%d(n)‖₂ ≤ ρ^{n+1} · M_∞ = %.6e^{n+1} · %.6e\n",
            NUM_EIGS, ρ_tail, tail_M_inf)
else
    @warn "Tail bound FAILED at all K levels."
end
println()

# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: NK Certification at K_HIGH
# ═══════════════════════════════════════════════════════════════════════════

println("="^80)
println("PHASE 2: NK CERTIFICATION AT K=$K_HIGH")
println("="^80)
println()

# Build/load Float64 BallMatrix at K_HIGH
cache_ball_high = joinpath(CACHE_DIR, "ball_matrix_K$(K_HIGH).jls")
local A_ball_high
if isfile(cache_ball_high)
    @info "Loading cached Float64 BallMatrix at K=$K_HIGH..."
    A_ball_high = deserialize(cache_ball_high)
else
    @info "Building GKW matrix at K=$K_HIGH..."
    t0 = time()
    M_arb_high = gkw_matrix_direct_fast(s; K=K_HIGH, threaded=true)
    @info "  Matrix built in $(round(time()-t0, digits=1))s"

    @info "Converting to Float64 BallMatrix..."
    A_ball_high = arb_to_ball_matrix(M_arb_high)
    serialize(cache_ball_high, A_ball_high)
    @info "  Cached."
end

n_high = K_HIGH + 1
@info "  Matrix size: $n_high × $n_high"
println()

eps_K_high = eps_K_table[K_HIGH]
@printf("  ε_{K=%d} = %.6e\n\n", K_HIGH, eps_K_high)

# NK certification results
struct NKResult
    eigenvalue_index::Int
    eigenvalue_center::ComplexF64
    eigenvector_center::Union{Vector{ComplexF64}, Nothing}
    nk_radius::Float64
    eigenvalue_radius::Float64
    eigenvector_radius::Float64
    qk::Float64
    C_norm::Float64
    q0::Float64
    y::Float64
    is_certified::Bool
end

nk_results = Vector{NKResult}(undef, NUM_EIGS)

for i in 1:NUM_EIGS
    @info "NK certification: eigenvalue $i / $NUM_EIGS..."
    t0 = time()

    local nk_res
    try
        nk_res = certify_eigenpair_nk(A_ball_high; K=K_HIGH, target_idx=i, N_C2=N_SPLITTING)
    catch e
        @warn "  j=$i: NK threw error: $(typeof(e)): $(e)"
        nk_results[i] = NKResult(i, lambda_centers[i], nothing, Inf, Inf, Inf,
                                  Inf, Inf, Inf, Inf, false)
        continue
    end
    dt = time() - t0

    nk_results[i] = NKResult(
        i, nk_res.eigenvalue_center, nk_res.eigenvector_center,
        nk_res.enclosure_radius, nk_res.eigenvalue_radius, nk_res.eigenvector_radius,
        nk_res.qk_bound, nk_res.C_bound, nk_res.q0_bound, nk_res.y_bound,
        nk_res.is_certified)

    if nk_res.is_certified
        @printf("  j=%2d: CERTIFIED, r_NK = %.4e, q₀ = %.4e [%.1fs]\n",
                i, nk_res.enclosure_radius, nk_res.q0_bound, dt)
    else
        @printf("  j=%2d: FAILED, q₀ = %.4e [%.1fs]\n",
                i, nk_res.q0_bound, dt)
    end
end

num_nk_certified = count(r -> r.is_certified, nk_results)
println()
println("NK Summary: $num_nk_certified / $NUM_EIGS certified")
println()

# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Transfer Bridge + Riesz Projector Errors
# ═══════════════════════════════════════════════════════════════════════════

println("="^80)
println("PHASE 3: TRANSFER BRIDGE + RIESZ PROJECTOR ERRORS")
println("="^80)
println()

struct TransferResult
    eigenvalue_index::Int
    transfer_resolvent::Float64    # ‖R_{A_{K_HIGH}}(z)‖ via reverse transfer
    transfer_alpha::Float64
    transfer_valid::Bool
    riesz_projector_error::Float64
    riesz_valid::Bool
end

transfer_results = Vector{TransferResult}(undef, NUM_EIGS)

for i in 1:NUM_EIGS
    rr = resolvent_results[i]
    if !rr.is_certified
        transfer_results[i] = TransferResult(i, Inf, Inf, false, Inf, false)
        continue
    end

    # Reverse transfer: M_∞ from Stage 1 + ε_{K_HIGH}
    t_resolvent, t_alpha, t_valid = reverse_transfer_resolvent_bound(
        rr.M_inf, eps_K_high)

    proj_error = Inf
    proj_valid = false
    if t_valid
        contour_length = 2π * rr.circle_radius
        proj_error, proj_valid = projector_approximation_error_rigorous(
            contour_length, t_resolvent, eps_K_high)
    end

    transfer_results[i] = TransferResult(
        i, t_resolvent, t_alpha, t_valid, proj_error, proj_valid)

    if t_valid && proj_valid
        @printf("  j=%2d: ‖R_{A_%d}‖ ≤ %.4f, Riesz error ≤ %.4e\n",
                i, K_HIGH, t_resolvent, proj_error)
    elseif t_valid
        @printf("  j=%2d: ‖R_{A_%d}‖ ≤ %.4f (Riesz FAILED)\n",
                i, K_HIGH, t_resolvent)
    else
        @printf("  j=%2d: Transfer FAILED (α_high = %.4e)\n",
                i, t_alpha)
    end
end

println()

# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Load BigFloat Spectral Results and Combine
# ═══════════════════════════════════════════════════════════════════════════

println("="^80)
println("PHASE 4: COMBINING WITH BIGFLOAT SPECTRAL RESULTS")
println("="^80)
println()

cache_bigfloat = joinpath(CACHE_DIR, "bigfloat_spectral_K$(K_HIGH)_P$(P_HIGH).jls")
has_bigfloat = isfile(cache_bigfloat)

if has_bigfloat
    @info "Loading BigFloat spectral results from $cache_bigfloat..."
    bf_results = deserialize(cache_bigfloat)
    bf_eigenvalues = bf_results[:eigenvalues]
    bf_eigenvectors = bf_results[:eigenvectors]
    bf_ell_center = bf_results[:ell_center]
    bf_ell_radius = bf_results[:ell_radius]
    bf_E_bound = bf_results[:E_bound]
    @info "  Loaded: $(length(bf_eigenvalues)) eigenvalues"
else
    @warn "BigFloat spectral results not found at $cache_bigfloat"
    @warn "Run bigfloat_spectral_K512.jl first to compute v_j, ℓ_j(1)."
end
println()

# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: Summary + Output
# ═══════════════════════════════════════════════════════════════════════════

println("="^80)
println("FINAL RESULTS")
println("="^80)
println()

num_fully_certified = 0

println("-"^130)
@printf("  %3s  %22s  %10s  %10s  %10s  %10s  %10s  %22s  %5s\n",
    "j", "λ̂_j", "K_resolv", "α₁", "NK r", "Proj err", "α_high", "ℓ_j(1)", "Full")
println("-"^130)

for i in 1:NUM_EIGS
    rr = resolvent_results[i]
    nk = nk_results[i]
    tr = transfer_results[i]

    is_full = rr.is_certified && nk.is_certified && tr.transfer_valid
    if is_full
        num_fully_certified += 1
    end

    # ℓ_j(1) from BigFloat if available
    ell_str = "N/A"
    if has_bigfloat && i ≤ length(bf_ell_center)
        ell_str = @sprintf("%+.14e", Float64(bf_ell_center[i]))
    end

    @printf("  %3d  %+.14e  K=%3d  %10.2e  %10.2e  %10.2e  %10.2e  %22s  %5s\n",
        i,
        real(rr.is_certified ? rr.lambda_center : lambda_centers[i]),
        rr.is_certified ? rr.K_certified : 0,
        rr.alpha,
        nk.nk_radius,
        tr.riesz_projector_error,
        tr.transfer_alpha,
        ell_str,
        is_full ? "YES" : "NO")
end

println("-"^130)
println()
println("CERTIFICATION STATISTICS:")
println("  Stage 1 (resolvent): $num_resolvent_certified / $NUM_EIGS")
println("  Stage 2 (NK):        $num_nk_certified / $NUM_EIGS")
println("  Transfer bridge:     $(count(r -> r.transfer_valid, transfer_results)) / $NUM_EIGS")
println("  Fully certified:     $num_fully_certified / $NUM_EIGS")
if tail_certified
    @printf("  Tail bound: ‖R_%d(n)‖₂ ≤ %.4e · (%.6e)^n\n",
            NUM_EIGS, tail_M_inf, ρ_tail)
end
println()

# ═══════════════════════════════════════════════════════════════════════════
# Save all results
# ═══════════════════════════════════════════════════════════════════════════

results_file = joinpath(CACHE_DIR, "full_certification_50eigs.jls")
@info "Saving all results to $results_file..."

save_data = Dict(
    :num_eigs => NUM_EIGS,
    :K_high => K_HIGH,
    :P_high => P_HIGH,
    :timestamp => now(),
    # Stage 1
    :resolvent_certified => [r.is_certified for r in resolvent_results],
    :resolvent_K => [r.K_certified for r in resolvent_results],
    :resolvent_alpha => [r.alpha for r in resolvent_results],
    :resolvent_M_inf => [r.M_inf for r in resolvent_results],
    :resolvent_Ak => [r.resolvent_Ak for r in resolvent_results],
    :circle_radii => [r.circle_radius for r in resolvent_results],
    :lambda_centers => [r.lambda_center for r in resolvent_results],
    # Tail
    :tail_certified => tail_certified,
    :tail_rho => ρ_tail,
    :tail_M_inf => tail_certified ? tail_M_inf : Inf,
    :tail_K => tail_certified ? tail_K : 0,
    # Stage 2
    :nk_certified => [r.is_certified for r in nk_results],
    :nk_radius => [r.nk_radius for r in nk_results],
    :nk_eigenvalue_center => [r.eigenvalue_center for r in nk_results],
    :nk_q0 => [r.q0 for r in nk_results],
    # Transfer
    :transfer_valid => [r.transfer_valid for r in transfer_results],
    :transfer_resolvent => [r.transfer_resolvent for r in transfer_results],
    :riesz_projector_error => [r.riesz_projector_error for r in transfer_results],
    # BigFloat
    :has_bigfloat => has_bigfloat,
)

if has_bigfloat
    save_data[:bf_eigenvalues] = bf_eigenvalues
    save_data[:bf_ell_center] = bf_ell_center
    save_data[:bf_ell_radius] = bf_ell_radius
    save_data[:bf_E_bound] = bf_E_bound
end

serialize(results_file, save_data)
@info "  Saved."

# ═══════════════════════════════════════════════════════════════════════════
# Export portable text file
# ═══════════════════════════════════════════════════════════════════════════

txt_file = joinpath(CACHE_DIR, "full_certification_50eigs.txt")
open(txt_file, "w") do io
    println(io, "# Full certification results for 50 GKW eigenvalues")
    println(io, "# Generated: $(now())")
    println(io, "# K_HIGH = $K_HIGH, P_HIGH = $P_HIGH")
    println(io, "#")
    println(io, "# Columns: j, lambda_j, K_resolvent, alpha1, M_inf, nk_radius, proj_error, ell_j(1), ell_radius, certified")
    println(io, "#")
    for i in 1:NUM_EIGS
        rr = resolvent_results[i]
        nk = nk_results[i]
        tr = transfer_results[i]
        is_full = rr.is_certified && nk.is_certified && tr.transfer_valid

        ell_c = has_bigfloat && i ≤ length(bf_ell_center) ? string(bf_ell_center[i]) : "NaN"
        ell_r = has_bigfloat && i ≤ length(bf_ell_radius) ? string(bf_ell_radius[i]) : "NaN"

        λ_str = has_bigfloat && i ≤ length(bf_eigenvalues) ? string(bf_eigenvalues[i]) : @sprintf("%.15e", real(lambda_centers[i]))

        @printf(io, "%d\t%s\t%d\t%.15e\t%.15e\t%.15e\t%.15e\t%s\t%s\t%s\n",
            i, λ_str,
            rr.is_certified ? rr.K_certified : 0,
            rr.alpha, rr.M_inf,
            nk.nk_radius,
            tr.riesz_projector_error,
            ell_c, ell_r,
            is_full ? "YES" : "NO")
    end
end
@info "Portable text file written to $txt_file"

println()
println("="^80)
println("PIPELINE COMPLETE")
println("="^80)
