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

using Distributed
const N_WORKERS = 14        # 16 cores → 14 workers + main process + OS
addprocs(N_WORKERS; exeflags="--project=$(Base.active_project())")

using GKWExperiments
using ArbNumerics
using BallArithmetic
using GenericSchur          # enables direct BigFloat Schur in run_certification
using LinearAlgebra
using Printf
using Serialization
using Dates

@info "Launched $N_WORKERS workers: $(workers())"

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
let K_cur = K_START
    while K_cur ≤ K_MAX_RESOLVENT
        push!(K_values, K_cur)
        K_cur *= 2
    end
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

    # Compute Schur data ONCE for this K level, reuse for all circles
    @info "  Computing Schur data for K=$K_level (done once, reused for all circles)..."
    t_schur = time()
    local sd_k
    try
        sd_k = BallArithmetic.CertifScripts.compute_schur_and_error(A_ball_k)
    catch e
        @warn "  Schur computation failed at K=$K_level: $(typeof(e)): $e"
        continue
    end
    @info "  Schur done in $(round(time()-t_schur, digits=1))s"

    # Eigenvalue locations from this matrix's Schur
    S_k = sd_k[1]
    λ_k = diag(S_k.T)
    sorted_idx_k = sortperm(abs.(λ_k), rev=true)

    # Try each uncertified eigenvalue
    newly_certified = Int[]
    last_alpha = 0.0  # track α trend for early exit
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

        # Early exit: if previous α >> 1, remaining eigenvalues (smaller |λ|)
        # will be even worse. Skip to next K level to avoid hanging workers.
        if last_alpha > 10.0
            @printf("  j=%2d: SKIPPED at K=%d (previous α=%.1e >> 1, remaining deferred to higher K)\n",
                    i, K_level, last_alpha)
            continue
        end

        circle = CertificationCircle(λ_center_k, r_circle; samples=CIRCLE_SAMPLES)
        t1 = time()

        local cert_data
        try
            cert_data = run_certification(A_ball_k, circle, workers(); schur_data=sd_k)
        catch e
            @warn "  j=$i: resolvent certification threw error: $(typeof(e))"
            last_alpha = Inf  # trigger early exit for remaining
            continue
        end
        dt = time() - t1

        resolvent_Ak = cert_data.resolvent_original

        # Small-gain check
        alpha = setrounding(Float64, RoundUp) do
            eps_K * resolvent_Ak
        end
        last_alpha = alpha

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

# Summary of Float64 resolvent
num_resolvent_f64 = count(r -> r.is_certified, resolvent_results)
println()
println("  Float64 resolvent certified: $num_resolvent_f64 / $NUM_EIGS")
if !isempty(uncertified)
    println("  Remaining for BigFloat: $(length(uncertified)) eigenvalues")
end
println()

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1c: BigFloat Resolvent for remaining eigenvalues
# ═══════════════════════════════════════════════════════════════════════════

if !isempty(uncertified)
    println("-"^80)
    println("Phase 1c: BigFloat RESOLVENT for remaining eigenvalues")
    println("-"^80)
    println()

    # Try BigFloat BallMatrix at increasing K: K=256 (fast), then K=512 if needed.
    # K=128 could also work but we might not have it cached in BigFloat.
    bigfloat_K_candidates = [
        (256, 512, "bigfloat_ball_matrix_K256_P512.jls"),
        (K_HIGH, P_HIGH, "bigfloat_ball_matrix_K$(K_HIGH)_P$(P_HIGH).jls"),
    ]

    # Eigenvalue locations from BigFloat Schur at K_HIGH
    cache_bf_spec = joinpath(CACHE_DIR, "bigfloat_spectral_K$(K_HIGH)_P$(P_HIGH).jls")
    local bf_eigs_for_resolvent
    if isfile(cache_bf_spec)
        bf_spec = deserialize(cache_bf_spec)
        bf_eigs_for_resolvent = bf_spec[:eigenvalues]
    else
        bf_eigs_for_resolvent = nothing
        @warn "No BigFloat spectral results — using Float64 eigenvalue centers."
    end

    for (bf_K, bf_P, bf_filename) in bigfloat_K_candidates
        if isempty(uncertified)
            break
        end

        cache_bf = joinpath(CACHE_DIR, bf_filename)
        if !isfile(cache_bf)
            @info "  No BigFloat BallMatrix at K=$bf_K ($cache_bf), skipping."
            continue
        end

        @info "Loading BigFloat BallMatrix at K=$bf_K (P=$bf_P)..."
        bf_cached = deserialize(cache_bf)
        A_bf_center = bf_cached[:A_bf]
        A_bf_radius = bf_cached[:A_rad_bf]
        A_ball_bf = BallMatrix(A_bf_center, A_bf_radius)
        n_bf = size(A_bf_center, 1)
        @info "  Loaded. Matrix size: $n_bf × $n_bf"

        # Need ε_K for this BigFloat K
        eps_K_bf = if haskey(eps_K_table, bf_K)
            eps_K_table[bf_K]
        else
            @info "  Computing ε_K for K=$bf_K..."
            eps_arb = compute_Δ(bf_K; N=N_SPLITTING)
            _arb_to_float64_upper(eps_arb)
        end
        @printf("  ε_{%d} = %.6e\n", bf_K, eps_K_bf)

        # Compute BigFloat Schur data ONCE (uses GenericSchur for direct BigFloat path)
        @info "  Computing BigFloat Schur data for K=$bf_K (done once, reused for all circles)..."
        t_schur_bf = time()
        local sd_bf
        try
            sd_bf = BallArithmetic.CertifScripts.compute_schur_and_error(A_ball_bf)
        catch e
            @warn "  BigFloat Schur computation failed at K=$bf_K: $(typeof(e)): $e"
            continue
        end
        @info "  BigFloat Schur done in $(round(time()-t_schur_bf, digits=1))s"

        newly_certified_bf = Int[]
        for i in copy(uncertified)
            # Eigenvalue center
            λ_center = if bf_eigs_for_resolvent !== nothing && i ≤ length(bf_eigs_for_resolvent)
                Float64(bf_eigs_for_resolvent[i])
            else
                real(lambda_centers[i])
            end
            r_circle_bf = circle_radii[i]

            @info "  j=$i: BigFloat resolvent at K=$bf_K, λ ≈ $(@sprintf("%.4e", λ_center)), r = $(@sprintf("%.4e", r_circle_bf))..."

            local cert_bf, dt_bf
            try
                circle_bf = CertificationCircle(ComplexF64(λ_center), r_circle_bf; samples=CIRCLE_SAMPLES)
                t_bf = time()
                cert_bf = run_certification(A_ball_bf, circle_bf, workers(); schur_data=sd_bf)
                dt_bf = time() - t_bf
            catch e
                @warn "  j=$i: BigFloat resolvent failed at K=$bf_K: $(typeof(e))"
                continue
            end

            resolvent_bf = Float64(cert_bf.resolvent_original)

            # Small-gain check
            alpha_bf = setrounding(Float64, RoundUp) do
                eps_K_bf * resolvent_bf
            end

            if alpha_bf < 1.0
                denom_bf = setrounding(Float64, RoundDown) do
                    1.0 - alpha_bf
                end
                M_inf_bf = setrounding(Float64, RoundUp) do
                    resolvent_bf / denom_bf
                end

                resolvent_results[i] = ResolventResult(
                    i, ComplexF64(λ_center), bf_K, r_circle_bf,
                    resolvent_bf, alpha_bf, eps_K_bf, M_inf_bf, true)
                push!(newly_certified_bf, i)

                @printf("  j=%2d: CERTIFIED (BigFloat K=%d), α=%.4e, ‖R‖=%.4e, M_∞=%.4e [%.1fs]\n",
                        i, bf_K, alpha_bf, resolvent_bf, M_inf_bf, dt_bf)
            else
                @printf("  j=%2d: FAILED (BigFloat K=%d), α=%.4e, ‖R‖=%.4e [%.1fs]\n",
                        i, bf_K, alpha_bf, resolvent_bf, dt_bf)
            end
        end

        filter!(i -> !(i in newly_certified_bf), uncertified)
        @info "  BigFloat K=$bf_K: certified $(length(newly_certified_bf)), $(length(uncertified)) remaining."
    end
end

# Final resolvent summary
num_resolvent_certified = count(r -> r.is_certified, resolvent_results)
println()
println("="^80)
println("PHASE 1 SUMMARY")
println("="^80)
println("  Resolvent certified (Float64): $num_resolvent_f64 / $NUM_EIGS")
println("  Resolvent certified (total):   $num_resolvent_certified / $NUM_EIGS")
if !isempty(uncertified)
    println("  UNCERTIFIED eigenvalues: $uncertified")
end
println()

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1b: Tail Bound (separating circle)
# ═══════════════════════════════════════════════════════════════════════════

println("="^80)
println("PHASE 1b: TAIL BOUND")
println("="^80)
println()

# Strategy: use a separating circle around the RESOLVENT-CERTIFIED eigenvalues.
# For eigenvalues beyond the Float64 precision limit (~10^{-16}), resolvent
# certification in Float64 BallArithmetic is impossible. Instead:
#   - Tail bound separates the first N_res eigenvalues from the rest
#   - Eigenvalues beyond N_res are bounded via BigFloat Bauer-Fike (Phase 4)

N_resolvent = count(r -> r.is_certified, resolvent_results)

# Find the separating circle: between |λ_{N_res}| and |λ_{N_res+1}|
λ_tail_inner = lambda_centers[N_resolvent]
λ_tail_outer = if N_resolvent + 1 ≤ length(λ_all)
    ComplexF64(λ_all[sorted_idx_global[N_resolvent + 1]])
else
    ComplexF64(0.0)
end
abs_λ_inner = abs(λ_tail_inner)
abs_λ_outer = abs(λ_tail_outer)
ρ_tail = (abs_λ_inner + abs_λ_outer) / 2.0

@printf("  Separating first %d resolvent-certified eigenvalues\n", N_resolvent)
@printf("  |λ_%d| = %.6e,  |λ_%d| = %.6e\n", N_resolvent, abs_λ_inner, N_resolvent+1, abs_λ_outer)
@printf("  Tail circle radius ρ = %.6e\n", ρ_tail)
@printf("  Gap to λ_%d: %.6e,  gap to λ_%d: %.6e\n",
        N_resolvent, abs_λ_inner - ρ_tail, N_resolvent+1, ρ_tail - abs_λ_outer)
println()

# Adaptive: try increasing K until tail resolvent certifies
tail_certified = false
tail_resolvent_Ak = Inf
tail_M_inf = Inf
tail_alpha = Inf
tail_K = 0
tail_N_separated = N_resolvent
for K_level in K_levels
    cache_k = joinpath(CACHE_DIR, "ball_matrix_K$(K_level).jls")
    if !isfile(cache_k)
        @warn "  No cached BallMatrix at K=$K_level, skipping."
        continue
    end
    A_ball_k = deserialize(cache_k)
    eps_K = eps_K_table[K_level]

    @info "Tail bound: trying K=$K_level..."
    circle_tail = CertificationCircle(ComplexF64(0.0), ρ_tail; samples=CIRCLE_SAMPLES)

    # Compute Schur data once for tail bound at this K
    local sd_tail_k
    try
        sd_tail_k = BallArithmetic.CertifScripts.compute_schur_and_error(A_ball_k)
    catch e
        @warn "  Tail Schur failed at K=$K_level: $(typeof(e))"
        continue
    end

    local cert_tail, dt_tail
    try
        t_tail = time()
        cert_tail = run_certification(A_ball_k, circle_tail, workers(); schur_data=sd_tail_k)
        dt_tail = time() - t_tail
    catch e
        @warn "  Tail resolvent failed at K=$K_level: $(typeof(e))"
        continue
    end

    global tail_resolvent_Ak = cert_tail.resolvent_original
    global tail_alpha = setrounding(Float64, RoundUp) do
        eps_K * tail_resolvent_Ak
    end

    if tail_alpha < 1.0
        denom = setrounding(Float64, RoundDown) do
            1.0 - tail_alpha
        end
        global tail_M_inf = setrounding(Float64, RoundUp) do
            tail_resolvent_Ak / denom
        end
        global tail_K = K_level
        global tail_certified = true

        @printf("  CERTIFIED at K=%d: α=%.4e, ‖R‖=%.4f, M_∞=%.4f [%.1fs]\n",
                K_level, tail_alpha, tail_resolvent_Ak, tail_M_inf, dt_tail)
        break
    else
        @printf("  FAILED at K=%d: α=%.4e, ‖R‖=%.4f [%.1fs]\n",
                K_level, tail_alpha, tail_resolvent_Ak, dt_tail)
    end
end

if !tail_certified
    # Try BigFloat BallMatrix for the tail bound: K=256 first, then K=512
    for (bf_K, bf_P, bf_filename) in [
            (256, 512, "bigfloat_ball_matrix_K256_P512.jls"),
            (K_HIGH, P_HIGH, "bigfloat_ball_matrix_K$(K_HIGH)_P$(P_HIGH).jls")]
        if tail_certified
            break
        end
        cache_bf_tail = joinpath(CACHE_DIR, bf_filename)
        if !isfile(cache_bf_tail)
            continue
        end

        @info "Tail bound: trying BigFloat BallMatrix at K=$bf_K..."
        bf_cached_tail = deserialize(cache_bf_tail)
        A_ball_bf_tail = BallMatrix(bf_cached_tail[:A_bf], bf_cached_tail[:A_rad_bf])

        eps_K_tail = if haskey(eps_K_table, bf_K)
            eps_K_table[bf_K]
        else
            _arb_to_float64_upper(compute_Δ(bf_K; N=N_SPLITTING))
        end

        # Compute BigFloat Schur for tail bound (uses GenericSchur)
        @info "  Computing BigFloat Schur for tail bound at K=$bf_K..."
        local sd_tail_bf
        try
            t_schur_tail = time()
            sd_tail_bf = BallArithmetic.CertifScripts.compute_schur_and_error(A_ball_bf_tail)
            @info "  BigFloat Schur for tail done in $(round(time()-t_schur_tail, digits=1))s"
        catch e
            @warn "  BigFloat Schur for tail failed at K=$bf_K: $(typeof(e)): $e"
            continue
        end

        circle_tail_bf = CertificationCircle(ComplexF64(0.0), ρ_tail; samples=CIRCLE_SAMPLES)
        try
            t_tail_bf = time()
            cert_tail_bf = run_certification(A_ball_bf_tail, circle_tail_bf, workers(); schur_data=sd_tail_bf)
            dt_tail_bf = time() - t_tail_bf

            global tail_resolvent_Ak = Float64(cert_tail_bf.resolvent_original)
            global tail_alpha = setrounding(Float64, RoundUp) do
                eps_K_tail * tail_resolvent_Ak
            end

            if tail_alpha < 1.0
                denom = setrounding(Float64, RoundDown) do
                    1.0 - tail_alpha
                end
                global tail_M_inf = setrounding(Float64, RoundUp) do
                    tail_resolvent_Ak / denom
                end
                global tail_K = bf_K
                global tail_certified = true

                @printf("  CERTIFIED (BigFloat K=%d): α=%.4e, ‖R‖=%.4e, M_∞=%.4e [%.1fs]\n",
                        bf_K, tail_alpha, tail_resolvent_Ak, tail_M_inf, dt_tail_bf)
            else
                @printf("  BigFloat tail FAILED at K=%d: α=%.4e [%.1fs]\n", bf_K, tail_alpha, dt_tail_bf)
            end
        catch e
            @warn "  BigFloat tail resolvent failed at K=$bf_K: $(typeof(e))"
        end
    end
end

if tail_certified
    @printf("\n  Tail bound: ‖R_%d(n)‖₂ ≤ ρ^{n+1} · M_∞ = %.6e^{n+1} · %.6e\n",
            tail_N_separated, ρ_tail, tail_M_inf)
else
    @warn "Tail bound FAILED at all K levels including BigFloat."
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
        global num_fully_certified += 1
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
