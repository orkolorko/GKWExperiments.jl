#!/usr/bin/env julia
"""
Script 1: Coarse eigenvalue enclosures + simplicity proof for the first 50 GKW eigenvalues.

Phase 0: Load/build matrices and constants
Phase 1: Resolvent certification (proves simplicity + resolvent bounds)
  1a: K=48 for eigenvalues 1-20 (standard CertifScripts)
  1b: K=256 for eigenvalues 21-50 (block Schur + CertifScripts for T22)
Phase 2: Transfer bridge + projector errors
Phase 3: Tail resolvent on separating circle (block Schur)
Phase 4: Summary + save

All heavy computations are memoized to data/*.jls files.
Rerunning the script resumes from the last completed phase.

NOTE: NK certification, l_j(1), P_j, v_j are computed in Script 2
(scripts/spectral_data_50eigs.jl) which loads results from this script.

Usage:
  julia --project --startup-file=no scripts/full_certification_50eigs.jl
"""

using Printf, Dates, Serialization

flush(stdout); flush(stderr)
println("Script 1 started: $(now())")
flush(stdout)

using GKWExperiments, BallArithmetic, ArbNumerics, LinearAlgebra
using BallArithmetic.CertifScripts: CertificationCircle, run_certification,
    compute_schur_and_error
using GenericSchur  # enables direct BigFloat Schur in compute_schur_and_error

@info "All packages loaded"
flush(stdout)

# ============================================================================
# Configuration
# ============================================================================

const PRECISION = 512
const K_LOW = 48
const K_HIGH = 256
const NUM_EIGS = 50
const N_EIGS_LOW = 20       # eigenvalues certified at K_LOW
const N_SPLITTING = 5000
const CIRCLE_SAMPLES = 256

setprecision(ArbFloat, PRECISION)
setprecision(BigFloat, PRECISION)

const DATA_DIR = joinpath(@__DIR__, "..", "data")
mkpath(DATA_DIR)

# Cache paths
const CACHE_BALL_K256 = joinpath(DATA_DIR, "ball_matrix_K256.jls")
const CACHE_SCHUR_K256 = joinpath(DATA_DIR, "bigfloat_schur_K256.jls")
const CACHE_PHASE1A = joinpath(DATA_DIR, "phase1a_resolvent_K48.jls")
const CACHE_PHASE1B = joinpath(DATA_DIR, "phase1b_schur_direct_K256.jls")
const CACHE_PHASE2 = joinpath(DATA_DIR, "phase2_transfer_bridge.jls")
const CACHE_PHASE3 = joinpath(DATA_DIR, "phase3_tail_resolvent.jls")
const RESULTS_PATH = joinpath(DATA_DIR, "script1_results.jls")

# Helper: rigorous Arb -> Float64 upper bound
const _arb_to_float64_upper = GKWExperiments.NewtonKantorovichCertification._arb_to_float64_upper

# Helper: BigFloat -> Float64 rigorous upper bound
function bf_to_f64_upper(x::BigFloat)
    f = Float64(x)
    BigFloat(f) < x && (f = nextfloat(f))
    return f
end

# Helper: Ball -> Float64 rigorous upper bound
function ball_to_f64_upper(x)
    m = abs(BallArithmetic.mid(x))
    r = BallArithmetic.rad(x)
    bf_to_f64_upper(BigFloat(m) + BigFloat(r))
end
ball_to_f64_upper(x::Float64) = x
ball_to_f64_upper(x::BigFloat) = bf_to_f64_upper(x)

s = ArbComplex(1.0, 0.0)  # Classical GKW (s=1)

# ============================================================================
# PHASE 0: Load data + constants
# ============================================================================

println("=" ^ 80)
println("SCRIPT 1: COARSE CERTIFICATION OF $NUM_EIGS GKW EIGENVALUES")
println("=" ^ 80)
println("Started: ", now())
println("K_LOW=$K_LOW, K_HIGH=$K_HIGH, NUM_EIGS=$NUM_EIGS")
println()
flush(stdout)

# --- Constants ---
@info "Computing truncation errors..."
C2_float = _arb_to_float64_upper(compute_C2(N_SPLITTING))
eps_K_low = _arb_to_float64_upper(compute_Δ(K_LOW; N=N_SPLITTING))
eps_K_high = _arb_to_float64_upper(compute_Δ(K_HIGH; N=N_SPLITTING))
@printf("  C2       <= %.6e\n", C2_float)
@printf("  eps_{K=%d} <= %.6e\n", K_LOW, eps_K_low)
@printf("  eps_{K=%d} <= %.6e\n", K_HIGH, eps_K_high)
println()
flush(stdout)

# --- K=48 BallMatrix ---
@info "Building K=$K_LOW BallMatrix..."
t0 = time()
M_arb_low = gkw_matrix_direct(s; K=K_LOW)
A_low = arb_to_ball_matrix(M_arb_low)
@info "  Built in $(round(time()-t0, digits=1))s"

# Float64 Schur for eigenvalue locations at K=48
A_center_low = BallArithmetic.mid(A_low)
S_low = schur(A_center_low)
eigenvalues_low = diag(S_low.T)
sorted_idx_low = sortperm(abs.(eigenvalues_low), rev=true)

# --- K=256 BallMatrix ---
@info "Loading K=$K_HIGH BallMatrix from $CACHE_BALL_K256..."
A_high = Serialization.deserialize(CACHE_BALL_K256)
n_high = size(A_high, 1)
@info "  Loaded: $(n_high)x$(n_high)"

# --- BigFloat Schur at K=256 ---
if isfile(CACHE_SCHUR_K256)
    @info "Loading cached BigFloat Schur from $CACHE_SCHUR_K256..."
    flush(stdout)
    sd_bf = Serialization.deserialize(CACHE_SCHUR_K256)
    @info "  Loaded"
else
    @info "Computing BigFloat Schur decomposition (will cache)..."
    flush(stdout)
    t0 = time()
    A_bf = float64_ball_to_bigfloat_ball(A_high)
    sd_bf = compute_schur_and_error(A_bf)
    @info "  Done in $(round(time()-t0, digits=1))s -- caching"
    Serialization.serialize(CACHE_SCHUR_K256, sd_bf)
end

S_bf = sd_bf[1]
norm_Z_f64 = ball_to_f64_upper(sd_bf[4])
norm_Z_inv_f64 = ball_to_f64_upper(sd_bf[5])

Q_bf = Complex{BigFloat}.(S_bf.Z)
T_bf = Complex{BigFloat}.(S_bf.T)
T_diag_bf = diag(T_bf)
n_bf = size(T_bf, 1)

# Verify GenericSchur ordering for the first NUM_EIGS positions
# (positions beyond NUM_EIGS may have nearly-degenerate eigenvalues with swapped order)
for i in 1:NUM_EIGS
    if abs(T_diag_bf[i]) < abs(T_diag_bf[i+1]) - 1e-10 * abs(T_diag_bf[i+1])
        @warn "Schur diagonal not sorted by magnitude at position $i" abs_i=Float64(abs(T_diag_bf[i])) abs_ip1=Float64(abs(T_diag_bf[i+1]))
    end
end
@info "Schur diagonal checked for first $NUM_EIGS positions"

# Rigorous Schur error bound
errF_bf = sd_bf[2]
E_bound_bf = BigFloat(BallArithmetic.mid(errF_bf)) + BigFloat(BallArithmetic.rad(errF_bf))
@printf("  Schur ||A - QTQ'||_2 <= %.4e\n", Float64(E_bound_bf))

# Eigenvalues from BigFloat Schur diagonal
eigenvalues_bf = [real(T_diag_bf[j]) for j in 1:NUM_EIGS]

println("\nEigenvalues (BigFloat Schur diagonal):")
for i in [1, 2, 5, 10, 20, 21, 30, 40, 50]
    i > n_bf && continue
    @printf("  j=%2d: lam = %+.15e  |lam| = %.6e\n",
            i, Float64(real(T_diag_bf[i])), Float64(abs(T_diag_bf[i])))
end
println()
flush(stdout)

# ============================================================================
# PHASE 1a: Circle certification at K=48 (eigenvalues 1-20)
# ============================================================================

println("=" ^ 80)
println("PHASE 1a: RESOLVENT CERTIFICATION AT K=$K_LOW (eigenvalues 1-$N_EIGS_LOW)")
println("=" ^ 80)
println()
flush(stdout)

# Storage: (circle_radius, resolvent_Ak, alpha, M_inf, certified)
resolvent_data = Dict{Int, @NamedTuple{circle_radius::Float64, resolvent_Ak::Float64,
    alpha::Float64, M_inf::Float64, certified::Bool}}()

if isfile(CACHE_PHASE1A)
    @info "Loading Phase 1a from cache..."
    phase1a_cache = Serialization.deserialize(CACHE_PHASE1A)
    for (k, v) in phase1a_cache
        resolvent_data[k] = v
    end
    n_cert_low = count(i -> resolvent_data[i].certified, 1:N_EIGS_LOW)
    @info "  Loaded $n_cert_low / $N_EIGS_LOW certified"
else
    for i in 1:N_EIGS_LOW
        idx = sorted_idx_low[i]
        lam_center = ComplexF64(eigenvalues_low[idx])

        # Gap-based circle radius
        circle_radius = max(abs(lam_center) * 0.01, eps_K_low * 10)
        for j in 1:min(NUM_EIGS, length(eigenvalues_low))
            j == i && continue
            other_idx = sorted_idx_low[j]
            dist = abs(lam_center - eigenvalues_low[other_idx])
            circle_radius = min(circle_radius, dist / 3)
        end

        circle = CertificationCircle(lam_center, circle_radius; samples=CIRCLE_SAMPLES)
        t1 = time()
        cert_data = run_certification(A_low, circle; log_io=devnull)
        dt = time() - t1

        resolvent_Ak = cert_data.resolvent_original
        alpha = setrounding(Float64, RoundUp) do; eps_K_low * resolvent_Ak; end
        is_cert = alpha < 1.0

        if is_cert
            denom = setrounding(Float64, RoundDown) do; 1.0 - alpha; end
            M_inf = setrounding(Float64, RoundUp) do; resolvent_Ak / denom; end
        else
            M_inf = Inf
        end

        resolvent_data[i] = (circle_radius=circle_radius, resolvent_Ak=resolvent_Ak,
                              alpha=alpha, M_inf=M_inf, certified=is_cert)

        status = is_cert ? "OK" : "FAIL"
        @printf("  j=%2d: lam=%+.10e  r=%.2e  ||R||=%.4f  alpha=%.2e  M_inf=%.4f  %s  [%.1fs]\n",
                i, real(lam_center), circle_radius, resolvent_Ak, alpha, M_inf, status, dt)
        flush(stdout)
    end

    # Save checkpoint
    phase1a_save = Dict(i => resolvent_data[i] for i in 1:N_EIGS_LOW)
    Serialization.serialize(CACHE_PHASE1A, phase1a_save)
    @info "Phase 1a cached to $CACHE_PHASE1A"

    n_cert_low = count(i -> resolvent_data[i].certified, 1:N_EIGS_LOW)
end

println("\nPhase 1a: $n_cert_low / $N_EIGS_LOW certified at K=$K_LOW\n")
flush(stdout)

# ============================================================================
# PHASE 1b: Schur direct certification at K=256 (eigenvalues 21-50)
# ============================================================================

println("=" ^ 80)
println("PHASE 1b: SCHUR DIRECT AT K=$K_HIGH (eigenvalues $(N_EIGS_LOW+1)-$NUM_EIGS)")
println("  Block Schur (no ordschur) + CertifScripts for T22")
println("=" ^ 80)
println()
flush(stdout)

# Track which eigenvalues were certified at K_HIGH (vs K_LOW)
certified_at_high = Set{Int}()

schur_direct_results = Dict{Int, Any}()

if isfile(CACHE_PHASE1B)
    @info "Loading Phase 1b from cache..."
    phase1b_cache = Serialization.deserialize(CACHE_PHASE1B)
    schur_direct_results = phase1b_cache[:schur_direct_results]
    for (j, rd) in phase1b_cache[:resolvent_entries]
        resolvent_data[j] = rd
        rd.certified && push!(certified_at_high, j)
    end
    n_cert_high = count(j -> haskey(resolvent_data, j) && resolvent_data[j].certified, (N_EIGS_LOW+1):NUM_EIGS)
    @info "  Loaded $n_cert_high / $(NUM_EIGS - N_EIGS_LOW) certified"
else
    for j in (N_EIGS_LOW+1):NUM_EIGS
        t1 = time()
        result = try
            certify_eigenvalue_schur_direct(
                A_high, j; K=K_HIGH, schur_data_bf=sd_bf, circle_samples=CIRCLE_SAMPLES)
        catch e
            @warn "  j=$j: certify_eigenvalue_schur_direct FAILED" exception=(e, catch_backtrace())
            nothing
        end
        dt = time() - t1

        if result !== nothing
            schur_direct_results[j] = result

            if result.is_certified
                denom = setrounding(Float64, RoundDown) do; 1.0 - result.small_gain_factor; end
                M_inf = setrounding(Float64, RoundUp) do; result.resolvent_Mr / denom; end
                push!(certified_at_high, j)
            else
                M_inf = Inf
            end
            resolvent_data[j] = (circle_radius=result.circle_radius, resolvent_Ak=result.resolvent_Mr,
                                  alpha=result.small_gain_factor, M_inf=M_inf, certified=result.is_certified)

            status = result.is_certified ? "OK" : "FAIL"
            @printf("  j=%2d: lam=%+.6e  alpha=%.2e  Mr=%.2e  r=%.2e  %s  [%.1fs]\n",
                    j, real(result.eigenvalue_center), result.small_gain_factor,
                    result.resolvent_Mr, result.circle_radius, status, dt)
        else
            resolvent_data[j] = (circle_radius=0.0, resolvent_Ak=Inf,
                                  alpha=Inf, M_inf=Inf, certified=false)
            @printf("  j=%2d: ERROR  [%.1fs]\n", j, dt)
        end
        flush(stdout)
    end

    # Save checkpoint
    resolvent_entries = Dict(j => resolvent_data[j] for j in (N_EIGS_LOW+1):NUM_EIGS if haskey(resolvent_data, j))
    Serialization.serialize(CACHE_PHASE1B, Dict(
        :schur_direct_results => schur_direct_results,
        :resolvent_entries => resolvent_entries))
    @info "Phase 1b cached to $CACHE_PHASE1B"

    n_cert_high = count(j -> haskey(resolvent_data, j) && resolvent_data[j].certified, (N_EIGS_LOW+1):NUM_EIGS)
end

n_cert_total = count(j -> haskey(resolvent_data, j) && resolvent_data[j].certified, 1:NUM_EIGS)
println("\nPhase 1b: $n_cert_high / $(NUM_EIGS - N_EIGS_LOW) certified at K=$K_HIGH")
println("Total resolvent certified: $n_cert_total / $NUM_EIGS\n")
flush(stdout)

# ============================================================================
# PHASE 1c: Fallback — certify any Phase 1a failures at K=256 block Schur
# ============================================================================

failed_1a = [j for j in 1:N_EIGS_LOW if !resolvent_data[j].certified]
if !isempty(failed_1a)
    println("=" ^ 80)
    println("PHASE 1c: FALLBACK K=$K_HIGH FOR FAILED K=$K_LOW EIGENVALUES $failed_1a")
    println("=" ^ 80)
    println()
    flush(stdout)

    for j in failed_1a
        t1 = time()
        result = try
            certify_eigenvalue_schur_direct(
                A_high, j; K=K_HIGH, schur_data_bf=sd_bf, circle_samples=CIRCLE_SAMPLES)
        catch e
            @warn "  j=$j: fallback certify FAILED" exception=(e, catch_backtrace())
            nothing
        end
        dt = time() - t1

        if result !== nothing && result.is_certified
            denom = setrounding(Float64, RoundDown) do; 1.0 - result.small_gain_factor; end
            M_inf = setrounding(Float64, RoundUp) do; result.resolvent_Mr / denom; end
            resolvent_data[j] = (circle_radius=result.circle_radius, resolvent_Ak=result.resolvent_Mr,
                                  alpha=result.small_gain_factor, M_inf=M_inf, certified=true)
            push!(certified_at_high, j)
            @printf("  j=%2d: lam=%+.6e  alpha=%.2e  Mr=%.2e  r=%.2e  OK  [%.1fs]\n",
                    j, real(result.eigenvalue_center), result.small_gain_factor,
                    result.resolvent_Mr, result.circle_radius, dt)
        elseif result !== nothing
            @printf("  j=%2d: lam=%+.6e  alpha=%.2e  FAIL  [%.1fs]\n",
                    j, real(result.eigenvalue_center), result.small_gain_factor, dt)
        else
            @printf("  j=%2d: ERROR  [%.1fs]\n", j, dt)
        end
        flush(stdout)
    end

    n_cert_total = count(j -> haskey(resolvent_data, j) && resolvent_data[j].certified, 1:NUM_EIGS)
    println("\nAfter Phase 1c: $n_cert_total / $NUM_EIGS certified total\n")
    flush(stdout)
end

# ============================================================================
# PHASE 2: Transfer bridge + Projector errors
# ============================================================================

println("=" ^ 80)
println("PHASE 2: TRANSFER BRIDGE + PROJECTOR ERRORS")
println("=" ^ 80)
println()
flush(stdout)

M_inf_all = Vector{Float64}(undef, NUM_EIGS)
proj_error_all = Vector{Float64}(undef, NUM_EIGS)

if isfile(CACHE_PHASE2)
    @info "Loading Phase 2 from cache..."
    phase2_cache = Serialization.deserialize(CACHE_PHASE2)
    M_inf_all .= phase2_cache[:M_inf_all]
    proj_error_all .= phase2_cache[:proj_error_all]
    @info "  Loaded"
else
    for j in 1:NUM_EIGS
        rd = resolvent_data[j]
        if !rd.certified
            M_inf_all[j] = Inf
            proj_error_all[j] = Inf
            continue
        end

        if j in certified_at_high
            # Already certified at K_HIGH: resolvent_Ak is the K_HIGH resolvent directly
            M_inf_all[j] = rd.M_inf
            contour_length = 2pi * rd.circle_radius
            pe, pe_valid = projector_approximation_error_rigorous(contour_length, rd.resolvent_Ak, eps_K_high)
            proj_error_all[j] = pe_valid ? pe : Inf
        else
            # Certified at K_LOW: reverse transfer to K_HIGH
            resolvent_Ak_high, alpha_high, valid = reverse_transfer_resolvent_bound(rd.M_inf, eps_K_high)
            M_inf_all[j] = rd.M_inf
            if valid
                contour_length = 2pi * rd.circle_radius
                pe, pe_valid = projector_approximation_error_rigorous(contour_length, resolvent_Ak_high, eps_K_high)
                proj_error_all[j] = pe_valid ? pe : Inf
            else
                proj_error_all[j] = Inf
            end
        end
    end

    Serialization.serialize(CACHE_PHASE2, Dict(
        :M_inf_all => M_inf_all, :proj_error_all => proj_error_all))
    @info "Phase 2 cached to $CACHE_PHASE2"
end

@info "Transfer bridge + projector errors computed"
for j in [1, 5, 10, 20, 21, 30, 40, 50]
    j > NUM_EIGS && continue
    @printf("  j=%2d: M_inf=%.4e  proj_err=%.4e\n", j, M_inf_all[j], proj_error_all[j])
end
println()
flush(stdout)

# ============================================================================
# PHASE 3: Tail resolvent on separating circle (block Schur)
# ============================================================================

println("=" ^ 80)
println("PHASE 3: TAIL RESOLVENT ON SEPARATING CIRCLE")
println("  (full tail bound needs projectors from Script 2)")
println("=" ^ 80)
println()
flush(stdout)

tail_certified = false
M_inf_tail = Inf
rho_tail = 0.0

if isfile(CACHE_PHASE3)
    @info "Loading Phase 3 from cache..."
    phase3_cache = Serialization.deserialize(CACHE_PHASE3)
    tail_certified = phase3_cache[:tail_certified]
    M_inf_tail = phase3_cache[:M_inf_tail]
    rho_tail = phase3_cache[:rho_tail]
    @info "  Loaded: tail_certified=$tail_certified, rho=$rho_tail, M_inf=$M_inf_tail"
else
    lam_N_bf = abs(T_diag_bf[NUM_EIGS])
    lam_Np1_bf = abs(T_diag_bf[NUM_EIGS + 1])
    lam_N_f64 = Float64(lam_N_bf)
    lam_Np1_f64 = Float64(lam_Np1_bf)
    rho_tail = (lam_N_f64 + lam_Np1_f64) / 2.0

    @printf("  |lam_%d| = %.6e,  |lam_%d| = %.6e\n", NUM_EIGS, lam_N_f64, NUM_EIGS+1, lam_Np1_f64)
    @printf("  rho_tail = %.6e  (arithmetic mean)\n", rho_tail)
    @printf("  Gap to lam_%d: %.6e,  gap to lam_%d: %.6e\n\n",
            NUM_EIGS, lam_N_f64 - rho_tail, NUM_EIGS+1, rho_tail - lam_Np1_f64)

    # Block Schur split at position NUM_EIGS
    k_tail = NUM_EIGS
    T11_tail_bf = T_bf[1:k_tail, 1:k_tail]
    T12_tail_bf = T_bf[1:k_tail, (k_tail+1):end]
    T22_tail_bf = T_bf[(k_tail+1):end, (k_tail+1):end]

    # ||T12||_F
    T12_tail_norm_bf = sqrt(sum(abs.(T12_tail_bf) .^ 2))
    T12_tail_norm = bf_to_f64_upper(T12_tail_norm_bf)
    @printf("  ||T12||_F = %.6e\n", T12_tail_norm)

    # sigma_min(T11) via BigFloat SVD + Miyajima certification
    @info "BigFloat SVD of T11 ($(k_tail)x$(k_tail))..."
    t0 = time()
    T11_center = Complex{BigFloat}.(T11_tail_bf)
    T11_ball = BallMatrix(T11_center)
    svdA_T11 = svd(T11_center)
    sv11_result = BallArithmetic._certify_svd(
        T11_ball, svdA_T11, BallArithmetic.MiyajimaM1(); apply_vbd=true)
    sig11_center_ball = sv11_result.singular_values[end]
    sig11_lower = max(Float64(BallArithmetic.mid(sig11_center_ball) - BallArithmetic.rad(sig11_center_ball)), 0.0)
    @printf("  sigma_min(T11) >= %.6e  [%.1fs]\n", sig11_lower, time()-t0)

    sig11_on_circle = sig11_lower - rho_tail
    @printf("  sigma_min(zI - T11) >= sigma_min(T11) - rho = %.6e\n", sig11_on_circle)

    if sig11_on_circle <= 0
        @error "sigma_min(zI-T11) <= 0: tail circle too close to T11 spectrum"
    else
        r11_tail = setrounding(Float64, RoundUp) do; 1.0 / sig11_on_circle; end

        # T22 via CertifScripts (with svdbox fallback if Inf or exception)
        @info "Certifying T22 resolvent on tail circle..."
        T22_tail_f64 = bigfloat_ball_to_float64_ball(BallMatrix(T22_tail_bf))
        circle_tail = CertificationCircle(ComplexF64(0.0), rho_tail; samples=CIRCLE_SAMPLES)
        t0 = time()

        cert_T22_tail = try
            run_certification(T22_tail_f64, circle_tail; log_io=devnull)
        catch e
            @warn "  T22: CertifScripts threw" exception=(e,)
            nothing
        end

        if cert_T22_tail !== nothing && isfinite(cert_T22_tail.resolvent_original)
            max_res_T22_tail = cert_T22_tail.resolvent_original
            @info "  T22: CertifScripts succeeded"
        else
            @info "  T22: CertifScripts cannot certify, using manual svdbox scan"
            d_max = setrounding(Float64, RoundUp) do
                2.0 * rho_tail * sin(π / (2 * CIRCLE_SAMPLES))
            end
            max_res_T22_tail = 0.0
            for s_idx in 0:(CIRCLE_SAMPLES - 1)
                θ = 2π * s_idx / CIRCLE_SAMPLES
                z = ComplexF64(rho_tail * cos(θ), rho_tail * sin(θ))
                sv = svdbox(T22_tail_f64 - z * I)
                σ_ball = sv[end]
                σ_at_sample = max(Float64(BallArithmetic.mid(σ_ball)) -
                                  Float64(BallArithmetic.rad(σ_ball)), 0.0)
                σ_lower = σ_at_sample - d_max
                if σ_lower <= 0
                    @error "σ_min(zI-T22) ≤ 0 at tail circle sample s=$s_idx" σ_at_sample d_max
                    max_res_T22_tail = Inf; break
                end
                r22 = setrounding(Float64, RoundUp) do; 1.0 / σ_lower; end
                max_res_T22_tail = max(max_res_T22_tail, r22)
            end
        end
        @printf("  T22 resolvent <= %.6e  [%.1fs]\n", max_res_T22_tail, time()-t0)

        # Block formula
        max_resolvent_tail = setrounding(Float64, RoundUp) do
            r11_tail * (1.0 + T12_tail_norm * max_res_T22_tail) + max_res_T22_tail
        end

        # Schur bridge
        resolvent_Ak_tail = setrounding(Float64, RoundUp) do
            norm_Z_f64 * max_resolvent_tail * norm_Z_inv_f64
        end

        # Small-gain
        alpha_tail = setrounding(Float64, RoundUp) do; eps_K_high * resolvent_Ak_tail; end

        @printf("\n  Block resolvent  <= %.6e\n", max_resolvent_tail)
        @printf("  Resolvent A_K    <= %.6e\n", resolvent_Ak_tail)
        @printf("  alpha = eps_K * ||R|| = %.6e\n", alpha_tail)

        if alpha_tail < 1.0
            denom_tail = setrounding(Float64, RoundDown) do; 1.0 - alpha_tail; end
            M_inf_tail = setrounding(Float64, RoundUp) do; resolvent_Ak_tail / denom_tail; end
            tail_certified = true
            @printf("  M_inf = ||R_{L_r}|| <= %.6e  CERTIFIED\n", M_inf_tail)
        else
            @error "Tail small-gain FAILED: alpha = $alpha_tail >= 1"
        end
    end

    Serialization.serialize(CACHE_PHASE3, Dict(
        :tail_certified => tail_certified, :M_inf_tail => M_inf_tail, :rho_tail => rho_tail))
    @info "Phase 3 cached to $CACHE_PHASE3"
end
println()
flush(stdout)

# ============================================================================
# PHASE 4: Summary + Save
# ============================================================================

println("=" ^ 80)
println("PHASE 4: SUMMARY")
println("=" ^ 80)
println()

println("-" ^ 95)
@printf("  %3s  %22s  %12s  %12s  %12s  %12s  %6s\n",
    "j", "lam_j", "circle_r", "||R_Ak||", "alpha", "M_inf", "cert")
println("-" ^ 95)

for j in 1:NUM_EIGS
    lam_j = Float64(eigenvalues_bf[j])
    rd = resolvent_data[j]
    status = rd.certified ? "YES" : "NO"
    @printf("  %3d  %+22.14e  %12.4e  %12.4e  %12.4e  %12.4e  %6s\n",
            j, lam_j, rd.circle_radius, rd.resolvent_Ak, rd.alpha, rd.M_inf, status)
end
println("-" ^ 95)
println()

n_resolvent = count(j -> resolvent_data[j].certified, 1:NUM_EIGS)
@printf("  Resolvent certified: %d / %d\n", n_resolvent, NUM_EIGS)
@printf("  Tail resolvent:      %s (rho=%.4e, M_inf=%.4e)\n",
        tail_certified ? "YES" : "NO", rho_tail, M_inf_tail)
println()

# Save unified results for Script 2
results_data = Dict(
    :NUM_EIGS => NUM_EIGS,
    :K_LOW => K_LOW,
    :K_HIGH => K_HIGH,
    :eps_K_low => eps_K_low,
    :eps_K_high => eps_K_high,
    :C2_float => C2_float,
    :eigenvalues_bf => Float64.(eigenvalues_bf),
    :resolvent_data => Dict(j => resolvent_data[j] for j in 1:NUM_EIGS),
    :M_inf_all => M_inf_all,
    :proj_error_all => proj_error_all,
    :tail_rho => rho_tail,
    :tail_M_inf => M_inf_tail,
    :tail_certified => tail_certified,
    :norm_Z => norm_Z_f64,
    :norm_Z_inv => norm_Z_inv_f64,
    :n => n_bf,
)
Serialization.serialize(RESULTS_PATH, results_data)
@info "Results saved to $RESULTS_PATH"

println()
println("=" ^ 80)
println("SCRIPT 1 DONE -- $(now())")
println("=" ^ 80)
