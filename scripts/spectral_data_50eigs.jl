#!/usr/bin/env julia
# Script 2: NK refinement + spectral projectors + tail bound for 50 GKW eigenvalues.
#
# Loads Script 1 results (resolvent certification, M_inf, proj errors)
# and computes:
#   Phase 1: NK at K=256 for all 50 eigenvalues → tight (lambda_j, v_j)
#   Phase 2: ordschur_ball + Sylvester → ell_j(1) with projector error bound
#   Phase 3: Tail bound ||R_N(n)||_2 <= rho^{n+1} * M_inf * ||Q_N * 1||
#   Phase 4: Summary + LaTeX + save
#
# All intermediate results are cached in data/ so the script can be
# interrupted and resumed.
#
# Usage:
#   julia --project --startup-file=no scripts/spectral_data_50eigs.jl

using GKWExperiments
using ArbNumerics
using BallArithmetic
using GenericSchur
using LinearAlgebra
using Printf
using Dates
using Serialization

# ═══════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════

const K         = 256
const PRECISION = 512
const NUM_EIGS  = 50
const N_SPLITTING = 5000
const n         = K + 1

setprecision(ArbFloat, PRECISION)
setprecision(BigFloat, PRECISION)

const DATA_DIR      = joinpath(@__DIR__, "..", "data")
mkpath(DATA_DIR)

# Script 1 results
const SCRIPT1_RESULTS = joinpath(DATA_DIR, "script1_results.jls")

# Cached data from Script 1 / previous runs
const CACHE_BALL_BF_K256   = joinpath(DATA_DIR, "ball_matrix_bf_K256.jls")
const CACHE_SCHUR_K256     = joinpath(DATA_DIR, "bigfloat_schur_K256.jls")
const CACHE_SCHUR_BALL_K256 = joinpath(DATA_DIR, "schur_ball_K256.jls")

# Per-phase caches for this script
const CACHE_NK        = joinpath(DATA_DIR, "script2_nk_K256.jls")
const CACHE_ELL       = joinpath(DATA_DIR, "script2_ell_K256.jls")
const CACHE_TAIL      = joinpath(DATA_DIR, "script2_tail_K256.jls")
const RESULTS_PATH    = joinpath(DATA_DIR, "script2_results.jls")

# Helper: compute ||Q_N * e_0|| (avoids Julia global-scope for-loop scoping issues)
function _compute_Q_N_norm(n, N_eigs, ell_center, ell_radius, q1_vectors, proj_error_all)
    q_center = zeros(Complex{BigFloat}, n)
    q_center[1] = one(Complex{BigFloat})
    q_radius_sum = BigFloat(0)

    for j in 1:N_eigs
        q_center .-= ell_center[j] .* q1_vectors[j]
        q1_norm = BigFloat(norm(q1_vectors[j]))
        q_radius_sum = setrounding(BigFloat, RoundUp) do
            q_radius_sum + abs(ell_radius[j]) * q1_norm
        end
    end

    norm_q_center = BigFloat(norm(real.(q_center)))
    norm_Q_N_1_galerkin = setrounding(BigFloat, RoundUp) do
        norm_q_center + q_radius_sum
    end

    total_proj_correction = setrounding(Float64, RoundUp) do
        sum(proj_error_all[j] for j in 1:N_eigs if isfinite(proj_error_all[j]); init=0.0)
    end

    norm_Q_N_1 = setrounding(Float64, RoundUp) do
        Float64(norm_Q_N_1_galerkin) + total_proj_correction
    end

    return norm_Q_N_1_galerkin, norm_Q_N_1
end

println("=" ^ 80)
println("SCRIPT 2: SPECTRAL DATA FOR $NUM_EIGS GKW EIGENVALUES")
println("  NK + ordschur + Sylvester + tail bound")
println("=" ^ 80)
println("Started: ", now())
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════
# Load Script 1 results
# ═══════════════════════════════════════════════════════════════════════

@info "Loading Script 1 results from $SCRIPT1_RESULTS..."
s1 = Serialization.deserialize(SCRIPT1_RESULTS)
eps_K_high   = s1[:eps_K_high]
M_inf_all    = s1[:M_inf_all]
proj_error_all = s1[:proj_error_all]
tail_rho     = s1[:tail_rho]
tail_M_inf   = s1[:tail_M_inf]
tail_certified = s1[:tail_certified]
resolvent_data = s1[:resolvent_data]
@printf("  eps_K=%d = %.4e\n", K, eps_K_high)
@printf("  Tail: rho=%.4e, M_inf=%.4e, certified=%s\n", tail_rho, tail_M_inf, tail_certified)
println()
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════
# Build or load K=256 BigFloat BallMatrix
# ═══════════════════════════════════════════════════════════════════════

if isfile(CACHE_BALL_BF_K256)
    @info "Loading BigFloat BallMatrix from $CACHE_BALL_BF_K256..."
    A_ball_bf = Serialization.deserialize(CACHE_BALL_BF_K256)
else
    @info "Building GKW matrix at K=$K with ArbNumerics (precision=$PRECISION)..."
    t0 = time()
    s_arb = ArbComplex(1.0, 0.0)
    M_arb = gkw_matrix_direct(s_arb; K=K)
    A_ball_bf = BallMatrix(BigFloat, M_arb)
    @info "  Built in $(round(time()-t0, digits=1))s"
    Serialization.serialize(CACHE_BALL_BF_K256, A_ball_bf)
    @info "  Cached to $CACHE_BALL_BF_K256"
end
@info "  Size: $(size(A_ball_bf, 1))x$(size(A_ball_bf, 2)), center eltype: $(eltype(BallArithmetic.mid(A_ball_bf)))"

# ── Rigorous BallMatrix Schur decomposition ──
if isfile(CACHE_SCHUR_BALL_K256)
    @info "Loading BallMatrix Schur from cache..."
    schur_ball_cache = Serialization.deserialize(CACHE_SCHUR_BALL_K256)
    Q_ball       = schur_ball_cache[:Q_ball]
    T_ball       = schur_ball_cache[:T_ball]
    E_bound_bf   = schur_ball_cache[:E_bound]
    delta_bf     = schur_ball_cache[:orth_defect]
    sorted_idx   = schur_ball_cache[:sorted_idx]
else
    # Use GenericSchur cache as seed if available, otherwise Float64 seed
    schur_seed = nothing
    if isfile(CACHE_SCHUR_K256)
        @info "Loading GenericSchur seed from $CACHE_SCHUR_K256..."
        sd_bf = Serialization.deserialize(CACHE_SCHUR_K256)
        schur_seed = (Complex{BigFloat}.(sd_bf[1].Z), Complex{BigFloat}.(sd_bf[1].T))
    end

    @info "Computing rigorous BallMatrix Schur decomposition..."
    t0 = time()
    Q_ball, T_ball, schur_result = rigorous_schur_bigfloat(
        A_ball_bf; target_precision=PRECISION, schur_seed=schur_seed)
    @printf("  Done in %.1fs  (iterations=%d, converged=%s)\n",
            time()-t0, schur_result.iterations, schur_result.converged)

    # Schur error and orthogonality defect (for display and output)
    A_mid_norm = upper_bound_L2_opnorm(BallMatrix(BallArithmetic.mid(A_ball_bf)))
    E_bound_bf = schur_result.residual_norm * A_mid_norm
    delta_bf = schur_result.orthogonality_defect

    # Sort eigenvalues by decreasing magnitude
    T_diag = diag(BallArithmetic.mid(T_ball))
    sorted_idx = sortperm(abs.(T_diag), rev=true)

    # Cache
    Serialization.serialize(CACHE_SCHUR_BALL_K256, Dict(
        :Q_ball => Q_ball, :T_ball => T_ball,
        :E_bound => E_bound_bf, :orth_defect => delta_bf,
        :sorted_idx => sorted_idx))
    @info "BallMatrix Schur cached to $CACHE_SCHUR_BALL_K256"
end

@printf("  Schur ||A - QTQ'||_F <= %.4e\n", Float64(E_bound_bf))
@printf("  Orthogonality ||Q'Q-I||_2 <= %.4e\n", Float64(delta_bf))
println()
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: NK certification at K=256 for all 50 eigenvalues
# ═══════════════════════════════════════════════════════════════════════

println("=" ^ 80)
println("PHASE 1: NK CERTIFICATION AT K=$K (all $NUM_EIGS eigenvalues, BigFloat)")
println("=" ^ 80)
println()
flush(stdout)

nk_radii     = Vector{BigFloat}(undef, NUM_EIGS)
nk_certified = Vector{Bool}(undef, NUM_EIGS)
nk_q0        = Vector{BigFloat}(undef, NUM_EIGS)

if isfile(CACHE_NK)
    @info "Loading NK results from cache..."
    nk_cache = Serialization.deserialize(CACHE_NK)
    nk_radii     .= nk_cache[:nk_radii]
    nk_certified .= nk_cache[:nk_certified]
    nk_q0        .= nk_cache[:nk_q0]
    n_nk = count(nk_certified)
    @info "  Loaded: $n_nk / $NUM_EIGS certified"
else
    for i in 1:NUM_EIGS
        t1 = time()
        nk_result = try
            certify_eigenpair_nk(A_ball_bf; K=K, target_idx=i, N_C2=N_SPLITTING)
        catch e
            @warn "  NK j=$i failed: $(typeof(e))" exception=(e, catch_backtrace())
            nothing
        end
        dt = time() - t1

        if nk_result !== nothing && nk_result.is_certified
            nk_radii[i]     = nk_result.enclosure_radius
            nk_certified[i] = true
            nk_q0[i]        = nk_result.q0_bound
            @printf("  j=%2d: OK  r_NK=%.4e  q0=%.4e  [%.1fs]\n",
                    i, Float64(nk_result.enclosure_radius), Float64(nk_result.q0_bound), dt)
        else
            nk_radii[i]     = BigFloat(Inf)
            nk_certified[i] = false
            nk_q0[i]        = nk_result !== nothing ? nk_result.q0_bound : BigFloat(Inf)
            reason = nk_result !== nothing ? @sprintf("q0=%.2e", Float64(nk_result.q0_bound)) : "error"
            @printf("  j=%2d: FAIL  %s  [%.1fs]\n", i, reason, dt)
        end
        flush(stdout)
    end

    Serialization.serialize(CACHE_NK, Dict(
        :nk_radii => nk_radii, :nk_certified => nk_certified, :nk_q0 => nk_q0))
    @info "NK results cached to $CACHE_NK"
end

n_nk_cert = count(nk_certified)
println("\nPhase 1: $n_nk_cert / $NUM_EIGS NK-certified\n")
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: ordschur_ball + Sylvester → ell_j(1) with projector error bound
# ═══════════════════════════════════════════════════════════════════════

println("=" ^ 80)
println("PHASE 2: ell_j(1) VIA ordschur_ball + SYLVESTER + PROJECTOR ERROR BOUND")
println("=" ^ 80)
println()
flush(stdout)

ell_center     = Vector{BigFloat}(undef, NUM_EIGS)
ell_radius     = Vector{BigFloat}(undef, NUM_EIGS)
eigenvalues_out = Vector{BigFloat}(undef, NUM_EIGS)
q1_vectors     = Vector{Vector{Complex{BigFloat}}}(undef, NUM_EIGS)

if isfile(CACHE_ELL)
    @info "Loading ell_j(1) results from cache..."
    ell_cache = Serialization.deserialize(CACHE_ELL)
    ell_center     .= ell_cache[:ell_center]
    ell_radius     .= ell_cache[:ell_radius]
    eigenvalues_out .= ell_cache[:eigenvalues_out]
    q1_vectors     .= ell_cache[:q1_vectors]
    n_ell = count(j -> abs(ell_center[j]) > ell_radius[j], 1:NUM_EIGS)
    @info "  Loaded: $n_ell / $NUM_EIGS sign-certified"
else
    for j in 1:NUM_EIGS
        t1 = time()

        # 1. ordschur_ball: move eigenvalue j to position 1
        target_pos = sorted_idx[j]
        select = falses(n)
        select[target_pos] = true
        ord_result = ordschur_ball(Q_ball, T_ball, select)

        T_ord_ball = ord_result.T
        Q_ord_ball = ord_result.Q
        eigenvalues_out[j] = real(BallArithmetic.mid(T_ord_ball)[1, 1])

        # Store Schur eigenvector (midpoint of first column of Q_ord)
        q1_vectors[j] = BallArithmetic.mid(Q_ord_ball)[:, 1]

        # 2. Certified Sylvester solve (BallMatrix overload — auto-propagates T radii)
        Y_transposed_ball = triangular_sylvester_miyajima_enclosure(T_ord_ball, 1)
        # Y_transposed_ball is (n-1)×1 BallMatrix; Y = transpose(Y_transposed) is the coupling

        # 3. ℓ_j(1) from spectral projector: P_S = [I, Y; 0, 0]
        Q_ord_mid = BallArithmetic.mid(Q_ord_ball)
        q = conj.(Q_ord_mid[1, :])       # Q_ord^H · e₁
        q1 = q[1]
        q_rest = q[2:end]

        Yt_mid = BallArithmetic.mid(Y_transposed_ball)[:, 1]   # (n-1) vector
        Yt_rad = BallArithmetic.rad(Y_transposed_ball)[:, 1]   # (n-1) vector

        ell_center[j] = real(q1 + transpose(Yt_mid) * q_rest)

        # 4. Sylvester error: componentwise propagation from BallMatrix radii
        sylv_err = setrounding(BigFloat, RoundUp) do
            sum(BigFloat(Yt_rad[i]) * abs(q_rest[i]) for i in 1:n-1)
        end

        # 5. Projector perturbation via resolvent bounds from Script 1
        rd = resolvent_data[j]
        proj_err = spectral_projector_error_bound(
            resolvent_bound_A = BigFloat(rd.resolvent_Ak),
            contour_radius    = BigFloat(rd.circle_radius),
            orth_defect       = BigFloat(ord_result.orth_defect),
            fact_defect       = BigFloat(ord_result.fact_defect)
        )

        # 6. NK eigenvector correction
        #    P_Schur = [I, Y; 0, 0] has ||P_Schur||₂ = sqrt(1 + ||Y||₂²)
        yt_norm = upper_bound_L2_opnorm(Y_transposed_ball)
        proj_schur_norm = setrounding(BigFloat, RoundUp) do
            sqrt(one(BigFloat) + yt_norm * yt_norm)
        end
        local nk_corr::BigFloat
        if nk_certified[j]
            nk_corr = setrounding(BigFloat, RoundUp) do
                proj_schur_norm * BigFloat(2) * BigFloat(nk_radii[j])
            end
        else
            nk_corr = BigFloat(Inf)
        end

        # 7. Total error
        ell_radius[j] = setrounding(BigFloat, RoundUp) do
            sylv_err + proj_err + nk_corr
        end

        sign_ok = abs(ell_center[j]) > ell_radius[j] ? "YES" : "NO"
        dt_j = time() - t1
        @printf("  j=%2d: lam=%+.10e  ell=%+.15e +/- %.2e  sylv=%.2e  proj=%.2e  nk=%.2e  %s  [%.1fs]\n",
                j, Float64(eigenvalues_out[j]), Float64(ell_center[j]), Float64(ell_radius[j]),
                Float64(sylv_err), Float64(proj_err), Float64(nk_corr), sign_ok, dt_j)
        flush(stdout)
    end

    Serialization.serialize(CACHE_ELL, Dict(
        :ell_center => ell_center, :ell_radius => ell_radius,
        :eigenvalues_out => eigenvalues_out, :q1_vectors => q1_vectors))
    @info "ell_j(1) results cached to $CACHE_ELL"
end

n_ell_cert = count(j -> abs(ell_center[j]) > ell_radius[j], 1:NUM_EIGS)
println("\nPhase 2: $n_ell_cert / $NUM_EIGS ell_j(1) sign-certified\n")
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Tail bound ||R_N(n)||_2 <= rho^{n+1} * M_inf * ||Q_N * 1||
# ═══════════════════════════════════════════════════════════════════════

println("=" ^ 80)
println("PHASE 3: TAIL BOUND")
println("=" ^ 80)
println()
flush(stdout)

norm_Q_N_1 = Inf
tail_prefactor = Inf

if isfile(CACHE_TAIL)
    @info "Loading tail bound from cache..."
    tail_cache = Serialization.deserialize(CACHE_TAIL)
    norm_Q_N_1 = tail_cache[:norm_Q_N_1]
    tail_prefactor = tail_cache[:tail_prefactor]
    @info "  ||Q_N * 1|| = $norm_Q_N_1, prefactor = $tail_prefactor"
else
    if !tail_certified
        @error "Tail resolvent not certified (from Script 1) — cannot compute tail bound"
    else
        # Q_N e_0 = e_0 - sum_j P_j e_0
        # where P_j e_0 = ell_j * q1_j (spectral projector applied to e_0)
        @info "Computing ||Q_$NUM_EIGS * e_0|| from spectral projectors..."

        norm_Q_N_1_galerkin, norm_Q_N_1 = _compute_Q_N_norm(
            n, NUM_EIGS, ell_center, ell_radius, q1_vectors, proj_error_all)

        @printf("  ||Q_%d(A_K) * e_0||  <= %.6e  (Galerkin)\n", NUM_EIGS, Float64(norm_Q_N_1_galerkin))
        @printf("  ||Q_%d(L_r) * 1||    <= %.6e  (rigorous)\n", NUM_EIGS, norm_Q_N_1)

        # Prefactor C = M_inf * ||Q_N * 1||
        tail_prefactor = setrounding(Float64, RoundUp) do
            tail_M_inf * norm_Q_N_1
        end

        println()
        println("TAIL BOUND: ||R_$(NUM_EIGS)(n)||_2 <= rho^{n+1} * C")
        println("-" ^ 55)
        @printf("  rho       = %.6e\n", tail_rho)
        @printf("  M_inf     = %.6e\n", tail_M_inf)
        @printf("  ||Q_%d*1|| = %.6e\n", NUM_EIGS, norm_Q_N_1)
        @printf("  C         = %.6e\n\n", tail_prefactor)

        @printf("  %5s   %20s\n", "n", "log10(bound)")
        println("-" ^ 35)
        for nn in [1, 2, 3, 5, 10, 20, 50, 100]
            log_bound = (nn + 1) * log10(tail_rho) + log10(tail_prefactor)
            @printf("  n=%3d   %.1f\n", nn, log_bound)
        end
        println("-" ^ 35)
    end

    Serialization.serialize(CACHE_TAIL, Dict(
        :norm_Q_N_1 => norm_Q_N_1, :tail_prefactor => tail_prefactor))
    @info "Tail bound cached to $CACHE_TAIL"
end
println()
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: Summary + LaTeX + save
# ═══════════════════════════════════════════════════════════════════════

println("=" ^ 80)
println("PHASE 4: SUMMARY")
println("=" ^ 80)
println()

println("-" ^ 130)
@printf("  %3s  %22s  %12s  %12s  %26s  %6s\n",
    "j", "lam_j", "NK radius", "proj error", "ell_j(1)", "sign")
println("-" ^ 130)

for j in 1:NUM_EIGS
    lam_j = Float64(eigenvalues_out[j])
    nk_rad = nk_certified[j] ? Float64(nk_radii[j]) : Inf
    pe = proj_error_all[j]
    sign_ok = abs(ell_center[j]) > ell_radius[j] ? "YES" : "NO"

    @printf("  %3d  %+22.14e  %12.4e  %12.4e  %+22.14e +/- %.2e  %6s\n",
            j, lam_j, nk_rad, pe, Float64(ell_center[j]), Float64(ell_radius[j]), sign_ok)
end
println("-" ^ 130)
println()

@printf("  NK certified:       %d / %d\n", n_nk_cert, NUM_EIGS)
@printf("  ell_j(1) sign:      %d / %d\n", n_ell_cert, NUM_EIGS)
@printf("  Tail bound valid:   %s\n", isfinite(tail_prefactor) ? "YES" : "NO")
println()

# Save unified results
Serialization.serialize(RESULTS_PATH, Dict(
    :K => K,
    :precision => PRECISION,
    :NUM_EIGS => NUM_EIGS,
    :eigenvalues => Float64.(eigenvalues_out),
    :eigenvalues_bf => eigenvalues_out,
    :nk_radii => Float64.(nk_radii),
    :nk_radii_bf => nk_radii,
    :nk_certified => nk_certified,
    :ell_center => Float64.(ell_center),
    :ell_radius => Float64.(ell_radius),
    :ell_center_bf => ell_center,
    :ell_radius_bf => ell_radius,
    :proj_error_all => proj_error_all,
    :M_inf_all => M_inf_all,
    :tail_rho => tail_rho,
    :tail_M_inf => tail_M_inf,
    :tail_norm_Q => norm_Q_N_1,
    :tail_prefactor => tail_prefactor,
    :E_bound => Float64(E_bound_bf),
))
@info "Results saved to $RESULTS_PATH"

# --- LaTeX output ---
latex_path = joinpath(DATA_DIR, "certified_spectral_data_50.tex")
open(latex_path, "w") do io
    println(io, "% Certified spectral data for $NUM_EIGS GKW eigenvalues")
    println(io, "% Generated: $(now())")
    println(io, "% K=$K, precision=$PRECISION bits")
    println(io)

    println(io, "\\begin{longtable}{rrrrrr}")
    println(io, "\\caption{Certified spectral data for the first $NUM_EIGS eigenvalues.}")
    println(io, "\\label{tab:spectral-data-50} \\\\")
    println(io, "\\toprule")
    println(io, "\$j\$ & \$\\hat\\lambda_j\$ & NK radius & \$\\ell_j(1)\$ & \$\\ell_j(1)\$ radius & sign \\\\")
    println(io, "\\midrule")
    println(io, "\\endfirsthead")
    println(io, "\\multicolumn{6}{c}{\\textit{continued}} \\\\")
    println(io, "\\toprule")
    println(io, "\$j\$ & \$\\hat\\lambda_j\$ & NK radius & \$\\ell_j(1)\$ & \$\\ell_j(1)\$ radius & sign \\\\")
    println(io, "\\midrule")
    println(io, "\\endhead")

    for j in 1:NUM_EIGS
        nk_rad = nk_certified[j] ? Float64(nk_radii[j]) : Inf
        sign_ok = abs(ell_center[j]) > ell_radius[j]
        @printf(io, "%d & \$%+.12e\$ & \$%.2e\$ & \$%+.14e\$ & \$%.2e\$ & %s \\\\\n",
                j, Float64(eigenvalues_out[j]), nk_rad,
                Float64(ell_center[j]), Float64(ell_radius[j]),
                sign_ok ? "\\checkmark" : "--")
    end

    println(io, "\\bottomrule")
    println(io, "\\end{longtable}")
    println(io)

    # Tail bound
    if isfinite(tail_prefactor)
        println(io, "% Tail bound: ||R_{$(NUM_EIGS)}(n)||_2 <= rho^{n+1} * C")
        @printf(io, "%% rho = %.6e, M_inf = %.6e, ||Q_%d * 1|| = %.6e, C = %.6e\n",
                tail_rho, tail_M_inf, NUM_EIGS, norm_Q_N_1, tail_prefactor)
    end
end
@info "LaTeX written to $latex_path"

# --- Portable text export ---
txt_path = joinpath(DATA_DIR, "spectral_coefficients_K$(K).txt")
open(txt_path, "w") do io
    println(io, "# GKW operator spectral expansion coefficients")
    println(io, "# K = $K, BigFloat precision = $PRECISION bits")
    println(io, "# ||E|| = ||A_true - QTQ'||_2 <= ", string(E_bound_bf))
    println(io, "#")
    println(io, "# Columns: j, lambda_j, ell_j(1), ell_radius, NK_radius, sign_certified")
    println(io, "#")
    for j in 1:NUM_EIGS
        sign_ok = abs(ell_center[j]) > ell_radius[j] ? "YES" : "NO"
        nk_rad = nk_certified[j] ? Float64(nk_radii[j]) : Inf
        println(io, j, "\t", string(eigenvalues_out[j]), "\t",
                string(ell_center[j]), "\t", string(ell_radius[j]), "\t",
                string(nk_rad), "\t", sign_ok)
    end
end
@info "Text export: $txt_path"

println()
println("=" ^ 80)
println("SCRIPT 2 DONE -- $(now())")
println("=" ^ 80)
