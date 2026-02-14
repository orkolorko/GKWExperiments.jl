#!/usr/bin/env julia
# Script 2: NK refinement + spectral projectors + tail bound for 50 GKW eigenvalues.
#
# Loads Script 1 results (resolvent certification, M_inf, proj errors)
# and computes:
#   Phase 1: NK at K=256 for all 50 eigenvalues → tight (lambda_j, v_j)
#   Phase 2: BigFloat ordschur + Sylvester → ell_j(1) with NK correction
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
const CACHE_BALL_K256  = joinpath(DATA_DIR, "ball_matrix_K256.jls")
const CACHE_SCHUR_K256 = joinpath(DATA_DIR, "bigfloat_schur_K256.jls")

# Per-phase caches for this script
const CACHE_NK        = joinpath(DATA_DIR, "script2_nk_K256.jls")
const CACHE_ELL       = joinpath(DATA_DIR, "script2_ell_K256.jls")
const CACHE_TAIL      = joinpath(DATA_DIR, "script2_tail_K256.jls")
const RESULTS_PATH    = joinpath(DATA_DIR, "script2_results.jls")

# Helper: rigorous Arb -> Float64 upper bound
const _arb_to_float64_upper = GKWExperiments.NewtonKantorovichCertification._arb_to_float64_upper

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
# Load K=256 BallMatrix and BigFloat Schur
# ═══════════════════════════════════════════════════════════════════════

@info "Loading K=$K BallMatrix from $CACHE_BALL_K256..."
A_ball = Serialization.deserialize(CACHE_BALL_K256)
@info "  Size: $(size(A_ball, 1))x$(size(A_ball, 2))"

@info "Loading BigFloat Schur from $CACHE_SCHUR_K256..."
sd_bf = Serialization.deserialize(CACHE_SCHUR_K256)
S_bf = sd_bf[1]
Q_bf = Complex{BigFloat}.(S_bf.Z)
T_bf = Complex{BigFloat}.(S_bf.T)
T_diag_bf = diag(T_bf)

# Rigorous Schur error bound
errF_bf = sd_bf[2]
E_bound_bf = BigFloat(BallArithmetic.mid(errF_bf)) + BigFloat(BallArithmetic.rad(errF_bf))
@printf("  Schur ||A - QTQ'||_2 <= %.4e\n", Float64(E_bound_bf))

# Orthogonality defect
I_n = Matrix{Complex{BigFloat}}(I, n, n)
orth_mat = Q_bf' * Q_bf - I_n
delta_bf = upper_bound_L2_opnorm(BallMatrix(orth_mat))
@printf("  Orthogonality ||Q'Q-I||_2 <= %.4e\n", Float64(delta_bf))

# GenericSchur sorts by decreasing magnitude
sorted_idx = sortperm(abs.(T_diag_bf), rev=true)
println()
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════
# BigFloat ordschur utilities
# ═══════════════════════════════════════════════════════════════════════

function swap_schur_1x1!(T::AbstractMatrix, Q::AbstractMatrix, k::Int)
    nn = size(T, 1)
    a, b, c = T[k, k], T[k+1, k+1], T[k, k+1]
    x = (b - a) / c
    nrm = sqrt(one(x) + x * conj(x))
    cs, sn = one(x) / nrm, x / nrm
    for j in 1:nn
        t1, t2 = T[k, j], T[k+1, j]
        T[k, j]   = conj(cs) * t1 + conj(sn) * t2
        T[k+1, j] = -sn * t1 + cs * t2
    end
    for i in 1:nn
        t1, t2 = T[i, k], T[i, k+1]
        T[i, k]   = t1 * cs + t2 * sn
        T[i, k+1] = -t1 * conj(sn) + t2 * cs
    end
    for i in 1:nn
        q1, q2 = Q[i, k], Q[i, k+1]
        Q[i, k]   = q1 * cs + q2 * sn
        Q[i, k+1] = -q1 * conj(sn) + q2 * cs
    end
    T[k+1, k] = zero(eltype(T))
end

function bigfloat_ordschur(T, Q, target_pos::Int)
    T_ord, Q_ord = copy(T), copy(Q)
    for k in (target_pos - 1):-1:1
        swap_schur_1x1!(T_ord, Q_ord, k)
    end
    for i in 2:size(T_ord, 1), j in 1:i-1
        T_ord[i, j] = zero(eltype(T_ord))
    end
    return T_ord, Q_ord
end

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: NK certification at K=256 for all 50 eigenvalues
# ═══════════════════════════════════════════════════════════════════════

println("=" ^ 80)
println("PHASE 1: NK CERTIFICATION AT K=$K (all $NUM_EIGS eigenvalues)")
println("=" ^ 80)
println()
flush(stdout)

nk_radii     = Vector{Float64}(undef, NUM_EIGS)
nk_certified = Vector{Bool}(undef, NUM_EIGS)
nk_q0        = Vector{Float64}(undef, NUM_EIGS)

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
            certify_eigenpair_nk(A_ball; K=K, target_idx=i, N_C2=N_SPLITTING)
        catch e
            @warn "  NK j=$i failed: $(typeof(e))"
            nothing
        end
        dt = time() - t1

        if nk_result !== nothing && nk_result.is_certified
            nk_radii[i]     = nk_result.enclosure_radius
            nk_certified[i] = true
            nk_q0[i]        = nk_result.q0_bound
            @printf("  j=%2d: OK  r_NK=%.4e  q0=%.4e  [%.1fs]\n",
                    i, nk_result.enclosure_radius, nk_result.q0_bound, dt)
        else
            nk_radii[i]     = Inf
            nk_certified[i] = false
            nk_q0[i]        = nk_result !== nothing ? nk_result.q0_bound : Inf
            reason = nk_result !== nothing ? @sprintf("q0=%.2e", nk_result.q0_bound) : "error"
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
# PHASE 2: BigFloat ordschur + Sylvester → ell_j(1) with NK correction
# ═══════════════════════════════════════════════════════════════════════

println("=" ^ 80)
println("PHASE 2: ell_j(1) VIA BigFloat ORDSCHUR + SYLVESTER + NK CORRECTION")
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
    const BF_ONE = BigFloat(1)
    const BF_TWO = BigFloat(2)
    const I_m = Matrix{Complex{BigFloat}}(I, n - 1, n - 1)

    for j in 1:NUM_EIGS
        t1 = time()

        target_pos = sorted_idx[j]
        T_ord, Q_ord = bigfloat_ordschur(T_bf, Q_bf, target_pos)

        lambda_j_bf = T_ord[1, 1]
        eigenvalues_out[j] = real(lambda_j_bf)

        # Store first column of Q_ord (Schur eigenvector)
        q1_vectors[j] = Q_ord[:, 1]

        sep_bf = minimum(abs(T_ord[1,1] - T_ord[i,i]) for i in 2:n)

        N_full = triu(T_ord, 1)
        N_norm_bf = upper_bound_L2_opnorm(BallMatrix(N_full))

        # Solve Sylvester in BigFloat
        T12 = T_ord[1, 2:n]
        T22 = T_ord[2:n, 2:n]
        M_tri = UpperTriangular(T22 - lambda_j_bf * I_m)

        w = Q_ord[1, :]
        w1, w_rest = w[1], w[2:n]

        z0 = M_tri \ w_rest
        ell_val_bf = w1 - dot(T12, z0)
        ell_center[j] = real(ell_val_bf)

        # --- Rigorous error bound ---

        # 1. Sylvester solve residual
        residual = w_rest - M_tri * z0
        res_norm_bf = BigFloat(norm(residual))

        diag_M = [abs(T22[i,i] - lambda_j_bf) for i in 1:n-1]
        sigma_min_bf = minimum(real, diag_M)
        T12_norm_bf = BigFloat(norm(T12))

        local sylv_err::BigFloat
        if sigma_min_bf > 0
            sylv_err = setrounding(BigFloat, RoundUp) do
                delta_z = res_norm_bf / sigma_min_bf
                bf_n = BigFloat(n)
                T12_norm_bf * delta_z + bf_n * eps(BigFloat) * abs(ell_center[j])
            end
        else
            sylv_err = BigFloat(Inf)
        end

        # 2. Perturbation correction (Schur error E_bound)
        local pert_corr::BigFloat = setrounding(BigFloat, RoundUp) do
            rho = sep_bf / BF_TWO
            kappa = (BF_ONE + delta_bf) / max(BF_ONE - delta_bf, BigFloat(1e-300))

            R_neumann = rho > N_norm_bf ? kappa / (rho - N_norm_bf) : BigFloat(Inf)
            R_sigma = BigFloat(Inf)
            if sigma_min_bf > rho
                R22 = BF_ONE / (sigma_min_bf - rho)
                R_sigma = kappa * (BF_ONE / rho) * (BF_ONE + T12_norm_bf * R22)
            end
            R_S = min(R_neumann, R_sigma)

            if isinf(R_S) || R_S * E_bound_bf >= BF_ONE
                BigFloat(Inf)
            else
                rho * R_S^2 * E_bound_bf / (BF_ONE - R_S * E_bound_bf)
            end
        end

        # 3. NK eigenvector correction
        z0_norm_bf = BigFloat(norm(z0))
        proj_norm_bf = setrounding(BigFloat, RoundUp) do
            BF_ONE + z0_norm_bf * T12_norm_bf
        end
        local nk_corr::BigFloat
        if nk_certified[j]
            nk_corr = setrounding(BigFloat, RoundUp) do
                proj_norm_bf * BF_TWO * BigFloat(nk_radii[j])
            end
        else
            nk_corr = BigFloat(Inf)
        end

        ell_radius[j] = setrounding(BigFloat, RoundUp) do
            sylv_err + pert_corr + nk_corr
        end

        sign_ok = abs(ell_center[j]) > ell_radius[j] ? "YES" : "NO"
        dt_j = time() - t1
        @printf("  j=%2d: lam=%+.10e  ell=%+.15e +/- %.2e  sep=%.2e  sylv=%.2e  pert=%.2e  nk=%.2e  %s  [%.1fs]\n",
                j, Float64(eigenvalues_out[j]), Float64(ell_center[j]), Float64(ell_radius[j]),
                Float64(sep_bf), Float64(sylv_err), Float64(pert_corr), Float64(nk_corr), sign_ok, dt_j)
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

        q_center = zeros(Complex{BigFloat}, n)
        q_center[1] = one(Complex{BigFloat})  # e_0
        q_radius_sum = BigFloat(0)

        for j in 1:NUM_EIGS
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

        # Correction for L_r vs A_K: ||Q_N(L_r)*1 - Q_N(A_K)*e_0|| <= sum proj_error_j
        total_proj_correction = setrounding(Float64, RoundUp) do
            sum(proj_error_all[j] for j in 1:NUM_EIGS if isfinite(proj_error_all[j]); init=0.0)
        end

        norm_Q_N_1 = setrounding(Float64, RoundUp) do
            Float64(norm_Q_N_1_galerkin) + total_proj_correction
        end

        @printf("  ||Q_%d(A_K) * e_0||  <= %.6e  (Galerkin)\n", NUM_EIGS, Float64(norm_Q_N_1_galerkin))
        @printf("  Projector correction  = %.6e\n", total_proj_correction)
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
    nk_rad = nk_certified[j] ? nk_radii[j] : Inf
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
    :nk_radii => nk_radii,
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
        nk_rad = nk_certified[j] ? nk_radii[j] : Inf
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
        nk_rad = nk_certified[j] ? nk_radii[j] : Inf
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
