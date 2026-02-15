# BigFloat spectral computation for K=2048, P=2048, 50 eigenvalues.
# NK certification + manual ordschur + Sylvester + standalone perturbation bound.
# Designed to run on ibis (32 cores, 125 GB RAM).
#
# Estimated timings (based on K=512/P=1024 scaling):
#   Matrix build:   ~50 min  (O(K²) assembly + O(K) Hurwitz zeta)
#   GenericSchur:   ~10-20 hours  (O(n³) at 2048-bit via GenericSchur)
#   SVD bounds:     ~hours  (optional, one-time O(n³) per matrix)
#   NK:             ~minutes  (tail bound negligible at K=2048)
#   ordschur loop:  ~2-3 days  (50 × O(n²) BigFloat Givens + Sylvester)
#
# Usage:
#   julia --project --startup-file=no -t auto scripts/bigfloat_spectral_K2048.jl
#
# All intermediate results are cached in data/ so the script can be
# interrupted and resumed.

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
const K           = 2048
const PRECISION   = 2048          # bits (≈616 decimal digits)
const NUM_EIGS    = 50
const N_SPLITTING = 5000          # C₂ splitting parameter for NK
const n           = K + 1         # matrix dimension

# Use SVD bounds for one-time norms (tighter but VERY slow for BigFloat)
# upper_bound_L2_opnorm is a good middle ground: tighter than Frobenius, much faster than SVD
const USE_SVD_BOUNDS = false

setprecision(ArbFloat, PRECISION)
setprecision(BigFloat, PRECISION)

const DATA_DIR = joinpath(@__DIR__, "..", "data")
mkpath(DATA_DIR)

# Cache paths (all K=2048, P=2048 specific)
const CACHE_BALL_BF   = joinpath(DATA_DIR, "ball_matrix_bf_K$(K)_P$(PRECISION).jls")
const CACHE_SCHUR     = joinpath(DATA_DIR, "bigfloat_schur_K$(K)_P$(PRECISION).jls")
const CACHE_SCHUR_CERT = joinpath(DATA_DIR, "schur_cert_K$(K)_P$(PRECISION).jls")
const CACHE_NK        = joinpath(DATA_DIR, "nk_K$(K)_P$(PRECISION).jls")
const CACHE_ELL       = joinpath(DATA_DIR, "ell_K$(K)_P$(PRECISION).jls")
const RESULTS_PATH    = joinpath(DATA_DIR, "spectral_K$(K)_P$(PRECISION).jls")

println("=" ^ 80)
println("K=$K SPECTRAL DATA: NK + ordschur + SYLVESTER ($NUM_EIGS eigenvalues)")
println("  Precision: $PRECISION bits (≈$(round(Int, PRECISION * log10(2))) decimal digits)")
println("  SVD bounds: $USE_SVD_BOUNDS")
println("=" ^ 80)
println("Started: ", now())
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════
# Manual BigFloat ordschur (proven fast for large K)
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
# Helper: compute rigorous operator norm (SVD or fast bound)
# ═══════════════════════════════════════════════════════════════════════

function rigorous_opnorm(M::BallMatrix, label::String; use_svd::Bool=USE_SVD_BOUNDS)
    fast_bound = upper_bound_L2_opnorm(M)
    if !use_svd
        return fast_bound
    end
    t0 = time()
    svd_bound = svd_bound_L2_opnorm(M)
    dt = time() - t0
    ratio = Float64(fast_bound / svd_bound)
    @printf("    %s: fast=%.4e, svd=%.4e (ratio=%.1fx) [%.1fs]\n",
            label, Float64(fast_bound), Float64(svd_bound), ratio, dt)
    flush(stdout)
    return svd_bound
end

# ═══════════════════════════════════════════════════════════════════════
# Step 1: Build or load BigFloat BallMatrix
# ═══════════════════════════════════════════════════════════════════════

local A_ball_bf
if isfile(CACHE_BALL_BF)
    @info "Loading BigFloat BallMatrix from cache..."
    t0 = time()
    A_ball_bf = Serialization.deserialize(CACHE_BALL_BF)
    @printf("  Loaded in %.1fs\n", time()-t0)
else
    @info "Building GKW matrix at K=$K with ArbNumerics (precision=$PRECISION)..."
    t0 = time()
    s_arb = ArbComplex(1.0, 0.0)
    M_arb = gkw_matrix_direct_fast(s_arb; K=K, threaded=true)
    @printf("  Arb matrix built in %.1fs\n", time()-t0)

    @info "Converting ArbNumerics → BigFloat BallMatrix..."
    t0 = time()
    A_ball_bf = BallMatrix(BigFloat, M_arb)
    @printf("  Converted in %.1fs\n", time()-t0)

    Serialization.serialize(CACHE_BALL_BF, A_ball_bf)
    @info "  Cached to $CACHE_BALL_BF"
end
@info "  Size: $(size(A_ball_bf, 1))×$(size(A_ball_bf, 2)), center eltype: $(eltype(BallArithmetic.mid(A_ball_bf)))"
println()
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════
# Step 2: Rigorous Schur decomposition
# ═══════════════════════════════════════════════════════════════════════

local Q_bf, T_bf, E_bound_bf, delta_bf, sorted_idx
if isfile(CACHE_SCHUR_CERT)
    @info "Loading certified Schur results from cache..."
    cert_cache = Serialization.deserialize(CACHE_SCHUR_CERT)
    Q_bf       = cert_cache[:Q_bf]
    T_bf       = cert_cache[:T_bf]
    E_bound_bf = cert_cache[:E_bound]
    delta_bf   = cert_cache[:orth_defect]
    sorted_idx = cert_cache[:sorted_idx]
else
    @info "Computing rigorous Schur decomposition..."
    t0 = time()
    A_real_center = real.(BallArithmetic.mid(A_ball_bf))

    # Check if GenericSchur cache exists
    need_recompute = true
    if isfile(CACHE_SCHUR)
        sd_bf = Serialization.deserialize(CACHE_SCHUR)
        cached_prec = precision(real(Complex{BigFloat}(sd_bf[:Q_bf][1,1])))
        if cached_prec >= PRECISION
            @info "Loading GenericSchur seed from cache (precision=$cached_prec bits)..."
            need_recompute = false
        else
            @info "GenericSchur cache has $cached_prec-bit precision, need $PRECISION — recomputing..."
        end
    end

    if need_recompute
        @info "Computing GenericSchur decomposition at $PRECISION-bit precision..."
        @info "  Matrix size: $n × $n — this may take HOURS..."
        t_schur = time()
        F_gs = schur(Complex{BigFloat}.(A_real_center))
        @printf("  GenericSchur done in %.1fs (%.1f hours)\n",
                time()-t_schur, (time()-t_schur)/3600)
        sd_bf = Dict(:Q_bf => F_gs.Z, :T_bf => F_gs.T)
        Serialization.serialize(CACHE_SCHUR, sd_bf)
        @info "  Cached to $CACHE_SCHUR"
    end

    Q_bf = Complex{BigFloat}.(sd_bf[:Q_bf])
    T_bf = Complex{BigFloat}.(sd_bf[:T_bf])

    # Compute Schur quality using rigorous norms
    @info "Computing Schur quality metrics..."
    A_complex = Complex{BigFloat}.(A_real_center)
    residual_mat = A_complex - Q_bf * T_bf * Q_bf'
    orth_mat = Q_bf' * Q_bf - Matrix{Complex{BigFloat}}(I, n, n)

    res_opnorm = rigorous_opnorm(BallMatrix(residual_mat), "||A-QTQ'||")
    orth_def   = rigorous_opnorm(BallMatrix(orth_mat), "||Q'Q-I||")
    A_norm     = rigorous_opnorm(BallMatrix(A_complex), "||A||")

    residual_norm = res_opnorm / A_norm
    delta_bf = orth_def

    @printf("  Residual ||A-QTQ'||_2           ≤ %.4e\n", Float64(res_opnorm))
    @printf("  Residual ||A-QTQ'||_2 / ||A||_2 ≤ %.4e\n", Float64(residual_norm))
    @printf("  Orthogonality ||Q'Q-I||_2       ≤ %.4e\n", Float64(delta_bf))

    # E_bound from Schur residual
    E_bound_bf = res_opnorm

    # Sort eigenvalues by decreasing magnitude
    lambda_all = real.(diag(T_bf))
    sorted_idx = sortperm(abs.(lambda_all), rev=true)

    @printf("  Step 2 done in %.1fs (%.1f hours)\n", time()-t0, (time()-t0)/3600)

    # Cache
    Serialization.serialize(CACHE_SCHUR_CERT, Dict(
        :Q_bf => Q_bf, :T_bf => T_bf,
        :E_bound => E_bound_bf, :orth_defect => delta_bf,
        :sorted_idx => sorted_idx))
    @info "Certified Schur cached to $CACHE_SCHUR_CERT"
end

@printf("  Schur ||A - QTQ'||_2 ≤ %.4e\n", Float64(E_bound_bf))
@printf("  Orthogonality ||Q'Q-I||_2 ≤ %.4e\n", Float64(delta_bf))
println()
flush(stdout)

# Rigorous total E_bound (Schur residual + Arb→BigFloat conversion)
@info "Computing ||A_rad||₂..."
A_rad_opnorm = rigorous_opnorm(BallMatrix(BallArithmetic.rad(A_ball_bf)), "||A_rad||")
E_bound_total = setrounding(BigFloat, RoundUp) do
    E_bound_bf + A_rad_opnorm
end
@printf("  Total ||E|| = ||A_true - QTQ'||_2 ≤ %.4e\n", Float64(E_bound_total))
@printf("    Schur residual = %.4e, input ||A_rad||_2 = %.4e\n",
        Float64(E_bound_bf), Float64(A_rad_opnorm))
println()
flush(stdout)

# Print first eigenvalues
println("First $NUM_EIGS eigenvalues (sorted by |λ|):")
println("-"^60)
lambda_sorted = real.(diag(T_bf))[sorted_idx]
for i in 1:NUM_EIGS
    @printf("  %3d: λ = %+.15e  |λ| = %.10e\n",
            i, Float64(lambda_sorted[i]), Float64(abs(lambda_sorted[i])))
end
println()
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: NK certification
# ═══════════════════════════════════════════════════════════════════════

println("=" ^ 80)
println("PHASE 1: NK CERTIFICATION AT K=$K (all $NUM_EIGS eigenvalues)")
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
# PHASE 2: ordschur + Sylvester → ℓ_j(1) with perturbation bound
# ═══════════════════════════════════════════════════════════════════════

println("=" ^ 80)
println("PHASE 2: ℓ_j(1) VIA ordschur + SYLVESTER + PERTURBATION BOUND")
println("=" ^ 80)
println()
flush(stdout)

const BF_ONE = one(BigFloat)
const BF_TWO = BigFloat(2)
const I_m = Matrix{Complex{BigFloat}}(I, n - 1, n - 1)

ell_center      = Vector{BigFloat}(undef, NUM_EIGS)
ell_radius      = Vector{BigFloat}(undef, NUM_EIGS)
eigenvalues_out = Vector{BigFloat}(undef, NUM_EIGS)
q1_vectors      = Vector{Vector{Complex{BigFloat}}}(undef, NUM_EIGS)

if isfile(CACHE_ELL)
    @info "Loading ℓ_j(1) results from cache..."
    ell_cache = Serialization.deserialize(CACHE_ELL)
    ell_center      .= ell_cache[:ell_center]
    ell_radius      .= ell_cache[:ell_radius]
    eigenvalues_out .= ell_cache[:eigenvalues_out]
    q1_vectors      .= ell_cache[:q1_vectors]
    n_ell = count(j -> abs(ell_center[j]) > ell_radius[j], 1:NUM_EIGS)
    @info "  Loaded: $n_ell / $NUM_EIGS sign-certified"
else
    for j in 1:NUM_EIGS
        t1 = time()

        # 1. Manual ordschur: move eigenvalue j to position 1
        target_pos = sorted_idx[j]
        T_ord, Q_ord = bigfloat_ordschur(T_bf, Q_bf, target_pos)

        lambda_j_bf = T_ord[1, 1]
        eigenvalues_out[j] = real(lambda_j_bf)
        q1_vectors[j] = Q_ord[:, 1]

        # 2. Eigenvalue separation and nilpotent norm
        sep_bf = minimum(abs(T_ord[1,1] - T_ord[i,i]) for i in 2:n)
        N_full = triu(T_ord, 1)
        N_norm = upper_bound_L2_opnorm(BallMatrix(N_full))

        # 3. Solve Sylvester in BigFloat: (T₂₂ - λI)z = w_rest
        T12 = T_ord[1, 2:n]
        T22 = T_ord[2:n, 2:n]
        M_tri = UpperTriangular(T22 - lambda_j_bf * I_m)

        w = Q_ord[1, :]       # first row of Q_ord = Q_ord^H · e₁
        w1, w_rest = w[1], w[2:n]

        z0 = M_tri \ w_rest
        ell_center[j] = real(w1 - dot(T12, z0))

        # 4. Sylvester residual error
        residual = w_rest - M_tri * z0
        res_norm_bf = BigFloat(norm(residual))  # Euclidean norm (vector)

        diag_M = [abs(T22[i,i] - lambda_j_bf) for i in 1:n-1]
        sigma_min_bf = minimum(real, diag_M)
        T12_norm_bf = BigFloat(norm(T12))       # Euclidean norm (row vector)

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

        # 5. Perturbation correction (standalone Neumann/sigma bound)
        local pert_err::BigFloat = setrounding(BigFloat, RoundUp) do
            rho = sep_bf / BF_TWO
            kappa = (BF_ONE + delta_bf) / max(BF_ONE - delta_bf, BigFloat(1e-300))

            R_neumann = rho > N_norm ? kappa / (rho - N_norm) : BigFloat(Inf)
            R_sigma = BigFloat(Inf)
            if sigma_min_bf > rho
                R22 = BF_ONE / (sigma_min_bf - rho)
                R_sigma = kappa * (BF_ONE / rho) * (BF_ONE + T12_norm_bf * R22)
            end
            R_S = min(R_neumann, R_sigma)

            if isinf(R_S) || R_S * E_bound_total >= BF_ONE
                BigFloat(Inf)
            else
                rho * R_S^2 * E_bound_total / (BF_ONE - R_S * E_bound_total)
            end
        end

        # 6. NK eigenvector correction
        local nk_corr::BigFloat
        if nk_certified[j]
            z0_norm = BigFloat(norm(z0))
            proj_schur_norm = setrounding(BigFloat, RoundUp) do
                sqrt(BF_ONE + z0_norm * z0_norm)
            end
            nk_corr = setrounding(BigFloat, RoundUp) do
                proj_schur_norm * BF_TWO * BigFloat(nk_radii[j])
            end
        else
            nk_corr = BigFloat(Inf)
        end

        # 7. Total error
        ell_radius[j] = setrounding(BigFloat, RoundUp) do
            sylv_err + pert_err + nk_corr
        end

        sign_ok = abs(ell_center[j]) > ell_radius[j] ? "YES" : "NO"
        dt_j = time() - t1
        @printf("  j=%2d: lam=%+.10e  ell=%+.15e +/- %.2e  sylv=%.2e  pert=%.2e  nk=%.2e  %s  [%.1fs]\n",
                j, Float64(eigenvalues_out[j]), Float64(ell_center[j]), Float64(ell_radius[j]),
                Float64(sylv_err), Float64(pert_err), Float64(nk_corr), sign_ok, dt_j)
        flush(stdout)
    end

    Serialization.serialize(CACHE_ELL, Dict(
        :ell_center => ell_center, :ell_radius => ell_radius,
        :eigenvalues_out => eigenvalues_out, :q1_vectors => q1_vectors))
    @info "ℓ_j(1) results cached to $CACHE_ELL"
end

n_ell_cert = count(j -> abs(ell_center[j]) > ell_radius[j], 1:NUM_EIGS)
println("\nPhase 2: $n_ell_cert / $NUM_EIGS ℓ_j(1) sign-certified\n")
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Summary + save + export
# ═══════════════════════════════════════════════════════════════════════

println("=" ^ 80)
println("SUMMARY  K=$K, P=$PRECISION, $NUM_EIGS eigenvalues")
println("=" ^ 80)
println()

println("-" ^ 130)
@printf("  %3s  %22s  %12s  %12s  %26s  %6s\n",
    "j", "lam_j", "NK radius", "pert error", "ell_j(1)", "sign")
println("-" ^ 130)

for j in 1:NUM_EIGS
    lam_j = Float64(eigenvalues_out[j])
    nk_rad = nk_certified[j] ? Float64(nk_radii[j]) : Inf
    sign_ok = abs(ell_center[j]) > ell_radius[j] ? "YES" : "NO"

    @printf("  %3d  %+22.14e  %12.4e  %12.4e  %+22.14e +/- %.2e  %6s\n",
            j, lam_j, nk_rad, Float64(ell_radius[j]),
            Float64(ell_center[j]), Float64(ell_radius[j]), sign_ok)
end
println("-" ^ 130)
println()

@printf("  NK certified:       %d / %d\n", n_nk_cert, NUM_EIGS)
@printf("  ell_j(1) sign:      %d / %d\n", n_ell_cert, NUM_EIGS)
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
    :E_bound => Float64(E_bound_total),
))
@info "Results saved to $RESULTS_PATH"

# --- LaTeX output ---
latex_path = joinpath(DATA_DIR, "certified_spectral_data_K$(K).tex")
open(latex_path, "w") do io
    println(io, "% Certified spectral data for $NUM_EIGS GKW eigenvalues at K=$K")
    println(io, "% Generated: $(now())")
    println(io, "% K=$K, precision=$PRECISION bits")
    println(io)

    println(io, "\\begin{longtable}{rrrrrr}")
    println(io, "\\caption{Certified spectral data (K=\$K, P=\$PRECISION).}")
    println(io, "\\label{tab:spectral-data-K$(K)} \\\\")
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
end
@info "LaTeX written to $latex_path"

# --- Portable text export ---
txt_path = joinpath(DATA_DIR, "spectral_coefficients_K$(K)_P$(PRECISION).txt")
open(txt_path, "w") do io
    println(io, "# GKW operator spectral expansion coefficients")
    println(io, "# K = $K, BigFloat precision = $PRECISION bits (≈$(round(Int, PRECISION * log10(2))) decimal digits)")
    println(io, "# ||E|| = ||A_true - QTQ'||_2 <= ", string(E_bound_total))
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

# --- Eigenvector coefficients ---
vec_file = joinpath(DATA_DIR, "eigenvectors_K$(K)_P$(PRECISION).txt")
open(vec_file, "w") do io
    println(io, "# GKW operator eigenvector coefficients in shifted monomial basis {(x-1)^k}")
    println(io, "# K = $K, BigFloat precision = $PRECISION bits")
    println(io, "# Each column j contains [v_j]_k for k = 0, 1, ..., $K")
    println(io, "# Eigenvectors have unit ℓ² norm")
    println(io, "#")
    println(io, "# Header: j values")
    print(io, "k")
    for j in 1:NUM_EIGS
        print(io, "\tv_", j)
    end
    println(io)
    for k in 0:K
        print(io, k)
        for j in 1:NUM_EIGS
            print(io, "\t", string(real(q1_vectors[j][k+1])))
        end
        println(io)
    end
end
@info "  Written: $vec_file"

println()
println("=" ^ 80)
@printf("K=$K SPECTRAL DATA DONE -- %s (total: %.1f hours)\n", now(), 0.0)
println("=" ^ 80)
