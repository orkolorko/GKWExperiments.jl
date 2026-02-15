# BigFloat spectral computation for K=512, P=1024, 50 eigenvalues.
# NK certification + ordschur_ball + Sylvester + standalone perturbation bound.
# Designed to run on ibis (32 cores, 125 GB RAM).
#
# Usage:
#   julia --project --startup-file=no scripts/bigfloat_spectral_K512.jl
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
const K           = 512
const PRECISION   = 1024          # bits (≈308 decimal digits)
const NUM_EIGS    = 50
const N_SPLITTING = 5000          # C₂ splitting parameter for NK
const n           = K + 1         # matrix dimension

setprecision(ArbFloat, PRECISION)
setprecision(BigFloat, PRECISION)

const DATA_DIR = joinpath(@__DIR__, "..", "data")
mkpath(DATA_DIR)

# Cache paths (all K=512, P=1024 specific)
const CACHE_BALL_BF   = joinpath(DATA_DIR, "ball_matrix_bf_K$(K)_P$(PRECISION).jls")
const CACHE_SCHUR     = joinpath(DATA_DIR, "bigfloat_schur_K$(K)_P$(PRECISION).jls")
const CACHE_SCHUR_BALL = joinpath(DATA_DIR, "schur_ball_K$(K)_P$(PRECISION).jls")
const CACHE_NK        = joinpath(DATA_DIR, "nk_K$(K)_P$(PRECISION).jls")
const CACHE_ELL       = joinpath(DATA_DIR, "ell_K$(K)_P$(PRECISION).jls")
const RESULTS_PATH    = joinpath(DATA_DIR, "spectral_K$(K)_P$(PRECISION).jls")

println("=" ^ 80)
println("K=$K SPECTRAL DATA: NK + ordschur_ball + SYLVESTER ($NUM_EIGS eigenvalues)")
println("  Precision: $PRECISION bits (≈$(round(Int, PRECISION * log10(2))) decimal digits)")
println("=" ^ 80)
println("Started: ", now())
flush(stdout)

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
# Step 2: Rigorous BallMatrix Schur decomposition
# ═══════════════════════════════════════════════════════════════════════

local Q_ball, T_ball, E_bound_bf, delta_bf, sorted_idx
if isfile(CACHE_SCHUR_BALL)
    @info "Loading BallMatrix Schur from cache..."
    schur_cache = Serialization.deserialize(CACHE_SCHUR_BALL)
    Q_ball     = schur_cache[:Q_ball]
    T_ball     = schur_cache[:T_ball]
    E_bound_bf = schur_cache[:E_bound]
    delta_bf   = schur_cache[:orth_defect]
    sorted_idx = schur_cache[:sorted_idx]
else
    @info "Computing rigorous BallMatrix Schur decomposition..."
    t0 = time()
    A_real_center = real.(BallArithmetic.mid(A_ball_bf))

    # Check if GenericSchur cache exists at the required precision
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
        @info "  Matrix size: $n × $n — this may take hours..."
        t_schur = time()
        F_gs = schur(Complex{BigFloat}.(A_real_center))
        @printf("  GenericSchur done in %.1fs\n", time() - t_schur)
        sd_bf = Dict(:Q_bf => F_gs.Z, :T_bf => F_gs.T)
        Serialization.serialize(CACHE_SCHUR, sd_bf)
        @info "  Cached to $CACHE_SCHUR"
    end

    Q0 = Complex{BigFloat}.(sd_bf[:Q_bf])
    T0 = Complex{BigFloat}.(sd_bf[:T_bf])

    # Compute Schur quality using upper_bound_L2_opnorm (NOT Frobenius!)
    A_complex = Complex{BigFloat}.(A_real_center)
    T_hat = Q0' * A_complex * Q0
    E = tril(T_hat, -1)   # strictly lower triangular = residual
    T_clean = T_hat - E   # upper triangular part

    E_norm = upper_bound_L2_opnorm(BallMatrix(E))
    A_norm = upper_bound_L2_opnorm(BallMatrix(A_complex))
    residual_norm = E_norm / A_norm

    Y = Q0' * Q0 - Matrix{Complex{BigFloat}}(I, n, n)
    orth_defect = upper_bound_L2_opnorm(BallMatrix(Y))

    @printf("  GenericSchur quality: residual_norm=%.2e, orth_defect=%.2e\n",
            Float64(residual_norm), Float64(orth_defect))

    schur_result = SchurRefinementResult(Q0, T_clean, 0, residual_norm, orth_defect, true)
    Q_ball, T_ball = certify_schur_decomposition(A_ball_bf, schur_result)
    @printf("  Done in %.1fs\n", time()-t0)

    # Schur error and orthogonality defect
    A_mid_norm = upper_bound_L2_opnorm(BallMatrix(BallArithmetic.mid(A_ball_bf)))
    E_bound_bf = schur_result.residual_norm * A_mid_norm
    delta_bf = schur_result.orthogonality_defect

    # Sort eigenvalues by decreasing magnitude
    T_diag = diag(BallArithmetic.mid(T_ball))
    sorted_idx = sortperm(abs.(T_diag), rev=true)

    # Cache
    Serialization.serialize(CACHE_SCHUR_BALL, Dict(
        :Q_ball => Q_ball, :T_ball => T_ball,
        :E_bound => E_bound_bf, :orth_defect => delta_bf,
        :sorted_idx => sorted_idx))
    @info "BallMatrix Schur cached to $CACHE_SCHUR_BALL"
end

@printf("  Schur ||A - QTQ'||_2 <= %.4e\n", Float64(E_bound_bf))
@printf("  Orthogonality ||Q'Q-I||_2 <= %.4e\n", Float64(delta_bf))
println()
flush(stdout)

# Rigorous total E_bound (Schur residual + Arb→BigFloat conversion)
A_rad_opnorm = upper_bound_L2_opnorm(BallMatrix(BallArithmetic.rad(A_ball_bf)))
E_bound_total = setrounding(BigFloat, RoundUp) do
    E_bound_bf + A_rad_opnorm
end
@printf("  Total ||E|| = ||A_true - QTQ'||_2 <= %.4e\n", Float64(E_bound_total))
@printf("    Schur residual = %.4e, input ||A_rad||_2 = %.4e\n",
        Float64(E_bound_bf), Float64(A_rad_opnorm))
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
# PHASE 2: ordschur_ball + Sylvester → ℓ_j(1) with perturbation bound
# ═══════════════════════════════════════════════════════════════════════

println("=" ^ 80)
println("PHASE 2: ℓ_j(1) VIA ordschur_ball + SYLVESTER + PERTURBATION BOUND")
println("=" ^ 80)
println()
flush(stdout)

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

        # 2. Certified Sylvester solve (BallMatrix overload)
        X_ball = triangular_sylvester_miyajima_enclosure(T_ord_ball, 1)
        # X_ball satisfies T₂₂^H X - X T₁₁^H = T₁₂^H
        # Coupling Y = -X^H (sign convention from BallArithmetic fix)

        # 3. ℓ_j(1) from spectral projector: P_S = [I, Y; 0, 0]
        Q_ord_mid = BallArithmetic.mid(Q_ord_ball)
        q = conj.(Q_ord_mid[1, :])       # Q_ord^H · e₁
        q1 = q[1]
        q_rest = q[2:end]

        X_mid = BallArithmetic.mid(X_ball)[:, 1]
        X_rad = BallArithmetic.rad(X_ball)[:, 1]

        # Y · q_rest = -X^H · q_rest = -dot(X_mid, q_rest)
        ell_center[j] = real(q1 - dot(X_mid, q_rest))

        # 4. Sylvester error: componentwise propagation from BallMatrix radii
        sylv_err = setrounding(BigFloat, RoundUp) do
            sum(BigFloat(X_rad[i]) * abs(q_rest[i]) for i in 1:n-1)
        end

        # 5. Standalone perturbation correction (Neumann/sigma resolvent bound)
        #    Does NOT require Script 1 resolvent data
        T_ord_mid = BallArithmetic.mid(T_ord_ball)
        sep_bf = minimum(abs(T_ord_mid[1,1] - T_ord_mid[i,i]) for i in 2:n)
        N_full = triu(T_ord_mid, 1)
        N_norm = upper_bound_L2_opnorm(BallMatrix(N_full))

        local pert_err::BigFloat = setrounding(BigFloat, RoundUp) do
            BF_ONE = one(BigFloat)
            BF_TWO = BigFloat(2)
            rho = sep_bf / BF_TWO
            kappa = (BF_ONE + delta_bf) / max(BF_ONE - delta_bf, BigFloat(1e-300))

            R_neumann = rho > N_norm ? kappa / (rho - N_norm) : BigFloat(Inf)
            # sigma-based bound using T22 diagonal separation
            diag_M = [abs(T_ord_mid[i,i] - T_ord_mid[1,1]) for i in 2:n]
            sigma_min = minimum(real, diag_M)
            T12_norm = BigFloat(norm(T_ord_mid[1, 2:n]))
            R_sigma = BigFloat(Inf)
            if sigma_min > rho
                R22 = BF_ONE / (sigma_min - rho)
                R_sigma = kappa * (BF_ONE / rho) * (BF_ONE + T12_norm * R22)
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
            yt_norm = upper_bound_L2_opnorm(X_ball)
            proj_schur_norm = setrounding(BigFloat, RoundUp) do
                sqrt(one(BigFloat) + yt_norm * yt_norm)
            end
            nk_corr = setrounding(BigFloat, RoundUp) do
                proj_schur_norm * BigFloat(2) * BigFloat(nk_radii[j])
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
    println(io, "\\caption{Certified spectral data (K=$K, P=$PRECISION).}")
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

println()
println("=" ^ 80)
println("K=$K SPECTRAL DATA DONE -- $(now())")
println("=" ^ 80)
