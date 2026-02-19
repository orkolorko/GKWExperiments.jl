# BigFloat spectral computation for K=1024, P=2048, 50 eigenvalues.
# ordschur + direct triangular Sylvester solve + perturbation bound.
# Designed to run on ibis (32 cores, 125 GB RAM).
#
# Spectral coefficient ℓ_j(1) = q₁ - T₁₂·(T₂₂-λI)⁻¹·q_rest
# where q = Q_ord^H e₁ = conj(first row of Q_ord).
# Error bound = Schur perturbation bound (triangular solve error is negligible).
#
# Usage:
#   julia --project --startup-file=no scripts/bigfloat_spectral_K1024.jl
#
# All intermediate results are cached in data/ so the script can be
# interrupted and resumed.
#
# IMPORTANT: Delete stale caches before re-running if previous results were wrong:
#   rm -f data/bigfloat_schur_K1024_P2048.jls data/schur_cert_K1024_P2048.jls data/ell_K1024_P2048.jls
#   (keep ball_matrix_bf — that's just the Arb→BigFloat conversion, known good)

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
const K         = 1024
const PRECISION = 2048          # bits (≈616 decimal digits)
const NUM_EIGS  = 50
const n         = K + 1         # matrix dimension

setprecision(ArbFloat, PRECISION)
setprecision(BigFloat, PRECISION)

const DATA_DIR = joinpath(@__DIR__, "..", "data")
mkpath(DATA_DIR)

const N_SPLITTING = 5000
const SCRIPT1_RESULTS = joinpath(DATA_DIR, "script1_results.jls")

# Cache paths (all K=1024, P=2048 specific)
const CACHE_BALL_BF   = joinpath(DATA_DIR, "ball_matrix_bf_K$(K)_P$(PRECISION).jls")
const CACHE_SCHUR     = joinpath(DATA_DIR, "bigfloat_schur_K$(K)_P$(PRECISION).jls")
const CACHE_SCHUR_CERT = joinpath(DATA_DIR, "schur_cert_K$(K)_P$(PRECISION).jls")
const CACHE_ELL       = joinpath(DATA_DIR, "ell_K$(K)_P$(PRECISION).jls")
const RESULTS_PATH    = joinpath(DATA_DIR, "spectral_K$(K)_P$(PRECISION).jls")

println("=" ^ 80)
println("K=$K SPECTRAL DATA: ordschur + SYLVESTER + PERTURBATION ($NUM_EIGS eigenvalues)")
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
        @printf("  GenericSchur done in %.1fs (%.1f hours)\n",
                time()-t_schur, (time()-t_schur)/3600)
        sd_bf = Dict(:Q_bf => F_gs.Z, :T_bf => F_gs.T)
        Serialization.serialize(CACHE_SCHUR, sd_bf)
        @info "  Cached to $CACHE_SCHUR"
    end

    Q_bf = Complex{BigFloat}.(sd_bf[:Q_bf])
    T_bf = Complex{BigFloat}.(sd_bf[:T_bf])

    # Compute Schur quality metrics
    A_complex = Complex{BigFloat}.(A_real_center)
    residual_mat = A_complex - Q_bf * T_bf * Q_bf'
    orth_mat = Q_bf' * Q_bf - Matrix{Complex{BigFloat}}(I, n, n)

    res_opnorm = upper_bound_L2_opnorm(BallMatrix(residual_mat))
    orth_def   = upper_bound_L2_opnorm(BallMatrix(orth_mat))
    A_norm     = upper_bound_L2_opnorm(BallMatrix(A_complex))

    E_bound_bf = res_opnorm
    delta_bf = orth_def

    @printf("  Residual ||A-QTQ'||_2           ≤ %.4e\n", Float64(res_opnorm))
    @printf("  Residual ||A-QTQ'||_2 / ||A||_2 ≤ %.4e\n", Float64(res_opnorm / A_norm))
    @printf("  Orthogonality ||Q'Q-I||_2       ≤ %.4e\n", Float64(delta_bf))

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
A_rad_opnorm = upper_bound_L2_opnorm(BallMatrix(BallArithmetic.rad(A_ball_bf)))
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
# PHASE 1: ordschur + direct triangular Sylvester → ℓ_j(1)
# ═══════════════════════════════════════════════════════════════════════

println("=" ^ 80)
println("PHASE 1: ℓ_j(1) VIA ordschur + direct triangular Sylvester solve")
println("=" ^ 80)
println()
flush(stdout)

const BF_ONE = one(BigFloat)
const BF_TWO = BigFloat(2)

# Wrap Schur factors as BallMatrices (zero radii) for ordschur_ball
const Q_ball_bf = BallMatrix(Q_bf)
const T_ball_bf = BallMatrix(T_bf)

ell_center      = Vector{BigFloat}(undef, NUM_EIGS)
ell_radius      = Vector{BigFloat}(undef, NUM_EIGS)
eigvec_radius   = Vector{BigFloat}(undef, NUM_EIGS)
eigenvalues_out = Vector{BigFloat}(undef, NUM_EIGS)
q1_vectors      = Vector{Vector{Complex{BigFloat}}}(undef, NUM_EIGS)

if isfile(CACHE_ELL)
    @info "Loading ℓ_j(1) results from cache..."
    ell_cache = Serialization.deserialize(CACHE_ELL)
    ell_center      .= ell_cache[:ell_center]
    ell_radius      .= ell_cache[:ell_radius]
    eigvec_radius   .= get(ell_cache, :eigvec_radius, fill(BigFloat(Inf), NUM_EIGS))
    eigenvalues_out .= ell_cache[:eigenvalues_out]
    q1_vectors      .= ell_cache[:q1_vectors]
    n_ell = count(j -> abs(ell_center[j]) > ell_radius[j], 1:NUM_EIGS)
    @info "  Loaded: $n_ell / $NUM_EIGS sign-certified"
else
    for j in 1:NUM_EIGS
        t1 = time()

        # 1. Ball-arithmetic ordschur: move eigenvalue j to position 1
        #    O(kn²) via incremental Givens — radii track rounding rigorously.
        target_pos = sorted_idx[j]
        select = falses(n)
        select[target_pos] = true
        ord_result = ordschur_ball(Q_ball_bf, T_ball_bf, select)
        Q_ord_ball = ord_result.Q   # BallMatrix
        T_ord_ball = ord_result.T   # BallMatrix

        # Extract midpoints for spectral coefficient computation
        Q_ord = BallArithmetic.mid(Q_ord_ball)
        T_ord = BallArithmetic.mid(T_ord_ball)

        lambda_j_bf = T_ord[1, 1]
        eigenvalues_out[j] = real(lambda_j_bf)
        q1_vectors[j] = Q_ord[:, 1]

        # 2. Direct triangular Sylvester solve for ℓ_j(1)
        #    Formula: ℓ_j(1) = q₁ - T₁₂·(T₂₂ - λI)⁻¹·q_rest
        #    where q = Q_ord^H e₁ = conj(first row of Q_ord)
        #    IMPORTANT: use conj() for complex Q, and sum(.*) not dot() to avoid
        #    Julia's dot(a,b) = conj(a)·b conjugation of the first argument.
        q = conj.(Q_ord[1, :])     # Q^H e₁
        q1 = q[1]
        q_rest = q[2:n]
        T12 = T_ord[1, 2:n]
        T22 = T_ord[2:n, 2:n]

        # Triangular solve: (T22 - λI) z = q_rest
        z = (T22 - lambda_j_bf * I) \ q_rest

        # ℓ_j(1) = real(q₁ - T₁₂·z) using unconjugated inner product
        ell_center[j] = real(q1 - sum(T12 .* z))

        # Solve residual (sanity check — should be ~eps at working precision)
        solve_resid = BigFloat(norm((T22 - lambda_j_bf * I) * z - q_rest))

        # 3. Ordschur error from Ball radii (O(n²), no matrix product)
        #    ordschur_ball tracks rounding in Q and T radii:
        #      ||ΔQ|| ≤ rad_Q_norm,  ||ΔT|| ≤ rad_T_norm
        #    Factorization error from ordschur rounding (first-order):
        #      ||Q_exact T_exact Q_exact' - Q_mid T_mid Q_mid'||
        #    Full expansion (no first-order truncation):
        #      = ΔQ·T·Q' + Q·ΔT·Q' + Q·T·ΔQ' + ΔQ·ΔT·Q' + ΔQ·T·ΔQ' + Q·ΔT·ΔQ' + ΔQ·ΔT·ΔQ'
        #      ≤ (q+εQ)²·(t+εT) - q²·t   where q=||Q||, t=||T||, εQ=||ΔQ||, εT=||ΔT||
        rad_Q_norm = upper_bound_L2_opnorm(BallMatrix(BallArithmetic.rad(Q_ord_ball)))
        rad_T_norm = upper_bound_L2_opnorm(BallMatrix(BallArithmetic.rad(T_ord_ball)))
        T_norm = upper_bound_L2_opnorm(T_ord_ball)
        Q_norm_bound = setrounding(BigFloat, RoundUp) do
            sqrt(BF_ONE + delta_bf)
        end
        E_ordschur = setrounding(BigFloat, RoundUp) do
            # Exact: ||(Q+ΔQ)(T+ΔT)(Q+ΔQ)' - QTQ'|| ≤ (q+εQ)²(t+εT) - q²t
            q = Q_norm_bound
            (q + rad_Q_norm)^2 * (T_norm + rad_T_norm) - q^2 * T_norm
        end
        E_combined = setrounding(BigFloat, RoundUp) do
            E_bound_total + E_ordschur
        end
        if j <= 3
            log_rQ = iszero(rad_Q_norm) ? -Inf : Float64(log10(rad_Q_norm))
            log_rT = iszero(rad_T_norm) ? -Inf : Float64(log10(rad_T_norm))
            log_Eo = iszero(E_ordschur) ? -Inf : Float64(log10(E_ordschur))
            @printf("  ordschur Ball: log₁₀(rad_Q)=%.0f  log₁₀(rad_T)=%.0f  log₁₀(E_ordschur)=%.0f\n",
                    log_rQ, log_rT, log_Eo)
        end

        diag_M = [abs(T22[i,i] - lambda_j_bf) for i in 1:n-1]
        sep_bf = minimum(real, diag_M)   # eigenvalue separation for λ_j
        # Rigorous ‖T₁₂‖₂ upper bound (T12 is a row vector, so spectral = L2)
        T12_norm_bf = upper_bound_L2_opnorm(BallMatrix(reshape(T12, 1, :)))
        N_full = triu(T_ord, 1)
        N_norm = upper_bound_L2_opnorm(BallMatrix(N_full))

        local pert_err::BigFloat, evec_err::BigFloat
        (pert_err, evec_err) = setrounding(BigFloat, RoundUp) do
            rho = sep_bf / BF_TWO
            kappa = (BF_ONE + delta_bf) / max(BF_ONE - delta_bf, BigFloat(1e-300))

            R_neumann = rho > N_norm ? kappa / (rho - N_norm) : BigFloat(Inf)
            R_sigma = BigFloat(Inf)
            if sep_bf > rho
                R22 = BF_ONE / (sep_bf - rho)
                R_sigma = kappa * (BF_ONE / rho) * (BF_ONE + T12_norm_bf * R22)
            end
            R_S = min(R_neumann, R_sigma)

            if isinf(R_S) || R_S * E_combined >= BF_ONE
                (BigFloat(Inf), BigFloat(Inf))
            else
                denom = BF_ONE - R_S * E_combined
                pe = rho * R_S^2 * E_combined / denom
                ve = R_S * E_combined / denom  # eigenvector L2 error bound
                (pe, ve)
            end
        end

        # 5. Total error = perturbation bound (solve error negligible)
        ell_radius[j] = pert_err
        eigvec_radius[j] = evec_err

        sign_ok = abs(ell_center[j]) > ell_radius[j] ? "YES" : "NO"
        dt_j = time() - t1
        @printf("  j=%2d: lam=%+.10e  ell=%+.15e +/- %.2e  ||dv||=%.2e  sep=%.2e  resid=%.1e  %s  [%.1fs]\n",
                j, Float64(eigenvalues_out[j]), Float64(ell_center[j]), Float64(ell_radius[j]),
                Float64(eigvec_radius[j]), Float64(sep_bf), Float64(solve_resid), sign_ok, dt_j)
        flush(stdout)
    end

    Serialization.serialize(CACHE_ELL, Dict(
        :ell_center => ell_center, :ell_radius => ell_radius,
        :eigvec_radius => eigvec_radius,
        :eigenvalues_out => eigenvalues_out, :q1_vectors => q1_vectors))
    @info "ℓ_j(1) results cached to $CACHE_ELL"
end

n_ell_cert = count(j -> abs(ell_center[j]) > ell_radius[j], 1:NUM_EIGS)
println("\nPhase 1: $n_ell_cert / $NUM_EIGS ℓ_j(1) sign-certified\n")
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Eigenvalue enclosures via projector control (Lemma 2.12)
# ═══════════════════════════════════════════════════════════════════════
#
# Uses M_∞,j (infinite-dimensional resolvent) from Script 1's results,
# transferred to K=1024 via reverse_transfer_resolvent_bound.
# Projector error: ϑ_j = (|Γ_j|/2π) · R_{A_1024}² · ε_1024 / (1 - R_{A_1024}·ε_1024)
# Eigenvalue enclosure: |λ*_j - λ̂_j| ≤ (ε_K(1+ϑ_j) + 2C·ϑ_j) / (1-ϑ_j)

println("=" ^ 80)
println("PHASE 2: EIGENVALUE ENCLOSURES (PROJECTOR CONTROL AT K=$K)")
println("=" ^ 80)
println()
flush(stdout)

# Load M_∞ and circle_radius from Script 1
@assert isfile(SCRIPT1_RESULTS) "Script 1 results not found at $SCRIPT1_RESULTS"
script1 = Serialization.deserialize(SCRIPT1_RESULTS)
M_inf_all = script1[:M_inf_all]::Vector{Float64}
resolvent_data_s1 = script1[:resolvent_data]

# Compute ε_1024
eps_K_1024 = _arb_to_float64_upper(compute_Δ(K; N=N_SPLITTING))
@printf("  ε_{K=%d} ≤ %.6e\n", K, eps_K_1024)

# Compute C = ||A_K||₂ (rigorous upper bound including BallMatrix radii)
C_AK = Float64(upper_bound_L2_opnorm(A_ball_bf))
@printf("  ||A_%d||₂ ≤ %.6e\n", K, C_AK)
println()

eval_encl = Vector{Float64}(undef, NUM_EIGS)
proj_error_1024 = Vector{Float64}(undef, NUM_EIGS)

for j in 1:NUM_EIGS
    M_inf_j = M_inf_all[j]
    rd = resolvent_data_s1[j]
    circle_radius_j = rd.circle_radius

    if !isfinite(M_inf_j)
        eval_encl[j] = Inf
        proj_error_1024[j] = Inf
        @printf("  j=%2d: M_inf=Inf → SKIP\n", j)
        continue
    end

    # Transfer: R_{A_1024} ≤ M_∞ / (1 - M_∞ · ε_1024)
    R_A1024, alpha_1024, valid = reverse_transfer_resolvent_bound(M_inf_j, eps_K_1024)
    if !valid
        eval_encl[j] = Inf
        proj_error_1024[j] = Inf
        @printf("  j=%2d: transfer FAILED (alpha=%.2e)\n", j, alpha_1024)
        continue
    end

    # Projector error: ϑ_j
    contour_length = 2π * circle_radius_j
    theta_j, theta_valid = projector_approximation_error_rigorous(
        contour_length, R_A1024, eps_K_1024)
    proj_error_1024[j] = theta_valid ? theta_j : Inf

    if !theta_valid || theta_j >= 1.0
        eval_encl[j] = Inf
        @printf("  j=%2d: projector error ϑ=%.2e ≥ 1 → FAIL\n", j, theta_j)
        continue
    end

    # Eigenvalue enclosure: |λ*_j - λ̂_j| ≤ (ε_K(1+ϑ) + 2C·ϑ) / (1-ϑ)
    eval_encl[j] = setrounding(Float64, RoundUp) do
        (eps_K_1024 * (1.0 + theta_j) + 2.0 * C_AK * theta_j) / (1.0 - theta_j)
    end

    @printf("  j=%2d: M_inf=%.4e  R_1024=%.4e  ϑ=%.2e  encl=%.2e\n",
            j, M_inf_j, R_A1024, theta_j, eval_encl[j])
    flush(stdout)
end

n_encl = count(isfinite, eval_encl)
println("\nPhase 2: $n_encl / $NUM_EIGS eigenvalue enclosures finite\n")
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Summary + save + export
# ═══════════════════════════════════════════════════════════════════════

println("=" ^ 80)
println("SUMMARY  K=$K, P=$PRECISION, $NUM_EIGS eigenvalues")
println("=" ^ 80)
println()

println("-" ^ 120)
@printf("  %3s  %22s  %12s  %26s  %6s  %12s\n",
    "j", "lam_j", "pert_err", "ell_j(1)", "sign", "eval_encl")
println("-" ^ 120)

for j in 1:NUM_EIGS
    sign_ok = abs(ell_center[j]) > ell_radius[j] ? "YES" : "NO"
    @printf("  %3d  %+22.14e  %12.4e  %+22.14e +/- %.2e  %6s  %12.4e\n",
            j, Float64(eigenvalues_out[j]),
            Float64(ell_radius[j]),
            Float64(ell_center[j]), Float64(ell_radius[j]), sign_ok,
            eval_encl[j])
end
println("-" ^ 120)
println()

n_encl_final = count(isfinite, eval_encl)
@printf("  ell_j(1) sign:      %d / %d\n", n_ell_cert, NUM_EIGS)
@printf("  eval enclosures:    %d / %d\n", n_encl_final, NUM_EIGS)
println()

# Save unified results
Serialization.serialize(RESULTS_PATH, Dict(
    :K => K,
    :precision => PRECISION,
    :NUM_EIGS => NUM_EIGS,
    :eigenvalues => Float64.(eigenvalues_out),
    :eigenvalues_bf => eigenvalues_out,
    :ell_center => Float64.(ell_center),
    :ell_radius => Float64.(ell_radius),
    :ell_center_bf => ell_center,
    :ell_radius_bf => ell_radius,
    :eigvec_radius => Float64.(eigvec_radius),
    :eigvec_radius_bf => eigvec_radius,
    :E_bound => Float64(E_bound_total),
    :eval_encl => eval_encl,
    :proj_error_1024 => proj_error_1024,
    :eps_K_1024 => eps_K_1024,
    :C_AK => C_AK,
))
@info "Results saved to $RESULTS_PATH"

# --- LaTeX output ---
latex_path = joinpath(DATA_DIR, "certified_spectral_data_K$(K).tex")
open(latex_path, "w") do io
    println(io, "% Certified spectral data for $NUM_EIGS GKW eigenvalues at K=$K")
    println(io, "% Generated: $(now())")
    println(io, "% K=$K, precision=$PRECISION bits, no NK (projector perturbation only)")
    println(io)

    println(io, "\\begin{longtable}{rrrrrr}")
    println(io, "\\caption{Certified spectral data (K=$K, P=$PRECISION).}")
    println(io, "\\label{tab:spectral-data-K$(K)} \\\\")
    println(io, "\\toprule")
    println(io, "\$j\$ & \$\\hat\\lambda_j\$ & \$\\ell_j(1)\$ & \$\\ell_j(1)\$ radius & sign & \$\\pm\\;\\delta_\\lambda\$ \\\\")
    println(io, "\\midrule")
    println(io, "\\endfirsthead")
    println(io, "\\multicolumn{6}{c}{\\textit{continued}} \\\\")
    println(io, "\\toprule")
    println(io, "\$j\$ & \$\\hat\\lambda_j\$ & \$\\ell_j(1)\$ & \$\\ell_j(1)\$ radius & sign & \$\\pm\\;\\delta_\\lambda\$ \\\\")
    println(io, "\\midrule")
    println(io, "\\endhead")

    for j in 1:NUM_EIGS
        sign_ok = abs(ell_center[j]) > ell_radius[j]
        encl_str = isfinite(eval_encl[j]) ? @sprintf("\$%.2e\$", eval_encl[j]) : "\$\\infty\$"
        @printf(io, "%d & \$%+.12e\$ & \$%+.14e\$ & \$%.2e\$ & %s & %s \\\\\n",
                j, Float64(eigenvalues_out[j]),
                Float64(ell_center[j]), Float64(ell_radius[j]),
                sign_ok ? "\\checkmark" : "--", encl_str)
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
    println(io, "# Method: Schur + ordschur + direct triangular Sylvester + Schur perturbation bound")
    println(io, "#")
    println(io, "# Columns: j, lambda_j, ell_j(1), ell_radius, sign_certified, eval_encl")
    println(io, "#")
    for j in 1:NUM_EIGS
        sign_ok = abs(ell_center[j]) > ell_radius[j] ? "YES" : "NO"
        println(io, j, "\t", string(eigenvalues_out[j]), "\t",
                string(ell_center[j]), "\t", string(ell_radius[j]), "\t", sign_ok,
                "\t", string(eval_encl[j]))
    end
end
@info "Text export: $txt_path"

# --- Eigenvector coefficients ---
vec_file = joinpath(DATA_DIR, "eigenvectors_K$(K)_P$(PRECISION).txt")
open(vec_file, "w") do io
    println(io, "# GKW operator eigenvector coefficients in shifted monomial basis {(x-1)^k}")
    println(io, "# K = $K, BigFloat precision = $PRECISION bits")
    println(io, "# Each column j contains [v_j]_k for k = 0, 1, ..., $K")
    println(io, "#")
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

# --- Theorem LaTeX snippet (eigenvalue enclosures for supplementary material) ---
function _format_latex_sci(x::Float64)
    if !isfinite(x)
        return "\\infty"
    end
    e = floor(Int, log10(abs(x)))
    m = x / 10.0^e
    @sprintf("%.2f \\times 10^{%d}", m, e)
end

function _format_lambda_center(x::Float64)
    ax = abs(x)
    if ax >= 0.001
        return @sprintf("%.12f", x)
    else
        e = floor(Int, log10(ax))
        m = x / 10.0^e
        return @sprintf("%.6f \\times 10^{%d}", m, e)
    end
end

theorem_path = joinpath(DATA_DIR, "theorem_enclosures_K$(K).tex")
open(theorem_path, "w") do io
    println(io, "% Eigenvalue enclosures for Theorem (K=$K)")
    println(io, "% Generated: $(now())")
    println(io, "% Paste this into supplementary_material.tex inside the theorem environment")
    println(io)
    println(io, "\\begin{enumerate}")
    for j in 1:NUM_EIGS
        lam_str = _format_lambda_center(Float64(eigenvalues_out[j]))
        encl_str = _format_latex_sci(eval_encl[j])
        proj_str = _format_latex_sci(proj_error_1024[j])
        @printf(io, "\\item \$\\lambda_{%d}^* \\in B(%s,\\; %s)\$, \\quad \$\\|P_{L_1}(\\Gamma_{%d}) - P_{(L_1)_{%d}}(\\Gamma_{%d})\\| \\leq %s\$\n",
                j, lam_str, encl_str, j, K, j, proj_str)
    end
    println(io, "\\end{enumerate}")
end
@info "Theorem snippet written to $theorem_path"

# --- Eigenvector coefficient LaTeX tables ---
const NUM_COEFFS_DISPLAY = 50  # number of coefficients per eigenvector to show
eigvec_latex_path = joinpath(DATA_DIR, "eigenvector_tables_K$(K).tex")
open(eigvec_latex_path, "w") do io
    println(io, "% Eigenvector coefficient tables for K=$K")
    println(io, "% Generated: $(now())")
    println(io, "% Each table shows the first $NUM_COEFFS_DISPLAY coefficients of v_j in {(w-1)^k}")
    println(io, "% with rigorous L2 error bound ||v̂_j - v_j||₂ ≤ δ_j from Schur perturbation")
    println(io)
    println(io, raw"\subsection*{Eigenvector Coefficients $[v_j]_k$}")
    println(io)
    println(io, "The following tables list the first \$$NUM_COEFFS_DISPLAY\$ coefficients of each unit-norm")
    println(io, raw"eigenvector $v_j$ in the shifted monomial basis $\{(w-1)^k\}_{k=0}^{K}$.")
    println(io, raw"The $\ell^2$ error $\|\hat{v}_j - v_j\|_2 \leq \delta_j$ shown for each")
    println(io, raw"eigenvector is controlled by the Schur orthogonality defect and the")
    println(io, raw"reconstruction error from the ordschur + Sylvester pipeline")
    println(io, "(see Table~\\ref{S-tab:spectral-data-K$(K)}).")
    evr_range = extrema(Float64.(eigvec_radius))
    @printf(io, "Eigenvectors at \$K = %d\$ (%d-bit precision) achieve radii \$\\sim 10^{%d}\$--\$10^{%d}\$.\n",
            K, PRECISION, floor(Int, log10(evr_range[1])), floor(Int, log10(evr_range[2])))
    println(io)

    nrows = NUM_COEFFS_DISPLAY ÷ 2

    for j in 1:NUM_EIGS
        lam_str = @sprintf("%.6e", Float64(eigenvalues_out[j]))
        ell_str = @sprintf("%+.6e", Float64(ell_center[j]))
        evr_str = @sprintf("%.2e", Float64(eigvec_radius[j]))

        println(io, "\\paragraph{\$j = $j\$: \$\\lambda_{$j} = $lam_str\$, \\quad \$\\ell_{$j}(\\mathbf{1}) = $ell_str\$, \\quad \$\\|\\hat{v}_{$j} - v_{$j}\\|_2 \\leq $evr_str\$}")
        println(io, "\\begin{center}")
        println(io, "{\\footnotesize")
        println(io, "\\begin{tabular}{r@{\\;\\;}l@{\\quad}r@{\\;\\;}l}")
        println(io, "\\toprule")
        println(io, "\$k\$ & \$[v_{$j}]_k\$ & \$k\$ & \$[v_{$j}]_k\$ \\\\")
        println(io, "\\midrule")

        v = q1_vectors[j]
        for row in 0:nrows-1
            k1 = row
            k2 = row + nrows
            c1 = Float64(real(v[k1+1]))
            c2 = Float64(real(v[k2+1]))
            @printf(io, "%d & \$%+.10e\$ & %d & \$%+.10e\$ \\\\\n", k1, c1, k2, c2)
        end

        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "}")
        println(io, "\\end{center}")
        println(io)
    end
end
@info "Eigenvector LaTeX tables written to $eigvec_latex_path"

println()
println("=" ^ 80)
println("K=$K SPECTRAL DATA DONE -- $(now())")
println("=" ^ 80)
