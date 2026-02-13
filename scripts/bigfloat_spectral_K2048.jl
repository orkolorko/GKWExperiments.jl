# BigFloat spectral computation for K=2048, P=1024, 50 eigenvalues.
# Designed to run on ibis (32 cores, 125 GB RAM).
#
# Estimated timings (based on K=512 scaling, O(n³)):
#   Matrix build: ~2-3 hours (O(K³) assembly + O(K) Hurwitz zeta)
#   BigFloat Schur: ~4-5 hours (O(n³) via GenericSchur)
#   Spectral coefficients: ~1-2 hours (50 ordschur + Sylvester solves)
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
using Serialization

# ═══════════════════════════════════════════════════════════════════════
# Parameters — change these to adjust the computation
# ═══════════════════════════════════════════════════════════════════════
const K         = 2048
const PRECISION = 2048          # bits (≈616 decimal digits)
const NUM_EIGS  = 50
const n         = K + 1         # matrix dimension

setprecision(ArbFloat, PRECISION)
setprecision(BigFloat, PRECISION)

const CACHE_DIR   = "data"
const CACHE_BALL  = joinpath(CACHE_DIR, "bigfloat_ball_matrix_K$(K)_P$(PRECISION).jls")
const CACHE_SCHUR = joinpath(CACHE_DIR, "bigfloat_schur_K$(K)_P$(PRECISION).jls")
const CACHE_RESULTS = joinpath(CACHE_DIR, "bigfloat_spectral_K$(K)_P$(PRECISION).jls")

mkpath(CACHE_DIR)

# ═══════════════════════════════════════════════════════════════════════
# BigFloat ordschur (swap eigenvalue to position 1)
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
# ArbNumerics → BigFloat BallMatrix conversion (rigorous)
# ═══════════════════════════════════════════════════════════════════════

function arb_to_bigfloat_ball_matrix(M_arb::Matrix{ArbComplex{P}}) where {P}
    nr, nc = size(M_arb)
    M_center = Matrix{Complex{BigFloat}}(undef, nr, nc)
    M_radius = Matrix{BigFloat}(undef, nr, nc)

    for i in 1:nr, j in 1:nc
        mid_re = ArbNumerics.midpoint(real(M_arb[i, j]))
        mid_im = ArbNumerics.midpoint(imag(M_arb[i, j]))
        rad_re = BigFloat(ArbNumerics.radius(real(M_arb[i, j])))
        rad_im = BigFloat(ArbNumerics.radius(imag(M_arb[i, j])))

        c_re = BigFloat(mid_re)
        c_im = BigFloat(mid_im)
        M_center[i, j] = Complex{BigFloat}(c_re, c_im)

        conv_re = eps(BigFloat) * abs(c_re)
        conv_im = eps(BigFloat) * abs(c_im)
        total_re = rad_re + conv_re
        total_im = rad_im + conv_im
        M_radius[i, j] = sqrt(total_re^2 + total_im^2)
    end
    return BallMatrix(M_center, M_radius)
end

# ═══════════════════════════════════════════════════════════════════════
println("="^80)
println("BigFloat spectral computation: K=$K, P=$PRECISION, $NUM_EIGS eigenvalues")
println("="^80)
println()

# ═══════════════════════════════════════════════════════════════════════
# Step 1: Build GKW matrix (with caching)
# ═══════════════════════════════════════════════════════════════════════

local A_bf, A_rad_bf
if isfile(CACHE_BALL)
    @info "Loading cached BigFloat BallMatrix from $CACHE_BALL..."
    t0 = time()
    cached = Serialization.deserialize(CACHE_BALL)
    A_bf = cached[:A_bf]
    A_rad_bf = cached[:A_rad_bf]
    @info "  Loaded in $(round(time()-t0, digits=1))s"
else
    s = ArbComplex(1.0, 0.0)
    @info "Building GKW matrix at K=$K in ArbNumerics (precision=$PRECISION)..."
    t0 = time()
    M_arb = gkw_matrix_direct_fast(s; K=K, threaded=true)
    dt = round(time()-t0, digits=1)
    @info "  Built in $(dt)s"

    @info "Converting ArbNumerics → BigFloat BallMatrix..."
    t0 = time()
    A_ball_bf = arb_to_bigfloat_ball_matrix(M_arb)
    A_bf = BallArithmetic.mid(A_ball_bf)
    A_rad_bf = BallArithmetic.rad(A_ball_bf)
    @info "  Converted in $(round(time()-t0, digits=1))s"

    @info "Caching to $CACHE_BALL..."
    Serialization.serialize(CACHE_BALL, Dict(:A_bf => A_bf, :A_rad_bf => A_rad_bf))
    @info "  Cached."
end

A_rad_ball = BallMatrix(A_rad_bf)
A_rad_opnorm = upper_bound_L2_opnorm(A_rad_ball)
@printf("  ||A_rad||_2  ≤ %.4e  (rigorous, via upper_bound_L2_opnorm)\n", Float64(A_rad_opnorm))
println()

# ═══════════════════════════════════════════════════════════════════════
# Step 2: BigFloat Schur via GenericSchur (with caching)
# ═══════════════════════════════════════════════════════════════════════

local Q_bf, T_bf
t0 = time()
if isfile(CACHE_SCHUR)
    @info "Loading cached BigFloat Schur from $CACHE_SCHUR..."
    cached_schur = Serialization.deserialize(CACHE_SCHUR)
    Q_bf = cached_schur[:Q_bf]
    T_bf = cached_schur[:T_bf]
    @info "  Loaded in $(round(time()-t0, digits=1))s"
else
    @info "Computing direct BigFloat Schur decomposition via GenericSchur..."
    @info "  Matrix size: $n × $n,  precision: $PRECISION bits"
    A_bf_real = real.(A_bf)
    F_bf = schur(A_bf_real)
    Q_bf = Complex{BigFloat}.(F_bf.Z)
    T_bf = Complex{BigFloat}.(F_bf.T)
    dt_schur = round(time()-t0, digits=1)
    @info "  BigFloat Schur in $(dt_schur)s"

    @info "Caching Schur to $CACHE_SCHUR..."
    Serialization.serialize(CACHE_SCHUR, Dict(:Q_bf => Q_bf, :T_bf => T_bf))
    @info "  Cached."
end
dt_schur = round(time()-t0, digits=1)
@info "  Step 2 complete in $(dt_schur)s"

# Residual and orthogonality defect
I_n = Matrix{Complex{BigFloat}}(I, n, n)
residual_mat = A_bf - Q_bf * T_bf * Q_bf'
res_opnorm = upper_bound_L2_opnorm(BallMatrix(residual_mat))
orth_mat = Q_bf' * Q_bf - I_n
orth_def = upper_bound_L2_opnorm(BallMatrix(orth_mat))
A_bf_opnorm = upper_bound_L2_opnorm(BallMatrix(A_bf))
@printf("  Residual ||A-QTQ'||_2           ≤ %.4e\n", Float64(res_opnorm))
@printf("  Residual ||A-QTQ'||_2 / ||A||_2 ≤ %.4e\n", Float64(res_opnorm / A_bf_opnorm))
@printf("  Orthogonality ||Q'Q-I||_2       ≤ %.4e\n", Float64(orth_def))

delta_bf = orth_def

# ═══════════════════════════════════════════════════════════════════════
# Step 3: Rigorous E_bound
# ═══════════════════════════════════════════════════════════════════════

E_bound = setrounding(BigFloat, RoundUp) do
    A_rad_opnorm + res_opnorm
end

@printf("\n  ||E|| = ||A_true - QTQ'||_2 ≤ %.4e\n", Float64(E_bound))
@printf("    of which: Schur residual = %.4e, input ||A_rad||_2 = %.4e\n",
        Float64(res_opnorm), Float64(A_rad_opnorm))
println()

# Eigenvalues sorted by |λ|
lambda_bf_all = real.(diag(T_bf))
sorted_idx = sortperm(abs.(lambda_bf_all), rev=true)

println("First $NUM_EIGS eigenvalues (sorted by |λ|):")
println("-"^60)
for i in 1:NUM_EIGS
    idx = sorted_idx[i]
    @printf("  %3d: λ = %+.15e  |λ| = %.10e\n",
            i, Float64(real(lambda_bf_all[idx])), Float64(abs(lambda_bf_all[idx])))
end
println()

# ═══════════════════════════════════════════════════════════════════════
# Step 4: ℓ_j(1) via BigFloat ordschur + Sylvester (no NK correction)
# ═══════════════════════════════════════════════════════════════════════
@info "Computing ℓ_j(1) via BigFloat ordschur + Sylvester..."
@info "  (NK eigenvector correction NOT applied — will be added in post-processing)"
println()

const BF_ONE = BigFloat(1)
const BF_TWO = BigFloat(2)
const I_m = Matrix{Complex{BigFloat}}(I, n - 1, n - 1)

ell_center     = Vector{BigFloat}(undef, NUM_EIGS)
ell_radius     = Vector{BigFloat}(undef, NUM_EIGS)
eigenvalues_out = Vector{BigFloat}(undef, NUM_EIGS)
eigenvectors_out = Matrix{BigFloat}(undef, n, NUM_EIGS)

for j in 1:NUM_EIGS
    t1 = time()

    target_pos = sorted_idx[j]
    T_ord, Q_ord = bigfloat_ordschur(T_bf, Q_bf, target_pos)

    lambda_j_bf = T_ord[1, 1]
    eigenvalues_out[j] = real(lambda_j_bf)

    # Store eigenvector (first column of Q_ord)
    eigenvectors_out[:, j] = real.(Q_ord[:, 1])

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

    # Rigorous error bound (Sylvester + perturbation, no NK correction)
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

    # Perturbation correction
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

        if isinf(R_S) || R_S * E_bound >= BF_ONE
            BigFloat(Inf)
        else
            rho * R_S^2 * E_bound / (BF_ONE - R_S * E_bound)
        end
    end

    ell_radius[j] = setrounding(BigFloat, RoundUp) do
        sylv_err + pert_corr
    end

    sign_ok = abs(ell_center[j]) > ell_radius[j] ? "YES" : "NO"
    dt_j = time() - t1
    @printf("  j=%2d: λ=%+.10e  ℓ=%+.15e ± %.2e  sep=%.2e  N=%.2e  σ≥%.2e  sylv=%.2e  pert=%.2e  %s  [%.1fs]\n",
            j, Float64(eigenvalues_out[j]), Float64(ell_center[j]), Float64(ell_radius[j]),
            Float64(sep_bf), Float64(N_norm_bf), Float64(sigma_min_bf),
            Float64(sylv_err), Float64(pert_corr), sign_ok, dt_j)
end

# ═══════════════════════════════════════════════════════════════════════
# Save results
# ═══════════════════════════════════════════════════════════════════════
@info "Saving results to $CACHE_RESULTS..."
Serialization.serialize(CACHE_RESULTS, Dict(
    :K => K,
    :precision => PRECISION,
    :num_eigs => NUM_EIGS,
    :eigenvalues => eigenvalues_out,
    :eigenvectors => eigenvectors_out,
    :ell_center => ell_center,
    :ell_radius => ell_radius,
    :E_bound => E_bound,
    :A_rad_opnorm => A_rad_opnorm,
    :res_opnorm => res_opnorm,
    :orth_def => orth_def,
))
@info "  Saved."

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
println()
println("="^80)
println("SUMMARY  K=$K, P=$PRECISION, $NUM_EIGS eigenvalues")
println("="^80)
println()
@printf("  ||E|| ≤ %.4e\n", Float64(E_bound))
println()
@printf("%-4s  %-22s  %-26s  %-12s  %-12s  %s\n",
        "j", "lambda_j", "ell_j(1)", "radius", "pert_corr", "sign?")
println("-"^95)

for j in 1:NUM_EIGS
    sign_ok = abs(ell_center[j]) > ell_radius[j] ? "YES" : "NO"
    @printf("%2d    %+.14e  %+.20e  %.4e  %.4e    %s\n",
            j, Float64(eigenvalues_out[j]), Float64(ell_center[j]),
            Float64(ell_radius[j]),
            isinf(ell_radius[j]) ? Inf : Float64(ell_radius[j]), sign_ok)
end

certified = sum(abs(ell_center[j]) > ell_radius[j] for j in 1:NUM_EIGS)
println()
println("  Certified: $certified / $NUM_EIGS")
println()
println("  BigFloat precision: $(precision(BigFloat)) bits ≈ $(round(Int, precision(BigFloat) * log10(2))) decimal digits")
println("  eps(BigFloat) = $(Float64(eps(BigFloat)))")
println()
println("="^80)
println("DONE")
println("="^80)

# ═══════════════════════════════════════════════════════════════════════
# Export portable text files for Harvard Dataverse
# ═══════════════════════════════════════════════════════════════════════
@info "Exporting portable text files for Dataverse..."

# --- Eigenvalues + ℓ_j(1) with full BigFloat precision ---
txt_file = joinpath(CACHE_DIR, "spectral_coefficients_K$(K)_P$(PRECISION).txt")
open(txt_file, "w") do io
    println(io, "# GKW operator spectral expansion coefficients")
    println(io, "# K = $K, BigFloat precision = $PRECISION bits (≈$(round(Int, PRECISION * log10(2))) decimal digits)")
    println(io, "# ||E|| = ||A_true - QTQ'||_2 ≤ ", string(E_bound))
    println(io, "# ||A_rad||_2 ≤ ", string(A_rad_opnorm))
    println(io, "# ||A_bf - QTQ'||_2 ≤ ", string(res_opnorm))
    println(io, "# ||Q'Q - I||_2 ≤ ", string(orth_def))
    println(io, "# NK correction: NOT applied (to be added in post-processing)")
    println(io, "#")
    println(io, "# Columns: j, lambda_j (BigFloat string), ell_j(1) (BigFloat string), radius (BigFloat string), certified (YES/NO)")
    println(io, "#")
    for j in 1:NUM_EIGS
        sign_ok = abs(ell_center[j]) > ell_radius[j] ? "YES" : "NO"
        println(io, j, "\t", string(eigenvalues_out[j]), "\t", string(ell_center[j]), "\t", string(ell_radius[j]), "\t", sign_ok)
    end
end
@info "  Written: $txt_file"

# --- Eigenvector coefficients (full precision) ---
vec_file = joinpath(CACHE_DIR, "eigenvectors_K$(K)_P$(PRECISION).txt")
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
            print(io, "\t", string(BigFloat(eigenvectors_out[k+1, j])))
        end
        println(io)
    end
end
@info "  Written: $vec_file"

@info "Export complete."
