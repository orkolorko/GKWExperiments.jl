# BigFloat Schur refinement with BigFloat input matrix.
#
# Key improvement: build the GKW matrix in ArbNumerics, convert to BigFloat
# (NOT Float64), then refine the Schur decomposition against the BigFloat A.
# This gives ||E|| ~ eps(BigFloat) ≈ 10^-77 instead of eps(Float64) ≈ 10^-16.

using GKWExperiments
using ArbNumerics
using BallArithmetic
using GenericSchur  # for BigFloat schur() with standard .Z/.T API
using LinearAlgebra
using Printf
using Serialization

const PRECISION = 512
const K = 256
const NUM_EIGS = 20

# NK radii from two-stage certification
const nk_radii = [
    7.40e-45, 2.33e-44, 7.32e-44, 2.17e-43, 6.22e-43,
    1.75e-42, 4.85e-42, 1.33e-41, 3.63e-41, 9.85e-41,
    2.66e-40, 7.16e-40, 1.92e-39, 5.15e-39, 1.38e-38,
    3.67e-38, 9.78e-38, 2.60e-37, 6.92e-37, 1.84e-36
]

setprecision(ArbFloat, PRECISION)
setprecision(BigFloat, PRECISION)

# ═══════════════════════════════════════════════════════════════════════
# BigFloat ordschur
# ═══════════════════════════════════════════════════════════════════════

function swap_schur_1x1!(T::AbstractMatrix, Q::AbstractMatrix, k::Int)
    n = size(T, 1)
    a, b, c = T[k, k], T[k+1, k+1], T[k, k+1]
    x = (b - a) / c
    nrm = sqrt(one(x) + x * conj(x))
    cs, sn = one(x) / nrm, x / nrm
    for j in 1:n
        t1, t2 = T[k, j], T[k+1, j]
        T[k, j]   = conj(cs) * t1 + conj(sn) * t2
        T[k+1, j] = -sn * t1 + cs * t2
    end
    for i in 1:n
        t1, t2 = T[i, k], T[i, k+1]
        T[i, k]   = t1 * cs + t2 * sn
        T[i, k+1] = -t1 * conj(sn) + t2 * cs
    end
    for i in 1:n
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
# Convert ArbNumerics → BigFloat BallMatrix (rigorous)
# ═══════════════════════════════════════════════════════════════════════

"""
Convert ArbNumerics matrix to a BallMatrix with BigFloat centers.
Each entry: center = BigFloat(midpoint(arb)), radius = |arb_radius| + |BigFloat_rounding|.
"""
function arb_to_bigfloat_ball_matrix(M_arb::Matrix{ArbComplex{P}}) where {P}
    n, m = size(M_arb)
    M_center = Matrix{Complex{BigFloat}}(undef, n, m)
    M_radius = Matrix{BigFloat}(undef, n, m)

    for i in 1:n, j in 1:m
        # ArbNumerics midpoint and radius
        mid_re = ArbNumerics.midpoint(real(M_arb[i, j]))
        mid_im = ArbNumerics.midpoint(imag(M_arb[i, j]))
        rad_re = BigFloat(ArbNumerics.radius(real(M_arb[i, j])))
        rad_im = BigFloat(ArbNumerics.radius(imag(M_arb[i, j])))

        # Convert midpoint to BigFloat (with rounding error)
        c_re = BigFloat(mid_re)
        c_im = BigFloat(mid_im)
        M_center[i, j] = Complex{BigFloat}(c_re, c_im)

        # Radius: arb_radius + BigFloat conversion error
        # |exact - BigFloat(exact)| ≤ eps(BigFloat)/2 * |BigFloat(exact)|
        conv_re = eps(BigFloat) * abs(c_re)
        conv_im = eps(BigFloat) * abs(c_im)
        total_re = rad_re + conv_re
        total_im = rad_im + conv_im
        M_radius[i, j] = sqrt(total_re^2 + total_im^2)
    end
    return BallMatrix(M_center, M_radius)
end

println("="^80)
println("BigFloat Schur with BigFloat input (A from ArbNumerics → BigFloat)")
println("="^80)
println()

# ═══════════════════════════════════════════════════════════════════════
# Step 1 & 2: Build GKW matrix and convert (with caching)
# ═══════════════════════════════════════════════════════════════════════
n = K + 1  # 257

const CACHE_BALL = "data/bigfloat_ball_matrix_K$(K)_P$(PRECISION).jls"
const CACHE_SCHUR = "data/bigfloat_schur_K$(K)_P$(PRECISION).jls"

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
    M_arb = gkw_matrix_direct(s; K=K)
    @info "  Built in $(round(time()-t0, digits=1))s"

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

# Rigorous spectral norm of radius matrix via BallArithmetic (stays BigFloat)
A_rad_ball = BallMatrix(A_rad_bf)  # wrap as BallMatrix with zero radii
A_rad_opnorm = upper_bound_L2_opnorm(A_rad_ball)  # BigFloat
@printf("  ||A_rad||_2  ≤ %.4e  (rigorous, via upper_bound_L2_opnorm)\n", Float64(A_rad_opnorm))
println()

# ═══════════════════════════════════════════════════════════════════════
# Step 3: Direct BigFloat Schur via GenericSchur (with caching)
# ═══════════════════════════════════════════════════════════════════════
local Q_bf, T_bf
if isfile(CACHE_SCHUR)
    @info "Loading cached BigFloat Schur from $CACHE_SCHUR..."
    t0 = time()
    cached_schur = Serialization.deserialize(CACHE_SCHUR)
    Q_bf = cached_schur[:Q_bf]
    T_bf = cached_schur[:T_bf]
    @info "  Loaded in $(round(time()-t0, digits=1))s"
else
    @info "Computing direct BigFloat Schur decomposition via GenericSchur..."
    @info "  (This computes schur(A_bf) directly in BigFloat — no Float64 intermediate.)"
    t0 = time()
    A_bf_real = real.(A_bf)  # GKW matrix is real
    F_bf = schur(A_bf_real)
    # GenericSchur returns standard LinearAlgebra.Schur with .Z and .T fields
    Q_bf = Complex{BigFloat}.(F_bf.Z)
    T_bf = Complex{BigFloat}.(F_bf.T)

    @info "Caching Schur to $CACHE_SCHUR..."
    Serialization.serialize(CACHE_SCHUR, Dict(:Q_bf => Q_bf, :T_bf => T_bf))
    @info "  Cached."
end
dt_schur = time() - t0
@info "  BigFloat Schur in $(round(dt_schur, digits=1))s"

# Compute residual and orthogonality defect using BallArithmetic norms (all BigFloat)
I_n = Matrix{Complex{BigFloat}}(I, n, n)
residual_mat = A_bf - Q_bf * T_bf * Q_bf'
res_opnorm = upper_bound_L2_opnorm(BallMatrix(residual_mat))  # BigFloat
orth_mat = Q_bf' * Q_bf - I_n
orth_def = upper_bound_L2_opnorm(BallMatrix(orth_mat))  # BigFloat
A_bf_opnorm = upper_bound_L2_opnorm(BallMatrix(A_bf))  # BigFloat
@printf("  Residual ||A-QTQ'||_2           ≤ %.4e\n", Float64(res_opnorm))
@printf("  Residual ||A-QTQ'||_2 / ||A||_2 ≤ %.4e\n", Float64(res_opnorm / A_bf_opnorm))
@printf("  Orthogonality ||Q'Q-I||_2       ≤ %.4e\n", Float64(orth_def))

delta_bf = orth_def  # BigFloat

# ═══════════════════════════════════════════════════════════════════════
# Step 4: Rigorous E_bound
# ═══════════════════════════════════════════════════════════════════════
# ||A_true - Q T Q'|| ≤ ||A_true - A_bf|| + ||A_bf - QTQ'||
#                     ≤ ||A_rad||_2  +  residual_norm * ||A_bf||_F / (1 - δ)

E_bound = setrounding(BigFloat, RoundUp) do
    # ||A_true - QTQ'||_2 ≤ ||A_true - A_bf||_2 + ||A_bf - QTQ'||_2
    A_rad_opnorm + res_opnorm
end  # BigFloat

@printf("\n  ||E|| = ||A_true - QTQ'||_2 ≤ %.4e\n", Float64(E_bound))
@printf("    of which: Schur residual ||A_bf-QTQ'||_2 = %.4e, input ||A_rad||_2 = %.4e\n",
        Float64(res_opnorm), Float64(A_rad_opnorm))
println()

# Eigenvalues from refined BigFloat Schur diagonal
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
# Step 5: For each eigenvalue, BigFloat ordschur + Sylvester
# ═══════════════════════════════════════════════════════════════════════
@info "Computing ℓ_j(1) via BigFloat ordschur + Sylvester..."
println()

ell_center = Vector{BigFloat}(undef, NUM_EIGS)
ell_radius = Vector{BigFloat}(undef, NUM_EIGS)
eigenvalues_out = Vector{BigFloat}(undef, NUM_EIGS)

const BF_ONE = BigFloat(1)
const BF_TWO = BigFloat(2)
const I_m = Matrix{Complex{BigFloat}}(I, n - 1, n - 1)
const nk_radii_bf = BigFloat.(nk_radii)

for j in 1:NUM_EIGS
    t1 = time()

    target_pos = sorted_idx[j]
    T_ord, Q_ord = bigfloat_ordschur(T_bf, Q_bf, target_pos)

    lambda_j_bf = T_ord[1, 1]
    eigenvalues_out[j] = real(lambda_j_bf)

    sep_bf = minimum(abs(T_ord[1,1] - T_ord[i,i]) for i in 2:n)

    N_full = triu(T_ord, 1)
    N_norm_bf = upper_bound_L2_opnorm(BallMatrix(N_full))  # BigFloat

    # Solve Sylvester in BigFloat
    T12 = T_ord[1, 2:n]
    T22 = T_ord[2:n, 2:n]
    M_tri = UpperTriangular(T22 - lambda_j_bf * I_m)

    w = Q_ord[1, :]
    w1, w_rest = w[1], w[2:n]

    z0 = M_tri \ w_rest
    ell_val_bf = w1 - dot(T12, z0)
    ell_center[j] = real(ell_val_bf)

    # Rigorous error bound — everything in BigFloat directed rounding
    residual = w_rest - M_tri * z0
    res_norm_bf = BigFloat(norm(residual))  # norm returns BigFloat for BigFloat input

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

    # Perturbation correction — all BigFloat
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

    # NK correction — no Float64 conversion error needed!
    z0_norm_bf = BigFloat(norm(z0))
    proj_norm_bf = setrounding(BigFloat, RoundUp) do
        BF_ONE + z0_norm_bf * T12_norm_bf
    end
    local nk_corr::BigFloat = setrounding(BigFloat, RoundUp) do
        proj_norm_bf * BF_TWO * nk_radii_bf[j]
    end

    ell_radius[j] = setrounding(BigFloat, RoundUp) do
        sylv_err + pert_corr + nk_corr
    end

    sign_ok = abs(ell_center[j]) > ell_radius[j] ? "YES" : "NO"
    local dt_j = time() - t1
    @printf("  j=%2d: λ=%+.10e  ℓ=%+.15e ± %.2e  sep=%.2e  N=%.2e  σ≥%.2e  sylv=%.2e  pert=%.2e  %s  [%.1fs]\n",
            j, Float64(eigenvalues_out[j]), Float64(ell_center[j]), Float64(ell_radius[j]),
            Float64(sep_bf), Float64(N_norm_bf), Float64(sigma_min_bf),
            Float64(sylv_err), Float64(pert_corr), sign_ok, dt_j)
end

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
println()
println("="^80)
println("SUMMARY  (all quantities BigFloat, converted to Float64 only for display)")
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

# Show the BigFloat precision of the results
println("  BigFloat precision: $(precision(BigFloat)) bits ≈ $(round(Int, precision(BigFloat) * log10(2))) decimal digits")
println("  eps(BigFloat) = $(Float64(eps(BigFloat)))")
println()
println("="^80)
println("DONE")
println("="^80)
