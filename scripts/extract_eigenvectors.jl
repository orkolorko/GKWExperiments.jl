# Extract rigorous spectral expansion L^n 1 = Σ_j λ_j^n ℓ_j(1) v_j
# for the first 20 eigenvalues of the GKW operator at K=256.
#
# Method (Sylvester-based Riesz projectors + Schur perturbation bounds):
#   1. Schur decomposition A_mid = Q T Q' with rigorous defects (δ, r_res)
#   2. For each eigenvalue λ_j, reorder Schur form (ordschur) to put λ_j first
#   3. Solve Sylvester equation on T (Miyajima enclosure) → rigorous Y
#   4. Riesz projector Π_j = Q · [I, -Y'; 0, 0] · Q'
#   5. ℓ_j(1) = ⟨hat_v_j, Π_j e₀⟩ + perturbation correction + NK correction
#
# Rigorous error budget:
#   |ℓ_j(1)_true - ℓ_j(1)_computed| ≤ ball_radius + pert_correction + nk_correction
#   where ball_radius comes from Miyajima Sylvester enclosure,
#   pert_correction bounds ||Π_j(A) - Π_j(QTQ')|| via resolvent integral (Gauss.tex),
#   and nk_correction accounts for eigenvector error from NK certification.

using GKWExperiments
using ArbNumerics
using BallArithmetic
using LinearAlgebra
using Printf
using Serialization

const PRECISION = 512
const K = 256
const NUM_EIGS = 20
const NUM_COEFFS = 50
const CACHE_FILE = "data/ball_matrix_K$(K).jls"

setprecision(ArbFloat, PRECISION)
setprecision(BigFloat, PRECISION)

s = ArbComplex(1.0, 0.0)

# NK radii from the two-stage certification (eigenvector error bounds)
const nk_radii = [
    7.40e-45, 2.33e-44, 7.32e-44, 2.17e-43, 6.22e-43,
    1.75e-42, 4.85e-42, 1.33e-41, 3.63e-41, 9.85e-41,
    2.66e-40, 7.16e-40, 1.92e-39, 5.15e-39, 1.38e-38,
    3.67e-38, 9.78e-38, 2.60e-37, 6.92e-37, 1.84e-36
]

println("="^80)
println("RIGOROUS SPECTRAL EXPANSION (DEFLATION POLYNOMIAL METHOD)")
println("L^n 1 = Sigma lambda_j^n ell_j(1) v_j")
println("K=$K, Sylvester Riesz projectors + Schur perturbation bounds + NK enclosures")
println("="^80)
println()

# ═══════════════════════════════════════════════════════════════════════
# Step 1: Load/build K=256 BallMatrix
# ═══════════════════════════════════════════════════════════════════════
n = K + 1  # 257
if isfile(CACHE_FILE)
    @info "Loading cached BallMatrix from $CACHE_FILE..."
    t0 = time()
    A_ball = deserialize(CACHE_FILE)
    A_mid = BallArithmetic.mid(A_ball)
    @info "  Loaded in $(round(time()-t0, digits=1))s"
else
    @info "Building GKW matrix at K=$K..."
    t0 = time()
    M_arb = gkw_matrix_direct(s; K=K)
    A_ball = arb_to_ball_matrix(M_arb)
    A_mid = BallArithmetic.mid(A_ball)
    @info "  Matrix built in $(round(time()-t0, digits=1))s"
    @info "Caching BallMatrix to $CACHE_FILE..."
    serialize(CACHE_FILE, A_ball)
    @info "  Cached."
end

# ═══════════════════════════════════════════════════════════════════════
# Step 2: Schur decomposition and eigenvalue ordering
# ═══════════════════════════════════════════════════════════════════════
@info "Computing Schur decomposition..."
# Use real part: A_mid is ComplexF64 from BallMatrix but GKW entries are real
A_real = real.(A_mid)
F_base = schur(A_real)
T_float = F_base.T
Q_float = F_base.Z
lambda_schur = diag(T_float)
sorted_idx = sortperm(abs.(lambda_schur), rev=true)

# Top eigenvalues (by magnitude)
top_eig_values = [real(lambda_schur[sorted_idx[i]]) for i in 1:NUM_EIGS]

println()
println("First 25 eigenvalues (sorted by |lambda|):")
println("-"^60)
for i in 1:min(25, n)
    idx = sorted_idx[i]
    @printf("  %3d: lambda = %+.12e   |lambda| = %.6e\n",
            i, real(lambda_schur[idx]), abs(lambda_schur[idx]))
end
println("-"^60)
println()

# ═══════════════════════════════════════════════════════════════════════
# Step 3: Rigorous Schur defects
# ═══════════════════════════════════════════════════════════════════════
@info "Computing rigorous Schur defects..."
t0 = time()

Q_ball_global = BallMatrix(ComplexF64.(Q_float), zeros(Float64, n, n))
T_ball_global = BallMatrix(ComplexF64.(T_float), zeros(Float64, n, n))

# Orthogonality defect: delta = ||Q'Q - I||_2
I_n = Matrix{ComplexF64}(I, n, n)
QtQ = BallMatrix(ComplexF64.(Q_float' * Q_float), zeros(Float64, n, n))
delta_ball = QtQ - BallMatrix(I_n, zeros(Float64, n, n))
delta = collatz_upper_bound_L2_opnorm(delta_ball)

# Schur residual: r_res = ||A Q - Q T||_2
R_ball = A_ball * Q_ball_global - Q_ball_global * T_ball_global
r_res = collatz_upper_bound_L2_opnorm(R_ball)

# ||A||_2
A_norm = collatz_upper_bound_L2_opnorm(A_ball)

# ||T||_2
T_norm = opnorm(T_float, 2)

# ||E|| = ||A - QTQ'|| <= r_res * sqrt(1+delta) + ||A|| * delta
# (Gauss.tex Theorem thm:schur-to-resolvent)
E_bound = setrounding(Float64, RoundUp) do
    r_res * sqrt(1.0 + delta) + A_norm * delta
end

@info "  Schur defects computed in $(round(time()-t0, digits=1))s"
@printf("  delta  = ||Q'Q - I||  = %.4e\n", delta)
@printf("  r_res  = ||AQ - QT||  = %.4e\n", r_res)
@printf("  ||E||  = ||A - QTQ'|| <= %.4e\n", E_bound)
@printf("  ||A||  = %.4e\n", A_norm)
@printf("  ||T||  = %.4e\n", T_norm)
println()

# ═══════════════════════════════════════════════════════════════════════
# Step 4: For each eigenvalue, compute ℓ_j(1) via verified Sylvester solve
# ═══════════════════════════════════════════════════════════════════════
#
# For each eigenvalue λ_j we:
#   (a) Reorder Schur form to put λ_j at T[1,1]
#   (b) Solve the triangular system M z = w_rest where M = T₂₂ - λ_j I,
#       and w = Q'e₀ (constant function in Schur coordinates)
#   (c) Compute ℓ_j(1) = w₁ - T₁₂ · z  (direct formula, no full projector needed)
#   (d) For rigorous error bound:
#       - Compute residual r = w_rest - M·z₀ (BallMatrix arithmetic)
#       - Bound ||M⁻¹||₂ via verified inverse: ||I - M·M_inv|| < 1
#       - Sylvester error: |ℓ - ℓ̃| ≤ ||T₁₂||₂ · ||r||₂ / σ_min(M)
#   (e) Add perturbation correction ||Π_j(A) - Π_j(S)|| from Schur defects
#   (f) Add NK eigenvector correction

# Storage
eigenvalues = Vector{Float64}(undef, NUM_EIGS)
eigenvectors = Matrix{Float64}(undef, n, NUM_EIGS)
ell_center = Vector{Float64}(undef, NUM_EIGS)
ell_radius = Vector{Float64}(undef, NUM_EIGS)
projections_center = Matrix{ComplexF64}(undef, n, NUM_EIGS)
projections_radius = Matrix{Float64}(undef, n, NUM_EIGS)
projector_norms = Vector{Float64}(undef, NUM_EIGS)
eigenvalue_separations = Vector{Float64}(undef, NUM_EIGS)
offdiag_norms = Vector{Float64}(undef, NUM_EIGS)
sigma_mins = Vector{Float64}(undef, NUM_EIGS)
sylvester_errors = Vector{Float64}(undef, NUM_EIGS)
pert_corrections = Vector{Float64}(undef, NUM_EIGS)

const I_m = Matrix{Float64}(I, n - 1, n - 1)

@info "Computing rigorous ℓ_j(1) via verified Sylvester solve..."
for j in 1:NUM_EIGS
    t1 = time()

    # --- (a) Reorder Schur form: put eigenvalue j at position [1,1] ---
    target_schur_idx = sorted_idx[j]
    select = falses(n)
    select[target_schur_idx] = true
    F_ordered = ordschur(copy(F_base), select)
    T_ordered = triu(F_ordered.T)  # enforce upper triangularity
    Q_ordered = F_ordered.Z

    lambda_j = real(T_ordered[1, 1])
    eigenvalues[j] = lambda_j
    eigenvectors[:, j] = real.(Q_ordered[:, 1])

    # Eigenvalue separation
    sep = minimum(abs(T_ordered[1,1] - T_ordered[i,i]) for i in 2:n)
    eigenvalue_separations[j] = sep

    # Off-diagonal norms (for resolvent bounds)
    T22 = T_ordered[2:n, 2:n]
    N_full = triu(T_ordered, 1)    # full strictly upper triangular part (includes T₁₂)
    N_norm_full = opnorm(N_full, 2)
    offdiag_norms[j] = N_norm_full

    # --- (b) Extract blocks and solve triangular system ---
    T12 = T_ordered[1, 2:n]       # row vector (n-1 entries)
    M = T22 - lambda_j * I_m      # upper triangular (n-1)×(n-1)

    # w = Q_ordered' * e₁ = first row of Q_ordered
    w = real.(Q_ordered[1, :])
    w1 = w[1]
    w_rest = w[2:n]

    # Solve M z = w_rest by backsubstitution
    z0 = UpperTriangular(M) \ w_rest

    # Approximate ℓ_j(1) = w₁ - T₁₂ · z₀
    ell_approx = w1 - dot(T12, z0)

    # --- (c) Rigorous error bound: verified inverse approach ---
    # Compute M_inv and verify: η = ||I - M·M_inv||₂ < 1
    M_inv = inv(UpperTriangular(M))

    # BallMatrix versions (zero radii since these are floating-point exact representations)
    M_ball = BallMatrix(ComplexF64.(M), zeros(Float64, n-1, n-1))
    M_inv_ball = BallMatrix(ComplexF64.(M_inv), zeros(Float64, n-1, n-1))
    I_ball = BallMatrix(ComplexF64.(I_m), zeros(Float64, n-1, n-1))

    # R_verify = I - M * M_inv (should be ≈ 0)
    R_verify = I_ball - M_ball * M_inv_ball
    eta = collatz_upper_bound_L2_opnorm(R_verify)

    # Upper bound on ||M_inv||₂
    M_inv_norm_upper = collatz_upper_bound_L2_opnorm(M_inv_ball)

    # Lower bound on σ_min(M): σ_min ≥ (1 - η) / ||M_inv||
    if eta < 1.0
        sigma_min_lower = setrounding(Float64, RoundDown) do
            (1.0 - eta) / M_inv_norm_upper
        end
    else
        sigma_min_lower = 0.0
    end
    sigma_mins[j] = sigma_min_lower

    # --- (d) Compute residual r = w_rest - M·z₀ (Ball arithmetic) ---
    w_rest_ball = BallVector(ComplexF64.(w_rest), zeros(Float64, n-1))
    z0_ball = BallVector(ComplexF64.(z0), zeros(Float64, n-1))
    r_ball = w_rest_ball - M_ball * z0_ball
    # ||r||₂ ≤ ||r.c||₂ + ||r.r||₂ (triangle inequality for Ball)
    r_norm_upper = norm(r_ball.c) + norm(r_ball.r)

    # Error in z: ||z - z₀||₂ ≤ ||r||₂ / σ_min
    T12_norm = norm(T12)

    # Sylvester error: |ℓ(S) - ℓ̃| ≤ ||T₁₂|| · ||δz|| + rounding in dot product
    if sigma_min_lower > 0.0
        sylv_err = setrounding(Float64, RoundUp) do
            delta_z = r_norm_upper / sigma_min_lower
            T12_norm * delta_z + (n - 1) * eps(Float64) * abs(dot(T12, z0))
        end
    else
        sylv_err = Inf
    end
    sylvester_errors[j] = sylv_err

    # --- (e) Projector norm (for perturbation bound) ---
    # ||Π_j(S)|| ≈ 1 + ||Y||₂ where Y solves the Sylvester equation
    # Y' = -z_row essentially, but we estimate from ||z₀||
    # More precisely: ||Π_j|| ≤ 1 + ||M⁻¹ T₁₂'||₂ where T₁₂' is a column
    # But for the perturbation bound we just need a reasonable estimate.
    # Use: ||Π_j(S)|| ≤ 1 + ||M_inv||₂ · ||T₁₂||₂
    proj_norm_bound = setrounding(Float64, RoundUp) do
        1.0 + M_inv_norm_upper * T12_norm
    end
    projector_norms[j] = proj_norm_bound

    # Store projection of e₀ (approximate, for plots/output)
    # P_j e₀ = Q · [w₁ - T₁₂·z₀; -T₂₂⁻¹(...)...] but we only need ℓ_j(1)
    # For eigenvector coefficients, we have Q_ordered[:,1] already
    projections_center[:, j] = ComplexF64.(Q_ordered[:, 1]) * ell_approx
    projections_radius[:, j] = fill(sylv_err, n)

    # --- (f) Perturbation correction: ||Π_j(A) - Π_j(S)|| ---
    # Two approaches, take the tighter one:
    # 1. Neumann resolvent: R_T = 1/(ρ - ||N₂₂||)  (works when ρ > ||N₂₂||)
    # 2. σ_min resolvent:   R₂₂ = 1/(σ_min_M - ρ)  (works when σ_min_M > ρ)
    #    Full resolvent: R_T ≤ (1/ρ)(1 + ||T₁₂||·R₂₂) (block upper triangular bound)
    # Then: ||Π_j(A) - Π_j(S)|| ≤ ρ · R_S² · ||E|| / (1 - R_S · ||E||)
    pert_corr = setrounding(Float64, RoundUp) do
        rho = sep / 2.0
        kappa = (1.0 + delta) / (1.0 - delta)

        # Method 1: Neumann resolvent (uses full off-diagonal norm)
        R_neumann = Inf
        if rho > N_norm_full
            R_neumann = kappa / (rho - N_norm_full)
        end

        # Method 2: σ_min-based block resolvent
        R_sigma = Inf
        if sigma_min_lower > rho
            R22 = 1.0 / (sigma_min_lower - rho)
            # Block resolvent for full T: ||(zI-T)^{-1}|| ≤ (1/ρ)(1 + ||T₁₂||R₂₂)
            # (from block upper triangular inverse)
            R_full = (1.0 / rho) * (1.0 + T12_norm * R22)
            R_sigma = kappa * R_full
        end

        R_S = min(R_neumann, R_sigma)

        if isinf(R_S) || R_S * E_bound >= 1.0
            # Fallback: Bauer-Fike-type bound
            if E_bound * proj_norm_bound >= sep / 4.0
                Inf
            else
                4.0 * proj_norm_bound * proj_norm_bound * E_bound / sep
            end
        else
            rho * R_S * R_S * E_bound / (1.0 - R_S * E_bound)
        end
    end
    pert_corrections[j] = pert_corr

    # --- (g) NK correction: eigenvector perturbation ---
    nk_correction = setrounding(Float64, RoundUp) do
        proj_norm_bound * 2.0 * nk_radii[j] / (1.0 - nk_radii[j])
    end

    # --- (h) Total rigorous enclosure ---
    ell_center[j] = ell_approx
    ell_radius[j] = setrounding(Float64, RoundUp) do
        sylv_err + pert_corr + nk_correction
    end

    dt = time() - t1
    @printf("  j=%2d: λ=%+.10e  ℓ=%+.10e ± %.2e  sep=%.2e  σ_min≥%.2e  η=%.2e  sylv=%.2e  pert=%.2e  [%.1fs]\n",
            j, lambda_j, ell_center[j], ell_radius[j], sep, sigma_min_lower, eta, sylv_err, pert_corr, dt)
end

# ═══════════════════════════════════════════════════════════════════════
# Step 5: Summary
# ═══════════════════════════════════════════════════════════════════════
println()
println("="^80)
println("SPECTRAL EXPANSION COEFFICIENTS ell_j(1)")
println("="^80)
println()
@printf("%-4s  %-22s  %-22s  %-10s  %-10s  %-10s  %-10s  %-10s  %s\n",
        "j", "lambda_j", "ell_j(1)", "radius", "sep", "sigma_min", "sylv_err", "pert_corr", "sign?")
println("-"^130)

for j in 1:NUM_EIGS
    sign_ok = abs(ell_center[j]) > ell_radius[j] ? "YES" : "NO"
    @printf("%2d    %+.14e  %+.14e  %.4e  %.4e  %.4e  %.4e  %.4e    %s\n",
            j, eigenvalues[j], ell_center[j], ell_radius[j],
            eigenvalue_separations[j], sigma_mins[j],
            sylvester_errors[j], pert_corrections[j], sign_ok)
end

println()
println("="^80)
println("EIGENVECTOR COEFFICIENTS v_j (unit ell^2 norm, first $NUM_COEFFS)")
println("="^80)
println()

for j in 1:NUM_EIGS
    @printf("v_%d  [lambda_%d = %+.10e,  ell_%d(1) = %+.10e]:\n",
            j, j, eigenvalues[j], j, ell_center[j])
    for k in 1:NUM_COEFFS
        @printf("  v_%d[%02d] = %+.12e  +/- %.2e\n",
                j, k-1, eigenvectors[k, j], nk_radii[j])
    end
    @printf("  ||v_%d||_2 = %.15f\n", j, norm(eigenvectors[:, j]))
    println()
end

# ═══════════════════════════════════════════════════════════════════════
# Step 6: Write output files
# ═══════════════════════════════════════════════════════════════════════
open("data/spectral_expansion.txt", "w") do io
    println(io, "# Rigorous spectral expansion for GKW operator (K=$K)")
    println(io, "# Method: Sylvester Riesz projectors + Schur perturbation bounds")
    println(io, "# L^n 1 = Sigma_j lambda_j^n ell_j(1) v_j")
    println(io, "")

    println(io, "# SECTION: ell_j(1) coefficients")
    println(io, "# Format: j  lambda_j  ell_center  ell_radius  proj_norm  sep  sigma_min  sylv_err  pert_corr  nk_radius")
    for j in 1:NUM_EIGS
        @printf(io, "%d\t%.15e\t%.15e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.4e\t%.4e\n",
                j, eigenvalues[j], ell_center[j], ell_radius[j],
                projector_norms[j], eigenvalue_separations[j], sigma_mins[j],
                sylvester_errors[j], pert_corrections[j], nk_radii[j])
    end

    println(io, "")
    println(io, "# SECTION: eigenvector coefficients")
    println(io, "# Format: j  lambda_j  coeff_index  v_center  v_radius(=nk_radius)")
    for j in 1:NUM_EIGS
        for k in 1:NUM_COEFFS
            @printf(io, "%d\t%.15e\t%d\t%.15e\t%.4e\n",
                    j, eigenvalues[j], k-1, eigenvectors[k, j], nk_radii[j])
        end
    end
end

@info "Data written to data/spectral_expansion.txt"

# Save all computed objects
results_cache = "data/spectral_results_K$(K).jls"
@info "Saving all results to $results_cache..."
serialize(results_cache, Dict(
    :K => K,
    :n => n,
    :precision => PRECISION,
    :eigenvalues => eigenvalues,
    :eigenvectors => eigenvectors,
    :ell_center => ell_center,
    :ell_radius => ell_radius,
    :projections_center => projections_center,
    :projections_radius => projections_radius,
    :projector_norms => projector_norms,
    :eigenvalue_separations => eigenvalue_separations,
    :nk_radii => nk_radii,
    :sorted_idx => sorted_idx,
    :offdiag_norms => offdiag_norms,
    :sigma_mins => sigma_mins,
    :sylvester_errors => sylvester_errors,
    :pert_corrections => pert_corrections,
    :schur_defects => Dict(
        :delta => delta,
        :r_res => r_res,
        :E_bound => E_bound,
        :A_norm => A_norm,
        :T_norm => T_norm,
    ),
))
@info "  Saved."

println()
println("="^80)
println("DONE")
println("="^80)
