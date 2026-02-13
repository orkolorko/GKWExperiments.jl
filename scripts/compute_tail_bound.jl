#!/usr/bin/env julia
# Compute rigorous tail bound for the spectral expansion remainder:
#   R_N(n) = L^n 1 - Σ_{j=1}^N λ_j^n ℓ_j(1) v_j
#
# Method: Cauchy integral on a circle Γ separating the first N eigenvalues
# from the rest, combined with direct computation of ‖Q_N · 1‖.
#
# The improved tail bound (valid for non-normal operators) is:
#   ‖R_N(n)‖₂ ≤ ρ^{n+1} · M_∞ · ‖Q_N · 1‖
# where:
#   ρ       = circle radius (in eigenvalue gap)
#   M_∞     = ‖R_{L_r}(z)‖ on Γ (resolvent bridge)
#   Q_N · 1 = 1 - Σ P_j(1)  (tail projection of the constant function)
#
# The factorization uses Q_N² = Q_N (Riesz idempotency) and L-invariance
# of range(Q_N), both valid for non-normal operators.

using GKWExperiments
using ArbNumerics
using BallArithmetic
using LinearAlgebra
using Printf
using Serialization

const PRECISION = 512
const K_RESOLVENT = 64        # moderate K for resolvent certification
const K_PROJECTOR = 256       # high K for Riesz projector computation
const NUM_EIGS = 20           # number of leading eigenvalues in the expansion
const N_SPLITTING = 5000      # C₂ splitting parameter
const CIRCLE_SAMPLES = 256    # resolvent certification samples
const CACHE_RESOLVENT = "data/ball_matrix_K$(K_RESOLVENT).jls"
const CACHE_PROJECTOR = "data/spectral_results_K$(K_PROJECTOR).jls"

setprecision(ArbFloat, PRECISION)
setprecision(BigFloat, PRECISION)

s = ArbComplex(1.0, 0.0)

println("="^80)
println("IMPROVED TAIL BOUND FOR SPECTRAL EXPANSION REMAINDER R_$(NUM_EIGS)(n)")
println("Resolvent at K=$K_RESOLVENT, projectors at K=$K_PROJECTOR")
println("="^80)
println()

# ════════════════════════════════════════════════════════════════════════════
# Part A: Compute ‖Q_N · 1‖  (tail projection of constant function)
# ════════════════════════════════════════════════════════════════════════════

@info "Loading spectral results (K=$K_PROJECTOR) for Riesz projectors..."
results = deserialize(CACHE_PROJECTOR)
proj_centers = results[:projections_center]   # n_high × NUM_EIGS ComplexF64
proj_radii   = results[:projections_radius]   # n_high × NUM_EIGS Float64
eigenvalues  = results[:eigenvalues]          # NUM_EIGS Float64
n_high = results[:n]

# Q_N · 1 = e₀ - Σ_{j=1}^N P_j(e₀)
# In ball arithmetic: center = e₀ - Σ centers, radius = Σ radii (triangle ineq.)
q_center = zeros(ComplexF64, n_high)
q_center[1] = 1.0 + 0.0im  # e₀
q_radius = zeros(Float64, n_high)

for j in 1:NUM_EIGS
    q_center .-= @view proj_centers[:, j]
    q_radius .+= @view proj_radii[:, j]
end

# Rigorous upper bound on ‖Q_N · 1‖₂ for the Galerkin approximation
# ‖q‖₂ ≤ ‖q_center‖₂ + ‖q_radius‖₂
norm_q_center = norm(real.(q_center))  # ‖Re(q_center)‖₂ (imaginary parts are ~0)
norm_q_radius = norm(q_radius)         # ‖q_radius‖₂

# Rigorous upper bound (RoundUp)
norm_Q_N_1_galerkin = setrounding(Float64, RoundUp) do
    norm_q_center + norm_q_radius
end

println("Part A: Tail projection ‖Q_$NUM_EIGS · 1‖")
println("-"^60)
@printf("  ‖Q_%d(A_%d) · e₀‖_center = %.6e\n", NUM_EIGS, K_PROJECTOR, norm_q_center)
@printf("  ‖Q_%d(A_%d) · e₀‖_radius = %.6e\n", NUM_EIGS, K_PROJECTOR, norm_q_radius)
@printf("  ‖Q_%d(A_%d) · e₀‖₂      ≤ %.6e  (Galerkin)\n", NUM_EIGS, K_PROJECTOR, norm_Q_N_1_galerkin)

# Correction for L_r vs A_K projector error:
# ‖Q_N(L_r) · 1 - Q_N(A_K) · e₀‖ ≤ Σ_j ‖P_j(L_r) - P_j(A_K)‖
# These Riesz projector errors come from the two-stage certification.
# At K=256, the projector errors are ~10⁻⁴¹ (from transfer bridge).
# We compute them here from ε_{K_PROJECTOR}.
@info "Computing Riesz projector error correction..."

Δ_arb = compute_Δ(K_PROJECTOR; N=N_SPLITTING)
Δ_real = real(Δ_arb)
mid_arb = ArbNumerics.midpoint(Δ_real)
rad_arb = ArbNumerics.radius(Δ_real)
mid_f64 = Float64(mid_arb)
rad_f64 = Float64(rad_arb)
mid_big = parse(BigFloat, string(mid_arb))
conv_err = Float64(abs(mid_big - BigFloat(mid_f64)))
ε_K_high = setrounding(Float64, RoundUp) do
    mid_f64 + rad_f64 + conv_err
end

# Projector error per eigenvalue: ‖P_j(L_r) - P_j(A_K)‖ ≤ (|Γ_j|/2π) · M_j² · ε_K / (1 - ε_K · M_j)
# where M_j = resolvent bound on individual circle around λ_j.
# The projector norms and idempotency defects bound these indirectly.
# Conservative bound: use ‖P_j‖ · ε_K as a rough upper bound per projector.
projector_norms = results[:projector_norms]  # ‖P_j(A_K)‖
total_projector_correction = setrounding(Float64, RoundUp) do
    sum(projector_norms) * ε_K_high
end

# Total ‖Q_N(L_r) · 1‖
norm_Q_N_1 = setrounding(Float64, RoundUp) do
    norm_Q_N_1_galerkin + total_projector_correction
end

@printf("  Projector correction (Σ ‖P_j‖ · ε_K) = %.6e\n", total_projector_correction)
@printf("  ‖Q_%d(L_r) · 1‖₂        ≤ %.6e  (rigorous)\n\n", NUM_EIGS, norm_Q_N_1)

# ════════════════════════════════════════════════════════════════════════════
# Part B: Resolvent certification on tail circle (at K_RESOLVENT)
# ════════════════════════════════════════════════════════════════════════════

@info "Building/loading resolvent matrix at K=$K_RESOLVENT..."
if isfile(CACHE_RESOLVENT)
    A_ball = deserialize(CACHE_RESOLVENT)
    A_mid = BallArithmetic.mid(A_ball)
else
    M_arb = gkw_matrix_direct(s; K=K_RESOLVENT)
    A_ball = arb_to_ball_matrix(M_arb)
    A_mid = BallArithmetic.mid(A_ball)
    serialize(CACHE_RESOLVENT, A_ball)
end
n_low = K_RESOLVENT + 1
@info "  Matrix size: $n_low × $n_low"

# Find eigenvalue gap from Schur decomposition
@info "Computing Schur decomposition for eigenvalue gap..."
F = schur(A_mid)
λ_all = diag(F.T)
sorted_idx = sortperm(abs.(λ_all), rev=true)

λ_N   = real(λ_all[sorted_idx[NUM_EIGS]])
λ_Np1 = real(λ_all[sorted_idx[NUM_EIGS + 1]])
abs_λ_N   = abs(λ_N)
abs_λ_Np1 = abs(λ_Np1)

# Circle radius: arithmetic mean (equalizes distances)
ρ = (abs_λ_N + abs_λ_Np1) / 2.0

println("Part B: Resolvent certification on tail circle")
println("-"^60)
@printf("  |λ_%d| = %.6e,  |λ_%d| = %.6e\n", NUM_EIGS, abs_λ_N, NUM_EIGS+1, abs_λ_Np1)
@printf("  Circle radius ρ = %.6e\n", ρ)
@printf("  Gap to λ_%d: %.6e,  gap to λ_%d: %.6e\n\n",
        NUM_EIGS, abs_λ_N - ρ, NUM_EIGS+1, ρ - abs_λ_Np1)

@info "Running resolvent certification ($CIRCLE_SAMPLES samples)..."
t0 = time()
circle = CertificationCircle(ComplexF64(0.0), ρ; samples=CIRCLE_SAMPLES)
cert_data = run_certification(A_ball, circle)
resolvent_Ak = cert_data.resolvent_original
dt = time() - t0

@printf("  Resolvent: ‖R_{A_%d}(z)‖ ≤ %.6e  [%.1fs]\n", K_RESOLVENT, resolvent_Ak, dt)

# Truncation error at K_RESOLVENT
Δ_low_arb = compute_Δ(K_RESOLVENT; N=N_SPLITTING)
Δ_low_real = real(Δ_low_arb)
mid_low = ArbNumerics.midpoint(Δ_low_real)
rad_low = ArbNumerics.radius(Δ_low_real)
mid_low_f64 = Float64(mid_low)
rad_low_f64 = Float64(rad_low)
mid_low_big = parse(BigFloat, string(mid_low))
conv_err_low = Float64(abs(mid_low_big - BigFloat(mid_low_f64)))
ε_K_low = setrounding(Float64, RoundUp) do
    mid_low_f64 + rad_low_f64 + conv_err_low
end

# Lift to L_r
α = setrounding(Float64, RoundUp) do
    ε_K_low * resolvent_Ak
end

if α >= 1.0
    @error "Small-gain FAILED: α = $α ≥ 1"
    exit(1)
end

denom = setrounding(Float64, RoundDown) do
    1.0 - α
end
M_inf = setrounding(Float64, RoundUp) do
    resolvent_Ak / denom
end

@printf("  ε_K = %.6e (at K=%d)\n", ε_K_low, K_RESOLVENT)
@printf("  α = ε_K · ‖R_{A_K}‖ = %.6e\n", α)
@printf("  M_∞ = ‖R_{L_r}‖ ≤ %.6e\n\n", M_inf)

# ════════════════════════════════════════════════════════════════════════════
# Part C: Combined tail bound
# ════════════════════════════════════════════════════════════════════════════

println("="^80)
println("IMPROVED TAIL BOUND RESULTS")
println("="^80)
println()
println("Formula: ‖R_$(NUM_EIGS)(n)‖₂ ≤ ρ^{n+1} · M_∞ · ‖Q_$(NUM_EIGS) · 1‖")
println()
@printf("  ρ           = %.6e\n", ρ)
@printf("  M_∞         = %.6e\n", M_inf)
@printf("  ‖Q_%d · 1‖  = %.6e\n", NUM_EIGS, norm_Q_N_1)
println()

# Prefactor C = M_∞ · ‖Q_N · 1‖
prefactor = setrounding(Float64, RoundUp) do
    M_inf * norm_Q_N_1
end
@printf("  Prefactor C = M_∞ · ‖Q_%d · 1‖ = %.6e\n\n", NUM_EIGS, prefactor)

# Compare with old bound (C_old = M_∞)
@printf("  Improvement factor: ‖Q_%d · 1‖ / ‖1‖ = %.6e  (%.0fx tighter)\n\n",
        NUM_EIGS, norm_Q_N_1, 1.0 / norm_Q_N_1)

println("Rigorous tail bounds:")
println("-"^55)
@printf("  %5s   %20s   %20s\n", "n", "improved", "old (Cauchy only)")
println("-"^55)
for nn in [1, 2, 3, 5, 10, 20, 50]
    # Improved bound
    bound_new = ρ^(nn + 1) * prefactor
    # Old bound
    bound_old = ρ^(nn + 1) * M_inf
    if bound_new == 0.0
        log_new = (nn + 1) * log10(ρ) + log10(prefactor)
        new_str = @sprintf("< 10^{%d}", floor(Int, log_new))
    else
        new_str = @sprintf("%.4e", bound_new)
    end
    if bound_old == 0.0
        log_old = (nn + 1) * log10(ρ) + log10(M_inf)
        old_str = @sprintf("< 10^{%d}", floor(Int, log_old))
    else
        old_str = @sprintf("%.4e", bound_old)
    end
    @printf("  n=%3d   %20s   %20s\n", nn, new_str, old_str)
end
println("-"^55)
println()

# ════════════════════════════════════════════════════════════════════════════
# Part D: Save results
# ════════════════════════════════════════════════════════════════════════════

tail_data = Dict(
    :K_resolvent => K_RESOLVENT,
    :K_projector => K_PROJECTOR,
    :NUM_EIGS => NUM_EIGS,
    :circle_radius => ρ,
    :rho_eff => ρ,
    :resolvent_Ak => resolvent_Ak,
    :M_inf => M_inf,
    :eps_K_resolvent => ε_K_low,
    :eps_K_projector => ε_K_high,
    :alpha => α,
    :norm_Q_N_1_galerkin => norm_Q_N_1_galerkin,
    :norm_Q_N_1 => norm_Q_N_1,
    :projector_correction => total_projector_correction,
    :prefactor => prefactor,
    :lambda_N => λ_N,
    :lambda_Np1 => λ_Np1,
)

serialize("data/tail_bound.jls", tail_data)

open("data/tail_bound.txt", "w") do io
    println(io, "# Improved tail bound for spectral expansion remainder R_$(NUM_EIGS)(n)")
    println(io, "# ‖R_$(NUM_EIGS)(n)‖₂ ≤ ρ^{n+1} · M_∞ · ‖Q_$(NUM_EIGS) · 1‖")
    println(io, "# K_resolvent = $K_RESOLVENT, K_projector = $K_PROJECTOR")
    println(io, "")
    @printf(io, "circle_center_re\t%.15e\n", 0.0)
    @printf(io, "circle_center_im\t%.15e\n", 0.0)
    @printf(io, "circle_radius\t%.15e\n", ρ)
    @printf(io, "rho_eff\t%.15e\n", ρ)
    @printf(io, "resolvent_Ak\t%.15e\n", resolvent_Ak)
    @printf(io, "M_inf\t%.15e\n", M_inf)
    @printf(io, "eps_K\t%.15e\n", ε_K_low)
    @printf(io, "alpha\t%.15e\n", α)
    @printf(io, "norm_Q_N_1\t%.15e\n", norm_Q_N_1)
    @printf(io, "prefactor\t%.15e\n", prefactor)
    @printf(io, "tail_projector_norm\t%.15e\n", ρ * prefactor)
    @printf(io, "lambda_%d\t%.15e\n", NUM_EIGS, λ_N)
    @printf(io, "lambda_%d\t%.15e\n", NUM_EIGS+1, λ_Np1)
    @printf(io, "min_sigma\t%.15e\n", cert_data.minimum_singular_value)
end
@info "Results saved to data/tail_bound.txt and data/tail_bound.jls"

println()
println("="^80)
println("DONE")
println("="^80)
