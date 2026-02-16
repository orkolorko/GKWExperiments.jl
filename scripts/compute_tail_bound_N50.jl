#!/usr/bin/env julia
# Compute rigorous tail bound for N=50 using:
#   - Phase 3 tail resolvent (ρ=1.014e-21, M_∞=6.062e+41) from full_certification_50eigs.jl
#   - K=512 spectral data: ℓ_j(1) centers/radii + q1 eigenvectors from bigfloat_spectral_K512.jl
#
# The tail bound is:
#   ‖R_50(n)‖₂ ≤ ρ^{n+1} · M_∞ · ‖Q_50 · 1‖
#
# Usage:
#   julia --project --startup-file=no scripts/compute_tail_bound_N50.jl

using GKWExperiments
using ArbNumerics
using BallArithmetic
using LinearAlgebra
using Printf
using Serialization

const PRECISION = 1024
const K = 512
const n = K + 1
const NUM_EIGS = 50
const N_SPLITTING = 5000
const DATA_DIR = joinpath(@__DIR__, "..", "data")

setprecision(ArbFloat, PRECISION)
setprecision(BigFloat, PRECISION)

println("=" ^ 80)
println("N=50 TAIL BOUND FROM K=$K DATA")
println("=" ^ 80)
println()
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════════
# Load phase3 tail resolvent
# ═══════════════════════════════════════════════════════════════════════════

@info "Loading Phase 3 tail resolvent..."
phase3 = Serialization.deserialize(joinpath(DATA_DIR, "phase3_tail_resolvent.jls"))
tail_rho   = phase3[:rho_tail]
tail_M_inf = phase3[:M_inf_tail]
tail_certified = phase3[:tail_certified]

@printf("  ρ_tail = %.6e\n", tail_rho)
@printf("  M_∞    = %.6e\n", tail_M_inf)
@printf("  certified = %s\n\n", tail_certified)

if !tail_certified
    @error "Tail resolvent not certified!"
    exit(1)
end

# ═══════════════════════════════════════════════════════════════════════════
# Load K=512 spectral data
# ═══════════════════════════════════════════════════════════════════════════

@info "Loading K=$K spectral data..."
ell_data = Serialization.deserialize(joinpath(DATA_DIR, "ell_K$(K)_P$(PRECISION).jls"))
ell_center    = ell_data[:ell_center]       # BigFloat[50]
ell_radius    = ell_data[:ell_radius]       # BigFloat[50]
eigenvalues_bf = ell_data[:eigenvalues_out] # BigFloat[50]
q1_vectors    = ell_data[:q1_vectors]       # Vector{Complex{BigFloat}}[50]

@printf("  ℓ_j(1) for j=1..%d loaded\n", NUM_EIGS)
@printf("  Eigenvector dimension: %d\n", length(q1_vectors[1]))

# Check all sign-certified
for j in 1:NUM_EIGS
    if abs(ell_center[j]) <= ell_radius[j]
        @warn "  j=$j NOT sign-certified: |center|=$(Float64(abs(ell_center[j]))), radius=$(Float64(ell_radius[j]))"
    end
end
println()
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════════
# Build projector ball vectors P_j(e₁) componentwise
#
# P_j(e₁) = ℓ_j(1) · v_j  where ℓ_j(1) = Ball(center, radius) and v_j ≈ BigFloat vector.
# For each component k:
#   P_j(e₁)[k] = Ball( ℓ_center · v_j[k],  |ℓ_radius · v_j[k]| + rounding )
#
# If the cache already has ball vectors, use them directly.
# ═══════════════════════════════════════════════════════════════════════════

@info "Building projector ball vectors P_j(e₁)..."

local pc::Matrix{BigFloat}   # proj_center[k, j]
local pr::Matrix{BigFloat}   # proj_radius[k, j]

if haskey(ell_data, :proj_center)
    @info "  Using cached ball vectors"
    pc = ell_data[:proj_center]
    pr = ell_data[:proj_radius]
else
    @info "  Constructing from ℓ_j × v_j..."
    pc = Matrix{BigFloat}(undef, n, NUM_EIGS)
    pr = Matrix{BigFloat}(undef, n, NUM_EIGS)
    for j in 1:NUM_EIGS
        local vj_real = real.(q1_vectors[j])
        for k in 1:n
            pc[k, j] = ell_center[j] * vj_real[k]
            pr[k, j] = setrounding(BigFloat, RoundUp) do
                abs(ell_radius[j]) * abs(vj_real[k]) +
                    eps(BigFloat) * abs(pc[k, j])
            end
        end
    end
end
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════════
# Compute Q_N · e₁ = e₁ - Σ P_j(e₁)  as a componentwise ball vector
#
# q_center[k] = δ_{k,1} - Σ_j pc[k,j]
# q_radius[k] = Σ_j pr[k,j]  (triangle inequality, directed rounding)
#
# This avoids the scalar-vector round-trip: the ℓ_j radius is propagated
# componentwise through the multiplication, and no eigenvector perturbation
# term is needed separately (it's already in pr from the Schur error).
# ═══════════════════════════════════════════════════════════════════════════

@info "Computing Q_$NUM_EIGS · e₁ as ball vector..."

q_center = zeros(BigFloat, n)
q_center[1] = one(BigFloat)
q_radius = zeros(BigFloat, n)

for j in 1:NUM_EIGS
    for k in 1:n
        q_center[k] -= pc[k, j]
    end
    for k in 1:n
        q_radius[k] = setrounding(BigFloat, RoundUp) do
            q_radius[k] + pr[k, j]
        end
    end
end

# Rigorous upper bound: ‖q‖₂ ≤ ‖q_center‖₂ + ‖q_radius‖₂
norm_q_center = BigFloat(norm(q_center))
norm_q_radius = BigFloat(norm(q_radius))

norm_Q_N_1_galerkin = setrounding(BigFloat, RoundUp) do
    norm_q_center + norm_q_radius
end

@printf("  ‖q_center‖₂  = %.6e\n", Float64(norm_q_center))
@printf("  ‖q_radius‖₂  = %.6e  (componentwise ball radii)\n", Float64(norm_q_radius))
@printf("  ‖Q_%d(A_%d)·e₁‖ ≤ %.6e  (Galerkin, ball vector)\n\n", NUM_EIGS, K, Float64(norm_Q_N_1_galerkin))
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════════
# Correction: Infinite-dimensional lift (L_r vs A_K)
#
# ‖Q_N(L_r)·1 - Q_N(A_K)·e₁‖ ≤ Σ_j ‖P_j(L_r) - P_j(A_K)‖
# Each projector perturbation: ‖P_j(L_r) - P_j(A_K)‖ ≤ ε_K / sep_j  (first order)
# ═══════════════════════════════════════════════════════════════════════════

@info "Computing truncation error correction (ε_K at K=$K)..."

# ε_K = ‖L_r - A_K‖ (Galerkin truncation error)
Δ_arb = compute_Δ(K; N=N_SPLITTING)
Δ_real = real(Δ_arb)
mid_arb = ArbNumerics.midpoint(Δ_real)
rad_arb = ArbNumerics.radius(Δ_real)
mid_big = parse(BigFloat, string(mid_arb))
ε_K = setrounding(BigFloat, RoundUp) do
    mid_big + BigFloat(string(rad_arb))
end
@printf("  ε_%d = %.6e\n", K, Float64(ε_K))

# Per-eigenvalue spectral gaps
sep = Vector{BigFloat}(undef, NUM_EIGS)
for j in 1:NUM_EIGS
    local s = typemax(BigFloat)
    for i in 1:NUM_EIGS
        i == j && continue
        local d = abs(eigenvalues_bf[j] - eigenvalues_bf[i])
        if d < s
            s = d
        end
    end
    sep[j] = s
end
@printf("  sep_min    = %.6e\n", Float64(minimum(sep)))

# Σ_j ε_K / sep_j
inf_dim_correction = zero(BigFloat)
for j in 1:NUM_EIGS
    local term = setrounding(BigFloat, RoundUp) do
        ε_K / sep[j]
    end
    global inf_dim_correction = setrounding(BigFloat, RoundUp) do
        inf_dim_correction + term
    end
end
@printf("  Inf-dim correction Σ ε_K/sep_j = %.6e\n", Float64(inf_dim_correction))

# ═══════════════════════════════════════════════════════════════════════════
# Total rigorous bound
#
# ‖Q_N(L_r)·1‖ ≤ ‖Q_N(A_K)·e₁‖_ball + Σ_j ε_K/sep_j
# ═══════════════════════════════════════════════════════════════════════════

norm_Q_N_1 = setrounding(BigFloat, RoundUp) do
    norm_Q_N_1_galerkin + inf_dim_correction
end

println()
println("  Error budget for ‖Q_$NUM_EIGS(L_r) · 1‖₂:")
@printf("    ‖e₁ - Σ P_j(e₁)‖₂     = %.6e  (Galerkin tail, exact at K=%d)\n", Float64(norm_q_center), K)
@printf("    ‖ball radii‖₂          = %.6e  (ℓ_j certification + rounding)\n", Float64(norm_q_radius))
@printf("    Σ ε_K/sep_j            = %.6e  (infinite-dim lift L_r vs A_K)\n", Float64(inf_dim_correction))
@printf("    ────────────────────────────────────\n")
@printf("    ‖Q_%d(L_r) · 1‖₂      ≤ %.6e  (rigorous)\n\n", NUM_EIGS, Float64(norm_Q_N_1))
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════════
# Combined tail bound
# ═══════════════════════════════════════════════════════════════════════════

println("=" ^ 80)
println("N=$NUM_EIGS TAIL BOUND RESULTS")
println("=" ^ 80)
println()

norm_Q_N_1_f64 = Float64(norm_Q_N_1)
prefactor = setrounding(Float64, RoundUp) do
    tail_M_inf * norm_Q_N_1_f64
end

@printf("  ρ₅₀           = %.6e\n", tail_rho)
@printf("  M_∞            = %.6e\n", tail_M_inf)
@printf("  ‖Q₅₀ · 1‖₂    ≤ %.6e\n", norm_Q_N_1_f64)
@printf("  Prefactor C    = %.6e\n\n", prefactor)

# Sample bounds
println("Rigorous tail bounds: ‖R₅₀(n)‖₂ ≤ ρ^{n+1} · C")
println("-" ^ 45)
@printf("  %5s   %20s\n", "n", "log₁₀(bound)")
println("-" ^ 45)
for nn in [1, 2, 5, 10, 20, 50, 100]
    log_bound = (nn + 1) * log10(tail_rho) + log10(prefactor)
    @printf("  n=%3d   %20.1f\n", nn, log_bound)
end
println("-" ^ 45)
println()

# ═══════════════════════════════════════════════════════════════════════════
# Also compute |λ₅₀|, |λ₅₁| for the table
# ═══════════════════════════════════════════════════════════════════════════

@printf("  |λ₅₀| = %.6e\n", Float64(abs(eigenvalues_bf[NUM_EIGS])))
# For |λ₅₁|, load Schur data
@info "Loading Schur data for |λ₅₁|..."
schur_cache = Serialization.deserialize(joinpath(DATA_DIR, "schur_cert_K$(K)_P$(PRECISION).jls"))
T_bf = schur_cache[:T_bf]
all_eigs = real.(diag(T_bf))
sorted_idx_all = sortperm(abs.(all_eigs), rev=true)
if length(sorted_idx_all) > NUM_EIGS
    @printf("  |λ₅₁| = %.6e\n", Float64(abs(all_eigs[sorted_idx_all[NUM_EIGS+1]])))
end
println()

# ═══════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════

tail_data = Dict(
    :K => K,
    :NUM_EIGS => NUM_EIGS,
    :rho_tail => tail_rho,
    :M_inf_tail => tail_M_inf,
    :norm_Q_N_1_galerkin => Float64(norm_Q_N_1_galerkin),
    :norm_Q_N_1 => norm_Q_N_1_f64,
    :norm_q_center => Float64(norm_q_center),
    :norm_q_radius => Float64(norm_q_radius),
    :inf_dim_correction => Float64(inf_dim_correction),
    :prefactor => prefactor,
    :eps_K => Float64(ε_K),
    :lambda_50 => Float64(eigenvalues_bf[NUM_EIGS]),
)
Serialization.serialize(joinpath(DATA_DIR, "tail_bound_N50_K512.jls"), tail_data)

open(joinpath(DATA_DIR, "tail_bound_N50.txt"), "w") do io
    println(io, "# Rigorous tail bound for spectral expansion remainder R_50(n)")
    println(io, "# ‖R_50(n)‖₂ ≤ ρ^{n+1} · M_∞ · ‖Q_50 · 1‖")
    println(io, "# Resolvent from full_certification_50eigs.jl (Phase 3)")
    println(io, "# Tail projection from K=$K, P=$PRECISION")
    println(io, "")
    @printf(io, "rho_tail\t%.15e\n", tail_rho)
    @printf(io, "M_inf\t%.15e\n", tail_M_inf)
    @printf(io, "norm_Q_N_1\t%.15e\n", norm_Q_N_1_f64)
    @printf(io, "prefactor\t%.15e\n", prefactor)
    @printf(io, "eps_K\t%.15e\n", Float64(ε_K))
    @printf(io, "lambda_50\t%.15e\n", Float64(eigenvalues_bf[NUM_EIGS]))
end

@info "Results saved to data/tail_bound_N50.txt and data/tail_bound_N50_K512.jls"

println()
println("=" ^ 80)
println("DONE")
println("=" ^ 80)
