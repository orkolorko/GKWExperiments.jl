#!/usr/bin/env julia
"""
Direct resolvent certification at K=256 — NO ordschur.

Uses the natural Schur diagonal order from GenericSchur.jl (sorted by magnitude).
For eigenvalue at diagonal position p, the block split is:
  T₁₁ = T[1:p-1, 1:p-1]  (large eigenvalues)
  T₂₂ = T[p:end, p:end]   (target + smaller)
  T₁₂ = T[1:p-1, p:end]   (coupling)

No ordschur → no untracked Givens rotation error. The only error source is
the original Schur residual from compute_schur_and_error (rigorously bounded).

Block triangular resolvent formula:
  ‖(zI-T)⁻¹‖ ≤ 1/σ₁₁ · (1 + ‖T₁₂‖/σ₂₂) + 1/σ₂₂

T₁₁: BigFloat SVD (GenericLinearAlgebra) + Weyl propagation
T₂₂: Float64 svdbox scanning circle
"""

using GKWExperiments, BallArithmetic, ArbNumerics, LinearAlgebra, Serialization, Dates
using BallArithmetic.CertifScripts: compute_schur_and_error

println("=" ^ 70)
println("Schur Direct Resolvent Certification (no ordschur) — K=256")
println("=" ^ 70)
println("Started: ", Dates.now())
flush(stdout)

# Load cached K=256 BallMatrix
const K = 256
const CACHE_PATH = joinpath(@__DIR__, "..", "data", "ball_matrix_K256.jls")
const SCHUR_CACHE_PATH = joinpath(@__DIR__, "..", "data", "bigfloat_schur_K256.jls")

println("\nLoading cached BallMatrix from $CACHE_PATH...")
A_ball = Serialization.deserialize(CACHE_PATH)
n = size(A_ball, 1)
println("  Matrix size: $(n)×$(n)")
flush(stdout)

# BigFloat Schur decomposition — memoized
if isfile(SCHUR_CACHE_PATH)
    println("\nLoading cached BigFloat Schur from $SCHUR_CACHE_PATH...")
    flush(stdout)
    sd_bf = Serialization.deserialize(SCHUR_CACHE_PATH)
    println("  Loaded successfully")
else
    println("\nComputing BigFloat Schur decomposition (will cache)...")
    flush(stdout)
    t_schur = @elapsed begin
        A_bf = float64_ball_to_bigfloat_ball(A_ball)
        sd_bf = compute_schur_and_error(A_bf)
    end
    println("  Done in $(round(t_schur, digits=1))s — saving cache")
    Serialization.serialize(SCHUR_CACHE_PATH, sd_bf)
end
S_bf = sd_bf[1]
println("  norm_Z = ", Float64(sd_bf[4]))
println("  norm_Z_inv = ", Float64(sd_bf[5]))

# Show Schur diagonal ordering
T_diag = diag(S_bf.T)
println("\n  Schur diagonal (natural order, by magnitude):")
for j in [1, 2, 5, 10, 20, 21, 30, 40, 50]
    j > length(T_diag) && continue
    println("    position $j: λ = $(Float64(real(T_diag[j])))  |λ| = $(Float64(abs(T_diag[j])))")
end
flush(stdout)

# === Configuration ===
const TEST_RANGE = 21:50  # positions on Schur diagonal to certify

println("\n" * "=" ^ 70)
println("Certifying eigenvalues at Schur positions $TEST_RANGE")
println("Natural Schur order — no ordschur — fully rigorous")
println("Block formula: ‖(zI-T)⁻¹‖ ≤ 1/σ₁₁·(1 + ‖T₁₂‖/σ₂₂) + 1/σ₂₂")
println("=" ^ 70)
flush(stdout)

results = Dict{Int, Any}()

for j in TEST_RANGE
    λ_j = Float64(real(T_diag[j]))

    println("\n--- Position j=$j: λ ≈ $(round(λ_j, sigdigits=6)) ---")
    println("  T₁₁ = T[1:$(j-1), 1:$(j-1)] ($(j-1) large eigenvalues)")
    flush(stdout)

    t_cert = @elapsed begin
        result = certify_eigenvalue_schur_direct(
            A_ball, j;
            K=K, schur_data_bf=sd_bf,
            circle_samples=256)
    end

    results[j] = result
    println("  method = ", result.certification_method)
    println("  certified = ", result.is_certified)
    println("  small_gain α = ", result.small_gain_factor)
    println("  resolvent_Mr = ", result.resolvent_Mr)
    println("  max_res_T11 = ", result.max_resolvent_T11)
    println("  max_res_T22 = ", result.max_resolvent_T22)
    println("  T12_norm = ", result.T12_norm)
    println("  circle_radius = ", result.circle_radius)
    println("  lambda_radius = ", result.eigenvalue_radius)
    println("  eps_K = ", result.truncation_error)
    println("  timing = ", round(t_cert, digits=2), "s")
    flush(stdout)
end

# Summary
println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)
n_certified = count(j -> results[j].is_certified, TEST_RANGE)
n_total = length(TEST_RANGE)
println("Certified: $n_certified / $n_total")
for j in TEST_RANGE
    r = results[j]
    status = r.is_certified ? "✓" : "✗"
    println("  j=$j: $status  λ=$(round(real(r.eigenvalue_center), sigdigits=6))  ",
            r.is_certified ? "radius=$(r.eigenvalue_radius)" : "α=$(r.small_gain_factor)")
end
println("\nFinished: ", Dates.now())
