#!/usr/bin/env julia
"""
Test ordschur + direct resolvent certification at K=256 on ibis.

After ordschur projects eigenvalues 1-20 to T₁₁, we certify the resolvent
DIRECTLY in the original λ-space (no polynomial, no rescaling).

Block triangular inversion formula:
  (zI-T)⁻¹ = [(zI-T₁₁)⁻¹    (zI-T₁₁)⁻¹ T₁₂ (zI-T₂₂)⁻¹]
              [0               (zI-T₂₂)⁻¹                  ]

  ‖(zI-T)⁻¹‖ ≤ 1/σ₁₁ · (1 + ‖T₁₂‖/σ₂₂) + 1/σ₂₂

T₁₁ and T₂₂ are well-conditioned separately, so svdbox works.
"""

using GKWExperiments, BallArithmetic, ArbNumerics, LinearAlgebra, Serialization, Dates
using BallArithmetic.CertifScripts: compute_schur_and_error

println("=" ^ 70)
println("Ordschur + Direct Resolvent Certification — K=256")
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

# Compute eigenvalues and sort by magnitude
eigs = eigvals(BallArithmetic.mid(A_ball))
sorted_idx = sortperm(abs.(eigs), rev=true)
sorted_eigs = eigs[sorted_idx]
println("  Top 5 eigenvalues: ", round.(real.(sorted_eigs[1:5]), sigdigits=8))
println("  Eigenvalue 20: ", real(sorted_eigs[20]))
println("  Eigenvalue 21: ", real(sorted_eigs[21]))
println("  Eigenvalue 30: ", real(sorted_eigs[30]))
println("  Eigenvalue 50: ", real(sorted_eigs[50]))
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
flush(stdout)

# === Configuration ===
const CERTIFIED_RANGE = 1:20  # already certified → move to T₁₁ via ordschur
const TEST_RANGE = 21:50      # targets for certification

println("\n" * "=" ^ 70)
println("Certifying eigenvalues $TEST_RANGE via ordschur + direct resolvent")
println("ordschur projects eigenvalues 1-20 to T₁₁")
println("Block formula: ‖(zI-T)⁻¹‖ ≤ 1/σ₁₁·(1 + ‖T₁₂‖/σ₂₂) + 1/σ₂₂")
println("=" ^ 70)
flush(stdout)

results = Dict{Int, Any}()
ordschur_set = collect(CERTIFIED_RANGE)

for j in TEST_RANGE
    λ_tgt = real(sorted_eigs[j])

    println("\n--- Eigenvalue j=$j: λ ≈ $(round(λ_tgt, sigdigits=6)) ---")
    println("  ordschur moves $(length(ordschur_set)) eigenvalues to T₁₁")
    flush(stdout)

    t_cert = @elapsed begin
        result = certify_eigenvalue_ordschur_direct(
            A_ball, λ_tgt, ordschur_set;
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

    # If certified, add to ordschur set for future eigenvalues
    if result.is_certified
        push!(ordschur_set, j)
    end
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
