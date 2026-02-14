#!/usr/bin/env julia
"""
Test BigFloat deflation certification at K=256 on ibis.

Uses cached BallMatrix, computes BigFloat Schur once, then certifies
eigenvalues via polynomial deflation with ordschur projection.

The ordschur approach moves certified eigenvalues to the top-left block
of the Schur form, then certifies only the smaller p(T₂₂) block via svdbox.
This avoids ill-conditioning when the full p(T) has eigenvalues spanning
many orders of magnitude (the original approach failed with σ_min = 0).
"""

using GKWExperiments, BallArithmetic, ArbNumerics, LinearAlgebra, Serialization, Dates

println("=" ^ 70)
println("BigFloat Deflation Certification (ordschur) — K=256")
println("=" ^ 70)
println("Started: ", Dates.now())
flush(stdout)

# Load cached K=256 BallMatrix
const K = 256
const CACHE_PATH = joinpath(@__DIR__, "..", "data", "ball_matrix_K256.jls")

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
println("  Eigenvalue 30: ", real(sorted_eigs[30]))
println("  Eigenvalue 50: ", real(sorted_eigs[50]))
flush(stdout)

# Compute BigFloat Schur decomposition (one-time cost)
println("\nComputing BigFloat Schur decomposition...")
flush(stdout)
t_schur = @elapsed begin
    A_bf = float64_ball_to_bigfloat_ball(A_ball)
    sd_bf = compute_schur_and_error(A_bf)
end
S_bf = sd_bf[1]
println("  Done in $(round(t_schur, digits=1))s")
println("  norm_Z = ", Float64(sd_bf[4]))
println("  norm_Z_inv = ", Float64(sd_bf[5]))
flush(stdout)

# Verify Schur eigenvalue ordering matches sorted Float64 eigenvalues
T_bf_diag = diag(S_bf.T)
schur_sorted_idx = sortperm(abs.(T_bf_diag), rev=true)
println("\n  Schur eigenvalue comparison (top 5):")
for j in 1:5
    bf_eig = Float64(real(T_bf_diag[schur_sorted_idx[j]]))
    f64_eig = real(sorted_eigs[j])
    println("    j=$j: Schur=$(round(bf_eig, sigdigits=8)), Float64=$(round(f64_eig, sigdigits=8))")
end
flush(stdout)

# Assume eigenvalues 1:20 are already certified (from previous BigFloat resolvent run)
# Test deflation certification for eigenvalues 21-50
const CERTIFIED_RANGE = 1:20
const TEST_RANGE = 21:50

println("\n" * "=" ^ 70)
println("Testing BigFloat deflation (ordschur) for eigenvalues $TEST_RANGE")
println("Using eigenvalues $CERTIFIED_RANGE as initial certified deflation zeros")
println("=" ^ 70)
flush(stdout)

results = Dict{Int, Any}()
certified_indices = collect(CERTIFIED_RANGE)

for j in TEST_RANGE
    λ_tgt = real(sorted_eigs[j])
    println("\n--- Eigenvalue j=$j: λ ≈ $(round(λ_tgt, sigdigits=6)) ---")
    flush(stdout)

    t_cert = @elapsed begin
        result = certify_eigenvalue_deflation_bigfloat(
            A_ball, λ_tgt, certified_indices;
            K=K, schur_data_bf=sd_bf,
            image_circle_radius=0.3,
            image_circle_samples=256,
            backmap_order=2,
            use_ordschur=true)
    end

    results[j] = result
    println("  method = ", result.certification_method)
    println("  certified = ", result.is_certified)
    println("  small_gain = ", result.small_gain_factor)
    println("  resolvent_Mr = ", result.resolvent_Mr)
    println("  bridge_const = ", result.bridge_constant)
    println("  eps_p = ", result.poly_perturbation_bound)
    println("  lambda_radius = ", result.eigenvalue_radius)
    println("  timing = ", round(t_cert, digits=2), "s")
    flush(stdout)

    # If certified, add to the set for incremental deflation
    if result.is_certified
        push!(certified_indices, j)
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
