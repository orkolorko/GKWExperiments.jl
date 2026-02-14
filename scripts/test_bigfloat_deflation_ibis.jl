#!/usr/bin/env julia
"""
Test BigFloat deflation certification at K=256 on ibis.

Uses cached BallMatrix, computes BigFloat Schur once, then certifies
eigenvalues via polynomial deflation with ordschur projection.

Key insight: deflating ALL 20 previously certified eigenvalues creates a
degree-20 polynomial with bridge constant ~10^85 (normalization factor
1/∏(λ_tgt - λᵢ) is enormous when zeros span [10⁻⁹, 1]). Instead, we
selectively deflate only the k nearest eigenvalues (k ≤ 5), keeping the
polynomial degree low and the bridge constant manageable.
"""

using GKWExperiments, BallArithmetic, ArbNumerics, LinearAlgebra, Serialization, Dates

println("=" ^ 70)
println("BigFloat Deflation Certification (ordschur + selective) — K=256")
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

# === Selective deflation strategy ===
# For each target eigenvalue, only deflate the k nearest certified eigenvalues.
# This keeps the polynomial degree low and the bridge constant manageable.
#
# Why: A degree-d polynomial with normalization 1/∏(λ_tgt - λᵢ) has
# bridge constant C_r ~ (1/gap)^d. For d=20 with zeros spanning [10⁻⁹, 1],
# C_r ~ 10^85. For d=3 with nearby zeros (gap ~ 10⁻⁸), C_r ~ 10^24,
# giving eps_p ~ 10⁻²⁰ << 1.

const CERTIFIED_RANGE = 1:20
const TEST_RANGE = 21:50
const MAX_DEFLATION_NEIGHBORS = 5  # max nearby eigenvalues to deflate

"""
Select the k nearest certified eigenvalues to the target for deflation.
Returns indices (in magnitude-sorted order) of eigenvalues to deflate.
"""
function select_nearby_deflation(target_j, certified_set, sorted_eigs; k_max=MAX_DEFLATION_NEIGHBORS)
    λ_tgt = abs(real(sorted_eigs[target_j]))
    # Sort certified eigenvalues by distance to target
    distances = [(j, abs(real(sorted_eigs[j]) - real(sorted_eigs[target_j]))) for j in certified_set]
    sort!(distances, by=x -> x[2])
    # Take the k_max nearest
    selected = [d[1] for d in distances[1:min(k_max, length(distances))]]
    return sort(selected)  # return in order
end

println("\n" * "=" ^ 70)
println("Testing BigFloat deflation (ordschur + selective) for eigenvalues $TEST_RANGE")
println("Max deflation neighbors: $MAX_DEFLATION_NEIGHBORS")
println("=" ^ 70)
flush(stdout)

results = Dict{Int, Any}()
certified_set = Set(CERTIFIED_RANGE)

for j in TEST_RANGE
    λ_tgt = real(sorted_eigs[j])
    deflation_indices = select_nearby_deflation(j, certified_set, sorted_eigs)

    println("\n--- Eigenvalue j=$j: λ ≈ $(round(λ_tgt, sigdigits=6)) ---")
    println("  Deflating $(length(deflation_indices)) nearby eigenvalues: $deflation_indices")
    println("  Deflation zeros: ", round.(real.(sorted_eigs[deflation_indices]), sigdigits=4))
    flush(stdout)

    t_cert = @elapsed begin
        result = certify_eigenvalue_deflation_bigfloat(
            A_ball, λ_tgt, deflation_indices;
            K=K, schur_data_bf=sd_bf,
            image_circle_radius=0.3,
            image_circle_samples=256,
            backmap_order=2,
            use_ordschur=true)
    end

    results[j] = result
    println("  method = ", result.certification_method)
    println("  certified = ", result.is_certified)
    println("  small_gain α = ", result.small_gain_factor)
    println("  resolvent_Mr = ", result.resolvent_Mr)
    println("  bridge_const = ", result.bridge_constant)
    println("  eps_p = ", result.poly_perturbation_bound)
    println("  lambda_radius = ", result.eigenvalue_radius)
    println("  timing = ", round(t_cert, digits=2), "s")
    flush(stdout)

    # If certified, add to the set for future deflation
    if result.is_certified
        push!(certified_set, j)
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
