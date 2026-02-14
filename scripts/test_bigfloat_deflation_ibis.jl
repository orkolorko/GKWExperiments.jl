#!/usr/bin/env julia
"""
Test BigFloat deflation certification at K=256 on ibis.

Uses cached BallMatrix and memoized BigFloat Schur, then certifies
eigenvalues via polynomial deflation with ordschur projection.

Key design choices:
1. **Selective deflation**: Only deflate the k nearest eigenvalues (k ≤ 5)
   to keep polynomial degree and bridge constant manageable.
2. **Full ordschur projection**: Move ALL certified eigenvalues (1-20) to T₁₁
   so that T₂₂ contains only small eigenvalues. This prevents the polynomial
   from mapping large eigenvalues (λ₁ = 1, etc.) to ~10^37 inside T₂₂.
3. **Memoized Schur**: Cache the BigFloat Schur decomposition to avoid
   recomputing it (150s) across runs.
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
    println("\nComputing BigFloat Schur decomposition (will cache for future runs)...")
    flush(stdout)
    t_schur = @elapsed begin
        A_bf = float64_ball_to_bigfloat_ball(A_ball)
        sd_bf = compute_schur_and_error(A_bf)
    end
    println("  Done in $(round(t_schur, digits=1))s — saving to $SCHUR_CACHE_PATH")
    Serialization.serialize(SCHUR_CACHE_PATH, sd_bf)
end
S_bf = sd_bf[1]
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

# === Configuration ===
const CERTIFIED_RANGE = 1:20       # already certified (BigFloat resolvent)
const TEST_RANGE = 21:50           # targets for deflation certification
const MAX_DEFLATION_NEIGHBORS = 5  # max nearby eigenvalues to deflate
const ALL_CERTIFIED = collect(CERTIFIED_RANGE)  # for ordschur: move ALL to T₁₁

"""
Select the k nearest certified eigenvalues to the target for deflation.
Returns indices (in magnitude-sorted order) of eigenvalues to deflate.
"""
function select_nearby_deflation(target_j, certified_set, sorted_eigs; k_max=MAX_DEFLATION_NEIGHBORS)
    # Sort certified eigenvalues by distance to target
    distances = [(j, abs(real(sorted_eigs[j]) - real(sorted_eigs[target_j]))) for j in certified_set]
    sort!(distances, by=x -> x[2])
    selected = [d[1] for d in distances[1:min(k_max, length(distances))]]
    return sort(selected)
end

println("\n" * "=" ^ 70)
println("Testing BigFloat deflation (ordschur + selective) for eigenvalues $TEST_RANGE")
println("Max deflation neighbors: $MAX_DEFLATION_NEIGHBORS")
println("ordschur projects ALL eigenvalues 1-20 to T₁₁")
println("=" ^ 70)
flush(stdout)

results = Dict{Int, Any}()
certified_set = Set(CERTIFIED_RANGE)

for j in TEST_RANGE
    λ_tgt = real(sorted_eigs[j])
    deflation_indices = select_nearby_deflation(j, certified_set, sorted_eigs)

    # ordschur_indices: move ALL certified eigenvalues to T₁₁
    # This ensures T₂₂ only has small eigenvalues, so p(T₂₂) is well-conditioned
    ordschur_all = sort(collect(certified_set))

    println("\n--- Eigenvalue j=$j: λ ≈ $(round(λ_tgt, sigdigits=6)) ---")
    println("  Deflating $(length(deflation_indices)) nearby eigenvalues: $deflation_indices")
    println("  ordschur moves $(length(ordschur_all)) eigenvalues to T₁₁")
    flush(stdout)

    t_cert = @elapsed begin
        result = certify_eigenvalue_deflation_bigfloat(
            A_ball, λ_tgt, deflation_indices;
            K=K, schur_data_bf=sd_bf,
            image_circle_radius=0.3,
            image_circle_samples=256,
            backmap_order=2,
            use_ordschur=true,
            ordschur_indices=ordschur_all)
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
