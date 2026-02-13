# Supplementary Stage 1 at K_low=64 for eigenvalues 15-20
# (These failed at K_low=48 because |λ| ~ ε_{48} made α₁ ≥ 1)

using GKWExperiments
using ArbNumerics
using BallArithmetic
using LinearAlgebra
using Printf

const _arb_to_float64_upper = GKWExperiments.NewtonKantorovichCertification._arb_to_float64_upper

const PRECISION = 512
const K_LOW = 64
const K_HIGH = 256
const N_SPLITTING = 5000
const CIRCLE_SAMPLES = 256
const TARGET_EIGENVALUES = 16:20  # indices (sorted by magnitude, 1-based)

setprecision(ArbFloat, PRECISION)
setprecision(BigFloat, PRECISION)

s = ArbComplex(1.0, 0.0)

println("="^80)
println("SUPPLEMENTARY RESOLVENT CERTIFICATION AT K=$K_LOW")
println("Target eigenvalues: $(collect(TARGET_EIGENVALUES))")
println("="^80)
println()

# Phase 0: Constants
@info "Computing truncation errors..."
C2_float = _arb_to_float64_upper(compute_C2(N_SPLITTING))
eps_K_low = _arb_to_float64_upper(compute_Δ(K_LOW; N=N_SPLITTING))
eps_K_high = _arb_to_float64_upper(compute_Δ(K_HIGH; N=N_SPLITTING))

@printf("  C₂             = %.6e\n", C2_float)
@printf("  ε_{K_low=%d}  = %.6e\n", K_LOW, eps_K_low)
@printf("  ε_{K_high=%d} = %.6e\n", K_HIGH, eps_K_high)
println()

# Phase 1: Build K_low matrix and run resolvent
@info "Building GKW matrix at K=$K_LOW..."
t0 = time()
M_arb_low = gkw_matrix_direct(s; K=K_LOW)
A_low = arb_to_ball_matrix(M_arb_low)
@info "  Matrix built in $(round(time()-t0, digits=1))s"

A_center_low = BallArithmetic.mid(A_low)
S_low = schur(A_center_low)
eigenvalues_low = diag(S_low.T)
sorted_idx_low = sortperm(abs.(eigenvalues_low), rev=true)

num_eigs = length(eigenvalues_low)

println()
println("Eigenvalue overview (sorted by |λ|):")
println("-"^80)
for i in 1:min(25, num_eigs)
    idx = sorted_idx_low[i]
    λ = eigenvalues_low[idx]
    @printf("  %3d: λ = %+.12e   |λ| = %.6e\n", i, real(λ), abs(λ))
end
println("-"^80)
println()

# Run resolvent certification for target eigenvalues
struct Stage1Result
    index::Int
    lambda_center::ComplexF64
    circle_radius::Float64
    resolvent_Ak::Float64
    alpha1::Float64
    M_inf::Float64
    is_certified::Bool
end

stage1_results = Stage1Result[]

for i in TARGET_EIGENVALUES
    if i > num_eigs
        @warn "  Eigenvalue index $i out of range (only $num_eigs eigenvalues at K=$K_LOW)"
        continue
    end

    idx = sorted_idx_low[i]
    λ_center = ComplexF64(eigenvalues_low[idx])

    # Circle radius selection
    circle_radius = max(abs(λ_center) * 0.01, eps_K_low * 10)

    # Avoid overlap with neighbors
    for j in max(1, first(TARGET_EIGENVALUES)-2):min(num_eigs, last(TARGET_EIGENVALUES)+2)
        if j != i
            other_idx = sorted_idx_low[j]
            dist = abs(λ_center - eigenvalues_low[other_idx])
            if dist > 0
                circle_radius = min(circle_radius, dist / 3)
            end
        end
    end

    @info "Eigenvalue $i: λ ≈ $(@sprintf("%.10e", real(λ_center))), |λ| = $(@sprintf("%.4e", abs(λ_center))), circle_r = $(@sprintf("%.4e", circle_radius))"

    circle = CertificationCircle(λ_center, circle_radius; samples=CIRCLE_SAMPLES)
    t1 = time()
    cert_data = run_certification(A_low, circle)
    dt = time() - t1

    resolvent_Ak = cert_data.resolvent_original

    alpha1 = setrounding(Float64, RoundUp) do
        eps_K_low * resolvent_Ak
    end

    is_certified = alpha1 < 1.0

    if is_certified
        denom = setrounding(Float64, RoundDown) do
            1.0 - alpha1
        end
        M_inf = setrounding(Float64, RoundUp) do
            resolvent_Ak / denom
        end
        @info "  CERTIFIED: α₁ = $(@sprintf("%.4e", alpha1)), ‖R‖ = $(@sprintf("%.4f", resolvent_Ak)), M_∞ = $(@sprintf("%.4f", M_inf)) [$(round(dt, digits=1))s]"
    else
        M_inf = Inf
        @warn "  NOT CERTIFIED: α₁ = $(@sprintf("%.4e", alpha1)), ‖R‖ = $(@sprintf("%.4e", resolvent_Ak)) [$(round(dt, digits=1))s]"
    end

    push!(stage1_results, Stage1Result(i, λ_center, circle_radius, resolvent_Ak, alpha1, M_inf, is_certified))
end

num_certified = count(r -> r.is_certified, stage1_results)
println()
println("Stage 1 (K=$K_LOW) Summary: $num_certified / $(length(stage1_results)) eigenvalues certified")
println()

# Phase 2: Transfer bridge + NK (reuse K_HIGH=256 matrix if any certified)
certified_s1 = filter(r -> r.is_certified, stage1_results)

if !isempty(certified_s1)
    println("="^80)
    println("PHASE 2: NK AT K=$K_HIGH + TRANSFER BRIDGE")
    println("="^80)
    println()

    @info "Building GKW matrix at K=$K_HIGH..."
    t0 = time()
    M_arb_high = gkw_matrix_direct(s; K=K_HIGH)
    A_high = arb_to_ball_matrix(M_arb_high)
    @info "  Matrix built in $(round(time()-t0, digits=1))s"

    println()
    results = TwoStageCertificationResult[]

    for s1 in stage1_results
        i = s1.index

        # NK at K_high
        @info "Eigenvalue $i: NK at K=$K_HIGH..."
        t1 = time()
        nk_result = try
            certify_eigenpair_nk(A_high; K=K_HIGH, target_idx=i, N_C2=N_SPLITTING)
        catch e
            @warn "  NK failed: $(typeof(e)): $(e)"
            nothing
        end
        dt_nk = time() - t1

        nk_certified = nk_result !== nothing && nk_result.is_certified
        nk_radius = nk_certified ? nk_result.enclosure_radius : Inf
        nk_eig_radius = nk_certified ? nk_result.eigenvalue_radius : Inf
        nk_vec_radius = nk_certified ? nk_result.eigenvector_radius : Inf

        if nk_certified
            @info "  NK CERTIFIED: r_NK = $(@sprintf("%.4e", nk_radius)) [$(round(dt_nk, digits=1))s]"
        else
            @warn "  NK FAILED [$(round(dt_nk, digits=1))s]"
        end

        # Transfer bridge
        transfer_resolvent = Inf
        transfer_alpha = Inf
        transfer_valid = false
        proj_error = Inf

        if s1.is_certified
            transfer_resolvent, transfer_alpha, transfer_valid = reverse_transfer_resolvent_bound(
                s1.M_inf, eps_K_high)

            if transfer_valid
                contour_length = 2π * s1.circle_radius
                proj_error, proj_valid = projector_approximation_error_rigorous(
                    contour_length, transfer_resolvent, eps_K_high)

                @info "  Transfer: α_high = $(@sprintf("%.4e", transfer_alpha)), proj error = $(@sprintf("%.4e", proj_error))"
            end
        end

        result = TwoStageCertificationResult(
            s1.lambda_center, i, K_LOW, s1.circle_radius, s1.resolvent_Ak,
            s1.alpha1, eps_K_low, s1.M_inf, s1.is_certified,
            K_HIGH, eps_K_high, nk_radius, nk_eig_radius, nk_vec_radius, nk_certified,
            transfer_resolvent, transfer_alpha, transfer_valid,
            proj_error, 2π * s1.circle_radius, 1.0, C2_float)

        push!(results, result)
        println()
    end

    # Summary table
    println("="^80)
    println("RESULTS")
    println("="^80)
    println()
    println("-"^120)
    @printf("  %3s  %20s  %12s  %12s  %12s  %12s  %12s  %6s\n",
        "j", "λ̂_j", "α₁", "Circle r", "NK radius", "Proj error", "α_high", "Full")
    println("-"^120)

    for r in results
        status = (r.stage1_is_certified && r.stage2_is_certified && r.transfer_is_valid) ? "YES" : "NO"
        @printf("  %3d  %20.12e  %12.4e  %12.4e  %12.4e  %12.4e  %12.4e  %6s\n",
            r.eigenvalue_index, real(r.eigenvalue_center),
            r.stage1_alpha, r.stage1_circle_radius, r.stage2_nk_radius,
            r.riesz_projector_error, r.transfer_alpha_high, status)
    end
    println("-"^120)

    num_full = count(r -> r.stage1_is_certified && r.stage2_is_certified && r.transfer_is_valid, results)
    println()
    println("Fully certified: $num_full / $(length(results))")
else
    println("No eigenvalues certified at Stage 1 — skipping Phase 2")
end

println("="^80)
println("DONE")
println("="^80)
