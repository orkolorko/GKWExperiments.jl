# # Two-Stage Certification Pipeline for GKW Operator
#
# Stage 1 (K_low): Resolvent certification on excluding circles
#   → proves simplicity of eigenvalues + bounds ‖R_{L_r}(z)‖
#
# Stage 2 (K_high): NK certification → tight eigenpair enclosures (ε_{K_high} ≈ 10⁻⁴⁴)
#
# Transfer bridge: Stage 1 resolvent + Stage 2 truncation error
#   → Riesz projector error bounds (~ 10⁻⁸⁰)
#
# Mathematical framework:
#   Stage 1: α₁ = ε_{K_low} · ‖R_{A_{K_low}}(z)‖ < 1 on Γ_j
#     ⟹ M_∞ := ‖R_{L_r}(z)‖ ≤ ‖R_{A_{K_low}}(z)‖ / (1 - α₁)
#
#   Transfer to K_high:
#     ‖R_{A_{K_high}}(z)‖ ≤ M_∞ / (1 - M_∞ · ε_{K_high})
#
#   Riesz projector error:
#     ‖P_{L_r}(Γ) - P_{A_{K_high}}(Γ)‖ ≤ (|Γ|/2π) · ‖R_{A_{K_high}}‖² · ε_{K_high} / (1 - α_high)

using GKWExperiments
using ArbNumerics
using BallArithmetic
using GenericSchur
using LinearAlgebra
using Printf
using Dates

# Access internal helper for rigorous Arb → Float64 conversion
const _arb_to_float64_upper = GKWExperiments.NewtonKantorovichCertification._arb_to_float64_upper

# ============================================================================
# Configuration
# ============================================================================

const PRECISION = 512
const K_LOW = 48           # Stage 1: moderate K for resolvent certification
const K_HIGH = 256         # Stage 2: high K for NK refinement
const NUM_EIGENVALUES = 20
const N_SPLITTING = 5000   # C₂ splitting parameter
const CIRCLE_SAMPLES = 256
const CIRCLE_RADIUS_FACTOR = 0.01

setprecision(ArbFloat, PRECISION)
setprecision(BigFloat, PRECISION)

println("="^80)
println("TWO-STAGE CERTIFICATION PIPELINE FOR GKW OPERATOR")
println("="^80)
println("Timestamp: $(now())")
println("Precision: $PRECISION bits ($(round(Int, PRECISION * log10(2))) decimal digits)")
println()
println("Configuration:")
println("  K_low  = $K_LOW  (Stage 1: resolvent certification)")
println("  K_high = $K_HIGH (Stage 2: NK refinement)")
println("  Number of eigenvalues: $NUM_EIGENVALUES")
println("  C₂ splitting parameter: $N_SPLITTING")
println("  Circle samples: $CIRCLE_SAMPLES")
println("  Circle radius factor: $CIRCLE_RADIUS_FACTOR")
println()

s = ArbComplex(1.0, 0.0)  # Classical GKW (s=1)

# ============================================================================
# Phase 0: Compute Constants (K-independent)
# ============================================================================

println("="^80)
println("PHASE 0: COMPUTING CONSTANTS")
println("="^80)
println()

@info "Computing C₂ bound (N=$N_SPLITTING)..."
C2_arb = compute_C2(N_SPLITTING)
C2_float = _arb_to_float64_upper(C2_arb)
@info "  C₂ ≤ $C2_float (rigorous upper bound)"

@info "Computing ε_{K_low} for K=$K_LOW..."
eps_K_low_arb = compute_Δ(K_LOW; N=N_SPLITTING)
eps_K_low = _arb_to_float64_upper(eps_K_low_arb)
@info "  ε_{$K_LOW} ≤ $eps_K_low"

@info "Computing ε_{K_high} for K=$K_HIGH..."
eps_K_high_arb = compute_Δ(K_HIGH; N=N_SPLITTING)
eps_K_high = _arb_to_float64_upper(eps_K_high_arb)
@info "  ε_{$K_HIGH} ≤ $eps_K_high"

println()
println("Truncation Error Summary:")
@printf("  C₂             = %.6e\n", C2_float)
@printf("  ε_{K_low=%d}  = %.6e\n", K_LOW, eps_K_low)
@printf("  ε_{K_high=%d} = %.6e\n", K_HIGH, eps_K_high)
@printf("  Ratio ε_low/ε_high = %.2e\n", eps_K_low / eps_K_high)
println()

# Save constants checkpoint
const PHASE0_DONE = true

# ============================================================================
# Phase 1: Stage 1 — Resolvent Certification at K_low
# ============================================================================

println("="^80)
println("PHASE 1: RESOLVENT CERTIFICATION AT K=$K_LOW")
println("="^80)
println()

@info "Building GKW matrix at K=$K_LOW..."
t0 = time()
M_arb_low = gkw_matrix_direct(s; K=K_LOW)
A_low = arb_to_ball_matrix(M_arb_low)
@info "  Matrix built in $(round(time()-t0, digits=1))s"

# Get eigenvalue locations via Schur decomposition of the center matrix
A_center_low = BallArithmetic.mid(A_low)
S_low = schur(A_center_low)
eigenvalues_low = diag(S_low.T)
sorted_idx_low = sortperm(abs.(eigenvalues_low), rev=true)

eigenvalues_to_process = min(NUM_EIGENVALUES, length(eigenvalues_low))

# Stage 1 results storage
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

println()
println("Processing $eigenvalues_to_process eigenvalues...")
println("-"^80)

for i in 1:eigenvalues_to_process
    idx = sorted_idx_low[i]
    λ_center = ComplexF64(eigenvalues_low[idx])

    # Circle radius selection (gap-based)
    circle_radius = max(abs(λ_center) * CIRCLE_RADIUS_FACTOR, eps_K_low * 10)

    # Avoid overlap with other eigenvalues
    for j in 1:eigenvalues_to_process
        if j != i
            other_idx = sorted_idx_low[j]
            dist = abs(λ_center - eigenvalues_low[other_idx])
            circle_radius = min(circle_radius, dist / 3)
        end
    end

    @info "Eigenvalue $i: λ ≈ $(round(real(λ_center), sigdigits=10)), circle_r = $(@sprintf("%.4e", circle_radius))"

    # Run resolvent certification
    circle = CertificationCircle(λ_center, circle_radius; samples=CIRCLE_SAMPLES)
    t1 = time()
    cert_data = run_certification(A_low, circle)
    dt = time() - t1

    resolvent_Ak = cert_data.resolvent_original

    # Compute α₁ = ε_{K_low} · resolvent_Ak (rigorous upper bound)
    alpha1 = setrounding(Float64, RoundUp) do
        eps_K_low * resolvent_Ak
    end

    is_certified = alpha1 < 1.0

    if is_certified
        # M_∞ = resolvent_Ak / (1 - α₁) (rigorous upper bound)
        denom = setrounding(Float64, RoundDown) do
            1.0 - alpha1
        end
        M_inf = setrounding(Float64, RoundUp) do
            resolvent_Ak / denom
        end

        @info "  CERTIFIED: α₁ = $(@sprintf("%.4e", alpha1)), ‖R‖ = $(@sprintf("%.4f", resolvent_Ak)), M_∞ = $(@sprintf("%.4f", M_inf)) [$(round(dt, digits=1))s]"
    else
        M_inf = Inf
        @warn "  NOT CERTIFIED: α₁ = $(@sprintf("%.4e", alpha1)) ≥ 1, ‖R‖ = $(@sprintf("%.4f", resolvent_Ak)) [$(round(dt, digits=1))s]"

        # Try deflation fallback for clustered eigenvalues
        if i > 1
            certified_eigs = [r.lambda_center for r in stage1_results if r.is_certified]
            if !isempty(certified_eigs)
                @info "  Attempting deflation with $(length(certified_eigs)) certified eigenvalues..."
                try
                    defl_result = certify_eigenvalue_deflation(
                        A_low, λ_center, certified_eigs;
                        K=K_LOW, N=N_SPLITTING, q=1,
                        image_circle_radius=0.5,
                        image_circle_samples=CIRCLE_SAMPLES,
                        method=:direct)

                    if defl_result.is_certified
                        @info "  DEFLATION CERTIFIED: radius = $(@sprintf("%.4e", defl_result.eigenvalue_radius))"
                        # We still need M_inf for the transfer — run resolvent on a larger circle
                        larger_radius = circle_radius * 2
                        circle2 = CertificationCircle(λ_center, larger_radius; samples=CIRCLE_SAMPLES)
                        cert_data2 = run_certification(A_low, circle2)
                        resolvent_Ak = cert_data2.resolvent_original
                        alpha1 = setrounding(Float64, RoundUp) do
                            eps_K_low * resolvent_Ak
                        end
                        if alpha1 < 1.0
                            denom = setrounding(Float64, RoundDown) do
                                1.0 - alpha1
                            end
                            M_inf = setrounding(Float64, RoundUp) do
                                resolvent_Ak / denom
                            end
                            is_certified = true
                            circle_radius = larger_radius
                            @info "  Re-certified on larger circle: α₁ = $(@sprintf("%.4e", alpha1)), M_∞ = $(@sprintf("%.4f", M_inf))"
                        end
                    end
                catch e
                    @warn "  Deflation failed: $(typeof(e))"
                end
            end
        end
    end

    push!(stage1_results, Stage1Result(i, λ_center, circle_radius, resolvent_Ak, alpha1, M_inf, is_certified))
end

num_stage1_certified = count(r -> r.is_certified, stage1_results)
println()
println("Stage 1 Summary: $num_stage1_certified / $eigenvalues_to_process eigenvalues certified")
println()

# ============================================================================
# Phase 2: Stage 2 — NK at K_high + Transfer + Projector Error
# ============================================================================

println("="^80)
println("PHASE 2: NK CERTIFICATION AT K=$K_HIGH + TRANSFER BRIDGE")
println("="^80)
println()

@info "Building GKW matrix at K=$K_HIGH (this may take a while)..."
t0 = time()
M_arb_high = gkw_matrix_direct(s; K=K_HIGH)

# Convert directly to BigFloat BallMatrix (no Float64 truncation)
A_high_bf = BallMatrix(BigFloat, M_arb_high)
@info "  Matrix built in $(round(time()-t0, digits=1))s, center eltype: $(eltype(BallArithmetic.mid(A_high_bf)))"

# Storage for two-stage results
two_stage_results = TwoStageCertificationResult[]

println()
println("Processing certified eigenvalues...")
println("-"^80)

for s1 in stage1_results
    i = s1.index

    # Stage 2: NK at K_high (BigFloat)
    @info "Eigenvalue $i: NK certification at K=$K_HIGH (BigFloat)..."
    t1 = time()
    nk_result = try
        certify_eigenpair_nk(A_high_bf; K=K_HIGH, target_idx=i, N_C2=N_SPLITTING)
    catch e
        @warn "  NK failed with error: $(typeof(e)): $(e)"
        nothing
    end
    dt_nk = time() - t1

    nk_certified = nk_result !== nothing && nk_result.is_certified
    nk_radius = nk_certified ? Float64(nk_result.enclosure_radius) : Inf
    nk_eig_radius = nk_certified ? Float64(nk_result.eigenvalue_radius) : Inf
    nk_vec_radius = nk_certified ? Float64(nk_result.eigenvector_radius) : Inf

    if nk_certified
        @info "  NK CERTIFIED: r_NK = $(@sprintf("%.4e", nk_radius)) [$(round(dt_nk, digits=1))s]"
    else
        reason = nk_result !== nothing ? "q₀ = $(Float64(nk_result.q0_bound))" : "error"
        @warn "  NK FAILED ($reason) [$(round(dt_nk, digits=1))s]"
    end

    # Transfer bridge (only if Stage 1 certified)
    transfer_resolvent = Inf
    transfer_alpha = Inf
    transfer_valid = false
    proj_error = Inf
    proj_valid = false

    if s1.is_certified
        transfer_resolvent, transfer_alpha, transfer_valid = reverse_transfer_resolvent_bound(
            s1.M_inf, eps_K_high)

        if transfer_valid
            @info "  Transfer: ‖R_{A_{K_high}}‖ ≤ $(@sprintf("%.6f", transfer_resolvent)), α_high = $(@sprintf("%.4e", transfer_alpha))"

            # Riesz projector error
            contour_length = 2π * s1.circle_radius
            proj_error, proj_valid = projector_approximation_error_rigorous(
                contour_length, transfer_resolvent, eps_K_high)

            if proj_valid
                @info "  Riesz projector error: $(@sprintf("%.4e", proj_error))"
            else
                @warn "  Riesz projector error computation failed"
            end
        else
            @warn "  Transfer failed: α_high = $(@sprintf("%.4e", transfer_alpha)) ≥ 1"
        end
    end

    # Assemble result
    result = TwoStageCertificationResult(
        s1.lambda_center,
        i,
        K_LOW,
        s1.circle_radius,
        s1.resolvent_Ak,
        s1.alpha1,
        eps_K_low,
        s1.M_inf,
        s1.is_certified,
        K_HIGH,
        eps_K_high,
        nk_radius,
        nk_eig_radius,
        nk_vec_radius,
        nk_certified,
        transfer_resolvent,
        transfer_alpha,
        transfer_valid,
        proj_error,
        2π * s1.circle_radius,
        1.0,
        C2_float
    )

    push!(two_stage_results, result)
    println()
end

# ============================================================================
# Phase 3: Summary and LaTeX Output
# ============================================================================

println("="^80)
println("PHASE 3: RESULTS SUMMARY")
println("="^80)
println()

num_s1_cert = count(r -> r.stage1_is_certified, two_stage_results)
num_s2_cert = count(r -> r.stage2_is_certified, two_stage_results)
num_transfer = count(r -> r.transfer_is_valid, two_stage_results)
num_full = count(r -> r.stage1_is_certified && r.stage2_is_certified && r.transfer_is_valid, two_stage_results)

println("Certification Statistics:")
println("  Stage 1 (resolvent at K=$K_LOW): $num_s1_cert / $(length(two_stage_results)) certified")
println("  Stage 2 (NK at K=$K_HIGH):       $num_s2_cert / $(length(two_stage_results)) certified")
println("  Transfer bridge:                  $num_transfer / $(length(two_stage_results)) valid")
println("  Fully certified:                  $num_full / $(length(two_stage_results))")
println()

# Detailed results table
println("-"^120)
@printf("  %3s  %20s  %12s  %12s  %12s  %12s  %12s  %6s\n",
    "j", "λ̂_j", "α₁", "Circle r", "NK radius", "Proj error", "α_high", "Full")
println("-"^120)

for r in two_stage_results
    status = (r.stage1_is_certified && r.stage2_is_certified && r.transfer_is_valid) ? "YES" : "NO"
    @printf("  %3d  %20.12f  %12.4e  %12.4e  %12.4e  %12.4e  %12.4e  %6s\n",
        r.eigenvalue_index,
        real(r.eigenvalue_center),
        r.stage1_alpha,
        r.stage1_circle_radius,
        r.stage2_nk_radius,
        r.riesz_projector_error,
        r.transfer_alpha_high,
        status)
end
println("-"^120)
println()

# Transfer bridge table
println("Transfer Bridge Details:")
println("-"^100)
@printf("  %3s  %16s  %16s  %16s  %16s\n",
    "j", "‖R_{A_{K_low}}‖", "M_∞", "α_high", "‖R_{A_{K_high}}‖")
println("-"^100)

for r in two_stage_results
    @printf("  %3d  %16.6f  %16.6f  %16.4e  %16.6f\n",
        r.eigenvalue_index,
        r.stage1_resolvent_Ak,
        r.stage1_M_inf,
        r.transfer_alpha_high,
        r.transfer_resolvent_Ak_high)
end
println("-"^100)
println()

# ============================================================================
# LaTeX Output
# ============================================================================

mkpath("data")

open("data/supplementary_material.tex", "w") do io
    println(io, "% Two-Stage Certification Results for GKW Operator")
    println(io, "% Generated: $(now())")
    println(io, "% K_low = $K_LOW, K_high = $K_HIGH, N = $N_SPLITTING")
    println(io)

    # Table 1: Parameters
    println(io, "\\begin{table}[ht]")
    println(io, "\\centering")
    println(io, "\\caption{Two-stage certification parameters}")
    println(io, "\\label{tab:two-stage-params}")
    println(io, "\\begin{tabular}{ll}")
    println(io, "\\toprule")
    println(io, "Parameter & Value \\\\")
    println(io, "\\midrule")
    @printf(io, "\$K_{\\mathrm{low}}\$ & %d \\\\\n", K_LOW)
    @printf(io, "\$K_{\\mathrm{high}}\$ & %d \\\\\n", K_HIGH)
    @printf(io, "\$C_2\$ & \$%.6e\$ \\\\\n", C2_float)
    @printf(io, "\$\\varepsilon_{K_{\\mathrm{low}}}\$ & \$%.6e\$ \\\\\n", eps_K_low)
    @printf(io, "\$\\varepsilon_{K_{\\mathrm{high}}}\$ & \$%.6e\$ \\\\\n", eps_K_high)
    @printf(io, "Circle samples & %d \\\\\n", CIRCLE_SAMPLES)
    @printf(io, "\$N\$ (splitting) & %d \\\\\n", N_SPLITTING)
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")
    println(io)

    # Table 2: Main Results
    println(io, "\\begin{table}[ht]")
    println(io, "\\centering")
    println(io, "\\caption{Two-stage certification results for the first $(length(two_stage_results)) eigenvalues}")
    println(io, "\\label{tab:two-stage-results}")
    println(io, "\\begin{tabular}{rrrrrr}")
    println(io, "\\toprule")
    println(io, "\$j\$ & \$\\hat\\lambda_j\$ & \$\\alpha_1\$ & Circle \$r\$ & NK radius & Riesz proj.~error \\\\")
    println(io, "\\midrule")

    for r in two_stage_results
        @printf(io, "%d & \$%.10f\$ & \$%.2e\$ & \$%.2e\$ & \$%.2e\$ & \$%.2e\$ \\\\\n",
            r.eigenvalue_index,
            real(r.eigenvalue_center),
            r.stage1_alpha,
            r.stage1_circle_radius,
            r.stage2_nk_radius,
            r.riesz_projector_error)
    end

    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")
    println(io)

    # Table 3: Transfer Bridge
    println(io, "\\begin{table}[ht]")
    println(io, "\\centering")
    println(io, "\\caption{Transfer bridge details}")
    println(io, "\\label{tab:transfer-bridge}")
    println(io, "\\begin{tabular}{rrrrr}")
    println(io, "\\toprule")
    println(io, "\$j\$ & \$\\|R_{A_{K_{\\mathrm{low}}}}\\|\$ & \$M_\\infty\$ & \$\\alpha_{\\mathrm{high}}\$ & \$\\|R_{A_{K_{\\mathrm{high}}}}\\|\$ \\\\")
    println(io, "\\midrule")

    for r in two_stage_results
        @printf(io, "%d & \$%.4f\$ & \$%.4f\$ & \$%.2e\$ & \$%.4f\$ \\\\\n",
            r.eigenvalue_index,
            r.stage1_resolvent_Ak,
            r.stage1_M_inf,
            r.transfer_alpha_high,
            r.transfer_resolvent_Ak_high)
    end

    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")
    println(io)

    # Theorem statement
    println(io, "\\begin{theorem}[Rigorous spectral certification of the GKW operator]")
    println(io, "\\label{thm:two-stage}")
    println(io, "Let \$L_1 := SL : H^2(D_1) \\to H^2(D_1)\$ be the GKW transfer operator for \$s=1\$.")
    println(io, "The first $num_full eigenvalues \\(\\lambda_1, \\ldots, \\lambda_{$num_full}\\)")
    println(io, "are simple, with the following rigorous enclosures:")
    println(io, "\\begin{enumerate}")

    for r in two_stage_results
        if r.stage1_is_certified && r.stage2_is_certified && r.transfer_is_valid
            @printf(io, "\\item \$\\lambda_{%d} \\in B(%.12f, %.2e)\$", r.eigenvalue_index, real(r.eigenvalue_center), r.stage2_nk_radius)
            @printf(io, ", \\quad \\|P_{L_1} - P_{(L_1)_{%d}}\\| \\leq %.2e\$\n", K_HIGH, r.riesz_projector_error)
        end
    end

    println(io, "\\end{enumerate}")
    println(io, "\\end{theorem}")
end

@info "LaTeX output written to data/supplementary_material.tex"

println()
println("="^80)
println("TWO-STAGE CERTIFICATION COMPLETE")
println("="^80)
println("  Fully certified eigenvalues: $num_full / $(length(two_stage_results))")
if num_full > 0
    best_nk = minimum(r.stage2_nk_radius for r in two_stage_results if r.stage2_is_certified)
    worst_nk = maximum(r.stage2_nk_radius for r in two_stage_results if r.stage2_is_certified)
    @printf("  NK radius range: [%.2e, %.2e]\n", best_nk, worst_nk)

    valid_proj = [r.riesz_projector_error for r in two_stage_results if r.transfer_is_valid && isfinite(r.riesz_projector_error)]
    if !isempty(valid_proj)
        @printf("  Projector error range: [%.2e, %.2e]\n", minimum(valid_proj), maximum(valid_proj))
    end
end
println("="^80)
