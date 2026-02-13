# # Certification of GKW Eigenvalues and Projections
#
# This script demonstrates the rigorous certification of eigenvalues for the
# Gauss-Kuzmin-Wirsing (GKW) transfer operator, including:
# - Eigenvalue bounds
# - Projection of the constant function 1 onto each eigenspace
# - Eigenvector bounds
#
# Since the GKW operator is **non-normal**, we use Schur-based spectral
# projectors rather than simple eigenvector projections.

# ## Setup

using GKWExperiments
using ArbNumerics
using BallArithmetic
using LinearAlgebra
using Dates
using CairoMakie

setprecision(ArbFloat, 256)

# ## Parameters

s = ArbComplex(1.0, 0.0)  # Classical GKW (s=1)
K = 16                      # Discretization size
num_eigenvalues = 5         # Number of eigenvalues to certify
r = 1.0                     # Hardy space radius
N_splitting = 1000          # C₂ splitting parameter
circle_samples = 128        # Resolvent certification samples
circle_radius_factor = 0.02

# ## Certification Log Structures

struct ProjectionCertificationEntry
    cluster_idx::Int
    projection_coefficient::ComplexF64      # (P_k · 1)[1] = leading coefficient
    projection_coefficient_radius::Float64  # Rigorous radius
    projection_vector_norm::Float64         # ‖P_k · 1‖
    projection_vector_norm_radius::Float64
end

struct EigenvectorCertificationEntry
    cluster_idx::Int
    schur_vector_norm::Float64             # ‖Q[:, cluster]‖
    eigenvector_error_bound::Float64       # From resolvent lift bound
end

struct NKRefinementEntry
    cluster_idx::Int
    nk_radius::Float64                     # r_NK from Newton–Kantorovich
    nk_eigenvalue_radius::Float64
    nk_eigenvector_radius::Float64
    qk_bound::Float64
    q0_bound::Float64
    y_bound::Float64
    discriminant::Float64
    is_certified::Bool
end

struct EigenvalueCertificationEntry
    cluster_idx::Int
    eigenvalue_center::ComplexF64
    eigenvalue_radius::Float64
    vbd_interval_radius::Float64
    resolvent_bound_Ak::Float64
    resolvent_bound_Lr::Float64
    small_gain_alpha::Float64
    is_certified::Bool
    circle_radius::Float64
end

struct FullCertificationLog
    timestamp::DateTime
    s_parameter::ComplexF64
    discretization_K::Int
    C2_bound::Float64
    truncation_error::Float64
    vbd_residual_norm::Float64
    vbd_orthogonality_defect::Float64
    eigenvalue_entries::Vector{EigenvalueCertificationEntry}
    projection_entries::Vector{ProjectionCertificationEntry}
    eigenvector_entries::Vector{EigenvectorCertificationEntry}
    nk_entries::Vector{NKRefinementEntry}
end

function Base.show(io::IO, log::FullCertificationLog)
    println(io, "="^70)
    println(io, "GKW FULL CERTIFICATION LOG")
    println(io, "="^70)
    println(io, "Timestamp: $(log.timestamp)")
    println(io, "GKW parameter s = $(log.s_parameter)")
    println(io, "Discretization K = $(log.discretization_K)")
    println(io, "Matrix size = $(log.discretization_K + 1) × $(log.discretization_K + 1)")
    println(io, "C₂ bound = $(log.C2_bound)")
    println(io, "Truncation error ε_K = $(log.truncation_error)")
    println(io)
    println(io, "VBD VERIFICATION:")
    println(io, "  Residual ‖A - QTQ'‖ = $(log.vbd_residual_norm)")
    println(io, "  Orthogonality ‖Q'Q - I‖ = $(log.vbd_orthogonality_defect)")
    println(io)
    println(io, "EIGENVALUE CERTIFICATION:")
    println(io, "-"^70)

    for entry in log.eigenvalue_entries
        status = entry.is_certified ? "CERTIFIED" : "FAILED"
        println(io, "Eigenvalue $(entry.cluster_idx):")
        println(io, "  λ = $(entry.eigenvalue_center)")
        println(io, "  Certified radius: $(entry.eigenvalue_radius)")
        println(io, "  Small-gain α: $(entry.small_gain_alpha)")
        println(io, "  Status: $status")
    end

    println(io)
    println(io, "PROJECTION OF CONSTANT FUNCTION 1:")
    println(io, "-"^70)

    for entry in log.projection_entries
        println(io, "Cluster $(entry.cluster_idx):")
        println(io, "  (P_k · 1)[1] = $(entry.projection_coefficient) ± $(entry.projection_coefficient_radius)")
        println(io, "  ‖P_k · 1‖ = $(entry.projection_vector_norm) ± $(entry.projection_vector_norm_radius)")
    end

    println(io)
    println(io, "EIGENVECTOR BOUNDS:")
    println(io, "-"^70)

    for entry in log.eigenvector_entries
        println(io, "Cluster $(entry.cluster_idx):")
        println(io, "  Schur vector norm: $(entry.schur_vector_norm)")
        println(io, "  Eigenvector error bound: $(entry.eigenvector_error_bound)")
    end

    if !isempty(log.nk_entries)
        println(io)
        println(io, "NEWTON–KANTOROVICH REFINEMENT (Stage 2):")
        println(io, "-"^70)

        for entry in log.nk_entries
            status = entry.is_certified ? "CERTIFIED" : "FAILED"
            println(io, "Eigenvalue $(entry.cluster_idx):")
            println(io, "  NK enclosure radius r_NK = $(entry.nk_radius)")
            println(io, "  Discrete defect q_k = $(entry.qk_bound)")
            println(io, "  Infinite-dim defect q₀ = $(entry.q0_bound)")
            println(io, "  Residual bound y = $(entry.y_bound)")
            println(io, "  Discriminant = $(entry.discriminant)")
            println(io, "  Status: $status")
        end
    end

    println(io, "="^70)
end

# ## Step 1: Compute Constants

@info "Computing certification constants..."
C2 = Float64(real(compute_C2(N_splitting)))
eps_K = Float64(real(compute_Δ(K; N=N_splitting)))

@info "C₂ = $C2"
@info "ε_K = $eps_K"

# ## Step 2: Finite-Dimensional Certification (VBD + Projections)

@info "Computing finite-dimensional certification..."
finite_result = certify_gkw_eigenspaces(s; K=K)

vbd = finite_result.block_schur.vbd_result
clusters = finite_result.block_schur.clusters
num_clusters = length(clusters)

@info "VBD found $num_clusters clusters"
@info "Residual norm: $(finite_result.block_schur.residual_norm)"
@info "Orthogonality defect: $(finite_result.block_schur.orthogonality_defect)"

# ## Step 3: Infinite-Dimensional Certification

eigenvalue_entries = EigenvalueCertificationEntry[]
projection_entries = ProjectionCertificationEntry[]
eigenvector_entries = EigenvectorCertificationEntry[]

eigenvalues_to_certify = min(num_eigenvalues, num_clusters)

for idx in 1:eigenvalues_to_certify
    cluster = clusters[idx]
    λ_ball = vbd.cluster_intervals[cluster[1]]
    λ_center = ComplexF64(BallArithmetic.mid(λ_ball))
    λ_vbd_radius = Float64(BallArithmetic.rad(λ_ball))

    @info "Processing cluster $idx: λ ≈ $λ_center"

    # Resolvent certification
    circle_radius = max(abs(λ_center) * circle_radius_factor, eps_K * 10)
    A = finite_result.gkw_matrix
    circle = CertificationCircle(λ_center, circle_radius; samples=circle_samples)
    cert_data = run_certification(A, circle)

    resolvent_Ak = cert_data.resolvent_original

    # Infinite-dimensional lift
    inf_result = certify_eigenvalue_lift(finite_result, cert_data, idx;
                                          r=r, N=N_splitting)

    push!(eigenvalue_entries, EigenvalueCertificationEntry(
        idx, λ_center, inf_result.eigenvalue_radius, λ_vbd_radius,
        resolvent_Ak, inf_result.resolvent_bound, inf_result.small_gain_factor,
        inf_result.is_certified, circle_radius
    ))

    # Projection of constant function 1
    # The projection P_k · 1 is computed in certify_gkw_eigenspaces
    proj_vec = finite_result.projections_of_one[idx]
    proj_coeff = finite_result.projection_coefficients[idx]

    # Compute projection vector norm
    proj_norm_sq = sum(abs2(BallArithmetic.mid(proj_vec[i])) for i in 1:length(proj_vec))
    proj_norm = sqrt(proj_norm_sq)

    # Radius on the projection coefficient
    proj_coeff_center = ComplexF64(BallArithmetic.mid(proj_coeff))
    proj_coeff_radius = Float64(BallArithmetic.rad(proj_coeff))

    # Radius on the norm (propagate uncertainties)
    proj_norm_radius_sq = sum(BallArithmetic.rad(proj_vec[i])^2 for i in 1:length(proj_vec))
    proj_norm_radius = sqrt(proj_norm_radius_sq)

    push!(projection_entries, ProjectionCertificationEntry(
        idx, proj_coeff_center, proj_coeff_radius, proj_norm, proj_norm_radius
    ))

    # Eigenvector bounds
    Q = finite_result.block_schur.Q
    Q_cluster = BallArithmetic.mid(Q)[:, cluster]
    schur_vec_norm = norm(Q_cluster)

    push!(eigenvector_entries, EigenvectorCertificationEntry(
        idx, schur_vec_norm, inf_result.eigenvector_error
    ))

    if inf_result.is_certified
        @info "  CERTIFIED: λ ∈ $(λ_center) ± $(inf_result.eigenvalue_radius)"
        @info "  Projection (P_$idx · 1)[1] = $(proj_coeff_center) ± $(proj_coeff_radius)"
    else
        @warn "  NOT CERTIFIED: α = $(inf_result.small_gain_factor) ≥ 1"
    end
end

# ## Step 4: Newton–Kantorovich Refinement (Stage 2)
#
# For eigenvalues certified by the resolvent method (Stage 1), we now
# apply the direct NK argument to obtain much tighter enclosure radii.

@info "Running Newton–Kantorovich refinement (Stage 2)..."

nk_entries = NKRefinementEntry[]

for idx in 1:eigenvalues_to_certify
    entry = eigenvalue_entries[idx]
    if !entry.is_certified
        @warn "  Skipping eigenvalue $idx (not certified by resolvent method)"
        continue
    end

    @info "  NK refinement for eigenvalue $idx: λ ≈ $(entry.eigenvalue_center)"
    nk_result = certify_eigenpair_nk(s; K=K, r=r, target_idx=idx, N_C2=N_splitting)

    push!(nk_entries, NKRefinementEntry(
        idx,
        nk_result.enclosure_radius,
        nk_result.eigenvalue_radius,
        nk_result.eigenvector_radius,
        nk_result.qk_bound,
        nk_result.q0_bound,
        nk_result.y_bound,
        nk_result.discriminant,
        nk_result.is_certified
    ))

    if nk_result.is_certified
        improvement = entry.eigenvalue_radius / nk_result.enclosure_radius
        @info "  NK CERTIFIED: r_NK = $(nk_result.enclosure_radius) ($(round(improvement, sigdigits=3))× tighter)"
    else
        @warn "  NK FAILED (q₀ = $(nk_result.q0_bound), disc = $(nk_result.discriminant))"
    end
end

# ## Create Full Certification Log

cert_log = FullCertificationLog(
    now(),
    ComplexF64(ArbNumerics.midpoint(real(s)), ArbNumerics.midpoint(imag(s))),
    K,
    C2,
    eps_K,
    finite_result.block_schur.residual_norm,
    finite_result.block_schur.orthogonality_defect,
    eigenvalue_entries,
    projection_entries,
    eigenvector_entries,
    nk_entries
)

# ## Display Results

println(cert_log)

# ## Spectral Expansion: L^n · 1
#
# For a non-normal operator, the spectral expansion gives:
#   L^n · 1 = Σ λ_i^n P_i(1)
#
# where P_i is the spectral projector onto the i-th eigenspace.
# This shows how iterates of the constant function decompose into
# contributions from each eigenspace.

println()
println("="^70)
println("SPECTRAL EXPANSION: L^n · 1 = Σ λᵢⁿ Pᵢ(1)")
println("="^70)
println()

# Extract eigenvalues and projection vectors for the expansion
certified_eigenvalues = [e.eigenvalue_center for e in eigenvalue_entries]
certified_projections = [finite_result.projections_of_one[i] for i in 1:length(eigenvalue_entries)]

# Show the expansion coefficients for each eigenspace
println("Spectral decomposition of 1:")
for (i, (λ, proj)) in enumerate(zip(certified_eigenvalues, certified_projections))
    proj_coeff = ComplexF64(BallArithmetic.mid(finite_result.projection_coefficients[i]))
    println("  Pᵢ(1) for λ_$i = $λ:")
    println("    Leading coefficient: $proj_coeff")
end
println()

# Show L^n · 1 for various powers
powers_to_show = [1, 2, 3, 4, 5, 6, 10, 20]

println("Approximate L^n · 1 (leading coefficients from spectral expansion):")
println("-"^70)

for n in powers_to_show
    # Compute L^n · 1 ≈ Σ λ_i^n P_i(1)
    # We compute the full vector, but display key information

    # Compute spectral expansion result (as a vector in coefficient space)
    expansion_result = zeros(ComplexF64, K + 1)

    for (i, (λ, proj_vec)) in enumerate(zip(certified_eigenvalues, certified_projections))
        λ_n = λ^n
        for j in 1:(K + 1)
            expansion_result[j] += λ_n * ComplexF64(BallArithmetic.mid(proj_vec[j]))
        end
    end

    # The leading coefficient [1] represents the "constant part" in the monomial basis
    leading_coeff = expansion_result[1]

    # Also show individual contributions
    contributions = [certified_eigenvalues[i]^n * ComplexF64(BallArithmetic.mid(finite_result.projection_coefficients[i]))
                     for i in 1:length(certified_eigenvalues)]

    println("n = $n:")
    println("  L^$n · 1 ≈ $(round(real(leading_coeff), sigdigits=10)) (leading coeff)")

    # Show breakdown by eigenvalue
    for (i, (λ, c)) in enumerate(zip(certified_eigenvalues, contributions))
        λ_n = λ^n
        println("    λ_$i^$n · P_$i(1)[1] = $(round(real(λ_n), sigdigits=6)) × $(round(real(finite_result.projection_coefficients[i].c), sigdigits=6)) = $(round(real(c), sigdigits=10))")
    end
    println()
end

# ## Convergence to Invariant Measure
#
# As n → ∞, the contribution from λ₁ = 1 dominates (since |λ₂| < 1),
# so L^n · 1 → P₁(1), the projection onto the invariant measure.

println("-"^70)
println("Asymptotic behavior:")
println("  As n → ∞: L^n · 1 → P₁(1) (invariant measure)")
println("  P₁(1)[1] = $(round(real(ComplexF64(BallArithmetic.mid(finite_result.projection_coefficients[1]))), sigdigits=10))")
println()
println("  Rate of convergence determined by |λ₂/λ₁| = $(round(abs(certified_eigenvalues[2] / certified_eigenvalues[1]), sigdigits=10))")
println("="^70)

# ## Plot: L^k · 1 for the first iterations
#
# We evaluate the functions L^k · 1 on the interval [0, 1] and compare
# the spectral approximation to the actual iterates.

# Evaluate polynomial in monomial basis (w-1)^k at point x
function eval_monomial_basis(coeffs, x)
    result = zero(eltype(coeffs))
    w = x - 1  # Change of variables: monomial basis is (w-1)^k
    w_power = one(w)
    for c in coeffs
        result += c * w_power
        w_power *= w
    end
    return result
end

# Evaluate spectral approximation: L^n · 1 ≈ Σ λ_i^n P_i(1)
function eval_spectral_approx(n, eigenvalues, projections, x)
    result = 0.0 + 0.0im
    for (λ, proj) in zip(eigenvalues, projections)
        λ_n = λ^n
        proj_coeffs = [ComplexF64(BallArithmetic.mid(proj[j])) for j in 1:length(proj)]
        result += λ_n * eval_monomial_basis(proj_coeffs, x)
    end
    return real(result)
end

# Points for evaluation
x_points = range(0.01, 0.99, length=200)

# Compute L^k · 1 for different k using the spectral approximation
iterations_to_plot = [0, 1, 2, 3, 5, 10]

# Create figure with 2x2 layout
fig = Figure(size = (1200, 900))

# Compute the invariant density P₁(1)
y_invariant = [eval_spectral_approx(100, certified_eigenvalues, certified_projections, x)
               for x in x_points]

# Plot 1: L^k · 1 for different iterations
ax1 = Axis(fig[1, 1],
    xlabel = "x",
    ylabel = "L^k · 1 (x)",
    title = "Iterates of the constant function under GKW operator"
)

colors = [:blue, :red, :green, :orange, :purple, :brown]

for (i, k) in enumerate(iterations_to_plot)
    if k == 0
        # L^0 · 1 = 1 (identity)
        y_vals = ones(length(x_points))
    else
        y_vals = [eval_spectral_approx(k, certified_eigenvalues, certified_projections, x)
                  for x in x_points]
    end
    lines!(ax1, x_points, y_vals, label = "k = $k", color = colors[i], linewidth = 2)
end

lines!(ax1, x_points, y_invariant, label = "k → ∞ (invariant)", color = :black,
       linewidth = 3, linestyle = :dash)

axislegend(ax1, position = :rt)

# Plot 2: Error L^k · 1 - P₁(1) as a function of x
ax2 = Axis(fig[1, 2],
    xlabel = "x",
    ylabel = "L^k · 1 (x) - P₁(1)(x)",
    title = "Error: deviation from invariant density"
)

error_iterations = [1, 2, 3, 4, 5, 6]
error_colors = cgrad(:viridis, length(error_iterations), categorical=true)

for (i, k) in enumerate(error_iterations)
    y_vals = [eval_spectral_approx(k, certified_eigenvalues, certified_projections, x)
              for x in x_points]
    error_vals = y_vals .- y_invariant
    lines!(ax2, x_points, error_vals, label = "k = $k", color = error_colors[i], linewidth = 2)
end

hlines!(ax2, [0.0], color = :black, linewidth = 1, linestyle = :dot)
axislegend(ax2, position = :rt)

# Plot 3: Convergence of leading coefficient
ax3 = Axis(fig[2, 1],
    xlabel = "Iteration k",
    ylabel = "(L^k · 1)[1]",
    title = "Convergence of leading coefficient"
)

k_range = 0:20
leading_coeffs = Float64[]

for k in k_range
    if k == 0
        push!(leading_coeffs, 1.0)
    else
        # Compute L^k · 1 via spectral expansion
        val = sum(certified_eigenvalues[i]^k *
                  real(ComplexF64(BallArithmetic.mid(finite_result.projection_coefficients[i])))
                  for i in 1:length(certified_eigenvalues))
        push!(leading_coeffs, val)
    end
end

scatter!(ax3, collect(k_range), leading_coeffs, color = :blue, markersize = 10)
lines!(ax3, collect(k_range), leading_coeffs, color = :blue, linewidth = 1)

# Add horizontal line at the limit
limit_val = real(ComplexF64(BallArithmetic.mid(finite_result.projection_coefficients[1])))
hlines!(ax3, [limit_val], color = :red, linewidth = 2, linestyle = :dash,
        label = "P₁(1)[1] = $(round(limit_val, sigdigits=6))")

axislegend(ax3, position = :rb)

# Plot 4: Error decay (log scale)
ax4 = Axis(fig[2, 2],
    xlabel = "Iteration k",
    ylabel = "|L^k · 1 - P₁(1)|  (log scale)",
    title = "Error decay: exponential convergence",
    yscale = log10
)

k_error_range = 1:15
errors_leading = Float64[]
errors_L2 = Float64[]

for k in k_error_range
    # Leading coefficient error
    val = sum(certified_eigenvalues[i]^k *
              real(ComplexF64(BallArithmetic.mid(finite_result.projection_coefficients[i])))
              for i in 1:length(certified_eigenvalues))
    push!(errors_leading, abs(val - limit_val))

    # L² error (approximate)
    y_vals = [eval_spectral_approx(k, certified_eigenvalues, certified_projections, x)
              for x in x_points]
    error_vals = y_vals .- y_invariant
    l2_error = sqrt(sum(error_vals.^2) * (x_points[2] - x_points[1]))
    push!(errors_L2, l2_error)
end

scatter!(ax4, collect(k_error_range), errors_leading, color = :blue, markersize = 10,
         label = "Leading coeff error")
lines!(ax4, collect(k_error_range), errors_leading, color = :blue, linewidth = 1)

scatter!(ax4, collect(k_error_range), errors_L2, color = :green, markersize = 10,
         label = "L² error (approx)")
lines!(ax4, collect(k_error_range), errors_L2, color = :green, linewidth = 1)

# Add theoretical decay rate |λ₂|^k
λ2_abs = abs(certified_eigenvalues[2])
theoretical_decay = [λ2_abs^k * 0.3 for k in k_error_range]  # Scaled for visibility
lines!(ax4, collect(k_error_range), theoretical_decay, color = :red, linewidth = 2,
       linestyle = :dash, label = "|λ₂|^k × const")

axislegend(ax4, position = :rt)

# Save figure
save("scripts/Lk_iterates.png", fig, px_per_unit = 2)
println("\nPlot saved to scripts/Lk_iterates.png")

fig

# ## Method Comparison: Resolvent Bridge vs Newton–Kantorovich
#
# Stage 1 (resolvent bridge) proves simplicity via contour integration.
# Stage 2 (NK) refines the enclosure radius using the eigenpair map directly.

println()
println("="^70)
println("METHOD COMPARISON: RESOLVENT BRIDGE vs NEWTON–KANTOROVICH")
println("="^70)
println()

for nk_entry in nk_entries
    idx = nk_entry.cluster_idx
    resolvent_entry = eigenvalue_entries[idx]
    λ = resolvent_entry.eigenvalue_center

    println("Eigenvalue $idx: λ ≈ $(round(real(λ), sigdigits=10))")
    println("  Resolvent radius: $(resolvent_entry.eigenvalue_radius)")
    println("  NK radius:        $(nk_entry.nk_radius)")
    if nk_entry.is_certified && resolvent_entry.is_certified
        improvement = resolvent_entry.eigenvalue_radius / nk_entry.nk_radius
        println("  Improvement:      $(round(improvement, sigdigits=3))×")
    end
    println()
end

# ## Hardcoded Reference Results (for K=16, s=1)
#
# These are the expected results for the classical GKW operator:
#
# **Eigenvalue 1** (Perron-Frobenius):
# - λ₁ ≈ 1.0
# - Certified radius ≈ 0.102 (determined by truncation error via resolvent)
# - NK radius gives much tighter enclosure
# - (P₁ · 1)[1] ≈ 0.789 (projection of 1 onto eigenspace of λ=1)
#
# **Eigenvalue 2** (Wirsing constant):
# - λ₂ ≈ -0.3036630029...
# - Certified radius ≈ 0.102 (resolvent), much tighter via NK
# - (P₂ · 1)[1] ≈ 0.211 gives the component of 1 in the second eigenspace
#
# **Two-Stage Pipeline:**
# Stage 1 (resolvent bridge): proves eigenvalue simplicity inside a contour
# Stage 2 (NK refinement): tightens enclosure via defect-based argument
#
# **Spectral Expansion:**
# L^n · 1 ≈ λ₁ⁿ P₁(1) + λ₂ⁿ P₂(1) + ... (higher eigenspaces)

cert_log
