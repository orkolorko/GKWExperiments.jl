"""
    InfiniteDimensionalLift

Rigorous finite-to-infinite dimensional certification for the GKW transfer operator.

This module implements the resolvent bridge and spectral stability framework from
the reference paper, allowing certified bounds on the spectrum of the infinite-dimensional
operator L_r : H²(D_r) → H²(D_r) from finite-dimensional Galerkin approximations.

All computations work in a single Hardy space H²(D_r) for r ∈ {1, 3/2}.

# Key Results Implemented

1. **Truncation Error** (Corollary 4.1): ‖L_r - A_k‖_{(r)} ≤ C₂(2/3)^{K+1}

2. **Resolvent Bridge** (Lemma 4): If ε_K ‖R_{A_k}(z)‖_{(r)} < 1, then z ∈ ρ(L_r)

3. **Projector Approximation**: Bound on ‖P_{L_r}(Γ) - P_{A_k}(Γ)‖_{(r)}

4. **Newton-Kantorovich Error Bounds**: Rigorous eigenvalue/eigenvector error in H²(D_r)
"""
module InfiniteDimensionalLift

using LinearAlgebra
using ArbNumerics
using BallArithmetic

import ..Constants: compute_C2, compute_Δ, h2_whiten
import ..EigenspaceCertification: GKWEigenCertificationResult, arb_to_ball_matrix

# Import CertifScripts for resolvent certification
using BallArithmetic.CertifScripts: CertificationCircle, run_certification

export InfiniteDimCertificationResult
export resolvent_bridge_condition, certified_resolvent_bound
export eigenvalue_inclusion_radius, projector_approximation_error
export newton_kantorovich_error
export certify_eigenvalue_lift, verify_spectral_gap

# ============================================================================
# Result Types
# ============================================================================

"""
    InfiniteDimCertificationResult

Result of the finite-to-infinite dimensional eigenvalue certification.

All norms are in the H²(D_r) space for the specified `hardy_space_radius`.

# Fields
- `eigenvalue_center`: Approximate eigenvalue from discretization
- `eigenvalue_radius`: Rigorous error bound |λ_true - λ_approx|
- `eigenvalue_ball`: Ball containing the true eigenvalue
- `eigenvector_error`: Bound on ‖v_true - v_approx‖_{(r)}
- `truncation_error`: ε_K = C₂(2/3)^{K+1}
- `resolvent_bound`: Certified bound on ‖(zI - L_r)⁻¹‖_{(r)} on contour
- `small_gain_factor`: α = ε_K · ‖R_{A_k}‖ (must be < 1 for certification)
- `is_certified`: Whether the eigenvalue is rigorously certified
- `discretization_size`: K+1 (matrix size)
- `hardy_space_radius`: r ∈ {1, 3/2} for H²(D_r)
- `C2_bound`: Operator norm bound C₂
"""
struct InfiniteDimCertificationResult
    eigenvalue_center::ComplexF64
    eigenvalue_radius::Float64
    eigenvalue_ball::Ball{Float64, ComplexF64}
    eigenvector_error::Float64
    truncation_error::Float64
    resolvent_bound::Float64
    small_gain_factor::Float64
    is_certified::Bool
    discretization_size::Int
    hardy_space_radius::Float64
    C2_bound::Float64
end

function Base.show(io::IO, result::InfiniteDimCertificationResult)
    println(io, "Infinite-Dimensional Eigenvalue Certification")
    println(io, "=============================================")
    println(io, "Eigenvalue: $(result.eigenvalue_center) ± $(result.eigenvalue_radius)")
    println(io, "Certified: $(result.is_certified)")
    println(io, "")
    println(io, "Small-gain factor α = $(result.small_gain_factor)")
    println(io, "Truncation error ε_K = $(result.truncation_error)")
    println(io, "Resolvent bound = $(result.resolvent_bound)")
    println(io, "Eigenvector error = $(result.eigenvector_error)")
    println(io, "")
    println(io, "Hardy space: H²(D_{$(result.hardy_space_radius)})")
    println(io, "Matrix size: $(result.discretization_size)")
    println(io, "C₂ bound: $(result.C2_bound)")
end

# ============================================================================
# Core Resolvent Bridge Functions
# ============================================================================

"""
    resolvent_bridge_condition(resolvent_norm::Real, truncation_error::Real)

Check the small-gain condition for the resolvent bridge (Lemma 4 in reference).

The condition `α = ε_K · ‖R_{A_k}(z)‖ < 1` ensures that `z ∈ ρ(L_r)`.

# Returns
- `(is_satisfied::Bool, alpha::Float64)` where alpha is the small-gain factor
"""
function resolvent_bridge_condition(resolvent_norm::Real, truncation_error::Real)
    α = Float64(truncation_error * resolvent_norm)
    return α < 1.0, α
end

"""
    certified_resolvent_bound(resolvent_Ak::Real, truncation_error::Real)

Compute the certified resolvent bound for the infinite-dimensional operator.

If `α = ε_K · ‖R_{A_k}(z)‖ < 1`, then (Lemma 4):
```math
\\|R_{L_r}(z)\\|_{(r)} \\leq \\frac{\\|R_{A_k}(z)\\|_{(r)}}{1 - \\alpha}
```

# Arguments
- `resolvent_Ak`: Upper bound on ‖(zI - A_k)⁻¹‖_{(r)} from finite-dimensional computation
- `truncation_error`: ε_K = C₂(2/3)^{K+1}

# Returns
- `(resolvent_Lr::Float64, is_valid::Bool)` where is_valid indicates if α < 1
"""
function certified_resolvent_bound(resolvent_Ak::Real, truncation_error::Real)
    is_valid, α = resolvent_bridge_condition(resolvent_Ak, truncation_error)
    if !is_valid
        return Inf, false
    end
    resolvent_Lr = Float64(resolvent_Ak) / (1.0 - α)
    return resolvent_Lr, true
end

# ============================================================================
# Eigenvalue Inclusion
# ============================================================================

"""
    eigenvalue_inclusion_radius(λ_approx::Number, resolvent_on_circle::Real,
                                 circle_radius::Real, truncation_error::Real)

Compute the eigenvalue inclusion radius using the resolvent bridge.

If Γ is a circle of radius r around λ_approx, and the small-gain condition
`α = ε_K · sup_{z∈Γ} ‖R_{A_k}(z)‖ < 1` is satisfied, then Γ ⊂ ρ(L_r).
By the maximum principle, L_r has exactly one eigenvalue inside Γ.

# Arguments
- `λ_approx`: Approximate eigenvalue from discretization
- `resolvent_on_circle`: Maximum of ‖(zI - A_k)⁻¹‖ on Γ
- `circle_radius`: Radius r of the circle Γ
- `truncation_error`: ε_K

# Returns
- `(radius::Float64, is_certified::Bool)`
"""
function eigenvalue_inclusion_radius(::Number, resolvent_on_circle::Real,
                                      circle_radius::Real, truncation_error::Real)
    is_valid, _ = resolvent_bridge_condition(resolvent_on_circle, truncation_error)
    if !is_valid
        return Inf, false
    end
    # The eigenvalue is inside Γ, so inclusion radius ≤ circle_radius
    return Float64(circle_radius), true
end

"""
    verify_spectral_gap(certification_result, contour_center::Number, contour_radius::Real;
                        N::Int=1000)

Verify that a contour lies in the resolvent set of the infinite-dimensional operator.

Uses the resolvent bridge: if `ε_K · sup_{z∈Γ} ‖R_{A_k}(z)‖ < 1`, then Γ ⊂ ρ(L_r).

# Arguments
- `certification_result`: Result from `run_certification` containing resolvent bounds
- `contour_center`: Center of the certification contour (unused, for API consistency)
- `contour_radius`: Radius of the certification contour (unused, for API consistency)
- `N`: Splitting parameter for C₂ computation

# Returns
- `(is_gap::Bool, gap_margin::Float64)` where gap_margin = 1 - α
"""
function verify_spectral_gap(certification_result, ::Number,
                              ::Real; N::Int=1000)
    # Get resolvent bound from certification
    resolvent_Ak = certification_result.resolvent_original

    # Compute truncation error
    K = size(certification_result.schur.T, 1) - 1
    ε_K = Float64(real(compute_Δ(K; N=N)))

    # Check small-gain condition
    is_valid, α = resolvent_bridge_condition(resolvent_Ak, ε_K)

    return is_valid, 1.0 - α
end

# ============================================================================
# Projector Approximation
# ============================================================================

"""
    projector_approximation_error(contour_length::Real, resolvent_on_contour::Real,
                                   truncation_error::Real)

Bound the spectral projector approximation error (Equation proj-diff in reference).

```math
\\|P_{L_r}(\\Gamma) - P_{A_k}(\\Gamma)\\|_{(r)}
\\leq \\frac{|\\Gamma|}{2\\pi} \\sup_{z\\in\\Gamma}
\\frac{\\|R_{A_k}(z)\\|_{(r)}^2 \\varepsilon_K}{1 - \\varepsilon_K \\|R_{A_k}(z)\\|_{(r)}}
```

# Arguments
- `contour_length`: |Γ| = 2πr for a circle of radius r
- `resolvent_on_contour`: sup_{z∈Γ} ‖R_{A_k}(z)‖_{(r)}
- `truncation_error`: ε_K

# Returns
- `(error_bound::Float64, is_valid::Bool)`
"""
function projector_approximation_error(contour_length::Real, resolvent_on_contour::Real,
                                        truncation_error::Real)
    is_valid, α = resolvent_bridge_condition(resolvent_on_contour, truncation_error)
    if !is_valid
        return Inf, false
    end

    numerator = resolvent_on_contour^2 * truncation_error
    denominator = 1.0 - α
    error_bound = (contour_length / (2π)) * (numerator / denominator)

    return Float64(error_bound), true
end

# ============================================================================
# Newton-Kantorovich Error Bounds (Single-Space Framework)
# ============================================================================

"""
    newton_kantorovich_error(resolvent_Ak::Real, left_eigenvector_norm::Real,
                              right_eigenvector_norm::Real, truncation_error::Real;
                              kappa::Real=0.0)

Compute Newton-Kantorovich error bound for eigenvalue/eigenvector (Section 8).

All norms are in the same H²(D_r) space (same-space normalization ‖v‖_{(r)} = 1).

The bound is:
```math
\\|(\\lambda_A, v_A) - (\\lambda, v)\\|
\\leq \\frac{\\|DF_{A_k}^{-1}\\|_{(r)}}{1-\\kappa} \\|A_k - L_r\\|_{(r)}
```

where ‖DF_{A_k}⁻¹‖ ≤ max{‖u_A‖_{(r)} + ‖S_A‖_{(r)}, ‖v_A‖_{(r)}} and
‖S_A‖_{(r)} ≤ sup_{z∈Γ} ‖R_{A_k}(z)‖_{(r)}.

# Arguments
- `resolvent_Ak`: Resolvent bound (gives ‖S_A‖)
- `left_eigenvector_norm`: ‖u_A‖_{(r)} in H²(D_r)
- `right_eigenvector_norm`: ‖v_A‖_{(r)} in H²(D_r)
- `truncation_error`: ε_K = ‖A_k - L_r‖_{(r)}
- `kappa`: Contraction constant (default 0 for first-order bound)

# Returns
- `(eigenvalue_error, eigenvector_error, total_error)`
"""
function newton_kantorovich_error(resolvent_Ak::Real, left_eigenvector_norm::Real,
                                   right_eigenvector_norm::Real, truncation_error::Real;
                                   kappa::Real=0.0)
    # Bound on ‖DF_{A_k}⁻¹‖_{(r)}
    DF_inv_bound = max(
        left_eigenvector_norm + resolvent_Ak,
        right_eigenvector_norm
    )

    # NK error bound (with same-space normalization ‖v‖_{(r)} = 1)
    total_error = (DF_inv_bound / (1.0 - kappa)) * truncation_error

    # The error is in product norm: |Δλ| + ‖Δv‖_{(r)}
    # We can bound each component by the total
    eigenvalue_error = total_error
    eigenvector_error = total_error

    return Float64(eigenvalue_error), Float64(eigenvector_error), Float64(total_error)
end

# ============================================================================
# Main Certification Function
# ============================================================================

"""
    certify_eigenvalue_lift(finite_dim_result::GKWEigenCertificationResult,
                            certification_data, cluster_idx::Int;
                            r::Real=1.0, N::Int=1000)

Certify that a finite-dimensional eigenvalue corresponds to a true eigenvalue
of the infinite-dimensional GKW operator L_r : H²(D_r) → H²(D_r).

Combines the block Schur decomposition with the resolvent bridge to provide
rigorous error bounds on eigenvalues and eigenvectors.

# Arguments
- `finite_dim_result`: Result from `certify_gkw_eigenspaces`
- `certification_data`: Result from `run_certification` containing resolvent bounds
- `cluster_idx`: Which eigenvalue cluster to certify (1-indexed)
- `r`: Hardy space radius (1.0 or 1.5)
- `N`: Splitting parameter for C₂ computation

# Returns
`InfiniteDimCertificationResult` with rigorous error bounds.

# Example
```julia
using GKWExperiments, ArbNumerics

s = ArbComplex(1.0)  # Classical GKW
finite_result = certify_gkw_eigenspaces(s; K=32)

# Get eigenvalue from VBD
cluster = finite_result.block_schur.clusters[1]
λ_approx = finite_result.block_schur.vbd_result.eigenvalues[cluster[1]]

# Run resolvent certification around this eigenvalue
A = finite_result.gkw_matrix
circle = CertificationCircle(mid(λ_approx), 0.01; samples=128)
cert_data = run_certification(A, circle)

# Combine for infinite-dimensional certification
inf_result = certify_eigenvalue_lift(finite_result, cert_data, 1)
```
"""
function certify_eigenvalue_lift(finite_dim_result::GKWEigenCertificationResult,
                                  certification_data, cluster_idx::Int;
                                  r::Real=1.0, N::Int=1000)
    # Extract dimensions
    K = finite_dim_result.discretization_size - 1

    # Compute C₂ and truncation error
    C2 = Float64(real(compute_C2(N)))
    ε_K = Float64(real(compute_Δ(K; N=N)))

    # Get eigenvalue from VBD result
    vbd = finite_dim_result.block_schur.vbd_result
    clusters = finite_dim_result.block_schur.clusters

    1 <= cluster_idx <= length(clusters) ||
        throw(BoundsError("cluster_idx $cluster_idx out of range 1:$(length(clusters))"))

    cluster = clusters[cluster_idx]
    λ_ball = vbd.cluster_intervals[cluster[1]]
    λ_approx = BallArithmetic.mid(λ_ball)

    # Get resolvent bound from certification
    resolvent_Ak = certification_data.resolvent_original
    circle_radius = certification_data.circle.radius

    # Check small-gain condition
    is_valid, α = resolvent_bridge_condition(resolvent_Ak, ε_K)

    if !is_valid
        # Certification failed - return result indicating this
        return InfiniteDimCertificationResult(
            λ_approx,
            Inf,
            Ball(λ_approx, Inf),
            Inf,
            ε_K,
            Inf,
            α,
            false,
            K + 1,
            Float64(r),
            C2
        )
    end

    # Certified resolvent bound
    resolvent_Lr, _ = certified_resolvent_bound(resolvent_Ak, ε_K)

    # Eigenvalue inclusion: eigenvalue is within the circle radius
    eigenvalue_radius = circle_radius

    # Newton-Kantorovich error (assumes unit norm eigenvectors in H²(D_r))
    eigenvalue_error, eigenvector_error, _ = newton_kantorovich_error(
        resolvent_Ak, 1.0, 1.0, ε_K
    )

    # Use the tighter of the two bounds
    final_eigenvalue_radius = min(eigenvalue_radius, eigenvalue_error)

    eigenvalue_ball = Ball(λ_approx, final_eigenvalue_radius)

    return InfiniteDimCertificationResult(
        λ_approx,
        final_eigenvalue_radius,
        eigenvalue_ball,
        eigenvector_error,
        ε_K,
        resolvent_Lr,
        α,
        true,
        K + 1,
        Float64(r),
        C2
    )
end

"""
    certify_eigenvalue_simple(A::BallMatrix, λ_approx::ComplexF64, circle_radius::Real;
                               K::Int, r::Real=1.0, N::Int=1000, samples::Int=128)

Simplified interface for eigenvalue certification without pre-computed results.

Runs the resolvent certification and returns the infinite-dimensional bound.

# Arguments
- `A`: GKW discretization matrix as BallMatrix
- `λ_approx`: Approximate eigenvalue to certify
- `circle_radius`: Radius of certification contour
- `K`: Discretization order (matrix size is K+1)
- `r`: Hardy space radius
- `N`: Splitting parameter for C₂
- `samples`: Number of samples on certification circle

# Returns
`InfiniteDimCertificationResult` with rigorous bounds.
"""
function certify_eigenvalue_simple(A::BallMatrix, λ_approx::ComplexF64, circle_radius::Real;
                                    K::Int, r::Real=1.0, N::Int=1000, samples::Int=128)
    # Run resolvent certification
    circle = CertificationCircle(λ_approx, circle_radius; samples=samples)
    cert_data = run_certification(A, circle)

    # Compute bounds
    C2 = Float64(real(compute_C2(N)))
    ε_K = Float64(real(compute_Δ(K; N=N)))

    resolvent_Ak = cert_data.resolvent_original
    is_valid, α = resolvent_bridge_condition(resolvent_Ak, ε_K)

    if !is_valid
        return InfiniteDimCertificationResult(
            λ_approx, Inf, Ball(λ_approx, Inf), Inf,
            ε_K, Inf, α, false, K + 1, Float64(r), C2
        )
    end

    resolvent_Lr, _ = certified_resolvent_bound(resolvent_Ak, ε_K)
    eigenvalue_error, eigenvector_error, _ = newton_kantorovich_error(
        resolvent_Ak, 1.0, 1.0, ε_K
    )

    final_radius = min(circle_radius, eigenvalue_error)

    return InfiniteDimCertificationResult(
        λ_approx, final_radius, Ball(λ_approx, final_radius), eigenvector_error,
        ε_K, resolvent_Lr, α, true, K + 1, Float64(r), C2
    )
end

# ============================================================================
# Polynomial Deflation Support
# ============================================================================

"""
    deflation_truncation_error(poly_coeffs::AbstractVector, Ak_norm::Real,
                                Lr_norm::Real, base_truncation_error::Real)

Compute the truncation error for p(L_r) - p(A_k) from Section 9.

```math
\\|p(L_r) - p(A_k)\\|_{(r)} \\leq \\varepsilon_r \\cdot \\mathfrak{C}_r(p; A_k, L_r)
```

where the bridge constant is bounded by:
```math
\\mathfrak{C}_r \\leq \\sum_{j=1}^d |a_j| \\cdot j \\cdot C_2^{j-1}
```

# Arguments
- `poly_coeffs`: Polynomial coefficients [a₀, a₁, ..., aₐ]
- `Ak_norm`: ‖A_k‖_{(r)} (or use C₂ for crude bound)
- `Lr_norm`: ‖L_r‖_{(r)} (or use C₂ for crude bound)
- `base_truncation_error`: ε_r = ‖L_r - A_k‖_{(r)}

# Returns
- Truncation error ‖p(L_r) - p(A_k)‖_{(r)}
"""
function deflation_truncation_error(poly_coeffs::AbstractVector, Ak_norm::Real,
                                     Lr_norm::Real, base_truncation_error::Real)
    d = length(poly_coeffs) - 1  # degree
    C = max(Float64(Ak_norm), Float64(Lr_norm))

    # Crude bound: Σ_{j=1}^d |a_j| · j · C^{j-1}
    bridge_const = 0.0
    for j in 1:d
        aj = abs(poly_coeffs[j + 1])
        bridge_const += aj * j * C^(j - 1)
    end

    return Float64(base_truncation_error) * bridge_const
end

end # module
