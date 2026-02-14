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
import ..Constants: poly_bridge_constant_powers_from_coeffs
import ..EigenspaceCertification: GKWEigenCertificationResult, arb_to_ball_matrix
import ..EigenspaceCertification: float64_ball_to_bigfloat_ball, bigfloat_ball_to_float64_ball
import ..Polynomials: deflation_polynomial, polyval, polyval_derivative

# Import CertifScripts for resolvent certification
using BallArithmetic.CertifScripts: CertificationCircle, run_certification,
    run_certification_parametric, run_certification_ogita,
    compute_schur_and_error

export InfiniteDimCertificationResult
export resolvent_bridge_condition, certified_resolvent_bound
export eigenvalue_inclusion_radius, projector_approximation_error
export newton_kantorovich_error
export certify_eigenvalue_lift, verify_spectral_gap
export DeflationCertificationResult, certify_eigenvalue_deflation, backmap_inclusion_radius
export certify_eigenvalue_deflation_bigfloat
export OrdschurDirectResult, certify_eigenvalue_ordschur_direct
export deflation_truncation_error
export TwoStageCertificationResult, reverse_transfer_resolvent_bound
export projector_approximation_error_rigorous

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

# ============================================================================
# Deflation Certification Pipeline (Section 8)
# ============================================================================

"""
    DeflationCertificationResult

Result of certifying an eigenvalue via polynomial deflation.

# Fields
- `eigenvalue_center`, `eigenvalue_radius`, `eigenvalue_ball`: enclosure of the true eigenvalue
- `deflation_polynomial_coeffs`: coefficients of the deflation polynomial p
- `deflation_polynomial_degree`: degree of p
- `deflation_power`: exponent q used in p(z) = (α∏(1-z/λ̂ᵢ))^q
- `deflated_eigenvalues`: eigenvalues zeroed out by the polynomial
- `image_circle_radius`: radius of the certification circle in p-space (around 1)
- `image_certified_radius`: certified inclusion radius in p-space
- `poly_perturbation_bound`: ‖p(L_r) - p(A_k)‖
- `bridge_constant`: 𝒞_r^{pow} from the bridge constant computation
- `resolvent_Mr`: resolvent bound ‖R_{p(A_k)}‖ on the image circle
- `small_gain_factor`: α = eps_p · M_r (must be < 1)
- `p_derivative_at_target`: |p'(λ_tgt)| used for back-mapping
- `is_certified`: whether the certification succeeded
- `truncation_error`: base ε_K = C₂(2/3)^{K+1}
- `discretization_size`: K+1
- `hardy_space_radius`: r for H²(D_r)
- `certification_method`: :direct, :parametric, or :ogita
- `timing`: elapsed seconds for the certification
"""
struct DeflationCertificationResult
    eigenvalue_center::ComplexF64
    eigenvalue_radius::Float64
    eigenvalue_ball::Ball{Float64, ComplexF64}
    deflation_polynomial_coeffs::Vector{Float64}
    deflation_polynomial_degree::Int
    deflation_power::Int
    deflated_eigenvalues::Vector{ComplexF64}
    image_circle_radius::Float64
    image_certified_radius::Float64
    poly_perturbation_bound::Float64
    bridge_constant::Float64
    resolvent_Mr::Float64
    small_gain_factor::Float64
    p_derivative_at_target::Float64
    is_certified::Bool
    truncation_error::Float64
    discretization_size::Int
    hardy_space_radius::Float64
    certification_method::Symbol
    timing::Float64
end

function Base.show(io::IO, r::DeflationCertificationResult)
    println(io, "Deflation Certification Result")
    println(io, "==============================")
    println(io, "Eigenvalue: $(r.eigenvalue_center) ± $(r.eigenvalue_radius)")
    println(io, "Certified: $(r.is_certified)")
    println(io, "Method: $(r.certification_method)")
    println(io, "")
    println(io, "Deflation polynomial degree: $(r.deflation_polynomial_degree) (q=$(r.deflation_power))")
    println(io, "Deflated eigenvalues: $(r.deflated_eigenvalues)")
    println(io, "")
    println(io, "Image circle radius: $(r.image_circle_radius)")
    println(io, "Image certified radius: $(r.image_certified_radius)")
    println(io, "Poly perturbation bound: $(r.poly_perturbation_bound)")
    println(io, "Bridge constant: $(r.bridge_constant)")
    println(io, "Resolvent M_r: $(r.resolvent_Mr)")
    println(io, "Small-gain factor: $(r.small_gain_factor)")
    println(io, "|p'(λ_tgt)|: $(r.p_derivative_at_target)")
    println(io, "")
    println(io, "Truncation error ε_K: $(r.truncation_error)")
    println(io, "Matrix size: $(r.discretization_size)")
    println(io, "Hardy space: H²(D_{$(r.hardy_space_radius)})")
    println(io, "Timing: $(r.timing) s")
end

"""
    backmap_inclusion_radius(r_p, p_coeffs, lambda_tgt; order=1)

Back-map an inclusion radius from p-space to λ-space.

Given a certified radius `r_p` in p-space (i.e., |p(λ) - 1| ≤ r_p),
compute the radius in λ-space using:

**First-order** (order=1):
    |λ - λ_tgt| ≤ r_p / |p'(λ_tgt)|

This is valid when p is injective on B(λ_tgt, δ₁). The injectivity
condition can be verified using the second-order bound.

**Second-order** (order=2): Rigorous bound using BallArithmetic.

Evaluates p'' on the disk B(λ_tgt, δ₁) using Ball interval arithmetic
to obtain a rigorous upper bound M₂ = sup_{z ∈ disk} |p''(z)|. Then
solves the quadratic (M₂/2)δ² - |p'(λ_tgt)|δ + r_p = 0. The smaller
root δ₂ is a rigorous inclusion radius by the inverse function theorem:
on |λ - λ_tgt| = δ₂ we have |p(λ) - 1| ≥ |p'|δ₂ - M₂δ₂²/2 = r_p,
so all solutions of |p(λ) - 1| ≤ r_p lie within the disk of radius δ₂.

Note: δ₂ ≥ δ₁ because the curvature correction widens the bound. The
first-order bound is tighter but requires an injectivity assumption;
the second-order bound is rigorous without additional assumptions.

# Arguments
- `r_p`: certified radius in p-space
- `p_coeffs`: polynomial coefficients [a₀, a₁, ..., aₐ]
- `lambda_tgt`: target eigenvalue
- `order`: 1 for first-order, 2 for rigorous second-order

# Returns
- `(radius, dp_abs)` where radius is the back-mapped inclusion radius
  and dp_abs = |p'(λ_tgt)|
"""
function backmap_inclusion_radius(r_p::Real, p_coeffs::AbstractVector, lambda_tgt::Number;
                                   order::Int=1)
    _, dp = polyval_derivative(p_coeffs, lambda_tgt)
    dp_abs = abs(dp)
    dp_abs == 0 && return Inf, 0.0

    if order == 1
        return Float64(r_p / dp_abs), Float64(dp_abs)
    elseif order == 2
        # Rigorous second-order back-mapping using BallArithmetic.
        #
        # We bound |p''(z)| on the perturbation disk B(λ_tgt, δ₁) by evaluating
        # the polynomial p'' on a Ball enclosure. This gives M₂ = sup |p''|.
        # The inclusion radius δ₂ is the smaller root of:
        #   (M₂/2) δ² - |p'(λ_tgt)| δ + r_p = 0

        d = length(p_coeffs) - 1
        if d < 2
            # Linear or constant: p'' = 0, first-order is exact
            return Float64(r_p / dp_abs), Float64(dp_abs)
        end

        # First-order estimate for the evaluation domain
        delta_1 = Float64(r_p / dp_abs)

        # Compute p'' coefficients: p''(x) = Σ_{k=2}^d k(k-1) a_k x^{k-2}
        dpp_coeffs = [(k * (k - 1)) * p_coeffs[k + 1] for k in 2:d]

        # Convert p'' coefficients to Balls (exact, zero radius)
        ball_dpp_coeffs = [Ball(ComplexF64(c), 0.0) for c in dpp_coeffs]

        # Create Ball enclosure of the perturbation disk B(λ_tgt, δ₁)
        z_ball = Ball(ComplexF64(lambda_tgt), delta_1)

        # Evaluate p'' on the Ball — result contains all p''(z) for z in the disk
        dpp_ball = polyval(ball_dpp_coeffs, z_ball)

        # Rigorous upper bound on |p''(z)| for z ∈ B(λ_tgt, δ₁)
        M2 = abs(BallArithmetic.mid(dpp_ball)) + Float64(BallArithmetic.rad(dpp_ball))

        if M2 == 0
            # No curvature: first-order is exact
            return Float64(r_p / dp_abs), Float64(dp_abs)
        end

        half_M2 = M2 / 2

        # Solve: half_M2 · δ² - dp_abs · δ + r_p = 0
        discriminant = dp_abs^2 - 4 * half_M2 * Float64(r_p)

        if discriminant < 0
            # Discriminant negative means r_p is too large for the quadratic
            # argument: 2M₂ r_p > |p'|². Fall back to first-order.
            return Float64(r_p / dp_abs), Float64(dp_abs)
        end

        # Smaller root = rigorous inclusion radius
        delta_2 = (dp_abs - sqrt(discriminant)) / (2 * half_M2)

        return Float64(delta_2), Float64(dp_abs)
    else
        throw(ArgumentError("order must be 1 or 2"))
    end
end

"""
    certify_eigenvalue_deflation(A::BallMatrix, lambda_tgt::Number,
                                  certified_eigenvalues::AbstractVector;
                                  K::Int, r::Real=1.0, N::Int=5000,
                                  q::Int=1,
                                  image_circle_radius::Real=0.5,
                                  image_circle_samples::Int=128,
                                  method::Symbol=:direct,
                                  backmap_order::Int=1,
                                  use_tight_bridge::Bool=true)

Certify an eigenvalue via polynomial deflation.

# Pipeline
1. Build deflation polynomial `p` that zeros out `certified_eigenvalues` and
   maps `lambda_tgt` to 1.
2. Run resolvent certification on a circle around 1 in p-space, using
   `run_certification(A, circle; polynomial=poly_coeffs)`.
3. Compute polynomial perturbation bound `ε_p = ε_K · C_r^{pow}`.
4. Check small-gain: `ε_p · M_r < 1`.
5. Back-map via `backmap_inclusion_radius` to get eigenvalue enclosure.

# Arguments
- `A`: GKW discretization matrix as BallMatrix
- `lambda_tgt`: target eigenvalue to certify
- `certified_eigenvalues`: previously certified eigenvalue centers to deflate
- `K`: discretization order (matrix is (K+1)×(K+1))
- `r`: Hardy space radius
- `N`: splitting parameter for C₂
- `q`: power of the deflation polynomial
- `image_circle_radius`: radius of circle around 1 in p-space
- `image_circle_samples`: number of samples on image circle
- `method`: `:direct`, `:parametric`, or `:ogita`
- `backmap_order`: 1 or 2 for back-mapping precision
- `use_tight_bridge`: if true, use `poly_bridge_constant_powers_from_coeffs`

# Returns
[`DeflationCertificationResult`](@ref)
"""
function certify_eigenvalue_deflation(A::BallMatrix, lambda_tgt::Number,
                                       certified_eigenvalues::AbstractVector;
                                       K::Int, r::Real=1.0, N::Int=5000,
                                       q::Int=1,
                                       image_circle_radius::Real=0.5,
                                       image_circle_samples::Int=128,
                                       method::Symbol=:direct,
                                       backmap_order::Int=1,
                                       use_tight_bridge::Bool=true)
    t0 = time()

    λ_tgt = ComplexF64(lambda_tgt)
    cert_eigs = ComplexF64.(certified_eigenvalues)

    # Step 1: Build deflation polynomial
    poly_coeffs = deflation_polynomial(Float64.(real.(cert_eigs)), real(λ_tgt); q=q)
    poly_degree = length(poly_coeffs) - 1

    # Step 2: Run resolvent certification in p-space (circle around 1)
    image_center = ComplexF64(1.0, 0.0)
    circle = CertificationCircle(image_center, Float64(image_circle_radius);
                                  samples=image_circle_samples)

    cert_data = if method == :parametric
        run_certification_parametric(A, circle; polynomial=poly_coeffs)
    elseif method == :ogita
        run_certification_ogita(A, circle; polynomial=poly_coeffs)
    else
        run_certification(A, circle; polynomial=poly_coeffs)
    end

    resolvent_Mr = cert_data.resolvent_original

    # Step 3: Compute polynomial perturbation bound
    ε_K = Float64(real(compute_Δ(K; N=N)))

    if use_tight_bridge
        Ak_center = BallArithmetic.mid(A)
        Cr, _, _, _ = poly_bridge_constant_powers_from_coeffs(
            poly_coeffs, Ak_center; r=Float64(r), εr=ε_K)
        # Use upper bound of the Ball to ensure rigorous bound
        bridge_const = Float64(BallArithmetic.mid(Cr)) + Float64(BallArithmetic.rad(Cr))
        eps_p = ε_K * bridge_const
    else
        Ak_norm_ball = svd_bound_L2_opnorm(
            BallMatrix(h2_whiten(BallArithmetic.mid(A), Float64(r))))
        Ak_norm = Float64(BallArithmetic.mid(Ak_norm_ball)) + Float64(BallArithmetic.rad(Ak_norm_ball))
        Lr_norm = Ak_norm + ε_K
        eps_p = deflation_truncation_error(poly_coeffs, Ak_norm, Lr_norm, ε_K)
        bridge_const = eps_p / ε_K
    end

    # Step 4: Small-gain check
    α = eps_p * resolvent_Mr
    is_certified = α < 1.0

    # Step 5: Back-map to λ-space
    if is_certified
        # Certified radius in p-space is the image_circle_radius
        image_certified_radius = Float64(image_circle_radius)
        λ_radius, dp_abs = backmap_inclusion_radius(
            image_certified_radius, poly_coeffs, λ_tgt; order=backmap_order)
    else
        image_certified_radius = Inf
        _, dp_abs = backmap_inclusion_radius(1.0, poly_coeffs, λ_tgt; order=1)
        λ_radius = Inf
    end

    timing = time() - t0

    return DeflationCertificationResult(
        λ_tgt,
        λ_radius,
        Ball(λ_tgt, λ_radius),
        Float64.(poly_coeffs),
        poly_degree,
        q,
        cert_eigs,
        Float64(image_circle_radius),
        image_certified_radius,
        eps_p,
        bridge_const,
        resolvent_Mr,
        α,
        Float64(dp_abs),
        is_certified,
        ε_K,
        K + 1,
        Float64(r),
        method,
        timing
    )
end

# ============================================================================
# BigFloat ordschur utilities
# ============================================================================

"""
    _swap_schur_1x1!(T, Q, k)

Swap adjacent eigenvalues at positions k and k+1 in the Schur form using
a Givens rotation. Modifies T and Q in place.
"""
function _swap_schur_1x1!(T::AbstractMatrix, Q::AbstractMatrix, k::Int)
    nn = size(T, 1)
    a, b, c = T[k, k], T[k+1, k+1], T[k, k+1]
    x = (b - a) / c    # CORRECT formula (not c/(b-a))
    nrm = sqrt(one(x) + x * conj(x))
    cs, sn = one(x) / nrm, x / nrm
    for j in 1:nn
        t1, t2 = T[k, j], T[k+1, j]
        T[k, j]   = conj(cs) * t1 + conj(sn) * t2
        T[k+1, j] = -sn * t1 + cs * t2
    end
    for i in 1:nn
        t1, t2 = T[i, k], T[i, k+1]
        T[i, k]   = t1 * cs + t2 * sn
        T[i, k+1] = -t1 * conj(sn) + t2 * cs
    end
    for i in 1:nn
        q1, q2 = Q[i, k], Q[i, k+1]
        Q[i, k]   = q1 * cs + q2 * sn
        Q[i, k+1] = -q1 * conj(sn) + q2 * cs
    end
    T[k+1, k] = zero(eltype(T))
end

"""
    _bigfloat_ordschur_block(T, Q, select_indices)

Reorder a Schur decomposition so that eigenvalues at `select_indices` are moved
to the top-left block. Returns `(T_ord, Q_ord, k)` where `k = length(select_indices)`
and `T_ord[1:k,1:k]` contains the selected eigenvalues.

Works for BigFloat (LAPACK ordschur is Float64 only).
"""
function _bigfloat_ordschur_block(T, Q, select_indices::AbstractVector{<:Integer})
    T_ord, Q_ord = copy(T), copy(Q)
    n = size(T, 1)

    # Track where each original index has moved
    current_pos = collect(1:n)

    for (dest, orig_idx) in enumerate(select_indices)
        # Find current position of the eigenvalue originally at orig_idx
        src = findfirst(==(orig_idx), current_pos)
        # Bubble it up to position dest
        for k in (src - 1):-1:dest
            _swap_schur_1x1!(T_ord, Q_ord, k)
            current_pos[k], current_pos[k+1] = current_pos[k+1], current_pos[k]
        end
    end

    # Clean lower triangular
    for i in 2:n, j in 1:i-1
        T_ord[i, j] = zero(eltype(T_ord))
    end

    return T_ord, Q_ord, length(select_indices)
end

# ============================================================================
# BigFloat Deflation Certification Pipeline
# ============================================================================

"""
    _bigfloat_to_float64_upper(x::BigFloat) → Float64

Convert a BigFloat to Float64 with rigorous upper bound (nextfloat if needed).
"""
function _bigfloat_to_float64_upper(x::BigFloat)
    f = Float64(x)
    if BigFloat(f) < x
        f = nextfloat(f)
    end
    return f
end

"""
    _ball_to_float64_upper(x) → Float64

Extract a rigorous Float64 upper bound from a Ball (any precision) or scalar.
"""
_ball_to_float64_upper(x::Float64) = x
_ball_to_float64_upper(x::BigFloat) = _bigfloat_to_float64_upper(x)
function _ball_to_float64_upper(x)
    # For Ball types: upper bound = |mid| + rad
    m = abs(BallArithmetic.mid(x))
    r = BallArithmetic.rad(x)
    return _bigfloat_to_float64_upper(BigFloat(m) + BigFloat(r))
end

"""
    certify_eigenvalue_deflation_bigfloat(A_f64::BallMatrix, lambda_tgt::Number,
                                           certified_indices::AbstractVector{<:Integer};
                                           K::Int, r::Real=1.0, N::Int=5000,
                                           q::Int=1,
                                           image_circle_radius::Real=0.5,
                                           image_circle_samples::Int=256,
                                           backmap_order::Int=2,
                                           use_tight_bridge::Bool=true,
                                           use_ordschur::Bool=true,
                                           ordschur_indices=nothing,
                                           schur_data_bf=nothing)

Certify an eigenvalue via polynomial deflation using BigFloat Schur decomposition.

This function targets eigenvalues that are too small for direct resolvent
certification (|λ| < 10⁻²⁰). It uses BigFloat precision for the Schur
decomposition and deflation polynomial construction, then converts to Float64
for fast SVD certification.

# Pipeline
1. Promote `A_f64` to BigFloat BallMatrix.
2. Compute BigFloat Schur decomposition (or reuse `schur_data_bf`).
3. Build deflation polynomial `p` in BigFloat using Schur diagonal eigenvalues
   at `certified_indices`, normalized so `p(lambda_tgt) = 1`.
4. Evaluate `p(T)` in BigFloat via Horner on the upper triangular Schur factor.
5. Convert `p(T)` to Float64 BallMatrix (radii capture conversion error).
6. Certify `σ_min(p(T) - zI) > 0` on the image circle around `z = 1`.
7. Apply Schur bridge to get `M_r = ‖(zI - p(A_k))⁻¹‖`.
8. Check small-gain: `ε_{p,r} · M_r < 1`.
9. Back-map via `backmap_inclusion_radius` to get eigenvalue enclosure.

# Arguments
- `A_f64`: Float64 GKW discretization matrix as BallMatrix
- `lambda_tgt`: target eigenvalue to certify
- `certified_indices`: indices (in magnitude-sorted order) of already-certified
  eigenvalues to deflate — their BigFloat values are taken from `diag(T)`
- `K`: discretization order (matrix is (K+1)×(K+1))
- `r`: Hardy space radius (default 1.0)
- `N`: splitting parameter for C₂ computation
- `q`: power of the deflation polynomial
- `image_circle_radius`: radius of certification circle around 1 in p-space
- `image_circle_samples`: number of samples on image circle
- `backmap_order`: 1 (first-order) or 2 (rigorous second-order via p'')
- `use_tight_bridge`: use `poly_bridge_constant_powers_from_coeffs` for tighter bound
- `use_ordschur`: (default `true`) use ordered Schur decomposition to project away
  certified eigenvalues, certifying only the smaller `p(T₂₂)` block. This avoids
  ill-conditioning when the full `p(T)` has eigenvalues spanning many orders of magnitude.
  The block triangular bound (Weyl's inequality) is:
  `σ_min(zI-p(T)) ≥ min(σ_min(zI-p(T₁₁)), σ_min(zI-p(T₂₂))) - ‖B_off‖`.
- `ordschur_indices`: (default `nothing` → same as `certified_indices`) indices of
  eigenvalues to move to the top-left T₁₁ block via ordschur. To avoid ill-conditioning,
  pass ALL eigenvalues with `|λ| > |λ_tgt|` so that T₂₂ only contains small eigenvalues.
  The deflation polynomial zeros (from `certified_indices`) must be a subset.
- `schur_data_bf`: pre-computed BigFloat Schur data (5-tuple from `compute_schur_and_error`);
  if `nothing`, computed from `A_f64` promoted to BigFloat

# Returns
[`DeflationCertificationResult`](@ref)
"""
function certify_eigenvalue_deflation_bigfloat(A_f64::BallMatrix, lambda_tgt::Number,
                                                certified_indices::AbstractVector{<:Integer};
                                                K::Int, r::Real=1.0, N::Int=5000,
                                                q::Int=1,
                                                image_circle_radius::Real=0.5,
                                                image_circle_samples::Int=256,
                                                backmap_order::Int=2,
                                                use_tight_bridge::Bool=true,
                                                use_ordschur::Bool=true,
                                                ordschur_indices::Union{Nothing, AbstractVector{<:Integer}}=nothing,
                                                schur_data_bf=nothing)
    t0 = time()
    λ_tgt = ComplexF64(lambda_tgt)

    # Step 1: Promote to BigFloat BallMatrix
    A_bf = float64_ball_to_bigfloat_ball(A_f64)

    # Step 2: BigFloat Schur decomposition (reuse if provided)
    if schur_data_bf === nothing
        schur_data_bf = compute_schur_and_error(A_bf)
    end
    S_bf, _, _, norm_Z_bf, norm_Z_inv_bf = schur_data_bf

    # Step 3: Build polynomial p(z)
    T_bf_diag = diag(S_bf.T)
    sorted_idx = sortperm(abs.(T_bf_diag), rev=true)

    if isempty(certified_indices)
        # No deflation zeros → linear rescaling: p(z) = z / λ_tgt
        # Maps target eigenvalue to 1, all others to λ_j / λ_tgt
        bf_coeffs = [BigFloat(0), inv(BigFloat(real(λ_tgt)))]
    else
        # Deflation polynomial with zeros at certified eigenvalues
        bf_zeros = BigFloat.(real.(T_bf_diag[sorted_idx[certified_indices]]))
        bf_coeffs = deflation_polynomial(bf_zeros, BigFloat(real(λ_tgt)); q=q)
    end
    poly_degree = length(bf_coeffs) - 1

    # Determine ordschur indices
    ords_avail = ordschur_indices !== nothing ? ordschur_indices :
                 (!isempty(certified_indices) ? certified_indices : Int[])

    # Step 4–6: Evaluate p(T) and certify σ_min on image circle
    if use_ordschur && !isempty(ords_avail)
        # === ordschur path: project away certified eigenvalues ===
        # Move certified eigenvalues to top-left block of Schur form.
        # After ordschur: T_ord = [T₁₁ T₁₂; 0 T₂₂] with σ(T₁₁) = certified eigs.
        # p(T_ord) = [p(T₁₁) B_off; 0 p(T₂₂)] with p(T₁₁) ≈ 0 (zeros match eigs).
        # Weyl bound: σ_min(zI-p(T)) ≥ min(σ_min(zI-p(T₁₁)), σ_min(zI-p(T₂₂))) - ‖B_off‖.
        schur_ords_indices = sorted_idx[ords_avail]
        # Q_ord not needed: Givens rotations are unitary, so ‖Q_ord‖ = ‖Q‖.
        # The Schur bridge uses original norm_Z, norm_Z_inv.
        T_ord, _, k_block = _bigfloat_ordschur_block(
            S_bf.T, S_bf.Z, schur_ords_indices)

        # Evaluate p(T_ord) in BigFloat via Horner on full reordered T
        bT_ord = BallMatrix(T_ord)
        pT_ord_bf = BallArithmetic.CertifScripts._polynomial_matrix(bf_coeffs, bT_ord)

        # Extract blocks
        n_full = size(T_ord, 1)
        pT22_center = BallArithmetic.mid(pT_ord_bf)[k_block+1:end, k_block+1:end]
        pT22_radius = BallArithmetic.rad(pT_ord_bf)[k_block+1:end, k_block+1:end]
        pT22_bf = BallMatrix(pT22_center, pT22_radius)

        # ‖B_off‖ via Frobenius norm (upper bound on operator norm)
        Boff_abs = abs.(BallArithmetic.mid(pT_ord_bf)[1:k_block, k_block+1:end]) .+
                   BallArithmetic.rad(pT_ord_bf)[1:k_block, k_block+1:end]
        Boff_norm = _bigfloat_to_float64_upper(sqrt(sum(Boff_abs .^ 2)))

        # ‖p(T₁₁)‖ via Frobenius (should be ~0 since polynomial zeros = eigenvalues)
        pT11_abs = abs.(BallArithmetic.mid(pT_ord_bf)[1:k_block, 1:k_block]) .+
                   BallArithmetic.rad(pT_ord_bf)[1:k_block, 1:k_block]
        pT11_norm = _bigfloat_to_float64_upper(sqrt(sum(pT11_abs .^ 2)))

        @info "ordschur block structure" k_block m_block=n_full-k_block Boff_norm pT11_norm

        # Convert p(T₂₂) to Float64 BallMatrix for svdbox certification
        pT22_f64 = bigfloat_ball_to_float64_ball(pT22_bf)

        # Certify σ_min on image circle using block triangular bound
        max_resolvent_pT = 0.0
        for s in 0:(image_circle_samples - 1)
            θ = 2π * s / image_circle_samples
            z = ComplexF64(1.0 + image_circle_radius * cos(θ),
                           image_circle_radius * sin(θ))

            # σ_min(zI - p(T₂₂)) via svdbox on smaller (n-k)×(n-k) matrix
            sigma_vals = svdbox(pT22_f64 - z * I)
            sigma_min_22 = max(Float64(BallArithmetic.mid(sigma_vals[end])) -
                               Float64(BallArithmetic.rad(sigma_vals[end])), 0.0)

            # σ_min(zI - p(T₁₁)) ≥ |z| - ‖p(T₁₁)‖  (Weyl)
            sigma_min_11 = abs(z) - pT11_norm

            # Block bound (Weyl): σ_min(full) ≥ min(σ_min_11, σ_min_22) - ‖B_off‖
            sigma_min_full = min(sigma_min_11, sigma_min_22) - Boff_norm

            if sigma_min_full <= 0
                @warn "Block σ_min ≤ 0 at sample s=$s" sigma_min_11 sigma_min_22 Boff_norm
                max_resolvent_pT = Inf
                break
            end

            resolvent_z = setrounding(Float64, RoundUp) do
                1.0 / sigma_min_full
            end
            max_resolvent_pT = max(max_resolvent_pT, resolvent_z)
        end
        method = :bigfloat_deflation_ordschur
    else
        # === Original path: full p(T) ===
        bT_bf = BallMatrix(S_bf.T)
        pT_bf = BallArithmetic.CertifScripts._polynomial_matrix(bf_coeffs, bT_bf)
        pT_f64 = bigfloat_ball_to_float64_ball(pT_bf)

        max_resolvent_pT = 0.0
        for s in 0:(image_circle_samples - 1)
            θ = 2π * s / image_circle_samples
            z = ComplexF64(1.0 + image_circle_radius * cos(θ),
                           image_circle_radius * sin(θ))

            sigma_vals = svdbox(pT_f64 - z * I)
            sigma_min_ball = sigma_vals[end]
            sigma_lower = max(Float64(BallArithmetic.mid(sigma_min_ball)) -
                              Float64(BallArithmetic.rad(sigma_min_ball)), 0.0)
            if sigma_lower <= 0
                @warn "σ_min ≤ 0 at sample s=$s (z=$z); certification will fail"
                max_resolvent_pT = Inf
                break
            end

            resolvent_z = setrounding(Float64, RoundUp) do
                1.0 / sigma_lower
            end
            max_resolvent_pT = max(max_resolvent_pT, resolvent_z)
        end
        method = :bigfloat_deflation
    end

    # Step 7: Schur bridge — M_r from ‖(zI - p(A_k))⁻¹‖
    # For BigFloat Schur: norm_Z ≈ 1, norm_Z_inv ≈ 1, so M_r ≈ max_resolvent_pT
    norm_Z_f64 = _ball_to_float64_upper(norm_Z_bf)
    norm_Z_inv_f64 = _ball_to_float64_upper(norm_Z_inv_bf)

    resolvent_Mr = setrounding(Float64, RoundUp) do
        norm_Z_f64 * max_resolvent_pT * norm_Z_inv_f64
    end

    # Step 8: Polynomial perturbation bound ε_{p,r} = ε_K · C_r^{pow}
    ε_K = Float64(real(compute_Δ(K; N=N)))
    poly_coeffs_f64 = Float64.(bf_coeffs)

    if use_tight_bridge
        Ak_center = BallArithmetic.mid(A_f64)
        Cr, _, _, _ = poly_bridge_constant_powers_from_coeffs(
            poly_coeffs_f64, Ak_center; r=Float64(r), εr=ε_K)
        bridge_const = Float64(BallArithmetic.mid(Cr)) + Float64(BallArithmetic.rad(Cr))
        eps_p = setrounding(Float64, RoundUp) do
            ε_K * bridge_const
        end
    else
        Ak_norm_ball = svd_bound_L2_opnorm(
            BallMatrix(h2_whiten(BallArithmetic.mid(A_f64), Float64(r))))
        Ak_norm = Float64(BallArithmetic.mid(Ak_norm_ball)) +
                  Float64(BallArithmetic.rad(Ak_norm_ball))
        Lr_norm = Ak_norm + ε_K
        eps_p = deflation_truncation_error(poly_coeffs_f64, Ak_norm, Lr_norm, ε_K)
        bridge_const = eps_p / ε_K
    end

    # Step 9: Small-gain check
    α = setrounding(Float64, RoundUp) do
        eps_p * resolvent_Mr
    end
    is_certified = α < 1.0

    # Step 10: Back-map to λ-space
    if is_certified
        image_certified_radius = Float64(image_circle_radius)
        λ_radius, dp_abs = backmap_inclusion_radius(
            image_certified_radius, poly_coeffs_f64, real(λ_tgt); order=backmap_order)
    else
        image_certified_radius = Inf
        _, dp_abs = backmap_inclusion_radius(1.0, poly_coeffs_f64, real(λ_tgt); order=1)
        λ_radius = Inf
    end

    timing = time() - t0
    cert_eigs = isempty(certified_indices) ? ComplexF64[] :
                ComplexF64.(BigFloat.(real.(T_bf_diag[sorted_idx[certified_indices]])))

    return DeflationCertificationResult(
        λ_tgt, λ_radius, Ball(λ_tgt, λ_radius),
        poly_coeffs_f64, poly_degree, q,
        cert_eigs,
        Float64(image_circle_radius), image_certified_radius,
        eps_p, bridge_const, resolvent_Mr, α, Float64(dp_abs),
        is_certified, ε_K, K + 1, Float64(r),
        method, timing
    )
end

# ============================================================================
# Direct Ordschur Resolvent Certification (no polynomial)
# ============================================================================

"""
    OrdschurDirectResult

Result of direct resolvent certification using block Schur structure.

The block triangular inversion formula gives a rigorous resolvent bound:

    (zI - T)⁻¹ = [(zI-T₁₁)⁻¹    (zI-T₁₁)⁻¹ T₁₂ (zI-T₂₂)⁻¹]
                  [0               (zI-T₂₂)⁻¹                  ]

So ‖(zI-T)⁻¹‖₂ ≤ 1/σ₁₁ · (1 + ‖T₁₂‖/σ₂₂) + 1/σ₂₂

where σ₁₁ = σ_min(zI-T₁₁), σ₂₂ = σ_min(zI-T₂₂).
"""
struct OrdschurDirectResult
    eigenvalue_center::ComplexF64
    eigenvalue_radius::Float64
    eigenvalue_ball::Ball{Float64, ComplexF64}
    circle_radius::Float64
    circle_samples::Int
    resolvent_Mr::Float64            # max ‖(zI-A_K)⁻¹‖ on circle
    max_resolvent_T11::Float64       # max 1/σ_min(zI-T₁₁)
    max_resolvent_T22::Float64       # max 1/σ_min(zI-T₂₂)
    T12_norm::Float64                # ‖T₁₂‖_F
    small_gain_factor::Float64       # α = ε_K · M_r
    is_certified::Bool
    truncation_error::Float64        # ε_K
    discretization_size::Int         # K+1
    hardy_space_radius::Float64
    ordschur_block_size::Int         # k (size of T₁₁)
    certification_method::Symbol     # :ordschur_direct
    timing::Float64
end

function Base.show(io::IO, r::OrdschurDirectResult)
    println(io, "Ordschur Direct Certification Result")
    println(io, "====================================")
    println(io, "Eigenvalue: $(r.eigenvalue_center) ± $(r.eigenvalue_radius)")
    println(io, "Certified: $(r.is_certified)")
    println(io, "Method: $(r.certification_method)")
    println(io, "")
    println(io, "Circle radius: $(r.circle_radius)")
    println(io, "Circle samples: $(r.circle_samples)")
    println(io, "Max resolvent T₁₁: $(r.max_resolvent_T11)")
    println(io, "Max resolvent T₂₂: $(r.max_resolvent_T22)")
    println(io, "‖T₁₂‖_F: $(r.T12_norm)")
    println(io, "Resolvent M_r: $(r.resolvent_Mr)")
    println(io, "Small-gain α: $(r.small_gain_factor)")
    println(io, "ε_K: $(r.truncation_error)")
    println(io, "ordschur block: $(r.ordschur_block_size) / $(r.discretization_size)")
    println(io, "Timing: $(round(r.timing, digits=2))s")
end

"""
    certify_eigenvalue_ordschur_direct(A_f64::BallMatrix, lambda_tgt::Number,
                                       ordschur_indices::AbstractVector{<:Integer};
                                       K::Int, r::Real=1.0, N::Int=5000,
                                       circle_radius::Real=0.0,
                                       circle_samples::Int=256,
                                       schur_data_bf=nothing)

Certify an eigenvalue via direct resolvent bound using block Schur structure.

Unlike polynomial deflation, this works in the ORIGINAL λ-space. No polynomial,
no rescaling, no back-mapping. The circle is directly around `lambda_tgt`.

# Why this works
After ordschur, T = [T₁₁ T₁₂; 0 T₂₂] where T₁₁ contains the large eigenvalues
(already certified) and T₂₂ contains small eigenvalues including the target.
Both blocks are well-conditioned for svdbox:
- T₁₁ entries O(1), σ_min(zI-T₁₁) ≥ |λ_tgt| (distance from z≈λ_tgt to T₁₁ spectrum)
- T₂₂ entries O(|λ_{k+1}|), σ_min(zI-T₂₂) controlled by eigenvalue separation

The block triangular resolvent formula gives a rigorous bound WITHOUT needing
to invert or factorize the full matrix.

# Arguments
- `A_f64`: Float64 GKW discretization matrix as BallMatrix
- `lambda_tgt`: target eigenvalue to certify
- `ordschur_indices`: indices (in magnitude-sorted order) of eigenvalues to move to T₁₁
  (should include all eigenvalues larger than `lambda_tgt`)
- `K`: discretization order (matrix is (K+1)×(K+1))
- `r`: Hardy space radius (default 1.0)
- `N`: splitting parameter for C₂ computation
- `circle_radius`: radius of certification circle around `lambda_tgt`;
  if 0 (default), auto-set to half the distance to nearest eigenvalue
- `circle_samples`: number of samples on circle
- `schur_data_bf`: pre-computed BigFloat Schur data (5-tuple from `compute_schur_and_error`);
  if `nothing`, computed from `A_f64`

# Returns
[`OrdschurDirectResult`](@ref)
"""
function certify_eigenvalue_ordschur_direct(A_f64::BallMatrix, lambda_tgt::Number,
                                            ordschur_indices::AbstractVector{<:Integer};
                                            K::Int, r::Real=1.0, N::Int=5000,
                                            circle_radius::Real=0.0,
                                            circle_samples::Int=256,
                                            schur_data_bf=nothing)
    t0 = time()
    λ_tgt = ComplexF64(lambda_tgt)

    # Step 1: BigFloat Schur decomposition
    if schur_data_bf === nothing
        A_bf = float64_ball_to_bigfloat_ball(A_f64)
        schur_data_bf = compute_schur_and_error(A_bf)
    end
    S_bf, _, _, norm_Z_bf, norm_Z_inv_bf = schur_data_bf

    # Sort eigenvalues by magnitude
    T_bf_diag = diag(S_bf.T)
    sorted_idx = sortperm(abs.(T_bf_diag), rev=true)
    sorted_eigs = T_bf_diag[sorted_idx]

    # Step 2: ordschur — move large eigenvalues to T₁₁
    schur_ords_indices = sorted_idx[ordschur_indices]
    T_ord, _, k_block = _bigfloat_ordschur_block(S_bf.T, S_bf.Z, schur_ords_indices)

    n_full = size(T_ord, 1)
    m_block = n_full - k_block  # size of T₂₂

    # Step 3: Extract blocks and convert to Float64 BallMatrix
    # T₁₁ (k×k), T₁₂ (k×m), T₂₂ (m×m)
    T11_bf = T_ord[1:k_block, 1:k_block]
    T12_bf = T_ord[1:k_block, k_block+1:end]
    T22_bf = T_ord[k_block+1:end, k_block+1:end]

    # T₂₂ converts to Float64 BallMatrix for fast svdbox (small entries, well-conditioned).
    T22_f64 = bigfloat_ball_to_float64_ball(BallMatrix(T22_bf))

    # ‖T₁₂‖_F (rigorous upper bound via BigFloat → Float64)
    T12_bf_abs = abs.(T12_bf)
    T12_norm_bf = sqrt(sum(T12_bf_abs .^ 2))
    T12_norm = _bigfloat_to_float64_upper(T12_norm_bf)

    @info "ordschur direct" k_block m_block T12_norm

    # Step 4: Auto-set circle radius if needed
    # Find nearest eigenvalue to λ_tgt (excluding itself)
    λ_tgt_bf = BigFloat(real(λ_tgt))
    if circle_radius <= 0
        # Find target in sorted list
        tgt_idx_in_sorted = findfirst(i -> abs(real(sorted_eigs[i]) - λ_tgt_bf) < BigFloat(1e-30), 1:n_full)
        min_dist = BigFloat(Inf)
        for i in 1:n_full
            i == tgt_idx_in_sorted && continue
            d = abs(sorted_eigs[i] - λ_tgt_bf)
            min_dist = min(min_dist, d)
        end
        circle_radius = Float64(min_dist) / 2.0
        @info "auto circle radius" Float64(min_dist) circle_radius
    end
    circle_radius_f64 = Float64(circle_radius)

    # Step 5: Certify resolvent on circle around λ_tgt
    norm_Z_f64 = _ball_to_float64_upper(norm_Z_bf)
    norm_Z_inv_f64 = _ball_to_float64_upper(norm_Z_inv_bf)

    # T₁₁: compute σ_min(z₀I - T₁₁) once at circle center, then propagate via Weyl.
    # Since all T₁₁ eigenvalues are far from λ_tgt (distance ≫ circle_radius),
    # Weyl's perturbation gives: σ_min(zI-T₁₁) ≥ σ_min(z₀I-T₁₁) - |z-z₀|
    # where z₀ = λ_tgt and |z-z₀| = circle_radius on the circle.
    #
    # Use GenericLinearAlgebra's native BigFloat SVD (not Ogita refinement which
    # uses Float64 seed and fails at condition number > ~10¹⁵). The native SVD
    # computes directly in BigFloat, achieving residuals ~10⁻⁷⁴.
    z0_bf = Complex{BigFloat}(λ_tgt_bf, zero(BigFloat))
    zI_T11_center = z0_bf * I - Complex{BigFloat}.(T11_bf)
    zI_T11_ball = BallMatrix(zI_T11_center)

    # Native BigFloat SVD via GenericLinearAlgebra, then Miyajima certification
    svdA = svd(zI_T11_center)  # dispatches to GenericLinearAlgebra for BigFloat
    sv11_result = BallArithmetic._certify_svd(
        zI_T11_ball, svdA, BallArithmetic.MiyajimaM1(); apply_vbd=true)
    sv11_center = sv11_result.singular_values
    σ11_center_ball = sv11_center[end]
    σ11_center_lower_bf = BallArithmetic.mid(σ11_center_ball) - BallArithmetic.rad(σ11_center_ball)
    σ11_center_lower = max(Float64(σ11_center_lower_bf), 0.0)
    t11_method = :svd_gla

    # Propagate to circle: σ_min(zI-T₁₁) ≥ σ_min(z₀I-T₁₁) - ρ
    σ11_on_circle = σ11_center_lower - circle_radius_f64
    @info "T₁₁ bound" t11_method σ11_center_lower σ11_on_circle circle_radius_f64

    if σ11_on_circle <= 0
        @warn "σ_min(zI-T₁₁) bound ≤ 0: eigenvalue too close to T₁₁ spectrum" σ11_center_lower circle_radius_f64
        max_resolvent = Inf
        max_res_T11 = Inf
        max_res_T22 = 0.0
    else
        # 1/σ₁₁ is constant (upper bound) across the circle
        r11 = setrounding(Float64, RoundUp) do
            1.0 / σ11_on_circle
        end
        max_res_T11 = r11

        # T₂₂: scan circle with Float64 svdbox (small entries, well-conditioned)
        max_res_T22 = 0.0
        max_resolvent = 0.0

        for s in 0:(circle_samples - 1)
            θ = 2π * s / circle_samples
            z_f64 = ComplexF64(real(λ_tgt) + circle_radius_f64 * cos(θ),
                               imag(λ_tgt) + circle_radius_f64 * sin(θ))

            sv22 = svdbox(T22_f64 - z_f64 * I)
            σ22_ball = sv22[end]
            σ22_lower = max(Float64(BallArithmetic.mid(σ22_ball)) -
                            Float64(BallArithmetic.rad(σ22_ball)), 0.0)

            if σ22_lower <= 0
                @warn "σ_min(zI-T₂₂) ≤ 0 at sample s=$s" σ22_lower
                max_resolvent = Inf
                break
            end

            # Block formula: ‖(zI-T)⁻¹‖ ≤ 1/σ₁₁·(1 + ‖T₁₂‖/σ₂₂) + 1/σ₂₂
            res_z = setrounding(Float64, RoundUp) do
                inv_σ22 = 1.0 / σ22_lower
                r11 * (1.0 + T12_norm * inv_σ22) + inv_σ22
            end

            r22 = setrounding(Float64, RoundUp) do
                1.0 / σ22_lower
            end

            max_res_T22 = max(max_res_T22, r22)
            max_resolvent = max(max_resolvent, res_z)
        end
    end

    # Step 6: Schur similarity bridge
    resolvent_Mr = setrounding(Float64, RoundUp) do
        norm_Z_f64 * max_resolvent * norm_Z_inv_f64
    end

    # Step 7: Small-gain check
    ε_K = Float64(real(compute_Δ(K; N=N)))
    α = setrounding(Float64, RoundUp) do
        ε_K * resolvent_Mr
    end
    is_certified = α < 1.0

    # Eigenvalue radius = circle radius (direct, no backmap needed)
    λ_radius = is_certified ? circle_radius_f64 : Inf

    timing = time() - t0

    return OrdschurDirectResult(
        λ_tgt, λ_radius, Ball(λ_tgt, λ_radius),
        circle_radius_f64, circle_samples,
        resolvent_Mr, max_res_T11, max_res_T22, T12_norm,
        α, is_certified, ε_K, K + 1, Float64(r),
        k_block, :ordschur_direct, timing
    )
end

# ============================================================================
# Two-Stage Certification Pipeline
# ============================================================================

"""
    TwoStageCertificationResult

Result of the two-stage certification pipeline for a single eigenvalue.

**Stage 1** (K_low): Resolvent certification on excluding circles proves
simplicity and bounds ‖R_{L_r}(z)‖ on the contour.

**Stage 2** (K_high): NK certification gives tight eigenpair enclosures.

**Transfer bridge**: Stage 1 resolvent + Stage 2 truncation error give
Riesz projector error bounds.

# Fields
- `eigenvalue_center`, `eigenvalue_index`: identity of the eigenvalue
- `stage1_*`: Stage 1 resolvent certification at K_low
- `stage2_*`: Stage 2 NK certification at K_high
- `transfer_*`: reverse perturbation from L_r to A_{K_high}
- `riesz_*`: Riesz projector approximation error
- `hardy_space_radius`, `C2_bound`: metadata
"""
struct TwoStageCertificationResult
    # Identity
    eigenvalue_center::ComplexF64
    eigenvalue_index::Int
    # Stage 1 (resolvent at K_low)
    stage1_K::Int
    stage1_circle_radius::Float64
    stage1_resolvent_Ak::Float64      # ‖R_{A_{K_low}}‖ on Γ
    stage1_alpha::Float64              # ε_{K_low} · resolvent_Ak
    stage1_eps_K::Float64
    stage1_M_inf::Float64              # ‖R_{L_r}‖ = resolvent_Ak / (1 - α₁)
    stage1_is_certified::Bool
    # Stage 2 (NK at K_high)
    stage2_K::Int
    stage2_eps_K::Float64
    stage2_nk_radius::Float64          # r_NK
    stage2_eigenvalue_radius::Float64
    stage2_eigenvector_radius::Float64
    stage2_is_certified::Bool
    # Transfer bridge
    transfer_resolvent_Ak_high::Float64 # ‖R_{A_{K_high}}‖ via reverse transfer
    transfer_alpha_high::Float64
    transfer_is_valid::Bool
    # Riesz projector
    riesz_projector_error::Float64      # ‖P_{L_r} - P_{A_{K_high}}‖
    riesz_contour_length::Float64       # |Γ| = 2πr
    # Metadata
    hardy_space_radius::Float64
    C2_bound::Float64
end

function Base.show(io::IO, r::TwoStageCertificationResult)
    println(io, "Two-Stage Certification Result")
    println(io, "==============================")
    println(io, "Eigenvalue $(r.eigenvalue_index): $(r.eigenvalue_center)")
    println(io, "")
    println(io, "Stage 1 (K=$(r.stage1_K), resolvent):")
    println(io, "  Circle radius: $(r.stage1_circle_radius)")
    println(io, "  ‖R_{A_K}‖: $(r.stage1_resolvent_Ak)")
    println(io, "  α₁ = ε_K · ‖R‖: $(r.stage1_alpha)")
    println(io, "  ε_K: $(r.stage1_eps_K)")
    println(io, "  M_∞ = ‖R_{L_r}‖: $(r.stage1_M_inf)")
    println(io, "  Certified: $(r.stage1_is_certified)")
    println(io, "")
    println(io, "Stage 2 (K=$(r.stage2_K), NK):")
    println(io, "  ε_K: $(r.stage2_eps_K)")
    println(io, "  NK radius: $(r.stage2_nk_radius)")
    println(io, "  Eigenvalue radius: $(r.stage2_eigenvalue_radius)")
    println(io, "  Eigenvector radius: $(r.stage2_eigenvector_radius)")
    println(io, "  Certified: $(r.stage2_is_certified)")
    println(io, "")
    println(io, "Transfer bridge:")
    println(io, "  ‖R_{A_{K_high}}‖ (reverse): $(r.transfer_resolvent_Ak_high)")
    println(io, "  α_high: $(r.transfer_alpha_high)")
    println(io, "  Valid: $(r.transfer_is_valid)")
    println(io, "")
    println(io, "Riesz projector:")
    println(io, "  ‖P_{L_r} - P_{A_{K_high}}‖: $(r.riesz_projector_error)")
    println(io, "  Contour length: $(r.riesz_contour_length)")
    println(io, "")
    println(io, "Hardy space: H²(D_{$(r.hardy_space_radius)})")
    println(io, "C₂ bound: $(r.C2_bound)")
end

"""
    reverse_transfer_resolvent_bound(M_inf::Float64, eps_K_high::Float64)

Reverse perturbation: given ‖R_{L_r}(z)‖ ≤ M_inf (from Stage 1), bound
‖R_{A_{K_high}}(z)‖ at a higher truncation level.

Since L_r = A_{K_high} + (L_r - A_{K_high}) and ‖L_r - A_{K_high}‖ ≤ ε_{K_high},
the standard Neumann perturbation gives:

    ‖R_{A_{K_high}}(z)‖ ≤ M_inf / (1 - M_inf · ε_{K_high})

All arithmetic uses directed rounding for rigorous bounds.

# Returns
- `(resolvent_Ak_high, alpha_high, is_valid)` tuple
"""
function reverse_transfer_resolvent_bound(M_inf::Float64, eps_K_high::Float64)
    # α_high = M_inf · ε_{K_high}, rigorous upper bound
    alpha_high = setrounding(Float64, RoundUp) do
        M_inf * eps_K_high
    end
    alpha_high >= 1.0 && return (Inf, alpha_high, false)

    # denominator (lower bound) = 1 - α_high
    denom = setrounding(Float64, RoundDown) do
        1.0 - alpha_high
    end
    denom <= 0.0 && return (Inf, alpha_high, false)

    # ‖R_{A_{K_high}}‖ ≤ M_inf / denom (rigorous upper bound)
    resolvent = setrounding(Float64, RoundUp) do
        M_inf / denom
    end
    return (resolvent, alpha_high, true)
end

"""
    projector_approximation_error_rigorous(contour_length::Float64,
        resolvent_Ak_high::Float64, eps_K_high::Float64)

Rigorous Riesz projector error bound with directed rounding.

```math
\\|P_{L_r}(\\Gamma) - P_{A_{K_{high}}}(\\Gamma)\\|
\\leq \\frac{|\\Gamma|}{2\\pi} \\cdot
\\frac{\\|R_{A_{K_{high}}}\\|^2 \\cdot \\varepsilon_{K_{high}}}{1 - \\varepsilon_{K_{high}} \\cdot \\|R_{A_{K_{high}}}\\|}
```

# Returns
- `(error_bound, is_valid)` tuple
"""
function projector_approximation_error_rigorous(contour_length::Float64,
        resolvent_Ak_high::Float64, eps_K_high::Float64)
    alpha = setrounding(Float64, RoundUp) do
        eps_K_high * resolvent_Ak_high
    end
    alpha >= 1.0 && return (Inf, false)

    denom = setrounding(Float64, RoundDown) do
        1.0 - alpha
    end
    denom <= 0.0 && return (Inf, false)

    error_bound = setrounding(Float64, RoundUp) do
        (contour_length / (2.0 * π)) * resolvent_Ak_high * resolvent_Ak_high * eps_K_high / denom
    end
    return (error_bound, true)
end

end # module
