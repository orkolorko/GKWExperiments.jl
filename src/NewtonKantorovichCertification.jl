"""
    NewtonKantorovichCertification

Direct defect-based Newton–Kantorovich/Krawczyk eigenpair certification for the
GKW transfer operator.

This module implements the two-stage certification pipeline:
1. **Stage 1 (external):** Use CertifScripts resolvent/contour methods to obtain
   an initial eigenvalue enclosure ball and prove simplicity.
2. **Stage 2 (this module):** Within each enclosure ball, apply a direct NK
   argument on the eigenpair map F(z,v) = (Mv - zv, u*v - 1) to obtain much
   tighter enclosure radii.

The NK argument avoids contour certification entirely for the refinement step
and works with the eigenpair map directly in a single Hardy space H²(D_r).

All norms are in the whitened coordinates where ‖M‖₂ = ‖M‖_{(r)}.

# References
See `reference/NK.md` for the full mathematical derivation.
"""
module NewtonKantorovichCertification

using LinearAlgebra
using ArbNumerics
using BallArithmetic

import ..Constants: compute_C2, compute_Δ, h2_whiten
import ..Constants: _arb_to_float64_upper, _arb_to_bigfloat_upper
import ..GKWDiscretization: gkw_matrix_direct

export NKCertificationResult, certify_eigenpair_nk
export assemble_eigenpair_jacobian, compute_nk_radius

# ============================================================================
# Result Type
# ============================================================================

"""
    NKCertificationResult

Result of certifying an eigenpair via the Newton–Kantorovich/Krawczyk argument.

The certified ball B((λ_A, v_A), r_NK) in the product space C × C^N contains a
unique true eigenpair (λ*, v*) of the infinite-dimensional operator L_r.

# Fields
- `eigenvalue_center`: Approximate eigenvalue λ_A from discretization
- `eigenvector_center`: Approximate (whitened) eigenvector v_A
- `enclosure_radius`: r_NK — radius of the NK enclosure ball in product norm
- `eigenvalue_radius`: Upper bound on |λ* - λ_A| (≤ r_NK)
- `eigenvector_radius`: Upper bound on ‖v* - v_A‖₂ (≤ r_NK)
- `qk_bound`: Certified ‖I - C·J_k‖₂ (discrete defect)
- `C_bound`: Certified ‖C‖₂ (preconditioner norm)
- `q0_bound`: q₀ = q_k + ‖C‖·ε (infinite-dimensional defect)
- `y_bound`: y = ‖C‖·ε·‖v_A‖₂ (residual size)
- `truncation_error`: ε_K = C₂·(2/3)^{K+1}
- `discriminant`: (1-q₀)² - 4·‖C‖·y (must be ≥ 0)
- `is_certified`: Whether the NK argument succeeds
- `discretization_size`: K+1 (matrix dimension)
- `hardy_space_radius`: r for H²(D_r)
- `C2_bound`: Operator norm constant C₂
- `v_norm`: ‖v_A‖₂ in whitened coordinates
"""
struct NKCertificationResult{CT<:Number, RT<:AbstractFloat}
    eigenvalue_center::CT
    eigenvector_center::Vector{CT}
    enclosure_radius::RT
    eigenvalue_radius::RT
    eigenvector_radius::RT
    qk_bound::RT
    C_bound::RT
    q0_bound::RT
    y_bound::RT
    truncation_error::RT
    discriminant::RT
    is_certified::Bool
    discretization_size::Int
    hardy_space_radius::RT
    C2_bound::RT
    v_norm::RT
end

function Base.show(io::IO, result::NKCertificationResult)
    println(io, "Newton–Kantorovich Eigenpair Certification")
    println(io, "==========================================")
    println(io, "Eigenvalue: $(result.eigenvalue_center) ± $(result.eigenvalue_radius)")
    println(io, "Certified: $(result.is_certified)")
    println(io, "")
    println(io, "NK enclosure radius r_NK = $(result.enclosure_radius)")
    println(io, "Discrete defect q_k = $(result.qk_bound)")
    println(io, "Preconditioner norm ‖C‖ = $(result.C_bound)")
    println(io, "Infinite-dim defect q₀ = $(result.q0_bound)")
    println(io, "Residual bound y = $(result.y_bound)")
    println(io, "Discriminant = $(result.discriminant)")
    println(io, "‖v_A‖₂ = $(result.v_norm)")
    println(io, "")
    println(io, "Truncation error ε_K = $(result.truncation_error)")
    println(io, "Hardy space: H²(D_{$(result.hardy_space_radius)})")
    println(io, "Matrix size: $(result.discretization_size)")
    println(io, "C₂ bound: $(result.C2_bound)")
end

# ============================================================================
# Whitening Utilities
# ============================================================================

"""
    h2_whiten_ball(A_ball::BallMatrix, r::Real) → BallMatrix

Apply the H²(D_r) whitening transform to a BallMatrix.

Computes C_r · A · C_r⁻¹ where C_r = diag(rⁿ), propagating both center
and radius through the diagonal scaling.
"""
function h2_whiten_ball(A_ball::BallMatrix, r::Real)
    N = size(A_ball, 1)
    @assert size(A_ball, 2) == N "Matrix must be square"

    A_center = BallArithmetic.mid(A_ball)
    A_radius = BallArithmetic.rad(A_ball)

    # Whiten center using existing function
    center_whitened = h2_whiten(A_center, r)

    # Whiten radius: |(C_r)_{ii}| · |A_r|_{ij} · |(C_r⁻¹)_{jj}| = r^(i-1) / r^(j-1) · rad_{ij}
    T = typeof(float(r))
    pow = [T(r)^k for k in 0:(N - 1)]
    radius_whitened = similar(A_radius)
    for i in 1:N, j in 1:N
        radius_whitened[i, j] = A_radius[i, j] * pow[i] / pow[j]
    end

    return BallMatrix(center_whitened, radius_whitened)
end

# ============================================================================
# Eigenpair Oracle
# ============================================================================

"""
    whiten_eigenpair(A_mid::AbstractMatrix, r::Real, target_idx::Int)
        → (λ_A, v_A, u_A, A_tilde)

Compute a numerical eigenpair of the whitened matrix and the biorthogonal
left eigenvector.

# Steps
1. Whiten: Ã = C_r · A · C_r⁻¹
2. Compute right eigenpair (λ_A, v_A) of Ã, sorted by descending magnitude
3. Compute left eigenvector u_A of Ã* for conj(λ_A)
4. Normalize so that u_A* · v_A = 1

Types are inferred from the input matrix: Float64 inputs give ComplexF64 outputs,
BigFloat inputs give Complex{BigFloat} outputs (requires `using GenericSchur`).

# Arguments
- `A_mid`: Center of the Galerkin matrix in monomial basis
- `r`: Hardy space radius
- `target_idx`: Which eigenvalue to target (1 = dominant)

# Returns
- `λ_A`: Target eigenvalue
- `v_A`: Right eigenvector (biorthogonally normalized)
- `u_A`: Left eigenvector with u_A* v_A = 1
- `A_tilde`: Whitened matrix
"""
function whiten_eigenpair(A_mid::AbstractMatrix, r::Real, target_idx::Int)
    # Step 1: Whiten (preserves element type)
    A_tilde = Matrix(complex(h2_whiten(A_mid, r)))

    # Step 2: Right eigenpair
    F = eigen(A_tilde)
    sorted_idx = sortperm(abs.(F.values), rev=true)
    idx = sorted_idx[target_idx]
    λ_A = F.values[idx]
    v_A = Vector(F.vectors[:, idx])

    # Step 3: Left eigenvector (eigenvector of A* for conj(λ_A))
    F_adj = eigen(Matrix(A_tilde'))
    # Match eigenvalue closest to conj(λ_A)
    dists = abs.(F_adj.values .- conj(λ_A))
    left_idx = argmin(dists)
    u_A = Vector(F_adj.vectors[:, left_idx])

    # Step 4: Normalize so u_A* v_A = 1
    ip = dot(u_A, v_A)
    u_A ./= ip

    return λ_A, v_A, u_A, A_tilde
end

# ============================================================================
# Jacobian Assembly
# ============================================================================

"""
    assemble_eigenpair_jacobian(A_tilde_ball::BallMatrix, λ_A::Number,
                                 v_A::AbstractVector, u_A::AbstractVector)
        → BallMatrix

Assemble the (N+1)×(N+1) Jacobian of the eigenpair map F(z,v) = (Mv - zv, u*v - 1)
at the approximate eigenpair (λ_A, v_A):

    J = [[-v_A,  (Ã - λ_A I)];
         [ 0,    u_A*        ]]

The first column corresponds to δz, the remaining N columns to δv.
The top-right block (Ã - λ_A I) inherits the BallMatrix uncertainty from Ã,
while v_A, u_A, λ_A are exact floating-point (they define the base point x₀).

# Arguments
- `A_tilde_ball`: Whitened Galerkin matrix as BallMatrix
- `λ_A`: Approximate eigenvalue
- `v_A`: Right eigenvector
- `u_A`: Left eigenvector (biorthogonally normalized)

# Returns
- `BallMatrix`: The (N+1)×(N+1) Jacobian with rigorous error tracking
"""
function assemble_eigenpair_jacobian(A_tilde_ball::BallMatrix, λ_A::Number,
                                      v_A::AbstractVector, u_A::AbstractVector)
    N = size(A_tilde_ball, 1)
    @assert length(v_A) == N
    @assert length(u_A) == N

    A_center = BallArithmetic.mid(A_tilde_ball)
    A_radius = BallArithmetic.rad(A_tilde_ball)

    # Infer types from BallMatrix
    CT = eltype(A_center)   # ComplexF64 or Complex{BigFloat}
    RT = real(CT)            # Float64 or BigFloat

    # Build (N+1)×(N+1) Jacobian
    J_center = zeros(CT, N + 1, N + 1)
    J_radius = zeros(RT, N + 1, N + 1)

    # First column: [-v_A; 0]  (exact: defines the base point)
    for i in 1:N
        J_center[i, 1] = -v_A[i]
    end

    # Top-right N×N block: (Ã - λ_A I)
    # Uncertainty comes only from Ã; λ_A is the exact floating-point base point.
    λ_c = CT(λ_A)
    for i in 1:N, j in 1:N
        if i == j
            J_center[i, j + 1] = A_center[i, j] - λ_c
        else
            J_center[i, j + 1] = A_center[i, j]
        end
        J_radius[i, j + 1] = A_radius[i, j]
    end

    # Bottom row: [0, u_A*]  (exact: defines the functional ℓ)
    for j in 1:N
        J_center[N + 1, j + 1] = conj(u_A[j])
    end

    return BallMatrix(J_center, J_radius)
end

# ============================================================================
# NK Radius Computation
# ============================================================================

"""
    compute_nk_radius(Jk::BallMatrix, C_float::AbstractMatrix,
                       v_A::AbstractVector, epsilon::Real)
        → NamedTuple

Compute the Newton–Kantorovich enclosure radius using the Krawczyk fixed-point
argument (Sections 4–7 of NK.md).

All scalar bounds use directed rounding (`setrounding`) to ensure rigor.
The final radius uses the rationalized formula
    r_NK = 2y / ((1-q₀) + √((1-q₀)² - 4·‖C‖·y))
which avoids cancellation and simplifies directed rounding (numerator up,
denominator down → rigorous upper bound).

The real type `T` (Float64 or BigFloat) is inferred from the BallMatrix.

# Arguments
- `Jk`: (N+1)×(N+1) eigenpair Jacobian as BallMatrix
- `C_float`: Approximate inverse of mid(Jk) (preconditioner)
- `v_A`: Right eigenvector
- `epsilon`: Truncation error ε_K (rigorous upper bound on ‖L_r - A_k‖_{(r)})

# Returns
NamedTuple with fields: `r_NK`, `qk`, `C_ub`, `q0`, `y`, `discriminant`, `is_certified`
"""
function compute_nk_radius(Jk::BallMatrix, C_float::AbstractMatrix,
                            v_A::AbstractVector, epsilon::Real)
    Np1 = size(Jk, 1)
    T = real(eltype(BallArithmetic.mid(Jk)))   # Float64 or BigFloat
    CT = complex(T)                             # ComplexF64 or Complex{BigFloat}

    # Step 1: Wrap preconditioner as BallMatrix (exact, zero radius)
    C_ball = BallMatrix(CT.(C_float), zeros(T, Np1, Np1))

    # Step 2: R_k = I - C·J_k, then q_k = ‖R_k‖₂
    # BallArithmetic subtraction includes ε·|center| term for rounding rigor.
    I_ball = BallMatrix(Matrix{CT}(I, Np1, Np1), zeros(T, Np1, Np1))
    CJ = C_ball * Jk
    Rk = I_ball - CJ

    # Certified upper bound on ‖R_k‖₂ via Rump-style SVD enclosure
    qk_ball = svd_bound_L2_opnorm(Rk)
    qk = T(BallArithmetic.mid(qk_ball)) + T(BallArithmetic.rad(qk_ball))

    # Step 3: Certified upper bound on ‖C‖₂
    C_ub_ball = svd_bound_L2_opnorm(C_ball)
    C_ub = T(BallArithmetic.mid(C_ub_ball)) + T(BallArithmetic.rad(C_ub_ball))

    # Compute ‖v_A‖₂
    v_norm = T(norm(v_A))

    # Convert epsilon to T with directed rounding (rigorous when BigFloat→Float64)
    epsilon_T = setrounding(T, RoundUp) do
        T(epsilon)
    end

    # Step 4: q₀ = q_k + ‖C‖·ε  (rigorous upper bound, NK.md §5.3)
    # All arithmetic under RoundUp to ensure the result is an upper bound.
    q0 = setrounding(T, RoundUp) do
        qk + C_ub * epsilon_T
    end

    # Step 5: y = ‖C‖·ε·‖v_A‖₂  (rigorous upper bound, NK.md §5.4)
    y = setrounding(T, RoundUp) do
        C_ub * epsilon_T * v_norm
    end

    # Step 6: Check q₀ < 1  (NK.md §6.1, contraction condition)
    if q0 >= one(T)
        return (r_NK=T(Inf), qk=qk, C_ub=C_ub, q0=q0, y=y,
                epsilon=epsilon_T, discriminant=T(-Inf), is_certified=false, v_norm=v_norm)
    end

    # Discriminant check: (1-q₀)² ≥ 4·‖C‖·y  (NK.md §6.1)
    # Compute rigorous LOWER bound on (1-q₀)².
    # Since q₀ is an upper bound, (1-q₀) is a lower bound on the true margin.
    # Squaring a positive lower bound with RoundDown gives a lower bound.
    omq_sq_lower = setrounding(T, RoundDown) do
        omq = one(T) - q0
        omq * omq
    end

    # Compute rigorous UPPER bound on 4·‖C‖·y
    four_Cy_upper = setrounding(T, RoundUp) do
        T(4) * C_ub * y
    end

    # Rigorous lower bound on discriminant
    disc = setrounding(T, RoundDown) do
        omq_sq_lower - four_Cy_upper
    end

    if disc < zero(T)
        return (r_NK=T(Inf), qk=qk, C_ub=C_ub, q0=q0, y=y,
                epsilon=epsilon_T, discriminant=disc, is_certified=false, v_norm=v_norm)
    end

    # Step 7: NK enclosure radius  (NK.md §6.2)
    # Use the rationalized formula for numerical stability:
    #   r_NK = 2y / ((1-q₀) + √((1-q₀)² - 4·‖C‖·y))
    # This is algebraically equal to the standard formula
    #   r_NK = ((1-q₀) - √((1-q₀)² - 4·‖C‖·y)) / (2·‖C‖)
    # but avoids catastrophic cancellation in the numerator.
    #
    # For a rigorous UPPER bound: numerator rounded UP, denominator rounded DOWN.

    numer = setrounding(T, RoundUp) do
        T(2) * y
    end

    # denominator = (1-q₀) + √disc
    # (1-q₀) is already a lower bound (since q₀ is upper); round down.
    # √disc: disc is a lower bound, √ is monotone, round down → lower bound.
    denom = setrounding(T, RoundDown) do
        omq = one(T) - q0
        omq + sqrt(disc)
    end

    if denom <= zero(T)
        return (r_NK=T(Inf), qk=qk, C_ub=C_ub, q0=q0, y=y,
                epsilon=epsilon_T, discriminant=disc, is_certified=false, v_norm=v_norm)
    end

    r_NK = setrounding(T, RoundUp) do
        numer / denom
    end

    if r_NK <= zero(T) || !isfinite(r_NK)
        return (r_NK=T(Inf), qk=qk, C_ub=C_ub, q0=q0, y=y,
                epsilon=epsilon_T, discriminant=disc, is_certified=false, v_norm=v_norm)
    end

    return (r_NK=r_NK, qk=qk, C_ub=C_ub, q0=q0, y=y,
            epsilon=epsilon_T, discriminant=disc, is_certified=true, v_norm=v_norm)
end

# ============================================================================
# Main Entry Point
# ============================================================================

"""
    certify_eigenpair_nk(s::ArbComplex; K::Int=32, r::Real=1.0,
                          target_idx::Int=1, N_C2::Int=1000)
        → NKCertificationResult

Certify an eigenpair of the GKW transfer operator using the Newton–Kantorovich
defect-based argument.

This is the **Stage 2** of the two-stage pipeline. Stage 1 uses CertifScripts
to obtain an initial eigenvalue enclosure proving simplicity. This function then
refines the enclosure using a direct NK argument on the eigenpair map.

# Pipeline
1. Build Galerkin matrix via `gkw_matrix_direct(s; K=K)`
2. Convert to BallMatrix with rigorous error bounds
3. Whiten and compute numerical eigenpair (λ_A, v_A, u_A)
4. Assemble eigenpair Jacobian J_k
5. Compute preconditioner C ≈ inv(mid(J_k))
6. Certify discrete defect q_k = ‖I - CJ_k‖₂ and ‖C‖₂
7. Transfer to infinite dimensions: q₀ = q_k + ‖C‖·ε, y = ‖C‖·ε·‖v_A‖₂
8. Compute NK enclosure radius r_NK

# Arguments
- `s`: GKW parameter (e.g., s=1 for classical Gauss map)
- `K`: Discretization order (matrix is (K+1)×(K+1))
- `r`: Hardy space radius (1.0 or 1.5)
- `target_idx`: Which eigenvalue to certify (1 = dominant, sorted by magnitude)
- `N_C2`: Splitting parameter for C₂ computation

# Returns
[`NKCertificationResult`](@ref) with rigorous enclosure bounds.

# Example
```julia
using GKWExperiments, ArbNumerics

# Stage 1: CertifScripts gives initial enclosure (external)
# Stage 2: NK refinement
result = certify_eigenpair_nk(ArbComplex(1.0); K=32)
result.is_certified  # true
result.enclosure_radius  # tight NK radius
```
"""
function certify_eigenpair_nk(s::ArbComplex; K::Int=32, r::Real=1.0,
                               target_idx::Int=1, N_C2::Int=1000)
    # Step 1: Build Galerkin matrix
    M_arb = gkw_matrix_direct(s; K=K)

    # Step 2: Convert to BigFloat BallMatrix (no Float64 truncation)
    A_ball = BallMatrix(BigFloat, M_arb)

    return certify_eigenpair_nk(A_ball; K=K, r=r, target_idx=target_idx, N_C2=N_C2)
end

"""
    certify_eigenpair_nk(A_ball::BallMatrix; K::Int, r::Real=1.0,
                          target_idx::Int=1, N_C2::Int=1000)
        → NKCertificationResult

Overload accepting a pre-computed BallMatrix (avoids rebuilding the matrix).

The real type (Float64 or BigFloat) is inferred from the BallMatrix center.
For BigFloat inputs, `using GenericSchur` must be active for `eigen` to work.
"""
function certify_eigenpair_nk(A_ball::BallMatrix; K::Int, r::Real=1.0,
                               target_idx::Int=1, N_C2::Int=1000)
    N = size(A_ball, 1)
    @assert N == K + 1 "Matrix size $(N) does not match K+1=$(K+1)"

    A_mid = BallArithmetic.mid(A_ball)
    T = real(eltype(A_mid))   # Float64 or BigFloat
    CT = complex(T)

    # Step 3: Whiten and compute eigenpair
    λ_A, v_A, u_A, _ = whiten_eigenpair(A_mid, T(r), target_idx)

    # Step 4: Whiten BallMatrix
    A_tilde_ball = h2_whiten_ball(A_ball, T(r))

    # Step 5: Assemble Jacobian
    Jk = assemble_eigenpair_jacobian(A_tilde_ball, λ_A, v_A, u_A)

    # Step 6: Compute preconditioner C ≈ inv(mid(Jk))
    Jk_mid = BallArithmetic.mid(Jk)
    C_float = inv(Jk_mid)

    # Step 7: Compute truncation error as rigorous BigFloat upper bound.
    # Always compute in BigFloat (no Arb→Float64 truncation).
    # compute_nk_radius will convert to T internally with directed rounding.
    C2_bf = _arb_to_bigfloat_upper(compute_C2(N_C2))
    ε_K_bf = _arb_to_bigfloat_upper(compute_Δ(K; N=N_C2))

    # Step 8: Compute NK radius
    nk = compute_nk_radius(Jk, C_float, v_A, ε_K_bf)

    return NKCertificationResult(
        CT(λ_A),
        Vector{CT}(v_A),
        nk.r_NK,
        nk.r_NK,       # eigenvalue_radius ≤ r_NK (product norm)
        nk.r_NK,       # eigenvector_radius ≤ r_NK
        nk.qk,
        nk.C_ub,
        nk.q0,
        nk.y,
        nk.epsilon,
        nk.discriminant,
        nk.is_certified,
        K + 1,
        T(r),
        T(C2_bf),
        T(nk.v_norm)
    )
end

end # module
