"""
    EigenspaceCertification

Certify GKW eigenvalues and project the constant function 1 onto eigenspaces.

This module provides the GKW-specific interface for eigenspace certification,
leveraging BallArithmetic.jl's rigorous block Schur decomposition (VBD) methods.
"""
module EigenspaceCertification

using LinearAlgebra
using ArbNumerics
using BallArithmetic

# Import GKW matrix construction from parent module
import ..GKWDiscretization: gkw_matrix_direct

export GKWEigenCertificationResult, certify_gkw_eigenspaces
export arb_to_ball_matrix

"""
    GKWEigenCertificationResult

Result of certifying GKW eigenvalues and projecting the constant function onto eigenspaces.

# Fields
- `block_schur`: The rigorous block Schur decomposition from BallArithmetic.jl
- `projections_of_one`: P_k * [1, 0, ..., 0] for each cluster
- `projection_coefficients`: Leading coefficient (first entry) of each projection
- `gkw_matrix`: The discretized L_s matrix with rigorous error bounds
- `s_parameter`: The s parameter used
- `discretization_size`: The matrix size K+1
"""
struct GKWEigenCertificationResult
    block_schur::RigorousBlockSchurResult
    projections_of_one::Vector{BallVector}
    projection_coefficients::Vector{Ball{Float64, ComplexF64}}
    gkw_matrix::BallMatrix
    s_parameter::ComplexF64
    discretization_size::Int
end

# Delegate to block_schur for common queries
eigenvalues(result::GKWEigenCertificationResult) = result.block_schur.vbd_result.eigenvalues
clusters(result::GKWEigenCertificationResult) = result.block_schur.clusters
num_clusters(result::GKWEigenCertificationResult) = length(result.block_schur.clusters)


"""
    arb_to_ball_matrix(M_arb::Matrix{ArbComplex{P}}) where {P}

Convert an ArbNumerics complex matrix to a BallMatrix{ComplexF64, Float64}.

The radius includes both the ArbNumerics uncertainty and any conversion error
from high-precision to Float64.

# Arguments
- `M_arb`: Matrix of ArbComplex numbers

# Returns
- `BallMatrix`: Ball matrix with centers and radii
"""
function arb_to_ball_matrix(M_arb::Matrix{ArbComplex{P}}) where {P}
    n, m = size(M_arb)
    M_center = Matrix{ComplexF64}(undef, n, m)
    M_radius = Matrix{Float64}(undef, n, m)

    for i in 1:n, j in 1:m
        # Get high-precision midpoint and radius from ArbNumerics
        mid_real_arb = ArbNumerics.midpoint(real(M_arb[i, j]))
        mid_imag_arb = ArbNumerics.midpoint(imag(M_arb[i, j]))
        rad_real_arb = ArbNumerics.radius(real(M_arb[i, j]))
        rad_imag_arb = ArbNumerics.radius(imag(M_arb[i, j]))

        # Convert midpoint to Float64
        center_real = Float64(mid_real_arb)
        center_imag = Float64(mid_imag_arb)
        M_center[i, j] = Complex{Float64}(center_real, center_imag)

        # Convert midpoint to BigFloat for error computation (avoiding ArbReal-BigFloat promotion)
        mid_real_big = parse(BigFloat, string(mid_real_arb))
        mid_imag_big = parse(BigFloat, string(mid_imag_arb))

        # Compute conversion error (difference between BigFloat and Float64)
        conv_err_real = Float64(abs(mid_real_big - BigFloat(center_real)))
        conv_err_imag = Float64(abs(mid_imag_big - BigFloat(center_imag)))

        # Total radius: ArbNumerics radius + conversion error
        total_rad_real = Float64(rad_real_arb) + conv_err_real
        total_rad_imag = Float64(rad_imag_arb) + conv_err_imag

        # Combined radius for complex ball
        M_radius[i, j] = sqrt(total_rad_real^2 + total_rad_imag^2)
    end

    return BallMatrix(M_center, M_radius)
end


"""
    project_constant_function(P::BallMatrix, K::Int)

Project the constant function 1 = [1, 0, 0, ..., 0] (in monomial basis) using projector P.

In the monomial basis {(w-1)^k} for k=0..K, the constant function 1 = (w-1)^0
is represented as the unit vector e_1 = [1, 0, 0, ..., 0].

# Arguments
- `P::BallMatrix`: Spectral projector matrix
- `K::Int`: Discretization size (matrix is (K+1) × (K+1))

# Returns
- `BallVector`: The projected vector P * e_1 with rigorous error bounds
"""
function project_constant_function(P::BallMatrix, K::Int)
    # Construct the constant function in monomial basis: e_1 = [1, 0, 0, ..., 0]
    one_vec_center = zeros(ComplexF64, K + 1)
    one_vec_center[1] = one(ComplexF64)
    one_vec_radius = zeros(Float64, K + 1)
    one_vec = BallVector(one_vec_center, one_vec_radius)

    # Project: P * e_1
    return P * one_vec
end


"""
    compute_cluster_projector(block_schur::RigorousBlockSchurResult, cluster_idx::Int)

Compute the spectral projector for a single cluster from the block Schur decomposition.

The projector P_k projects onto the invariant subspace corresponding to cluster k.
In the Schur basis, this is simply a block of the identity; we transform back to
the original coordinates via P_k = Q * P_schur_k * Q'.

# Arguments
- `block_schur`: Result from `rigorous_block_schur`
- `cluster_idx`: Index of the cluster (1-indexed)

# Returns
- `BallMatrix`: The spectral projector for the specified cluster
"""
function compute_cluster_projector(block_schur::RigorousBlockSchurResult, cluster_idx::Int)
    n = size(block_schur.A, 1)
    clusters = block_schur.clusters

    1 <= cluster_idx <= length(clusters) ||
        throw(BoundsError("cluster_idx $cluster_idx out of range 1:$(length(clusters))"))

    cluster = clusters[cluster_idx]

    # Build projector in Schur basis: P_schur has I in the cluster block, 0 elsewhere
    P_schur_c = zeros(ComplexF64, n, n)
    P_schur_r = zeros(Float64, n, n)
    P_schur_c[cluster, cluster] .= Matrix{ComplexF64}(I, length(cluster), length(cluster))
    P_schur = BallMatrix(P_schur_c, P_schur_r)

    # Transform back to original coordinates: P = Q * P_schur * Q'
    Q = block_schur.Q
    Q_adj = BallMatrix(adjoint(mid(Q)), adjoint(rad(Q)))

    return Q * P_schur * Q_adj
end


"""
    certify_gkw_eigenspaces(s::ArbComplex{P};
                            K::Int=64,
                            num_clusters::Union{Int,Nothing}=nothing,
                            hermitian::Bool=false) where {P}

Certify eigenvalue clusters of the GKW transfer operator and compute rigorous
projections of the constant function 1 onto each eigenspace.

Uses BallArithmetic.jl's rigorous block Schur decomposition (Miyajima VBD) to:
1. Automatically cluster eigenvalues based on Gershgorin disc overlap
2. Provide rigorous error bounds on the decomposition
3. Compute spectral projectors for each cluster

# Arguments
- `s::ArbComplex{P}`: The GKW parameter (e.g., s=1 for classical Gauss map)
- `K::Int=64`: Discretization size (matrix is (K+1) × (K+1))
- `num_clusters::Union{Int,Nothing}=nothing`: If specified, only compute projections
  for the first `num_clusters` clusters (sorted by the VBD ordering)
- `hermitian::Bool=false`: Whether to treat the matrix as Hermitian

# Returns
[`GKWEigenCertificationResult`](@ref) containing the block Schur decomposition,
projectors, and projections of the constant function.

# Example
```julia
using GKWExperiments, ArbNumerics

s = ArbComplex(1.0)  # Classical GKW operator
result = certify_gkw_eigenspaces(s; K=32)

# Access eigenvalue clusters
println("Number of clusters: ", num_clusters(result))
println("Cluster intervals: ", result.block_schur.vbd_result.cluster_intervals)

# Projection of constant function 1 onto first cluster
println("P₁(1) leading coeff = ", result.projection_coefficients[1])

# Verification bounds
println("Residual norm: ", result.block_schur.residual_norm)
println("Orthogonality defect: ", result.block_schur.orthogonality_defect)
```
"""
function certify_gkw_eigenspaces(s::ArbComplex{P};
                                  K::Int=64,
                                  num_clusters::Union{Int,Nothing}=nothing,
                                  hermitian::Bool=false) where {P}
    @assert K >= 1 "K must be at least 1"

    # Step 1: Build GKW matrix with ArbNumerics precision
    M_arb = gkw_matrix_direct(s; K=K)
    n = K + 1

    # Step 2: Convert to BallMatrix with rigorous error enclosure
    A = arb_to_ball_matrix(M_arb)

    # Step 3: Compute rigorous block Schur decomposition using VBD
    block_schur = rigorous_block_schur(A; hermitian=hermitian)

    # Step 4: Determine how many clusters to process
    total_clusters = length(block_schur.clusters)
    clusters_to_process = isnothing(num_clusters) ? total_clusters : min(num_clusters, total_clusters)

    # Step 5: Compute spectral projectors and project constant function for each cluster
    projections = Vector{BallVector}()
    coefficients = Vector{Ball{Float64, ComplexF64}}()

    for i in 1:clusters_to_process
        # Compute projector for this cluster
        P_i = compute_cluster_projector(block_schur, i)

        # Project the constant function
        proj_vec = project_constant_function(P_i, K)
        push!(projections, proj_vec)

        # Extract leading coefficient (first entry)
        push!(coefficients, proj_vec[1])
    end

    return GKWEigenCertificationResult(
        block_schur,
        projections,
        coefficients,
        A,
        ComplexF64(ArbNumerics.midpoint(real(s)), ArbNumerics.midpoint(imag(s))),
        n
    )
end


"""
    Base.show(io::IO, result::GKWEigenCertificationResult)

Pretty-print certification results.
"""
function Base.show(io::IO, result::GKWEigenCertificationResult)
    println(io, "GKW Eigenspace Certification Result")
    println(io, "====================================")
    println(io, "Parameter s = $(result.s_parameter)")
    println(io, "Matrix size = $(result.discretization_size) × $(result.discretization_size)")
    println(io, "")

    bs = result.block_schur
    println(io, "Block Schur decomposition:")
    println(io, "  Number of clusters: $(length(bs.clusters))")
    println(io, "  Residual norm ‖A - QTQ'‖: $(bs.residual_norm)")
    println(io, "  Orthogonality defect ‖Q'Q - I‖: $(bs.orthogonality_defect)")
    println(io, "")

    println(io, "Cluster information:")
    for (i, cluster) in enumerate(bs.clusters)
        interval = bs.vbd_result.cluster_intervals[cluster[1]]
        println(io, "  Cluster $i: indices $cluster, center ≈ $(mid(interval))")
    end
    println(io, "")

    if !isempty(result.projection_coefficients)
        println(io, "Projection coefficients (P_k * 1)[1]:")
        for (i, c) in enumerate(result.projection_coefficients)
            println(io, "  Cluster $i: $(mid(c)) ± $(rad(c))")
        end
    end
end

end # module
