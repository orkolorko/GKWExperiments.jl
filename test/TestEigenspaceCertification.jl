using Test
using GKWExperiments
using ArbNumerics
using LinearAlgebra
using BallArithmetic

# Note: s=0.5 (critical line) causes numerical issues with Hurwitz zeta in ArbNumerics.
# We use s=0.75+0.3i or s=1.0 (classical GKW) for testing.

@testset "EigenspaceCertification" begin

    @testset "arb_to_ball_matrix conversion" begin
        setprecision(256)
        s = ArbComplex(0.75, 0.3)
        K = 8
        M_arb = GKWExperiments.GKWDiscretization.gkw_matrix_direct(s; K=K)

        # Test conversion
        A = arb_to_ball_matrix(M_arb)
        @test size(A) == (K+1, K+1)
        @test all(isfinite.(BallArithmetic.mid(A)))
        @test all(rad(A) .>= 0)

        # Check that radii are reasonable (conversion from high precision to Float64)
        max_radius = maximum(rad(A))
        @test max_radius < 1e-14  # Should be small (mainly Float64 rounding)
    end

    @testset "certify_gkw_eigenspaces with VBD" begin
        setprecision(256)
        s = ArbComplex(0.75, 0.3)
        K = 16

        result = certify_gkw_eigenspaces(s; K=K)

        # Check result structure
        @test result isa GKWEigenCertificationResult
        @test result.discretization_size == K + 1

        # Check block Schur decomposition was computed
        bs = result.block_schur
        @test bs isa RigorousBlockSchurResult
        @test length(bs.clusters) >= 1

        # Check VBD verification bounds (relax tolerance for numerical errors)
        @test isfinite(bs.residual_norm)
        @test isfinite(bs.orthogonality_defect)
        @test bs.residual_norm < 1e-8
        @test bs.orthogonality_defect < 1e-8

        # Check projections were computed
        @test length(result.projections_of_one) == length(bs.clusters)
        @test length(result.projection_coefficients) == length(bs.clusters)
    end

    @testset "classical GKW s=1" begin
        setprecision(256)
        s = ArbComplex(1.0, 0.0)  # Classical GKW operator
        K = 12

        result = certify_gkw_eigenspaces(s; K=K)

        # Check result
        @test result isa GKWEigenCertificationResult
        @test result.s_parameter ≈ 1.0 + 0.0im

        # Block Schur should have reasonable bounds
        @test result.block_schur.residual_norm < 1e-8
        @test result.block_schur.orthogonality_defect < 1e-8

        # Should have at least one cluster
        @test length(result.block_schur.clusters) >= 1
    end

    @testset "num_clusters parameter" begin
        setprecision(256)
        s = ArbComplex(0.75, 0.3)
        K = 10

        # Compute with all clusters
        result_all = certify_gkw_eigenspaces(s; K=K)
        total = length(result_all.block_schur.clusters)

        # Compute with limited clusters
        if total >= 2
            result_limited = certify_gkw_eigenspaces(s; K=K, num_clusters=2)

            # Should have same block Schur but only 2 projections
            @test length(result_limited.projection_coefficients) == 2
            @test length(result_limited.block_schur.clusters) == total  # VBD finds all
        end
    end

    @testset "projection of constant function" begin
        setprecision(256)
        s = ArbComplex(0.75, 0.3)
        K = 12

        result = certify_gkw_eigenspaces(s; K=K)

        # Check that projections were computed
        @test !isempty(result.projections_of_one)

        # Projections should be vectors of correct size
        @test all(length(proj_vec) == K + 1 for proj_vec in result.projections_of_one)

        # First projection coefficient should be finite
        first_coeff = result.projection_coefficients[1]
        @test isfinite(BallArithmetic.mid(first_coeff))
    end

    @testset "eigenvalue information from VBD" begin
        setprecision(256)
        s = ArbComplex(0.75, 0.3)
        K = 10

        result = certify_gkw_eigenspaces(s; K=K)

        # Access eigenvalues through VBD result
        vbd = result.block_schur.vbd_result
        @test length(vbd.eigenvalues) == K + 1
        @test all(isfinite.(vbd.eigenvalues))

        # Cluster intervals should be valid
        @test length(vbd.cluster_intervals) == K + 1
        # Check all intervals at once
        @test all(isfinite(BallArithmetic.mid(interval)) for interval in vbd.cluster_intervals)
        @test all(rad(interval) >= 0 for interval in vbd.cluster_intervals)
    end

    @testset "verify_block_schur_properties" begin
        setprecision(256)
        s = ArbComplex(0.75, 0.3)
        K = 10

        result = certify_gkw_eigenspaces(s; K=K)

        # Use BallArithmetic's verification
        @test verify_block_schur_properties(result.block_schur; tol=1e-8)
    end

    @testset "different s parameters" begin
        setprecision(256)

        # Test with s = 1 + 0.1i
        s1 = ArbComplex(1.0, 0.1)
        result1 = certify_gkw_eigenspaces(s1; K=10)
        @test result1.block_schur.residual_norm < 1e-8

        # Test with s = 0.6 + 0.2i
        s2 = ArbComplex(0.6, 0.2)
        result2 = certify_gkw_eigenspaces(s2; K=10)
        @test result2.block_schur.residual_norm < 1e-8
    end

    @testset "show method" begin
        setprecision(256)
        s = ArbComplex(0.75, 0.3)
        K = 8

        result = certify_gkw_eigenspaces(s; K=K)

        # Just check that show doesn't error
        io = IOBuffer()
        show(io, result)
        output = String(take!(io))
        @test contains(output, "GKW Eigenspace Certification Result")
        @test contains(output, "Block Schur")
        @test contains(output, "Cluster")
    end

end
