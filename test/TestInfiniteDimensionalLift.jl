using Test
using GKWExperiments
using ArbNumerics
using LinearAlgebra
using BallArithmetic

@testset "InfiniteDimensionalLift" begin

    @testset "resolvent_bridge_condition" begin
        # Small-gain condition: α = ε · R < 1 (strict inequality)
        @test resolvent_bridge_condition(10.0, 0.05) == (true, 0.5)
        @test resolvent_bridge_condition(10.0, 0.1)[1] == false    # α = 1.0, fails strict inequality
        @test resolvent_bridge_condition(10.0, 0.2)[1] == false    # α > 1

        # Edge cases
        @test resolvent_bridge_condition(0.0, 1.0) == (true, 0.0)
        @test resolvent_bridge_condition(1e10, 1e-12)[1] == true
    end

    @testset "certified_resolvent_bound" begin
        # When α < 1, bound = R / (1 - α)
        bound, valid = certified_resolvent_bound(10.0, 0.05)
        @test valid
        @test bound ≈ 10.0 / (1 - 0.5) atol=1e-10

        # When α ≥ 1, bound is infinite
        bound2, valid2 = certified_resolvent_bound(10.0, 0.2)
        @test !valid2
        @test bound2 == Inf
    end

    @testset "eigenvalue_inclusion_radius" begin
        # If small-gain satisfied, radius = circle_radius
        radius, certified = eigenvalue_inclusion_radius(1.0+0im, 10.0, 0.01, 0.05)
        @test certified
        @test radius == 0.01

        # If small-gain fails, radius = Inf
        radius2, certified2 = eigenvalue_inclusion_radius(1.0+0im, 10.0, 0.01, 0.2)
        @test !certified2
        @test radius2 == Inf
    end

    @testset "projector_approximation_error" begin
        contour_length = 2π * 0.01  # Circle of radius 0.01
        resolvent = 10.0
        truncation_error = 0.05  # α = 0.5

        error_bound, valid = projector_approximation_error(contour_length, resolvent, truncation_error)
        @test valid
        @test error_bound > 0

        # Expected: (2πr / 2π) * (R² * ε) / (1 - α) = r * R² * ε / (1 - α)
        expected = 0.01 * 100.0 * 0.05 / 0.5
        @test error_bound ≈ expected atol=1e-10
    end

    @testset "newton_kantorovich_error" begin
        resolvent = 10.0
        left_norm = 1.0
        right_norm = 1.0
        truncation = 0.01

        eig_err, vec_err, total = newton_kantorovich_error(
            resolvent, left_norm, right_norm, truncation
        )

        # DF_inv_bound = max(left + resolvent, right) = max(11, 1) = 11
        # total = 11 * 0.01 = 0.11 (with unit norm eigenvector)
        @test total ≈ 0.11 atol=1e-10
        @test eig_err == total
        @test vec_err == total
    end

    @testset "compute_C2 integration" begin
        # Test that C₂ is computed correctly
        C2_N1 = compute_C2(1)
        C2_N10 = compute_C2(10)

        # C₂ should decrease (get tighter) as N increases
        @test real(C2_N10) < real(C2_N1)

        # C₂ should be positive and finite
        @test real(C2_N10) > 0
        @test isfinite(real(C2_N10))
    end

    @testset "compute_Δ integration" begin
        # Test truncation error Δ(K) = C₂ · (2/3)^{K+1}
        Δ_K10 = compute_Δ(10; N=100)
        Δ_K20 = compute_Δ(20; N=100)

        # Δ should decrease exponentially with K
        @test real(Δ_K20) < real(Δ_K10)

        # Ratio should be approximately (2/3)^10
        ratio = real(Δ_K20) / real(Δ_K10)
        expected_ratio = (2/3)^10
        @test ratio ≈ expected_ratio atol=1e-10
    end

    @testset "InfiniteDimCertificationResult structure" begin
        result = InfiniteDimCertificationResult(
            1.0 + 0.0im,    # eigenvalue_center
            0.01,           # eigenvalue_radius
            Ball(1.0 + 0.0im, 0.01),  # eigenvalue_ball
            0.05,           # eigenvector_error
            0.001,          # truncation_error
            100.0,          # resolvent_bound
            0.5,            # small_gain_factor
            true,           # is_certified
            33,             # discretization_size
            1.0,            # hardy_space_radius
            10.5            # C2_bound
        )

        @test result.is_certified
        @test result.eigenvalue_center == 1.0 + 0.0im
        @test result.eigenvalue_radius == 0.01
        @test result.small_gain_factor == 0.5
        @test result.discretization_size == 33

        # Test show method doesn't error
        io = IOBuffer()
        show(io, result)
        output = String(take!(io))
        @test contains(output, "Infinite-Dimensional")
        @test contains(output, "Certified: true")
    end

    @testset "full certification pipeline" begin
        setprecision(256)
        s = ArbComplex(1.0, 0.0)  # Classical GKW
        K = 16

        # Step 1: Finite-dimensional certification
        finite_result = certify_gkw_eigenspaces(s; K=K)
        @test finite_result isa GKWEigenCertificationResult

        # Step 2: Get approximate eigenvalue
        cluster = finite_result.block_schur.clusters[1]
        λ_ball = finite_result.block_schur.vbd_result.cluster_intervals[cluster[1]]
        λ_approx = ComplexF64(BallArithmetic.mid(λ_ball))

        # Step 3: Run resolvent certification on a circle around the eigenvalue
        A = finite_result.gkw_matrix
        circle_radius = 0.05  # Small circle
        circle = CertificationCircle(λ_approx, circle_radius; samples=64)
        cert_data = run_certification(A, circle)

        @test cert_data.resolvent_original > 0
        @test isfinite(cert_data.resolvent_original)

        # Step 4: Combine for infinite-dimensional certification
        inf_result = certify_eigenvalue_lift(finite_result, cert_data, 1; N=100)

        @test inf_result isa InfiniteDimCertificationResult
        @test inf_result.truncation_error > 0
        @test inf_result.C2_bound > 0

        # Check if certified (depends on whether α < 1)
        if inf_result.is_certified
            @test inf_result.small_gain_factor < 1.0
            @test isfinite(inf_result.eigenvalue_radius)
            @test isfinite(inf_result.resolvent_bound)
        else
            @test inf_result.small_gain_factor >= 1.0
        end
    end

    @testset "verify_spectral_gap" begin
        setprecision(256)
        s = ArbComplex(1.0, 0.0)
        K = 12

        # Get a certification result
        finite_result = certify_gkw_eigenspaces(s; K=K)
        A = finite_result.gkw_matrix

        # Choose a point likely in the resolvent set (far from spectrum)
        test_center = 0.5 + 0.5im
        circle = CertificationCircle(test_center, 0.1; samples=64)
        cert_data = run_certification(A, circle)

        # Verify spectral gap
        is_gap, margin = verify_spectral_gap(cert_data, test_center, 0.1; N=100)

        # The margin should tell us how far from the small-gain boundary we are
        @test margin isa Float64
        if is_gap
            @test margin > 0
        end
    end

    @testset "deflation_truncation_error" begin
        # Polynomial p(z) = z - 0.5 (deflating eigenvalue 0.5)
        poly_coeffs = [-0.5, 1.0]
        Ak_norm = 10.0
        Lr_norm = 10.0
        base_error = 0.01

        deflation_error = GKWExperiments.InfiniteDimensionalLift.deflation_truncation_error(
            poly_coeffs, Ak_norm, Lr_norm, base_error
        )

        # For degree 1 polynomial: bridge_const = |a_1| · 1 · C^0 = 1.0
        # deflation_error = 0.01 * 1.0 = 0.01
        @test deflation_error ≈ 0.01 atol=1e-12

        # Higher degree polynomial
        poly_coeffs2 = [1.0, -1.0, 0.5]  # 1 - z + 0.5z²
        deflation_error2 = GKWExperiments.InfiniteDimensionalLift.deflation_truncation_error(
            poly_coeffs2, Ak_norm, Lr_norm, base_error
        )

        # bridge_const = |a_1| · 1 · C^0 + |a_2| · 2 · C^1 = 1.0 + 0.5 * 2 * 10 = 11.0
        expected = 0.01 * 11.0
        @test deflation_error2 ≈ expected atol=1e-10
    end

end
