using Test
using GKWExperiments
using GKWExperiments.Polynomials: polyval, polyval_derivative
using ArbNumerics
using BallArithmetic
using LinearAlgebra

@testset "DeflationCertification" begin

    @testset "polyval_derivative" begin
        # p(x) = 3 + 2x + x² → p'(x) = 2 + 2x
        coeffs = [3.0, 2.0, 1.0]
        p, dp = polyval_derivative(coeffs, 1.0)
        @test p ≈ 6.0   # 3 + 2 + 1
        @test dp ≈ 4.0   # 2 + 2

        p, dp = polyval_derivative(coeffs, 0.0)
        @test p ≈ 3.0
        @test dp ≈ 2.0

        p, dp = polyval_derivative(coeffs, -1.0)
        @test p ≈ 2.0    # 3 - 2 + 1
        @test dp ≈ 0.0    # 2 - 2

        # constant polynomial: p(x) = 5 → p'(x) = 0
        p, dp = polyval_derivative([5.0], 42.0)
        @test p ≈ 5.0
        @test dp ≈ 0.0

        # empty polynomial
        p, dp = polyval_derivative(Float64[], 1.0)
        @test p == 0.0
        @test dp == 0.0

        # linear polynomial: p(x) = 1 + 3x → p'(x) = 3
        p, dp = polyval_derivative([1.0, 3.0], 2.0)
        @test p ≈ 7.0
        @test dp ≈ 3.0

        # cubic: p(x) = x³ → p'(x) = 3x²
        p, dp = polyval_derivative([0.0, 0.0, 0.0, 1.0], 2.0)
        @test p ≈ 8.0
        @test dp ≈ 12.0

        # complex evaluation
        p, dp = polyval_derivative([1.0, 1.0, 1.0], 1.0im)
        # p(i) = 1 + i + i² = 1 + i - 1 = i
        # p'(i) = 1 + 2i
        @test p ≈ 0.0 + 1.0im
        @test dp ≈ 1.0 + 2.0im

        # consistency with polyval
        for coeffs in [[1.0, -2.0, 3.0, -0.5], [0.5, 1.0], [7.0]]
            for x in [-1.0, 0.0, 0.5, 2.0, 3.14]
                pv, _ = polyval_derivative(coeffs, x)
                @test pv ≈ polyval(coeffs, x)
            end
        end
    end

    @testset "backmap_inclusion_radius" begin
        # Identity polynomial p(x) = x: p'(x) = 1
        r_lambda, dp = backmap_inclusion_radius(0.1, [0.0, 1.0], 1.0)
        @test r_lambda ≈ 0.1  # r_p / |p'| = 0.1 / 1
        @test dp ≈ 1.0

        # Scaled polynomial p(x) = 2x: p'(x) = 2
        r_lambda, dp = backmap_inclusion_radius(0.1, [0.0, 2.0], 1.0)
        @test r_lambda ≈ 0.05  # 0.1 / 2
        @test dp ≈ 2.0

        # Quadratic p(x) = x² at x=3: p'(3) = 6
        r_lambda, dp = backmap_inclusion_radius(0.6, [0.0, 0.0, 1.0], 3.0)
        @test r_lambda ≈ 0.1  # 0.6 / 6
        @test dp ≈ 6.0

        # Second-order rigorous back-mapping using BallArithmetic
        # p(x) = x² at x=3: p'(3)=6, p''=2
        # First-order: δ₁ = 0.06/6 = 0.01
        # M₂ = sup|p''| on B(3, 0.01) = 2 (constant for quadratic)
        # Quadratic: δ² - 6δ + 0.06 = 0 → δ = (6-√(36-0.24))/2
        r_lambda2, _ = backmap_inclusion_radius(0.06, [0.0, 0.0, 1.0], 3.0; order=2)
        r_lambda1, _ = backmap_inclusion_radius(0.06, [0.0, 0.0, 1.0], 3.0; order=1)
        @test r_lambda2 > 0
        @test isfinite(r_lambda2)
        # Second-order bound ≥ first-order (accounts for curvature rigorously)
        @test r_lambda2 >= r_lambda1 - 1e-15

        # For a higher-degree polynomial, second-order correction is more significant
        # p(x) = x⁴ at x=1: p'(1)=4, p''(x)=12x² → M₂ on B(1, 0.0025) > 12
        r2, _ = backmap_inclusion_radius(0.01, [0.0, 0.0, 0.0, 0.0, 1.0], 1.0; order=2)
        r1, _ = backmap_inclusion_radius(0.01, [0.0, 0.0, 0.0, 0.0, 1.0], 1.0; order=1)
        @test r2 > 0
        @test r1 > 0
        @test r2 >= r1 - 1e-15  # rigorous bound is wider

        # For linear polynomial, second-order equals first-order (p'' = 0)
        r2_lin, _ = backmap_inclusion_radius(0.1, [0.0, 3.0], 2.0; order=2)
        r1_lin, _ = backmap_inclusion_radius(0.1, [0.0, 3.0], 2.0; order=1)
        @test r2_lin ≈ r1_lin  # exact match for linear

        # Verify analytical formula for p(x) = x²
        # δ₁ = r_p/|p'| = 0.06/6 = 0.01
        # p'' = 2 everywhere, M₂ = 2
        # Quadratic: δ² - 6δ + 0.06 = 0 → δ = 3 - √(9-0.06) = 3 - √8.94
        expected_delta2 = 3.0 - sqrt(9.0 - 0.06)
        @test r_lambda2 ≈ expected_delta2 atol=1e-12

        # Edge case: zero derivative
        r_lambda, dp = backmap_inclusion_radius(0.1, [1.0], 0.0)  # constant poly
        @test r_lambda == Inf
        @test dp == 0.0
    end

    @testset "DeflationCertificationResult construction and show" begin
        result = DeflationCertificationResult(
            -0.3 + 0.0im,     # eigenvalue_center
            0.01,              # eigenvalue_radius
            Ball(-0.3 + 0.0im, 0.01),  # eigenvalue_ball
            [2.0, -1.0],      # deflation_polynomial_coeffs
            1,                 # deflation_polynomial_degree
            1,                 # deflation_power
            [1.0 + 0.0im],    # deflated_eigenvalues
            0.5,               # image_circle_radius
            0.5,               # image_certified_radius
            1e-10,             # poly_perturbation_bound
            100.0,             # bridge_constant
            50.0,              # resolvent_Mr
            0.5,               # small_gain_factor
            2.0,               # p_derivative_at_target
            true,              # is_certified
            1e-8,              # truncation_error
            33,                # discretization_size
            1.0,               # hardy_space_radius
            :direct,           # certification_method
            1.5                # timing
        )

        @test result.is_certified
        @test result.eigenvalue_center ≈ -0.3
        @test result.deflation_polynomial_degree == 1
        @test result.certification_method == :direct

        # Test show method
        io = IOBuffer()
        show(io, result)
        output = String(take!(io))
        @test contains(output, "Deflation Certification")
        @test contains(output, "Certified: true")
    end

    @testset "certify_eigenvalue_deflation basic" begin
        setprecision(256)
        s = ArbComplex(1.0, 0.0)
        K = 16

        # Build GKW matrix
        finite_result = certify_gkw_eigenspaces(s; K=K)
        A = finite_result.gkw_matrix

        # Get eigenvalues from Schur (sorted by magnitude)
        A_center = BallArithmetic.mid(A)
        S = schur(A_center)
        eigs = diag(S.T)
        sorted_idx = sortperm(abs.(eigs), rev=true)

        λ1 = ComplexF64(eigs[sorted_idx[1]])  # ≈ 1.0
        λ2 = ComplexF64(eigs[sorted_idx[2]])  # ≈ -0.3037

        # Deflate λ₁ to certify λ₂
        result = certify_eigenvalue_deflation(
            A, real(λ2), [real(λ1)];
            K=K, N=100,
            image_circle_radius=0.5,
            image_circle_samples=64,
            method=:direct,
            backmap_order=1,
            use_tight_bridge=false  # faster for test
        )

        @test result isa DeflationCertificationResult
        @test result.deflation_polynomial_degree >= 1
        @test result.truncation_error > 0
        @test result.resolvent_Mr > 0

        # The certification might or might not succeed at K=16
        # but the pipeline should run without errors
        if result.is_certified
            @test result.small_gain_factor < 1.0
            @test isfinite(result.eigenvalue_radius)
            @test result.eigenvalue_radius > 0
        end
    end

end
