using Test
using LinearAlgebra
using BallArithmetic
using GKWExperiments

# Note: These tests use the serial certification API from BallArithmetic.CertifScripts
# The distributed API has been removed in favor of BallArithmetic's implementation

@testset "compute_schur_and_error" begin
    A = BallMatrix([1.0 + 2.0im  0.2 - 0.1im; -0.3 + 0.05im  0.7 - 1.0im])

    S, errF, errT, norm_Z, norm_Z_inv = compute_schur_and_error(A)

    @test isa(S, LinearAlgebra.Schur)
    @test isa(errF, Real)
    @test isa(errT, Real)
    @test isa(norm_Z, Real)
    @test isa(norm_Z_inv, Real)

    # Identity polynomial should give same error as no polynomial
    _, _, errT_identity, _, _ = compute_schur_and_error(A; polynomial = (0.0, 1.0))
    @test errT_identity ≈ errT

    # Test with a nontrivial polynomial
    coeffs = (0.3, -0.4, 0.1)
    S_poly, errF_poly, errT_poly, norm_Z_poly, norm_Z_inv_poly = compute_schur_and_error(A; polynomial = coeffs)

    @test errF_poly ≈ errF
    @test norm_Z_poly ≈ norm_Z
    @test norm_Z_inv_poly ≈ norm_Z_inv
end

@testset "bound_res_original" begin
    A = BallMatrix([1.0 + 2.0im  0.2 - 0.1im; -0.3 + 0.05im  0.7 - 1.0im])
    _, errF, errT, norm_Z, norm_Z_inv = compute_schur_and_error(A)

    l2pseudo = Ball(1.1, 1e-3)
    η = 0.25
    N = size(A, 1)

    bound = bound_res_original(l2pseudo, η, norm_Z, norm_Z_inv, errF, errT, N)

    @test isa(bound, Real)
    @test bound > 0
end

@testset "CertificationCircle" begin
    center = 0.9 + 0.0im
    radius = 0.2
    samples = 8

    circle = CertificationCircle(center, radius; samples = samples)

    @test circle.center == center
    @test circle.radius == radius
    @test circle.samples == samples

    pts = points_on(circle)
    @test length(pts) == samples
    @test all(isapprox.(abs.(pts .- center), radius; atol = 1e-10))

    # Test error conditions
    @test_throws ArgumentError CertificationCircle(0.0, 0.0)  # zero radius
    @test_throws ArgumentError CertificationCircle(0.0, -1.0)  # negative radius
    @test_throws ArgumentError CertificationCircle(0.0, 1.0; samples = 2)  # too few samples
end

@testset "run_certification (serial)" begin
    # Simple 2x2 matrix with well-separated eigenvalues
    A = BallMatrix([1.0 + 0.0im  0.05 - 0.01im; 0.0 + 0.0im  0.9 + 0.0im])
    circle = CertificationCircle(0.9 + 0.0im, 0.15; samples = 8)

    # Run serial certification (no num_workers parameter in BallArithmetic API)
    result = run_certification(A, circle;
        polynomial = (0.0, 1.0),
        η = 0.6,
        check_interval = 10)

    @test !isempty(result.certification_log)
    @test result.minimum_singular_value > 0
    @test result.resolvent_schur > 0
    @test result.resolvent_original > 0
    @test result.schur_matrix isa BallMatrix
    @test result.circle == circle
    @test result.polynomial == [0.0, 1.0]
end

@testset "run_certification with polynomial" begin
    A = BallMatrix([0.8 + 0.0im  0.1 + 0.05im; 0.0 + 0.0im  0.6 + 0.2im])
    circle = CertificationCircle(0.7 + 0.1im, 0.2; samples = 8)
    coeffs = (0.2, -0.3, 0.05)

    result = run_certification(A, circle;
        polynomial = coeffs,
        η = 0.55,
        check_interval = 10)

    @test result.polynomial == collect(coeffs)
    @test result.minimum_singular_value > 0
    @test result.resolvent_schur > 0
    @test result.resolvent_original > 0
end

@testset "poly_from_roots" begin
    # (x - 1)(x - 2) = x² - 3x + 2 = 2 - 3x + x²
    coeffs = poly_from_roots([1.0, 2.0])
    @test length(coeffs) == 3
    @test coeffs[1] ≈ 2.0   # constant term
    @test coeffs[2] ≈ -3.0  # x coefficient
    @test coeffs[3] ≈ 1.0   # x² coefficient

    # Single root: (x - 3) = -3 + x
    coeffs_single = poly_from_roots([3.0])
    @test coeffs_single ≈ [-3.0, 1.0]

    # Empty roots: constant 1
    coeffs_empty = poly_from_roots(Float64[])
    @test coeffs_empty ≈ [1.0]
end
