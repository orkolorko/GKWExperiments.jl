using Test
using LinearAlgebra
using BallArithmetic
using GKWExperiments.CertifScripts

@testset "compute_schur_and_error" begin
    A = BallArithmetic.BallMatrix([1 + 2im  0.2 - 0.1im; -0.3 + 0.05im  0.7 - 1.0im])

    S, errF, errT, norm_Z, norm_Z_inv = compute_schur_and_error(A)

    @test isa(S, LinearAlgebra.Schur)
    @test isa(errF, Float64)
    @test isa(errT, Float64)
    @test isa(norm_Z, Float64)
    @test isa(norm_Z_inv, Float64)

    _, _, errT_identity, _, _ = compute_schur_and_error(A; polynomial = (0.0, 1.0))
    @test errT_identity == errT

    coeffs = (0.3, -0.4, 0.1)
    S_poly, errF_poly, errT_poly, norm_Z_poly, norm_Z_inv_poly = compute_schur_and_error(A; polynomial = coeffs)

    bZ = BallArithmetic.BallMatrix(S_poly.Z)
    bT = BallArithmetic.BallMatrix(S_poly.T)

    pA = CertifScripts._polynomial_matrix(coeffs, A)
    pT = CertifScripts._polynomial_matrix(coeffs, bT)
    manual_err = BallArithmetic.svd_bound_L2_opnorm(bZ * pT * bZ' - pA)

    @test errF_poly == errF
    @test errT_poly == manual_err
    @test norm_Z_poly == norm_Z
    @test norm_Z_inv_poly == norm_Z_inv
end

@testset "bound_res_original" begin
    A = BallArithmetic.BallMatrix([1 + 2im  0.2 - 0.1im; -0.3 + 0.05im  0.7 - 1.0im])
    _, _, errT, norm_Z, norm_Z_inv = compute_schur_and_error(A)

    l2pseudo = BallArithmetic.Ball(1.1, 1e-3)
    η = 0.25
    N = size(A, 1)

    bound = bound_res_original(l2pseudo, η, norm_Z, norm_Z_inv, errT, errT, N)

    @test isa(bound, Float64)
    @test bound > 0
end

@testset "run_certification" begin
    A = BallArithmetic.BallMatrix([1.0 + 0.0im  0.05 - 0.01im; 0.0 + 0.0im  0.9 + 0.0im])
    circle = CertificationCircle(0.9 + 0.0im, 0.2; samples = 4)

    pts = points_on(circle)
    @test length(pts) == circle.samples
    @test all(isapprox.(abs.(pts .- circle.center), circle.radius; atol = 1e-8))

    result = run_certification(A, circle, 1;
        polynomial = (0.0, 1.0),
        η = 0.6,
        check_interval = circle.samples + 5,
        channel_capacity = 8)

    @test !isempty(result.certification_log)
    @test result.minimum_singular_value > 0
    @test result.l2_resolvent_bound > 0
    @test result.l1_resolvent_bound > 0
    @test result.schur_matrix isa BallArithmetic.BallMatrix
    @test result.circle == circle
    @test result.polynomial == [0.0, 1.0]
end
