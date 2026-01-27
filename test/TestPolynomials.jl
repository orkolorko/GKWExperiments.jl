using Test
using GKWExperiments.Polynomials

@testset "polyconv" begin
    # (1 + x) * (1 - x) = 1 - x²
    @test polyconv([1.0, 1.0], [1.0, -1.0]) ≈ [1.0, 0.0, -1.0]

    # (1 + 2x) * (3 + 4x) = 3 + 10x + 8x²
    @test polyconv([1.0, 2.0], [3.0, 4.0]) ≈ [3.0, 10.0, 8.0]

    # constant * polynomial
    @test polyconv([2.0], [1.0, 2.0, 3.0]) ≈ [2.0, 4.0, 6.0]
end

@testset "polyval" begin
    # p(x) = 1 + 2x + 3x² at x = 2 → 1 + 4 + 12 = 17
    @test polyval([1.0, 2.0, 3.0], 2.0) ≈ 17.0

    # p(x) = 5 (constant) at any x
    @test polyval([5.0], 100.0) ≈ 5.0

    # complex evaluation
    @test polyval([1.0, 1.0], 1.0im) ≈ 1.0 + 1.0im

    # empty polynomial
    @test polyval(Float64[], 1.0) == 0.0
end

@testset "poly_scale" begin
    @test poly_scale([1.0, 2.0, 3.0], 2.0) ≈ [2.0, 4.0, 6.0]
    @test poly_scale([1.0, 2.0], 0.5) ≈ [0.5, 1.0]
end

@testset "polypow" begin
    # (1 + x)^0 = 1
    @test polypow([1.0, 1.0], 0) ≈ [1.0]

    # (1 + x)^1 = 1 + x
    @test polypow([1.0, 1.0], 1) ≈ [1.0, 1.0]

    # (1 + x)^2 = 1 + 2x + x²
    @test polypow([1.0, 1.0], 2) ≈ [1.0, 2.0, 1.0]

    # (1 + x)^3 = 1 + 3x + 3x² + x³
    @test polypow([1.0, 1.0], 3) ≈ [1.0, 3.0, 3.0, 1.0]

    @test_throws ArgumentError polypow([1.0], -1)
end

@testset "deflation_polynomial" begin
    # deflate single zero at z=2, target λ=1
    # P(x) = 1 - x/2, P(1) = 1/2, α = 2
    # p(x) = 2*(1 - x/2) = 2 - x
    p = deflation_polynomial([2.0], 1.0)
    @test polyval(p, 1.0) ≈ 1.0  # normalized so p(λ_tgt) = 1
    @test polyval(p, 2.0) ≈ 0.0  # zeros at the deflated eigenvalue

    # deflate two zeros
    p2 = deflation_polynomial([2.0, 3.0], 1.0)
    @test polyval(p2, 1.0) ≈ 1.0
    @test abs(polyval(p2, 2.0)) < 1e-14
    @test abs(polyval(p2, 3.0)) < 1e-14

    # with power q=2
    p3 = deflation_polynomial([2.0], 1.0; q=2)
    @test polyval(p3, 1.0) ≈ 1.0
    @test abs(polyval(p3, 2.0)) < 1e-14

    # empty zeros list
    p_empty = deflation_polynomial(Float64[], 1.0)
    @test p_empty == [1.0]
end

@testset "coeffs_about_c_from_about_0" begin
    # p(z) = 1 + 2z + 3z² centered at c=1
    # p(z) = p(1 + (z-1)) = 1 + 2(1+(z-1)) + 3(1+(z-1))²
    #      = 1 + 2 + 2(z-1) + 3(1 + 2(z-1) + (z-1)²)
    #      = 6 + 8(z-1) + 3(z-1)²
    a = [1.0, 2.0, 3.0]
    b = coeffs_about_c_from_about_0(a, 1.0)
    @test b ≈ [6.0, 8.0, 3.0]

    # verify by evaluation
    for z in [0.0, 0.5, 1.0, 1.5, 2.0]
        @test polyval(a, z) ≈ polyval(b, z - 1.0)
    end
end

@testset "coeffs_about_0_from_about_c" begin
    # inverse of the above
    b = [6.0, 8.0, 3.0]
    a = coeffs_about_0_from_about_c(b, 1.0)
    @test a ≈ [1.0, 2.0, 3.0]

    # round-trip test
    original = [1.0, -2.0, 0.5, 3.0]
    c = 2.5
    shifted = coeffs_about_c_from_about_0(original, c)
    recovered = coeffs_about_0_from_about_c(shifted, c)
    @test recovered ≈ original
end
