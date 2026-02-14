using Test
using GKWExperiments
using ArbNumerics
using LinearAlgebra
using BallArithmetic

@testset "TwoStageCertification" begin

    @testset "reverse_transfer_resolvent_bound" begin
        # Basic case: M_inf = 100, ε = 1e-10
        # α_high = 100 * 1e-10 = 1e-8
        # resolvent = 100 / (1 - 1e-8) ≈ 100
        resolvent, alpha, valid = reverse_transfer_resolvent_bound(100.0, 1e-10)
        @test valid
        @test alpha < 1.0
        @test alpha ≈ 1e-8 atol=1e-15
        @test resolvent > 100.0  # Must be strictly larger due to perturbation
        @test resolvent ≈ 100.0 rtol=1e-6

        # Edge case: α ≥ 1 should return Inf
        resolvent2, alpha2, valid2 = reverse_transfer_resolvent_bound(100.0, 0.01)
        @test !valid2
        @test resolvent2 == Inf
        @test alpha2 >= 1.0

        # Edge case: α exactly at boundary
        resolvent3, alpha3, valid3 = reverse_transfer_resolvent_bound(1.0, 1.0)
        @test !valid3
        @test resolvent3 == Inf

        # Directed rounding correctness: result ≥ exact BigFloat computation
        M = 50.0
        eps_val = 1e-20
        resolvent4, _, valid4 = reverse_transfer_resolvent_bound(M, eps_val)
        @test valid4
        exact = BigFloat(M) / (BigFloat(1) - BigFloat(M) * BigFloat(eps_val))
        @test resolvent4 >= Float64(exact)

        # Very small ε: resolvent should be very close to M_inf
        resolvent5, alpha5, valid5 = reverse_transfer_resolvent_bound(50.0, 1e-44)
        @test valid5
        @test alpha5 < 1e-40
        @test resolvent5 ≈ 50.0 rtol=1e-10
    end

    @testset "projector_approximation_error_rigorous" begin
        # Basic case: small ε gives small error
        contour_length = 2π * 0.01  # Circle of radius 0.01
        resolvent = 50.0
        eps_val = 1e-10

        error_bound, valid = projector_approximation_error_rigorous(
            contour_length, resolvent, eps_val)
        @test valid
        @test error_bound > 0
        @test error_bound < 1.0  # Should be very small

        # Compare with non-rigorous version: rigorous ≥ non-rigorous
        error_nonrigorous, valid_nr = projector_approximation_error(
            contour_length, resolvent, eps_val)
        @test valid_nr
        @test error_bound >= error_nonrigorous || isapprox(error_bound, error_nonrigorous; atol=1e-15)

        # Edge case: α ≥ 1
        error2, valid2 = projector_approximation_error_rigorous(
            contour_length, 1e10, 1.0)
        @test !valid2
        @test error2 == Inf

        # Directed rounding: result ≥ exact BigFloat computation
        L = 2π * 0.005
        R = 30.0
        ε = 1e-15
        error3, valid3 = projector_approximation_error_rigorous(L, R, ε)
        @test valid3
        alpha_exact = BigFloat(ε) * BigFloat(R)
        denom_exact = BigFloat(1) - alpha_exact
        exact_error = (BigFloat(L) / (2 * BigFloat(π))) * BigFloat(R)^2 * BigFloat(ε) / denom_exact
        @test error3 >= Float64(exact_error)

        # Very small ε: error should be dominated by R² * ε * (L / 2π)
        error4, valid4 = projector_approximation_error_rigorous(
            2π * 0.01, 50.0, 1e-44)
        @test valid4
        # Approximate: 0.01 * 2500 * 1e-44 = 2.5e-43
        @test error4 < 1e-40
        @test error4 > 0
    end

    @testset "TwoStageCertificationResult structure" begin
        result = TwoStageCertificationResult(
            1.0 + 0.0im,    # eigenvalue_center
            1,               # eigenvalue_index
            48,              # stage1_K
            0.01,            # stage1_circle_radius
            50.0,            # stage1_resolvent_Ak
            5e-7,            # stage1_alpha
            1e-8,            # stage1_eps_K
            50.00003,        # stage1_M_inf
            true,            # stage1_is_certified
            256,             # stage2_K
            1e-44,           # stage2_eps_K
            1e-42,           # stage2_nk_radius
            1e-42,           # stage2_eigenvalue_radius
            1e-42,           # stage2_eigenvector_radius
            true,            # stage2_is_certified
            50.00003,        # transfer_resolvent_Ak_high
            5e-43,           # transfer_alpha_high
            true,            # transfer_is_valid
            2.5e-41,         # riesz_projector_error
            2π * 0.01,       # riesz_contour_length
            1.0,             # hardy_space_radius
            10.5             # C2_bound
        )

        @test result.stage1_is_certified
        @test result.stage2_is_certified
        @test result.transfer_is_valid
        @test result.eigenvalue_center == 1.0 + 0.0im
        @test result.eigenvalue_index == 1
        @test result.stage1_K == 48
        @test result.stage2_K == 256
        @test result.riesz_projector_error < 1e-40

        # Test show method doesn't error
        io = IOBuffer()
        show(io, result)
        output = String(take!(io))
        @test contains(output, "Two-Stage Certification")
        @test contains(output, "Stage 1")
        @test contains(output, "Stage 2")
        @test contains(output, "Transfer bridge")
        @test contains(output, "Riesz projector")
    end

    @testset "integration: small-K two-stage pipeline" begin
        setprecision(ArbFloat, 256)
        setprecision(BigFloat, 256)

        s = ArbComplex(1.0, 0.0)
        K_low = 16
        K_high = 24
        N_splitting = 100

        # Compute truncation errors
        eps_K_low = _arb_to_float64_upper(compute_Δ(K_low; N=N_splitting))
        eps_K_high = _arb_to_float64_upper(compute_Δ(K_high; N=N_splitting))
        C2_val = _arb_to_float64_upper(compute_C2(N_splitting))

        @test eps_K_high < eps_K_low  # Higher K gives smaller error

        # Stage 1: Build matrix and run resolvent certification at K_low
        finite_result = certify_gkw_eigenspaces(s; K=K_low)
        A_low = finite_result.gkw_matrix
        A_center = BallArithmetic.mid(A_low)
        S = schur(A_center)
        eigenvalues = diag(S.T)
        sorted_idx = sortperm(abs.(eigenvalues), rev=true)

        # Test first eigenvalue (λ₁ ≈ 1)
        λ_center = ComplexF64(eigenvalues[sorted_idx[1]])
        circle_radius = 0.05

        # Avoid overlap with second eigenvalue
        λ2_center = ComplexF64(eigenvalues[sorted_idx[2]])
        dist = abs(λ_center - λ2_center)
        circle_radius = min(circle_radius, dist / 3)

        circle = CertificationCircle(λ_center, circle_radius; samples=64)
        cert_data = run_certification(A_low, circle)

        resolvent_Ak = cert_data.resolvent_original
        alpha1 = eps_K_low * resolvent_Ak

        if alpha1 < 1.0
            # Compute M_inf
            M_inf = resolvent_Ak / (1.0 - alpha1)

            # Stage 2: NK at K_high
            nk_result = certify_eigenpair_nk(s; K=K_high, target_idx=1, N_C2=N_splitting)

            # Transfer bridge
            resolvent_high, alpha_high, transfer_valid = reverse_transfer_resolvent_bound(
                M_inf, eps_K_high)

            if transfer_valid
                # Riesz projector error
                contour_length = 2π * circle_radius
                proj_error, proj_valid = projector_approximation_error_rigorous(
                    contour_length, resolvent_high, eps_K_high)

                @test proj_valid
                @test proj_error > 0
                @test isfinite(proj_error)

                # Projector error at K_high should be much smaller than at K_low
                proj_error_low, _ = projector_approximation_error_rigorous(
                    contour_length, resolvent_Ak, eps_K_low)
                @test proj_error < proj_error_low

                # alpha_high should be much smaller than alpha1
                @test alpha_high < alpha1
            end

            if nk_result.is_certified
                @test nk_result.enclosure_radius > 0
                @test isfinite(nk_result.enclosure_radius)
            end
        end
    end

end
