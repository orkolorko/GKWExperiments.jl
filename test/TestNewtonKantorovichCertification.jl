using Test
using GKWExperiments
using GKWExperiments.NewtonKantorovichCertification: whiten_eigenpair, h2_whiten_ball
using ArbNumerics
using BallArithmetic
using LinearAlgebra

@testset "NewtonKantorovichCertification" begin

    @testset "whiten_eigenpair biorthogonality" begin
        setprecision(256)
        s = ArbComplex(1.0, 0.0)
        K = 16
        M_arb = gkw_matrix_direct(s; K=K)
        A_mid = Float64.(real.(ArbNumerics.midpoint.(real.(M_arb))) +
                         im .* ArbNumerics.midpoint.(imag.(M_arb)))

        λ_A, v_A, u_A, A_tilde = whiten_eigenpair(A_mid, 1.0, 1)

        # Biorthogonality: u_A* v_A = 1
        @test abs(dot(u_A, v_A) - 1.0) < 1e-12

        # v_A is an eigenvector of A_tilde
        residual = norm(A_tilde * v_A - λ_A * v_A)
        @test residual < 1e-10

        # u_A is a left eigenvector of A_tilde for conj(λ_A)
        residual_left = norm(A_tilde' * u_A - conj(λ_A) * u_A)
        @test residual_left < 1e-10

        # Dominant eigenvalue of GKW at s=1 should be ≈ 1.0
        @test abs(λ_A - 1.0) < 0.01
    end

    @testset "whiten_eigenpair second eigenvalue" begin
        setprecision(256)
        s = ArbComplex(1.0, 0.0)
        K = 16
        M_arb = gkw_matrix_direct(s; K=K)
        A_mid = Float64.(real.(ArbNumerics.midpoint.(real.(M_arb))) +
                         im .* ArbNumerics.midpoint.(imag.(M_arb)))

        λ_A, v_A, u_A, A_tilde = whiten_eigenpair(A_mid, 1.0, 2)

        # Biorthogonality
        @test abs(dot(u_A, v_A) - 1.0) < 1e-12

        # Eigenvector residual
        @test norm(A_tilde * v_A - λ_A * v_A) < 1e-10

        # λ₂ ≈ -0.3037 for s=1
        @test abs(real(λ_A) - (-0.3037)) < 0.01
    end

    @testset "h2_whiten_ball" begin
        # For r=1, whitening should be identity
        N = 4
        center = randn(ComplexF64, N, N)
        radius = abs.(randn(N, N)) * 0.01
        A_ball = BallMatrix(center, radius)

        A_whitened = h2_whiten_ball(A_ball, 1.0)
        @test BallArithmetic.mid(A_whitened) ≈ center
        @test BallArithmetic.rad(A_whitened) ≈ radius

        # For r≠1, check scaling pattern
        r = 1.5
        A_whitened_r = h2_whiten_ball(A_ball, r)
        mid_w = BallArithmetic.mid(A_whitened_r)
        for i in 1:N, j in 1:N
            expected = center[i, j] * r^(i-1) / r^(j-1)
            @test abs(mid_w[i, j] - expected) < 1e-12
        end
    end

    @testset "assemble_eigenpair_jacobian structure" begin
        setprecision(256)
        s = ArbComplex(1.0, 0.0)
        K = 8
        M_arb = gkw_matrix_direct(s; K=K)
        A_ball = arb_to_ball_matrix(M_arb)
        A_tilde_ball = h2_whiten_ball(A_ball, 1.0)

        A_mid = BallArithmetic.mid(A_ball)
        λ_A, v_A, u_A, _ = whiten_eigenpair(A_mid, 1.0, 1)

        Jk = assemble_eigenpair_jacobian(A_tilde_ball, λ_A, v_A, u_A)
        N = K + 1

        @test size(Jk) == (N + 1, N + 1)

        Jk_mid = BallArithmetic.mid(Jk)

        # First column should be [-v_A; 0]
        for i in 1:N
            @test Jk_mid[i, 1] ≈ -v_A[i]
        end
        @test Jk_mid[N + 1, 1] ≈ 0.0

        # Last row should be [0, u_A*]
        @test Jk_mid[N + 1, 1] ≈ 0.0
        for j in 1:N
            @test Jk_mid[N + 1, j + 1] ≈ conj(u_A[j])
        end
    end

    @testset "NKCertificationResult construction and show" begin
        result = NKCertificationResult(
            1.0 + 0.0im,           # eigenvalue_center
            ones(ComplexF64, 5),    # eigenvector_center
            0.001,                  # enclosure_radius
            0.001,                  # eigenvalue_radius
            0.001,                  # eigenvector_radius
            0.01,                   # qk_bound
            10.0,                   # C_bound
            0.02,                   # q0_bound
            1e-5,                   # y_bound
            1e-8,                   # truncation_error
            0.9,                    # discriminant
            true,                   # is_certified
            33,                     # discretization_size
            1.0,                    # hardy_space_radius
            10.5,                   # C2_bound
            1.0                     # v_norm
        )

        @test result.is_certified
        @test result.eigenvalue_center ≈ 1.0
        @test result.enclosure_radius == 0.001
        @test result.discretization_size == 33

        # Test show method
        io = IOBuffer()
        show(io, result)
        output = String(take!(io))
        @test contains(output, "Newton")
        @test contains(output, "Certified: true")
        @test contains(output, "r_NK")
    end

    @testset "certify_eigenpair_nk dominant eigenvalue K=32" begin
        setprecision(256)
        s = ArbComplex(1.0, 0.0)
        K = 32

        result = certify_eigenpair_nk(s; K=K, target_idx=1, N_C2=100)

        @test result isa NKCertificationResult
        @test result.discretization_size == K + 1
        @test result.truncation_error > 0
        @test result.qk_bound >= 0
        @test result.C_bound > 0

        # Dominant eigenvalue should be ≈ 1.0
        @test abs(result.eigenvalue_center - 1.0) < 0.01

        # At K=32, certification should succeed
        @test result.is_certified
        @test isfinite(result.enclosure_radius)
        @test result.enclosure_radius > 0
        @test result.q0_bound < 1.0
        @test result.discriminant >= 0.0
    end

    @testset "certify_eigenpair_nk second eigenvalue" begin
        setprecision(256)
        s = ArbComplex(1.0, 0.0)
        K = 32

        result = certify_eigenpair_nk(s; K=K, target_idx=2, N_C2=100)

        @test result isa NKCertificationResult

        # λ₂ ≈ -0.3037
        @test abs(real(result.eigenvalue_center) - (-0.3037)) < 0.01

        if result.is_certified
            @test isfinite(result.enclosure_radius)
            @test result.enclosure_radius > 0
        end
    end

    @testset "certify_eigenpair_nk with pre-computed BallMatrix" begin
        setprecision(256)
        s = ArbComplex(1.0, 0.0)
        K = 16
        M_arb = gkw_matrix_direct(s; K=K)
        A_ball = arb_to_ball_matrix(M_arb)

        result = certify_eigenpair_nk(A_ball; K=K, target_idx=1, N_C2=100)

        @test result isa NKCertificationResult
        @test result.discretization_size == K + 1
        @test result.truncation_error > 0
    end

    @testset "convergence: r_NK decreases with K" begin
        setprecision(256)
        s = ArbComplex(1.0, 0.0)
        radii = Float64[]

        for K in [16, 24, 32]
            result = certify_eigenpair_nk(s; K=K, target_idx=1, N_C2=100)
            push!(radii, result.enclosure_radius)
        end

        # r_NK should decrease as K increases (tighter enclosure)
        # Only check if all certified
        if all(isfinite.(radii))
            @test radii[3] < radii[1]
        end
    end

end
