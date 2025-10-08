using Test
using LinearAlgebra
using Distributed
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

@testset "run_certification with polynomial" begin
    A = BallArithmetic.BallMatrix([0.8 + 0.0im  0.1 + 0.05im; 0.0 + 0.0im  0.6 + 0.2im])
    circle = CertificationCircle(0.7 + 0.1im, 0.25; samples = 5)
    coeffs = (0.2, -0.3, 0.05)

    result = run_certification(A, circle, 1;
        polynomial = coeffs,
        η = 0.55,
        check_interval = circle.samples + 3,
        channel_capacity = 8)

    @test result.polynomial == collect(coeffs)
    expected = CertifScripts._polynomial_matrix(collect(coeffs), BallArithmetic.BallMatrix(result.schur.T))
    @test all(isapprox.(expected.c, result.schur_matrix.c; atol = 1e-8))
    @test all(isapprox.(expected.r, result.schur_matrix.r; atol = 1e-8))
    @test result.minimum_singular_value > 0
    @test result.l2_resolvent_bound > 0
    @test result.l1_resolvent_bound > 0
end

@testset "resume_certification_from_snapshot" begin
    A = BallArithmetic.BallMatrix([1.0 + 0.0im  0.05 - 0.01im; 0.0 + 0.0im  0.9 + 0.0im])
    circle = CertificationCircle(0.9 + 0.0im, 0.2; samples = 6)
    η = 0.45

    schur, errF, errT, norm_Z, norm_Z_inv = compute_schur_and_error(A)
    schur_matrix = BallArithmetic.BallMatrix(schur.T)
    snapshot_base = tempname()

    added_workers = addprocs(1)
    certification_log = Any[]
    arcs = CertifScripts._initial_arcs(circle)
    cache = Dict{ComplexF64, Any}()
    pending = Dict{Int, Tuple{ComplexF64, ComplexF64}}()
    files = (snapshot_base * "_A.jld2", snapshot_base * "_B.jld2")

    try
        CertifScripts._load_certification_dependencies(added_workers)
        CertifScripts._set_schur_on_workers(added_workers, schur_matrix)

        job_channel = RemoteChannel(() -> Channel{Tuple{Int, ComplexF64}}(8))
        result_channel = RemoteChannel(() -> Channel{NamedTuple}(8))
        configure_certification!(; job_channel = job_channel, result_channel = result_channel,
            certification_log = certification_log, snapshot = snapshot_base)
        foreach(pid -> remote_do(CertifScripts.dowork, pid, job_channel, result_channel), added_workers)

        task = @async begin
            try
                adaptive_arcs!(arcs, cache, pending, η; check_interval = 1)
            catch e
                if !(e isa InterruptException)
                    rethrow(e)
                end
            end
        end

        progress = false
        snapshot_ready = false
        for _ in 1:200
            snapshot_ready = any(isfile, files)
            progress = length(certification_log) >= 1
            if progress && snapshot_ready
                break
            end
            sleep(0.05)
        end

        @test progress
        @test snapshot_ready

        Base.throwto(task, InterruptException())
        wait(task)
    finally
        rmprocs(added_workers)
    end

    snapshot = choose_snapshot_to_load(snapshot_base)
    @test snapshot !== nothing

    arcs_snapshot = snapshot["arcs"]
    cache_snapshot = snapshot["cache"]
    log_snapshot = snapshot["log"]
    pending_snapshot = snapshot["pending"]

    @test (!isempty(arcs_snapshot) || !isempty(pending_snapshot))

    initial_log_length = length(log_snapshot)

    added_workers = addprocs(1)
    try
        CertifScripts._load_certification_dependencies(added_workers)
        CertifScripts._set_schur_on_workers(added_workers, schur_matrix)

        job_channel = RemoteChannel(() -> Channel{Tuple{Int, ComplexF64}}(8))
        result_channel = RemoteChannel(() -> Channel{NamedTuple}(8))
        configure_certification!(; job_channel = job_channel, result_channel = result_channel,
            certification_log = log_snapshot, snapshot = snapshot_base)
        foreach(pid -> remote_do(CertifScripts.dowork, pid, job_channel, result_channel), added_workers)

        adaptive_arcs!(arcs_snapshot, cache_snapshot, pending_snapshot, η; check_interval = 10)
    finally
        rmprocs(added_workers)
    end

    @test isempty(arcs_snapshot)
    @test isempty(pending_snapshot)
    @test length(log_snapshot) > initial_log_length

    min_sigma = minimum(log -> log.lo_val, log_snapshot)
    l2pseudo = maximum(log -> log.hi_res, log_snapshot)
    resolvent_bound = bound_res_original(l2pseudo, η, norm_Z, norm_Z_inv, errF, errT, size(A, 1))

    @test min_sigma > 0
    @test resolvent_bound > 0

    CertifScripts._cleanup_snapshots(snapshot_base)
end
