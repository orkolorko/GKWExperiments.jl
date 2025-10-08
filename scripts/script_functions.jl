@inline compute_steps(ρ, r_pearl; arc = 2*pi) = ceil(Int64, (arc * ρ) / r_pearl)

function submit_job(
    λ, ρ, r_pearl, job_queue; start_angle = 0, stop_angle = 2*pi, N = compute_steps(ρ, r_pearl; arc = stop_angle-start_angle))
    for (i, θ) in enumerate(range(; start = start_angle, stop = stop_angle, length = N))
        # cis(x)
        # More efficient method for exp(im*x) by using Euler's formula: cos(x) + i sin(x) = \exp(i x)
        put!(job_queue, (i, θ,  λ + ρ * cis(θ), r_pearl))
    end
    return N
end

@everywhere function dowork(P, jobs, results)
    while true
        i,θ, c, r_pearl = take!(jobs)
        z = BallArithmetic.Ball(c, r_pearl)
        t = @elapsed Σ = BallArithmetic.svdbox(P - z * LinearAlgebra.I)
        put!(results,
            (i = i,
                val_c = Σ[end].c,
                val_r = Σ[end].r,
                second_val_c = Σ[end - 1].c,
                second_val_r = Σ[end - 1].r,
                c = c,
                radian = θ,
                r_pearl = r_pearl,
                t = t,
                id = Distributed.myid()))
    end
end

function compute_enclosure_arc(D, λ, ρ, r_pearl; csvfile = "", start_angle, stop_angle)
    jobs = RemoteChannel(() -> Channel{Tuple}(32))
    results = RemoteChannel(() -> Channel{NamedTuple}(32))

    Ntot = compute_steps(ρ, r_pearl; arc = stop_angle - start_angle)
    @info "$Ntot svd need to be computed to certify the arc centered at $(λ), with radius $(ρ), with pearls of size $(r_pearl)"
    @info "with start angle $(start_angle) and stop angle $(stop_angle)"

    @async submit_job(λ, ρ, r_pearl, jobs; start_angle = start_angle, stop_angle = stop_angle)

    @info "Jobs submitted to the queue"

    T = BallMatrix(D["S"].T)
    foreach(
        pid -> remote_do(dowork, pid, T, jobs, results),
        workers()
    )

    avg_time = 0.0
    N = Ntot
    count = 0
    min_svd = 100
    l2pseudo = 0.0

    if csvfile == ""
        csvfile = "results_$(now())_$(λ)_$(ρ)_$(r_pearl)_$(start_angle)_$(stop_angle).csv"
    end

    tot_time = @elapsed while N > 0 # print out results
        x = take!(results)

        sv = setrounding(Float64, RoundDown) do
            return x.val_c - x.val_r
        end

        l2pseudoloc = setrounding(Float64, RoundUp) do
            return 1.0 / sv
        end

        l2pseudo = max(l2pseudo, l2pseudoloc)
        min_svd = min(min_svd, sv)

        avg_time += x.t
        N = N - 1
        count += 1

        CSV.write(csvfile, [x]; append = isfile(csvfile))
    end

    @info "The minimum singular value along the arc centered at $(λ), with radius $(ρ), with pearls of size $(r_pearl) with start angle $(start_angle) and stop angle $(stop_angle) is $(min_svd), the maximum of the l2 pseudospectra is bounded by $(l2pseudo)"
    avg_time /= Ntot
    @info "Average time for certifying an SVD", avg_time
    @info "Total time for certifying the arc", tot_time
    return min_svd, l2pseudo
end