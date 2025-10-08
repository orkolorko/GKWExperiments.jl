module CertifScripts

using Distributed
using LinearAlgebra
using BallArithmetic
using JLD2
using Base: dirname, mod1

export dowork, adaptive_arcs!, bound_res_original, choose_snapshot_to_load,
       save_snapshot!, configure_certification!, set_schur_matrix!,
       compute_schur_and_error, CertificationCircle, points_on, run_certification

const _schur_matrix = Ref{Union{Nothing, BallArithmetic.BallMatrix}}(nothing)
const _job_channel = Ref{Union{Nothing, RemoteChannel}}(nothing)
const _result_channel = Ref{Union{Nothing, RemoteChannel}}(nothing)
const _certification_log = Ref{Any}(nothing)
const _snapshot_path = Ref{Union{Nothing, AbstractString}}(nothing)
const _log_io = Ref{IO}(stdout)

"""
    CertificationCircle(center, radius; samples = 256)

Discretisation of a circle with centre `center`, radius `radius`, and
`samples` equally spaced points used for certification runs.
"""
struct CertificationCircle
    center::ComplexF64
    radius::Float64
    samples::Int
end

function CertificationCircle(center::Number, radius::Real; samples::Integer = 256)
    samples < 3 && throw(ArgumentError("circle discretisation requires at least 3 samples"))
    radius <= 0 && throw(ArgumentError("circle radius must be positive"))
    return CertificationCircle(ComplexF64(center), Float64(radius), Int(samples))
end

"""
    points_on(circle)

Return the discretisation of `circle` used for certification.
"""
function points_on(circle::CertificationCircle)
    θs = range(0, 2π, length = circle.samples + 1)[1:(end - 1)]
    return circle.center .+ circle.radius .* exp.(θs .* im)
end

function _initial_arcs(circle::CertificationCircle)
    points = points_on(circle)
    n = length(points)
    arcs = Vector{Tuple{ComplexF64, ComplexF64}}(undef, n)
    for i in 1:n
        arcs[i] = (points[i], points[mod1(i + 1, n)])
    end
    return arcs
end

"""
    set_schur_matrix!(T)

Store the Schur factor `T` used by [`dowork`](@ref).
"""
function set_schur_matrix!(T::BallArithmetic.BallMatrix)
    _schur_matrix[] = T
    return T
end

"""
    configure_certification!(; job_channel, result_channel, certification_log, snapshot, io)

Cache common resources used by the certification helpers.  The stored values
are used as defaults by [`adaptive_arcs!`](@ref) and [`save_snapshot!`](@ref).
Any keyword may be omitted to keep its previous value.
"""
function configure_certification!(; job_channel = nothing, result_channel = nothing,
        certification_log = nothing, snapshot = nothing, io = nothing)
    job_channel !== nothing && (_job_channel[] = job_channel)
    result_channel !== nothing && (_result_channel[] = result_channel)
    certification_log !== nothing && (_certification_log[] = certification_log)
    snapshot !== nothing && (_snapshot_path[] = snapshot)
    io !== nothing && (_log_io[] = io)
    return nothing
end

function _require_config(ref::Base.RefValue, name::AbstractString)
    value = ref[]
    value === nothing && throw(ArgumentError("$name has not been configured"))
    return value
end

"""
    dowork(jobs, results)

Process tasks received on `jobs`, computing the SVD certification routine for
`T - zI`.  The Schur factor must have been registered in advance with
[`set_schur_matrix!`](@ref).
"""
function dowork(jobs, results)
    T = _require_config(_schur_matrix, "Schur factor")
    while true
        i, z = take!(jobs)
        @debug "Received and working on" z
        bz = BallArithmetic.Ball(z, 0.0)

        t = @elapsed Σ = BallArithmetic.svdbox(T - bz * LinearAlgebra.I)

        val = Σ[end]
        res = 1 / val

        lo_val = setrounding(Float64, RoundDown) do
            return val.c - val.r
        end

        hi_res = setrounding(Float64, RoundUp) do
            return res.c + res.r
        end

        put!(results,
            (
                i = i,
                val = val,
                lo_val = lo_val,
                res = res,
                hi_res = hi_res,
                second_val = Σ[end - 1],
                z = z,
                t = t,
                id = myid()
            ))
    end
end

function _resolve(value, ref::Base.RefValue, name::AbstractString)
    value === nothing || return value
    return _require_config(ref, name)
end

"""
    adaptive_arcs!(arcs, cache, pending, η; kwargs...)

Drive the adaptive refinement routine.  Channels and logging targets may be
passed explicitly via keywords or pre-configured with
[`configure_certification!`](@ref).
"""
function adaptive_arcs!(arcs::Vector{Tuple{ComplexF64, ComplexF64}},
        cache::Dict{ComplexF64, Any},
        pending::Dict{Int, Tuple{ComplexF64, ComplexF64}},
        η::Float64;
        check_interval = 1000,
        job_channel = nothing,
        result_channel = nothing,
        certification_log = nothing,
        snapshot = nothing,
        io = nothing)

    job_channel = _resolve(job_channel, _job_channel, "job_channel")
    result_channel = _resolve(result_channel, _result_channel, "result_channel")
    certification_log = _resolve(certification_log, _certification_log, "certification_log")
    snapshot = _resolve(snapshot, _snapshot_path, "snapshot path")
    io = io === nothing ? _log_io[] : io

    cycle = true
    @info "Starting adaptive refinement, arcs, $(length(arcs)), pending, $(length(pending))"
    flush(io)

    id_counter = maximum(collect(keys(pending)); init = 0) + 1
    @info "Pending from snapshot" length(pending) id_counter

    for (i, (z_a, _)) in pending
        put!(job_channel, (i, z_a))
    end

    while !isempty(pending)
        if isready(result_channel)
            result = take!(result_channel)
            z = result.z
            cache[z] = (result.val, result.second_val)
            z_a, z_b = pending[result.i]
            delete!(pending, result.i)
            push!(arcs, (z_a, z_b))
            push!(certification_log, result)
        else
            sleep(0.1)
        end
    end
    @info "Waited for all pending to be computed, arcs, $(length(arcs)), pending, $(length(pending))"

    flush(io)
    while !isempty(arcs)
        processed = 0
        new = 0

        while !isempty(arcs)
            z_a, z_b = pop!(arcs)

            if haskey(cache, z_a)
                σ_a = cache[z_a][1]
            else
                job_id = id_counter
                put!(job_channel, (job_id, z_a))
                pending[job_id] = (z_a, z_b)
                id_counter += 1
                continue
            end

            ℓ = abs(z_b - z_a)
            ε = ℓ / σ_a

            sup_ε = setrounding(Float64, RoundUp) do
                return ε.c + ε.r
            end

            if sup_ε > η
                z_m = (z_a + z_b) / 2
                push!(arcs, (z_m, z_b))
                push!(arcs, (z_a, z_m))
                new += 1
            end

            processed += 1
            if processed % check_interval == 0
                @info "Processed $processed arcs..."
                @info "Remaining arcs" length(arcs)
                @info "Pending jobs" length(pending)
                @info "New arcs ratio" new / check_interval
                new = 0
                flush(io)

                while isready(result_channel)
                    result = take!(result_channel)
                    z = result.z
                    cache[z] = (result.val, result.second_val)
                    z_a, z_b = pending[result.i]
                    delete!(pending, result.i)
                    push!(arcs, (z_a, z_b))
                    push!(certification_log, result)
                end
                cycle = save_snapshot!(arcs, cache, certification_log, pending, snapshot, cycle)
            end
        end

        @info "Waiting for all pending jobs..."
        while !isempty(pending)
            if isready(result_channel)
                result = take!(result_channel)
                z = result.z
                cache[z] = (result.val, result.second_val)
                z_a, z_b = pending[result.i]
                delete!(pending, result.i)
                push!(arcs, (z_a, z_b))
                push!(certification_log, result)
            else
                @info "Waiting for pending" length(pending)
                flush(io)
                sleep(0.1)
            end
        end

        @info "Restarting refinement cycle with new arcs: $(length(arcs))"
    end

    @info "Adaptive refinement complete"
    return nothing
end

"""
    bound_res_original(l2pseudo, η, norm_Z, norm_Z_inv, errF, errT, N;
        norm_constant = 1)

Return an upper bound on the resolvent norm of the original matrix given
the bounds obtained from the Schur form.  The optional `norm_constant`
scales the result and defaults to ``1``; set it to ``√N`` to recover the
classical ℓ₁ estimate based on the inequality ``‖·‖₁ ≤ √N ‖·‖₂``.
"""
function bound_res_original(l2pseudo, η, norm_Z, norm_Z_inv, errF, errT, N;
        norm_constant::Real = 1)
    norm_constant_val = Float64(norm_constant)
    norm_constant_val <= 0 && throw(ArgumentError("norm_constant must be positive"))

    l2pseudo_sup = _upper_bound(l2pseudo) / (1 - η)
    norm_Z_sup = max(_upper_bound_offset(norm_Z, 1), 0.0)
    norm_Z_inv_sup = max(_upper_bound_offset(norm_Z_inv, 1), 0.0)
    errF_sup = _upper_bound(errF)
    errT_sup = _upper_bound(errT)

    ϵ = max(max(errF_sup, errT_sup), max(norm_Z_sup, norm_Z_inv_sup))
    @info "The ϵ in the Schur theorems $ϵ"

    bound = setrounding(Float64, RoundUp) do
        numerator = 2 * norm_constant_val * (1 + ϵ^2) * l2pseudo_sup
        denominator = 1 - 2 * ϵ * (1 + ϵ^2) * l2pseudo_sup
        return numerator / denominator
    end
    return bound
end

"""
    choose_snapshot_to_load(basepath)

Return the most recent valid snapshot stored at `basepath`.
"""
function choose_snapshot_to_load(basepath::String)
    files = [basepath * "_A.jld2", basepath * "_B.jld2"]
    valid_files = filter(isfile, files)
    if isempty(valid_files)
        return nothing
    end
    sorted = sort(valid_files, by = f -> stat(f).mtime, rev = true)
    try
        snapshot = JLD2.load(sorted[1])
        return snapshot
    catch e
        @warn "Could not load snapshot file $(sorted[1]), possibly corrupted. Trying backup." exception=(e, catch_backtrace())
        if length(sorted) > 1
            try
                snapshot = JLD2.load(sorted[2])
                return snapshot
            catch e
                @warn "Both snapshot files failed to load." exception=(e, catch_backtrace())
                return nothing
            end
        else
            return nothing
        end
    end
end

"""
    save_snapshot!(arcs, cache, log, pending, basepath, toggle)

Persist the current certification state to disk using alternating files.
"""
function save_snapshot!(arcs, cache, log, pending, basepath::String, toggle::Bool)
    filename = basepath * (toggle ? "_A.jld2" : "_B.jld2")
    @info "Saved in $filename, arcs $(length(arcs)), pending $(length(pending))"
    JLD2.@save filename arcs cache log pending
    return !toggle
end

function _identity_ballmatrix(n::Integer)
    return BallArithmetic.BallMatrix(Matrix{ComplexF64}(I, n, n))
end

function _zero_ballmatrix(n::Integer)
    return BallArithmetic.BallMatrix(zeros(ComplexF64, n, n))
end

function _as_ball(value)
    value isa BallArithmetic.Ball && return value
    if value isa Complex
        return BallArithmetic.Ball(ComplexF64(value), 0.0)
    elseif value isa Real
        return BallArithmetic.Ball(Float64(value), 0.0)
    else
        throw(ArgumentError("unsupported value type $(typeof(value)) for ball conversion"))
    end
end

function _upper_bound(value)
    ball = _as_ball(value)
    return setrounding(Float64, RoundUp) do
        abs(ball.c) + ball.r
    end
end

function _upper_bound_offset(value, offset::Real)
    ball = _as_ball(value) - _as_ball(offset)
    return setrounding(Float64, RoundUp) do
        abs(ball.c) + ball.r
    end
end

function _normalize_polynomial(polynomial)
    if polynomial isa AbstractVector
        return collect(polynomial)
    elseif polynomial isa Tuple
        return collect(polynomial)
    elseif polynomial isa Number
        return [polynomial]
    end

    try
        return collect(polynomial)
    catch err
        throw(ArgumentError("unsupported polynomial coefficient container $(typeof(polynomial))")) from err
    end
end

function _is_identity_polynomial(coeffs::AbstractVector)
    length(coeffs) == 2 || return false
    return iszero(coeffs[1]) && coeffs[2] == one(coeffs[2])
end

function _polynomial_matrix(coeffs, M::BallArithmetic.BallMatrix)
    coeffs_vec = _normalize_polynomial(coeffs)
    n = size(M, 1)
    result = _zero_ballmatrix(n)
    identity = _identity_ballmatrix(n)
    for coeff in reverse(coeffs_vec)
        result = result * M
        if !iszero(coeff)
            value = _as_ball(coeff)
            result += value * identity
        end
    end
    return result
end

_polynomial_matrix(coeffs, M::AbstractMatrix) =
    _polynomial_matrix(coeffs, BallArithmetic.BallMatrix(M))

function _load_certification_dependencies(pids)
    Distributed.@sync begin
        for pid in pids
            Distributed.@async Distributed.remotecall_eval(Main, pid, :(begin
                using LinearAlgebra
                using BallArithmetic
                using GKWExperiments
                using GKWExperiments.CertifScripts
            end))
        end
    end
end

function _set_schur_on_workers(pids, matrix)
    Distributed.@sync begin
        for pid in pids
            Distributed.@async Distributed.remotecall_wait(pid, set_schur_matrix!, matrix)
        end
    end
end

function _cleanup_snapshots(basepath)
    for suffix in ("_A.jld2", "_B.jld2")
        filename = basepath * suffix
        isfile(filename) && rm(filename; force = true)
    end
end

"""
    compute_schur_and_error(A; polynomial = nothing)

Compute the Schur decomposition of `A` and certified bounds for the
orthogonality defect, the reconstruction error, and the norms of `Z` and
`Z⁻¹`.  When `polynomial` is provided (as coefficients in ascending order),
additional bounds are computed for `p(A)` and `p(T)`.
"""
function compute_schur_and_error(A::BallArithmetic.BallMatrix; polynomial = nothing)
    S = LinearAlgebra.schur(Complex{Float64}.(A.c))

    bZ = BallArithmetic.BallMatrix(S.Z)
    errF = BallArithmetic.svd_bound_L2_opnorm(bZ' * bZ - I)

    bT = BallArithmetic.BallMatrix(S.T)
    errT = BallArithmetic.svd_bound_L2_opnorm(bZ * bT * bZ' - A)

    sigma_Z = BallArithmetic.svdbox(bZ)
    max_sigma = sigma_Z[1]
    min_sigma = sigma_Z[end]

    norm_Z = setrounding(Float64, RoundUp) do
        return abs(max_sigma.c) + max_sigma.r
    end

    min_sigma_lower = setrounding(Float64, RoundDown) do
        return max(min_sigma.c - min_sigma.r, 0.0)
    end
    min_sigma_lower <= 0 && throw(ArgumentError("Schur factor has non-positive smallest singular value bound"))
    norm_Z_inv = setrounding(Float64, RoundUp) do
        return 1 / min_sigma_lower
    end

    if polynomial === nothing
        return S, errF, errT, norm_Z, norm_Z_inv
    end

    coeffs = _normalize_polynomial(polynomial)
    if _is_identity_polynomial(coeffs)
        return S, errF, errT, norm_Z, norm_Z_inv
    end

    pA = _polynomial_matrix(coeffs, A)
    pT = _polynomial_matrix(coeffs, bT)
    errT_poly = BallArithmetic.svd_bound_L2_opnorm(bZ * pT * bZ' - pA)

    return S, errF, errT_poly, norm_Z, norm_Z_inv
end

"""
    run_certification(A, circle, num_workers; polynomial = nothing, kwargs...)

Run the adaptive certification routine on `circle` by spawning `num_workers`
background workers.  When `polynomial` is supplied (as coefficients in
ascending order), the certification is performed on `p(T)` and the returned
error term corresponds to the reconstruction error of `p(A)`.

Additional keyword arguments:

* `η`: Threshold for the adaptive refinement (default `0.5`).
* `check_interval`: How often progress and snapshots are recorded.
* `snapshot_path`: Base filename used for alternating snapshot files.  When
  omitted, a temporary path is chosen and cleaned up automatically.
* `log_io`: IO target for log messages (default `stdout`).
* `channel_capacity`: Size of the job and result channels (default `1024`).

Returns a named tuple containing the Schur data, certification log, and the
computed resolvent bounds.
"""
function run_certification(A::BallArithmetic.BallMatrix, circle::CertificationCircle,
        num_workers::Integer; polynomial = nothing, η::Real = 0.5,
        check_interval::Integer = 100, snapshot_path::Union{Nothing, AbstractString} = nothing,
        log_io::IO = stdout, channel_capacity::Integer = 1024)

    num_workers < 1 && throw(ArgumentError("num_workers must be positive"))
    channel_capacity < 1 && throw(ArgumentError("channel_capacity must be positive"))
    check_interval < 1 && throw(ArgumentError("check_interval must be positive"))

    η = Float64(η)
    (η <= 0 || η >= 1) && throw(ArgumentError("η must belong to (0, 1)"))

    coeffs = polynomial === nothing ? nothing : _normalize_polynomial(polynomial)
    schur_data = coeffs === nothing ?
        compute_schur_and_error(A) :
        compute_schur_and_error(A; polynomial = coeffs)

    S, errF, errT, norm_Z, norm_Z_inv = schur_data
    bT = BallArithmetic.BallMatrix(S.T)
    schur_matrix = if coeffs === nothing
        bT
    elseif _is_identity_polynomial(coeffs)
        bT
    else
        _polynomial_matrix(coeffs, bT)
    end

    snapshot_base = snapshot_path === nothing ? tempname() : String(snapshot_path)
    mkpath(dirname(snapshot_base))
    cleanup_snapshot = snapshot_path === nothing

    added_workers = Int[]
    certification_log = Any[]
    job_channel = nothing
    result_channel = nothing

    try
        added_workers = addprocs(num_workers)
        isempty(added_workers) && throw(ArgumentError("no worker processes available for certification"))

        _load_certification_dependencies(added_workers)
        _set_schur_on_workers(added_workers, schur_matrix)

        job_channel = RemoteChannel(() -> Channel{Tuple{Int, ComplexF64}}(channel_capacity))
        result_channel = RemoteChannel(() -> Channel{NamedTuple}(channel_capacity))

        configure_certification!(; job_channel = job_channel, result_channel = result_channel,
            certification_log = certification_log, snapshot = snapshot_base, io = log_io)

        foreach(pid -> remote_do(dowork, pid, job_channel, result_channel), added_workers)

        arcs = _initial_arcs(circle)
        cache = Dict{ComplexF64, Any}()
        pending = Dict{Int, Tuple{ComplexF64, ComplexF64}}()

        adaptive_arcs!(arcs, cache, pending, η; check_interval = check_interval)

        isempty(certification_log) && throw(ErrorException("certification produced no samples"))

        min_sigma = minimum(log -> log.lo_val, certification_log)
        l2pseudo = maximum(log -> log.hi_res, certification_log)
        N = size(A, 1)
        resolvent_bound = bound_res_original(l2pseudo, η, norm_Z, norm_Z_inv, errF, errT, N;
            norm_constant = sqrt(N))

        return (; schur = S, schur_matrix, certification_log, minimum_singular_value = min_sigma,
            l2_resolvent_bound = l2pseudo, l1_resolvent_bound = resolvent_bound,
            errF, errT, norm_Z, norm_Z_inv, circle, polynomial = coeffs,
            snapshot_base)
    finally
        if !isempty(added_workers)
            rmprocs(added_workers)
        end
        _job_channel[] = nothing
        _result_channel[] = nothing
        _certification_log[] = nothing
        _snapshot_path[] = nothing
        _log_io[] = stdout
        if cleanup_snapshot && (isfile(snapshot_base * "_A.jld2") || isfile(snapshot_base * "_B.jld2"))
            _cleanup_snapshots(snapshot_base)
        end
    end
end

run_certification(A::AbstractMatrix, circle::CertificationCircle, num_workers::Integer; kwargs...) =
    run_certification(BallArithmetic.BallMatrix(A), circle, num_workers; kwargs...)

end # module
