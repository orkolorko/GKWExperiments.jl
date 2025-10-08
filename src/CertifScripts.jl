module CertifScripts

using Distributed
using LinearAlgebra
using BallArithmetic
using JLD2

export dowork, adaptive_arcs!, bound_res_original, choose_snapshot_to_load,
       save_snapshot!, configure_certification!, set_schur_matrix!,
       compute_schur_and_error

const _schur_matrix = Ref{Union{Nothing, BallArithmetic.BallMatrix}}(nothing)
const _job_channel = Ref{Union{Nothing, RemoteChannel}}(nothing)
const _result_channel = Ref{Union{Nothing, RemoteChannel}}(nothing)
const _certification_log = Ref{Any}(nothing)
const _snapshot_path = Ref{Union{Nothing, AbstractString}}(nothing)
const _log_io = Ref{IO}(stdout)

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
    bound_res_original(l2pseudo, η, norm_Z, norm_Z_inv, errF, errT, N)

Return an upper bound on the ℓ₁ resolvent norm of the original matrix
given the bounds obtained from the Schur form.
"""
function bound_res_original(l2pseudo, η, norm_Z, norm_Z_inv, errF, errT, N)
    bound = setrounding(Float64, RoundUp) do
        l2pseudo = l2pseudo * 1 / (1 - η)
        norm_Z_sup = (norm_Z - 1).c + (norm_Z - 1).r
        norm_Z_inv_sup = (norm_Z_inv - 1).c + (norm_Z_inv - 1).r

        ϵ = max(max(errF, errT), max(norm_Z_sup, norm_Z_inv_sup))
        @info "The ϵ in the Schur theorems $ϵ"
        return (2 * (1 + ϵ^2) * l2pseudo * sqrt(N)) / (1 - 2 * ϵ * (1 + ϵ^2) * l2pseudo)
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

function _polynomial_matrix(coeffs::AbstractVector, M::BallArithmetic.BallMatrix)
    n = size(M, 1)
    result = _zero_ballmatrix(n)
    identity = _identity_ballmatrix(n)
    for coeff in reverse(coeffs)
        result = result * M
        if !iszero(coeff)
            value = coeff isa BallArithmetic.Ball ? coeff : BallArithmetic.Ball(ComplexF64(coeff), 0.0)
            result += value * identity
        end
    end
    return result
end

_polynomial_matrix(coeffs::AbstractVector, M::AbstractMatrix) =
    _polynomial_matrix(coeffs, BallArithmetic.BallMatrix(M))

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

    norm_Z = sigma_Z[1]
    norm_Z_inv = 1.0 / sigma_Z[end]

    if polynomial === nothing
        return S, errF, errT, norm_Z, norm_Z_inv
    end

    coeffs = collect(polynomial)
    pA = _polynomial_matrix(coeffs, A)
    pT = _polynomial_matrix(coeffs, bT)
    errT_poly = BallArithmetic.svd_bound_L2_opnorm(bZ * pT * bZ' - pA)

    return S, errF, errT_poly, norm_Z, norm_Z_inv
end

end # module
