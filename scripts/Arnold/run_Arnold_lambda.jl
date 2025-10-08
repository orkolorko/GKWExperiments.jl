import Pkg;
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Logging, Dates, Distributed, LinearAlgebra, SlurmClusterManager, DataFrames, JLD2, CSV

datetime = Dates.now()

if haskey(ENV, "SLURM_NTASKS")
    procs = addprocs(SlurmManager())
    location = "slurm"
else
    procs = addprocs(4)
    location = "local"
end

nprocs = length(procs)

@everywhere import Pkg
@everywhere Pkg.activate(@__DIR__)
@everywhere Pkg.instantiate()

@everywhere using LinearAlgebra, BallArithmetic, JLD2
@everywhere D = JLD2.load("../../ArnoldMatrixSchur128.jld2")

λ = D["S"].values[1]
R = 0.01
N = 128

const filename = "./logs/log_$(location)_Arnold_$(λ)_$(R)_$datetime"

const snapshot = "./logs/new_snapshot_Arnold_$(λ)_$(R)"

include("../script_functions_2.jl")

# Choose the most recent working snapshot
load_snapshot = choose_snapshot_to_load(snapshot)

if load_snapshot !== nothing
    arcs = load_snapshot["arcs"]
    cache = load_snapshot["cache"]
    certification_log = load_snapshot["log"]
    pending = load_snapshot["pending"]
else
    @info "No previous snapshot found. Starting fresh."
    # Initialize fresh variables

    N = 128
    θs = range(0, 2π, length = N + 1)[1:(end - 1)]
    zs = λ .+ R .* exp.(1im .* θs)
    arcs = [(zs[i], zs[mod1(i + 1, N)]) for i in 1:N]
    cache = Dict{ComplexF64, Any}()
    certification_log = DataFrame(
        i = Int[],
        val = Ball{Float64, Float64}[],
        lo_val = Float64[],
        res = Ball{Float64, Float64}[],
        hi_res = Float64[],
        second_val = Ball{Float64, Float64}[],
        z = ComplexF64[],
        t = Float64[],
        id = Int[]
    )
    pending = Dict{Int, Tuple{ComplexF64, ComplexF64}}()
end

io = open(filename*".txt", "w+")
logger = SimpleLogger(io)
global_logger(logger)
@info "Added $nprocs processes"
Sys.cpu_summary(io)

@info "Schur decomposition errors"
errF = D["errF"]
errT = D["errT"]
norm_Z = D["norm_Z"]
norm_Z_inv = D["norm_Z_inv"]
@info "E_M", errF
@info "E_T", errT
@info "norm_Z", norm_Z
@info "norm_Z_inv", norm_Z_inv
N = size(D["P"])[1]

@everywhere const T_global = BallMatrix(D["S"].T)

const job_channel = RemoteChannel(()->Channel{Tuple{Int, ComplexF64}}(1024))
const result_channel = RemoteChannel(()->Channel{NamedTuple}(1024))



foreach(
        pid -> remote_do(dowork, pid, job_channel, result_channel),
        workers()
    )

@info "Certifying a circle of radius $R around $λ"

η = 0.5
@info "Size of balls < σ_min*$η"

id_counter = maximum(collect(keys(pending)); init=0) + 1
@info "Pending from snapshot", length(pending)

#@info arcs
adaptive_arcs!(arcs, cache, pending, η)

function lo(x::Ball)
    lo = setrounding(Float64, RoundUp) do
            return x.c - x.r
    end
    return lo
end

JLD2.@save "./logs/certification_log_$(location)_Arnold_$(λ)_$(R)_$datetime.jld2" certification_log
CSV.write("./logs/certification_log_$(location)_Arnold_$(λ)_$(R)_$datetime.csv", certification_log)

@info "The smallest singular value along the arc is bounded below by $(minimum(certification_log.lo_val))"
l2pseudo = maximum(certification_log.hi_res)
@info "The resolvent norm for the Schur matrix in l2 norm is bounded above by $(l2pseudo)"

bound = bound_res_original(l2pseudo, η, norm_Z, norm_Z_inv, errF, errT, N)

@info "The l1 resolvent norm for the original discretized operator is bounded above by $bound"


rmprocs(procs)