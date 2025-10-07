using GKWExperiments
using Test

@testset "GKWExperiments.jl" begin
    include("TestZeta.jl")
    include("TestGKWConsistency.jl")
end
