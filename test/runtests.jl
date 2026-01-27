using GKWExperiments
using Test

@testset "GKWExperiments.jl" begin
    include("TestZeta.jl")
    include("TestGKWConsistency.jl")
    include("TestPolynomials.jl")
    include("TestCertifScripts.jl")
    include("TestEigenspaceCertification.jl")
    include("TestInfiniteDimensionalLift.jl")
end
