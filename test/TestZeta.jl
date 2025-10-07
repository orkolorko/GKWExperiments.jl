setprecision(256)

@testset "ArbZeta: basic correctness" begin
    # --- real known values ---
    ζ2 = dirichlet_zeta(ArbComplex(2))
    ζ4 = dirichlet_zeta(ArbComplex(4))
    ζ0 = dirichlet_zeta(ArbComplex(0))
    ζm1 = dirichlet_zeta(ArbComplex(-1))

    @test isapprox(Float64(real(ζ2)), π^2 / 6; rtol=1e-15)
    @test isapprox(Float64(real(ζ4)), π^4 / 90; rtol=1e-15)
    @test isapprox(Float64(real(ζ0)), -0.5; atol=1e-15)
    @test isapprox(Float64(real(ζm1)), -1/12; atol=1e-15)

    # --- consistency: ζ(s) = ζ(s,1) ---
    for x in (0.5, 2.0, 3.5)
        s = ArbComplex(x)
        z1 = dirichlet_zeta(s)
        z2 = hurwitz_zeta(s, ArbComplex(1))
        @test abs(Float64(real(z1 - z2))) < 1e-30
        @test abs(Float64(imag(z1 - z2))) < 1e-30
    end

    # --- consistency on complex argument ---
    s_complex = ArbComplex(0.5, 14.134725141734693790)
    z_dir = dirichlet_zeta(s_complex)
    z_hur = hurwitz_zeta(s_complex, ArbComplex(1))
    diff = abs(Float64(real(z_dir - z_hur))) + abs(Float64(imag(z_dir - z_hur)))
    @test diff < 1e-25

    println("\nSample values:")
    println("ζ(2)   ≈ ", ζ2)
    println("ζ(4)   ≈ ", ζ4)
    println("ζ(0)   ≈ ", ζ0)
    println("ζ(-1)  ≈ ", ζm1)
    println("ζ(½+i14.13) ≈ ", z_dir)
end
