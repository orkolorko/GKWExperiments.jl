using ArbNumerics
using GKWExperiments  # assumes gkw_matrix_direct + build_Ls_matrix_arb are available

setprecision(256)

# --- helper: check if two ArbComplex balls overlap
@inline function arb_overlaps(a::ArbComplex{P}, b::ArbComplex{P}) where {P}
    re_a, im_a = real(a), imag(a)
    re_b, im_b = real(b), imag(b)
    # real components overlap?
    overlap_re = abs(midpoint(re_a) - midpoint(re_b)) ≤ (radius(re_a) + radius(re_b))
    # imaginary components overlap?
    overlap_im = abs(midpoint(im_a) - midpoint(im_b)) ≤ (radius(im_a) + radius(im_b))
    return overlap_re && overlap_im
end

# --- test parameters
s  = ArbComplex(0.75, 0.3)
K  = 10
N  = 16384

@testset "GKW operator: direct vs DFT consistency" begin
    @info "Building direct analytic matrix..."
    M_direct = gkw_matrix_direct(s; K=K)
    @info "Building FFT-based matrix..."
    M_fft    = build_Ls_matrix_arb(s; K=K, N=N)

    @test size(M_direct) == size(M_fft)

    maxdiff = 0.0
    overlap_fail = 0
    for i in 1:K+1, j in 1:K+1
        Δ = M_direct[i,j] - M_fft[i,j]
        midΔ = sqrt(abs2(Float64(real(mid(Δ)))) + abs2(Float64(imag(mid(Δ)))))
        maxdiff = max(maxdiff, midΔ)

        if !arb_overlaps(M_direct[i,j], M_fft[i,j])
            overlap_fail += 1
        end
    end

    @info "max midpoint difference ≈ $maxdiff"
    @info "entries with non-overlapping enclosures: $overlap_fail / $((K+1)^2)"

    @test maxdiff < 1e-6
    @test overlap_fail == 0
end
