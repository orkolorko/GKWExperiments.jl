# Quick test: verify gkw_matrix_direct_fast matches gkw_matrix_direct
using GKWExperiments
using ArbNumerics
using Printf

setprecision(ArbFloat, 512)

# Use ArbComplex(1.0, 0.0) without explicit type param — matches global precision
s = ArbComplex(1.0, 0.0)
const P = precision(s)   # actual precision including guard bits
@info "Working precision: $P bits"

for K in [64, 128]
    println("Testing K=$K ...")

    t0 = time()
    M_orig = gkw_matrix_direct(s; K=K)
    dt_orig = time() - t0

    t0 = time()
    M_fast = gkw_matrix_direct_fast(s; K=K, threaded=false)
    dt_fast = time() - t0

    t0 = time()
    M_fast_t = gkw_matrix_direct_fast(s; K=K, threaded=true)
    dt_fast_t = time() - t0

    # Compare entry-by-entry: midpoints should agree, radii may differ slightly
    max_mid_diff = 0.0
    max_rad_orig = 0.0
    max_rad_fast = 0.0
    for i in 1:K+1, j in 1:K+1
        mid_o = Float64(ArbNumerics.midpoint(real(M_orig[i,j])))
        mid_f = Float64(ArbNumerics.midpoint(real(M_fast[i,j])))
        mid_ft = Float64(ArbNumerics.midpoint(real(M_fast_t[i,j])))
        rad_o = Float64(ArbNumerics.radius(real(M_orig[i,j])))
        rad_f = Float64(ArbNumerics.radius(real(M_fast[i,j])))
        d = abs(mid_o - mid_f)
        if d > max_mid_diff
            max_mid_diff = d
        end
        if rad_o > max_rad_orig
            max_rad_orig = rad_o
        end
        if rad_f > max_rad_fast
            max_rad_fast = rad_f
        end
        # Check threaded matches serial
        dt = abs(mid_f - mid_ft)
        if dt > 1e-70
            @warn "Threaded mismatch at ($i,$j): $dt"
        end
    end

    @printf("  K=%3d  orig: %.2fs  fast: %.2fs  fast+threads: %.2fs  speedup: %.1f×\n",
            K, dt_orig, dt_fast, dt_fast_t, dt_orig / max(dt_fast, 1e-6))
    @printf("         max |mid_orig - mid_fast| = %.2e\n", max_mid_diff)
    @printf("         max rad(orig) = %.2e,  max rad(fast) = %.2e\n", max_rad_orig, max_rad_fast)
    println()
end
