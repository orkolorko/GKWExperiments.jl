#!/usr/bin/env julia
# Plot the integrated spectral expansion with eigenfunctions cumulatively removed:
#
#   F_n^{(j₀)}(x) = ∫₀ˣ Σ_{j≥j₀} λ_j^n ℓ_j(1) v_j(t) dt
#
# Five panels for j₀ = 2, 3, 4, 5, 6, each showing n = 1, 2, 5.
# This progressively reveals fine structure from higher eigenmodes.
#
# L^∞ error bound:  |F_n(x) - true| ≤ √π · ρ^{n+1} · C
#
# Usage:
#   julia --project --startup-file=no scripts/plot_integrated_spectral.jl

using CairoMakie, Printf

const DATA_DIR = joinpath(@__DIR__, "..", "data")
const NUM_EIGS = 50

# ──────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────

println("Loading eigenvector data...")
evec_lines = readlines(joinpath(DATA_DIR, "eigenvectors_K1024_P2048.txt"))
evec_data_lines = filter(l -> !startswith(l, "#") && !startswith(l, "k"), evec_lines)
K_plus_1 = length(evec_data_lines)
eigenvectors = zeros(K_plus_1, NUM_EIGS)
for (i, line) in enumerate(evec_data_lines)
    parts = split(line, '\t')
    for j in 1:NUM_EIGS
        eigenvectors[i, j] = parse(Float64, parts[j + 1])
    end
end
println("  $K_plus_1 coefficients × $NUM_EIGS eigenvectors")

println("Loading spectral coefficients...")
coeff_lines = readlines(joinpath(DATA_DIR, "spectral_coefficients_K1024_P2048.txt"))
eigenvalues = zeros(NUM_EIGS)
ell_center = zeros(NUM_EIGS)
for line in coeff_lines
    startswith(line, "#") && continue
    parts = split(line, '\t')
    length(parts) >= 4 || continue
    j = parse(Int, parts[1])
    j > NUM_EIGS && continue
    eigenvalues[j] = parse(Float64, parts[2])
    ell_center[j] = parse(Float64, parts[3])
end

println("Loading tail bound...")
tail_lines = readlines(joinpath(DATA_DIR, "tail_bound_N50.txt"))
tail_data = Dict{String,Float64}()
for line in tail_lines
    startswith(line, "#") && continue
    isempty(strip(line)) && continue
    parts = split(line, '\t')
    length(parts) >= 2 || continue
    tail_data[parts[1]] = parse(Float64, parts[2])
end
rho_tail = tail_data["rho_tail"]
prefactor_C = tail_data["prefactor"]

@printf("  λ₁..λ₆ = %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n",
    eigenvalues[1], eigenvalues[2], eigenvalues[3],
    eigenvalues[4], eigenvalues[5], eigenvalues[6])
@printf("  ρ = %.3e, C = %.3e\n", rho_tail, prefactor_C)

# ──────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────

function eval_integrated_poly(coeffs::AbstractVector, x::Real)
    w = x - 1.0
    result = 0.0
    w_pow = w           # (x-1)^{k+1}
    neg1_pow = -1.0     # (-1)^{k+1}
    for (k1, c) in enumerate(coeffs)   # k1 = k+1
        result += c * (w_pow - neg1_pow) / k1
        w_pow *= w
        neg1_pow = -neg1_pow
    end
    return result
end

function eval_Fn(x_pts, n::Int; j0=2, N=NUM_EIGS)
    vals = zeros(length(x_pts))
    for j in j0:N
        w = eigenvalues[j]^n * ell_center[j]
        cj = @view eigenvectors[:, j]
        for (i, x) in enumerate(x_pts)
            vals[i] += w * eval_integrated_poly(cj, x)
        end
    end
    return vals
end

linf_bound(n) = sqrt(π) * rho_tail^(n + 1) * prefactor_C

# ──────────────────────────────────────────────────────────────────────
# 5-panel figure: j₀ = 2, 3, 4, 5, 6
# ──────────────────────────────────────────────────────────────────────

x_pts = collect(range(0.005, 0.995, length=1000))
n_vals = [1, 2, 5]
colors = [:steelblue, :firebrick, :seagreen]
j0_vals = [2, 3, 4, 5, 6]

fig = Figure(size=(900, 1100), fontsize=12)

for (row, j0) in enumerate(j0_vals)
    removed = join(string.(1:j0-1), ", ")
    λ_dom = eigenvalues[j0]
    panel_title = @sprintf("Removed: v₁–v_%d  (dominant: λ_%d = %.4f)", j0 - 1, j0, λ_dom)

    ax = Axis(fig[row, 1];
        ylabel = L"F_n^{(%$j0)}(x)",
        title = panel_title,
        titlesize = 11)

    # Only show x-label on bottom panel
    if row == length(j0_vals)
        ax.xlabel = L"x"
    end

    for (i, n) in enumerate(n_vals)
        y = eval_Fn(x_pts, n; j0=j0)
        amp = maximum(abs, y)
        b = linf_bound(n)
        lb = floor(Int, log10(b))
        lbl = @sprintf("n = %d  (amp %.1e, err ≤ 10^%d)", n, amp, lb)
        lines!(ax, x_pts, y; color=colors[i], linewidth=1.5, label=lbl)
    end

    hlines!(ax, [0.0]; color=:gray70, linestyle=:dash, linewidth=0.5)

    # Markov partition lines
    for k in 2:8
        xk = 1.0 / k
        vlines!(ax, [xk]; color=(:gray40, 0.3), linestyle=:dot, linewidth=0.5)
    end

    axislegend(ax; position=:rt, framevisible=true, labelsize=9, nbanks=1)
end

fname = joinpath(DATA_DIR, "integrated_spectral_progressive.pdf")
save(fname, fig)
println("Saved → $fname")

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────

println()
println("Signal amplitudes (max|F_n|) by panel:")
println("-" ^ 60)
@printf("  %5s  %12s  %12s  %12s\n", "j₀", "n=1", "n=2", "n=5")
println("-" ^ 60)
for j0 in j0_vals
    amps = [maximum(abs, eval_Fn(x_pts, n; j0=j0)) for n in n_vals]
    @printf("  j≥%d   %12.3e  %12.3e  %12.3e\n", j0, amps...)
end
println("-" ^ 60)
println()
@printf("L^∞ error (all panels): n=1: ≤%.1e, n=2: ≤%.1e, n=5: ≤%.1e\n",
    linf_bound(1), linf_bound(2), linf_bound(5))
