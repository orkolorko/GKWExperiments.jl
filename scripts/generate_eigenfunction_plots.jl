#!/usr/bin/env julia
# Generate eigenfunction plots from K=512 spectral data.
# Produces eigenfunction batch plots (10 per figure) + spectral approximant + convergence.
#
# Usage:
#   julia --project --startup-file=no scripts/generate_eigenfunction_plots.jl

using CairoMakie, Printf, DelimitedFiles

# ──────────────────────────────────────────────────────────────────────
# Load K=512 eigenvector and spectral coefficient data from text files
# ──────────────────────────────────────────────────────────────────────

const DATA_DIR = joinpath(@__DIR__, "..", "data")
const NUM_EIGS = 50

# Parse eigenvector file: tab-separated, first column = k, then v_1 ... v_50
println("Loading eigenvector data...")
evec_lines = readlines(joinpath(DATA_DIR, "eigenvectors_K512_P1024.txt"))
evec_data_lines = filter(l -> !startswith(l, "#") && !startswith(l, "k"), evec_lines)
K_plus_1 = length(evec_data_lines)
eigenvectors = zeros(K_plus_1, NUM_EIGS)
for (i, line) in enumerate(evec_data_lines)
    parts = split(line, '\t')
    for j in 1:NUM_EIGS
        eigenvectors[i, j] = parse(Float64, parts[j + 1])
    end
end
println("  Loaded $(K_plus_1) coefficients × $NUM_EIGS eigenvectors")

# Parse spectral coefficients file
println("Loading spectral coefficient data...")
coeff_lines = readlines(joinpath(DATA_DIR, "spectral_coefficients_K512_P1024.txt"))
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
println("  Loaded $NUM_EIGS eigenvalues and ℓ_j(1) coefficients")

# ──────────────────────────────────────────────────────────────────────
# Evaluation on [0.01, 0.99]
# ──────────────────────────────────────────────────────────────────────
const x_pts = range(0.01, 0.99, length=500)

"""Evaluate polynomial in the shifted monomial basis {(x-1)^k}."""
function eval_poly(coeffs::AbstractVector, x::Real)
    w = x - 1.0
    result = 0.0
    w_power = 1.0
    for c in coeffs
        result += c * w_power
        w_power *= w
    end
    return result
end

"""Evaluate eigenfunction j at all x_pts."""
function eval_eigenfunction(j::Int)
    coeffs = @view eigenvectors[:, j]
    return [eval_poly(coeffs, x) for x in x_pts]
end

"""Evaluate spectral approximant S_N(n, x) = Σ λ_j^n ℓ_j(1) v_j(x)."""
function eval_spectral_approximant(n::Int; N=NUM_EIGS)
    vals = zeros(length(x_pts))
    for j in 1:N
        vj = eval_eigenfunction(j)
        coeff = eigenvalues[j]^n * ell_center[j]
        vals .+= coeff .* vj
    end
    return vals
end

# ──────────────────────────────────────────────────────────────────────
# Eigenfunction batch plots: 5 figures × 10 eigenfunctions (2×5 grid)
# ──────────────────────────────────────────────────────────────────────

batch_ranges = [1:10, 11:20, 21:30, 31:40, 41:50]

for batch in batch_ranges
    j_start, j_end = first(batch), last(batch)
    fname = "eigenfunctions_$(j_start)_$(j_end).pdf"
    println("Generating $fname …")

    fig = Figure(size=(1200, 500), fontsize=10)
    for (i, idx) in enumerate(batch)
        row, col = divrem(i - 1, 5) .+ (1, 1)
        λ = eigenvalues[idx]
        ℓ = ell_center[idx]

        # Adaptive title formatting
        if abs(λ) >= 0.01
            title = @sprintf("v_%d  (λ=%.4f, ℓ=%+.3f)", idx, λ, ℓ)
        elseif abs(λ) >= 1e-6
            title = @sprintf("v_%d  (λ=%.2e, ℓ=%+.2e)", idx, λ, ℓ)
        else
            title = @sprintf("v_%d  (λ=%.1e, ℓ=%+.1e)", idx, λ, ℓ)
        end

        ax = Axis(fig[row, col]; xlabel="x", ylabel="v_$(idx)(x)", title,
                  titlesize=9)
        y = eval_eigenfunction(idx)
        lines!(ax, collect(x_pts), y; color=:steelblue, linewidth=1.2)
        hlines!(ax, [0.0]; color=:gray70, linestyle=:dash, linewidth=0.5)
    end
    save(joinpath(DATA_DIR, fname), fig)
    println("  → data/$fname")
end

# ──────────────────────────────────────────────────────────────────────
# Overlay plots: 10-by-10 batches with zoomed x-range + Markov partition
#   Each batch zooms into the region where the eigenfunctions have
#   fine structure, with the corresponding Markov cylinders shown.
# ──────────────────────────────────────────────────────────────────────

overlay_colors = [:steelblue, :firebrick, :seagreen, :darkorange, :purple4,
                  :goldenrod, :deeppink, :teal, :sienna, :slateblue]

# (batch, x_range, Markov partition n-range)
overlay_specs = [
    (1:10,   (0.01,  0.99),  2:10),    # full interval, cylinders [1/2,1]...[1/10,1/9]
    (11:20,  (0.01,  0.5),   10:20),   # zoom to [0, 1/2], cylinders n=10..20
    (21:30,  (0.005, 0.25),  20:30),   # zoom to [0, 1/4], cylinders n=20..30
    (31:40,  (0.003, 0.125), 30:40),   # zoom to [0, 1/8], cylinders n=30..40
    (41:50,  (0.002, 0.1),   40:50),   # zoom to [0, 1/10], cylinders n=40..50
]

for (batch, (x_lo, x_hi), markov_range) in overlay_specs
    j_start, j_end = first(batch), last(batch)
    fname = "eigenfunctions_$(j_start)_$(j_end)_overlay.pdf"
    println("Generating $fname …")

    x_ov = range(x_lo, x_hi, length=500)

    fig_ov = Figure(size=(900, 500), fontsize=12)
    ax_ov = Axis(fig_ov[1, 1];
        xlabel="x", ylabel="v_j(x)",
        title="Eigenfunctions v_$(j_start) – v_$(j_end)  (x ∈ [$(@sprintf("%.3g", x_lo)), $(@sprintf("%.3g", x_hi))])")

    for (i, idx) in enumerate(batch)
        y = [eval_poly(@view(eigenvectors[:, idx]), x) for x in x_ov]
        lines!(ax_ov, collect(x_ov), y;
            color=overlay_colors[i], linewidth=1.3,
            label=@sprintf("v_%d", idx))
    end
    hlines!(ax_ov, [0.0]; color=:gray70, linestyle=:dash, linewidth=0.5)

    # Markov partition: x = 1/n for the relevant cylinders
    for n in markov_range
        xn = 1.0 / n
        if x_lo < xn < x_hi
            vlines!(ax_ov, [xn]; color=:gray30, linestyle=:dot, linewidth=0.7)
        end
    end

    axislegend(ax_ov; position=:lt, nbanks=2, framevisible=false, labelsize=9)
    save(joinpath(DATA_DIR, fname), fig_ov)
    println("  → data/$fname")
end

# ──────────────────────────────────────────────────────────────────────
# Overlay plot: all 50 eigenfunctions on same axes
# ──────────────────────────────────────────────────────────────────────
println("Generating eigenfunctions_all_overlay.pdf …")

fig_ov = Figure(size=(900, 500), fontsize=12)
ax_ov = Axis(fig_ov[1, 1];
    xlabel="x", ylabel="v_j(x)",
    title="Eigenfunctions v_1 – v_50 with Gauss map Markov partition")

colors_ov = Makie.wong_colors()
for idx in 1:NUM_EIGS
    y = eval_eigenfunction(idx)
    lines!(ax_ov, collect(x_pts), y;
        color=colors_ov[mod1(idx, length(colors_ov))],
        linewidth=(idx <= 6 ? 1.5 : 0.7))
end
hlines!(ax_ov, [0.0]; color=:gray70, linestyle=:dash, linewidth=0.5)

# Markov partition
for n in 2:10
    vlines!(ax_ov, [1.0 / n]; color=:gray30, linestyle=:dot, linewidth=0.7)
end

save(joinpath(DATA_DIR, "eigenfunctions_all_overlay.pdf"), fig_ov)
println("  → data/eigenfunctions_all_overlay.pdf")

# ──────────────────────────────────────────────────────────────────────
# Spectral approximant  S_50(n, x)  for n = 0, 1, 2, 5, 10
# ──────────────────────────────────────────────────────────────────────
println("Generating spectral_approximant.pdf …")

fig3 = Figure(size=(700, 450), fontsize=12)
ax3 = Axis(fig3[1, 1];
    xlabel="x", ylabel="S₅₀(n, x)",
    title="Spectral approximant  L^n 1  via 50 eigenvalues")

colors3 = [:gray50, :steelblue, :firebrick, :darkorange, :purple4]
n_vals = [0, 1, 2, 5, 10]

for (i, n) in enumerate(n_vals)
    y = eval_spectral_approximant(n)
    lines!(ax3, collect(x_pts), y;
        color=colors3[i], linewidth=1.5, label="n = $n")
end
axislegend(ax3; position=:rt)
save(joinpath(DATA_DIR, "spectral_approximant.pdf"), fig3)
println("  → data/spectral_approximant.pdf")

# ──────────────────────────────────────────────────────────────────────
# Convergence to invariant density
# ──────────────────────────────────────────────────────────────────────
println("Generating spectral_convergence.pdf …")

fig4 = Figure(size=(700, 450), fontsize=12)
ax4 = Axis(fig4[1, 1];
    xlabel="x", ylabel="S₅₀(n, x) / λ₁ⁿ",
    title="Convergence of  S₅₀(n, x) / λ₁ⁿ  to  ℓ₁(1) v₁(x)")

y_limit = ell_center[1] .* eval_eigenfunction(1)

n_conv = [0, 1, 2, 5, 10, 20]
colors4 = [:gray60, :cornflowerblue, :salmon, :goldenrod, :mediumpurple, :seagreen]

for (i, n) in enumerate(n_conv)
    y = eval_spectral_approximant(n) ./ eigenvalues[1]^n
    lines!(ax4, collect(x_pts), y;
        color=colors4[i], linewidth=1.2, label="n = $n")
end

lines!(ax4, collect(x_pts), y_limit;
    color=:black, linewidth=2.5, linestyle=:dash, label="ℓ₁(1) v₁(x)")

axislegend(ax4; position=:rt)
save(joinpath(DATA_DIR, "spectral_convergence.pdf"), fig4)
println("  → data/spectral_convergence.pdf")

println("\nAll plots generated successfully.")
