#!/usr/bin/env julia
# Generate eigenfunction and spectral approximant plots from cached data.
# Produces 5 PDF figures in data/:
#   eigenfunctions_1_6.pdf          — leading 6 eigenfunctions
#   eigenfunctions_7_12.pdf         — higher eigenfunctions 7–12
#   eigenfunctions_7_12_overlay.pdf — v_7–v_12 overlaid on same axes
#   spectral_approximant.pdf        — L^n 1 for several n
#   spectral_convergence.pdf        — convergence to invariant density

using Serialization, CairoMakie, Printf

# ──────────────────────────────────────────────────────────────────────
# Load cached spectral data
# ──────────────────────────────────────────────────────────────────────
data = deserialize("data/spectral_results_K256.jls")

eigenvalues  = data[:eigenvalues]       # 20 Float64
eigenvectors = data[:eigenvectors]      # 257×20 Float64 (shifted monomial coeffs)
ell_center   = data[:ell_center]        # 20 Float64

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
function eval_spectral_approximant(n::Int)
    N = length(eigenvalues)
    vals = zeros(length(x_pts))
    for j in 1:N
        vj = eval_eigenfunction(j)
        coeff = eigenvalues[j]^n * ell_center[j]
        vals .+= coeff .* vj
    end
    return vals
end

# ──────────────────────────────────────────────────────────────────────
# Figure 1: Leading eigenfunctions v_1 … v_6  (2×3 grid)
# ──────────────────────────────────────────────────────────────────────
println("Generating eigenfunctions_1_6.pdf …")

fig1 = Figure(size=(900, 550), fontsize=11)
for idx in 1:6
    row, col = divrem(idx - 1, 3) .+ (1, 1)
    λ = eigenvalues[idx]
    ℓ = ell_center[idx]
    title = @sprintf("v_%d   (λ = %.4f, ℓ = %+.3f)", idx, λ, ℓ)
    ax = Axis(fig1[row, col]; xlabel="x", ylabel="v_$(idx)(x)", title)
    y = eval_eigenfunction(idx)
    lines!(ax, collect(x_pts), y; color=:steelblue, linewidth=1.5)
    hlines!(ax, [0.0]; color=:gray70, linestyle=:dash, linewidth=0.5)
end
save("data/eigenfunctions_1_6.pdf", fig1)
println("  → data/eigenfunctions_1_6.pdf")

# ──────────────────────────────────────────────────────────────────────
# Figure 2: Higher eigenfunctions v_7 … v_12  (2×3 grid)
# ──────────────────────────────────────────────────────────────────────
println("Generating eigenfunctions_7_12.pdf …")

fig2 = Figure(size=(900, 550), fontsize=11)
for idx in 7:12
    row, col = divrem(idx - 7, 3) .+ (1, 1)
    λ = eigenvalues[idx]
    ℓ = ell_center[idx]
    title = @sprintf("v_%d   (λ = %.6f, ℓ = %+.4f)", idx, λ, ℓ)
    ax = Axis(fig2[row, col]; xlabel="x", ylabel="v_$(idx)(x)", title)
    y = eval_eigenfunction(idx)
    lines!(ax, collect(x_pts), y; color=:steelblue, linewidth=1.5)
    hlines!(ax, [0.0]; color=:gray70, linestyle=:dash, linewidth=0.5)
end
save("data/eigenfunctions_7_12.pdf", fig2)
println("  → data/eigenfunctions_7_12.pdf")

# ──────────────────────────────────────────────────────────────────────
# Figure 2b: v_4 … v_20 overlaid on same axes with Markov partition
# ──────────────────────────────────────────────────────────────────────
println("Generating eigenfunctions_4_20_overlay.pdf …")

fig2b = Figure(size=(900, 500), fontsize=12)
ax2b = Axis(fig2b[1, 1];
    xlabel="x", ylabel="v_j(x)",
    title="Eigenfunctions v_4 – v_20 with Gauss map Markov partition")

# 17 functions need distinguishable colors
colors2b = Makie.wong_colors()
n_funcs = 17
for (i, idx) in enumerate(4:20)
    y = eval_eigenfunction(idx)
    label = @sprintf("v_%d", idx)
    lines!(ax2b, collect(x_pts), y;
        color=colors2b[mod1(i, length(colors2b))],
        linewidth=(i <= 7 ? 1.5 : 0.9),
        label=label)
end
hlines!(ax2b, [0.0]; color=:gray70, linestyle=:dash, linewidth=0.5)

# Markov partition: vertical lines at x = 1/n for n = 2, 3, 4, …
for n in 2:10
    vlines!(ax2b, [1.0 / n]; color=:gray30, linestyle=:dot, linewidth=0.7)
end

axislegend(ax2b; position=:lt, nbanks=3, framevisible=false, labelsize=9)
save("data/eigenfunctions_4_20_overlay.pdf", fig2b)
println("  → data/eigenfunctions_4_20_overlay.pdf")

# ──────────────────────────────────────────────────────────────────────
# Figure 3: Spectral approximant  S_20(n, x)  for n = 0, 1, 2, 5, 10
# ──────────────────────────────────────────────────────────────────────
println("Generating spectral_approximant.pdf …")

fig3 = Figure(size=(700, 450), fontsize=12)
ax3 = Axis(fig3[1, 1];
    xlabel="x", ylabel="S₂₀(n, x)",
    title="Spectral approximant  L^n 1  via 20 eigenvalues")

colors3 = [:gray50, :steelblue, :firebrick, :darkorange, :purple4]
n_vals = [0, 1, 2, 5, 10]

for (i, n) in enumerate(n_vals)
    y = eval_spectral_approximant(n)
    lines!(ax3, collect(x_pts), y;
        color=colors3[i], linewidth=1.5, label="n = $n")
end
axislegend(ax3; position=:rt)
save("data/spectral_approximant.pdf", fig3)
println("  → data/spectral_approximant.pdf")

# ──────────────────────────────────────────────────────────────────────
# Figure 4: Convergence to invariant density
#   Plot S_20(n,x) / λ₁^n for several n → ℓ₁(1) v₁(x)
# ──────────────────────────────────────────────────────────────────────
println("Generating spectral_convergence.pdf …")

fig4 = Figure(size=(700, 450), fontsize=12)
ax4 = Axis(fig4[1, 1];
    xlabel="x", ylabel="S₂₀(n, x) / λ₁ⁿ",
    title="Convergence of  S₂₀(n, x) / λ₁ⁿ  to  ℓ₁(1) v₁(x)")

# The limit: ℓ₁(1) v₁(x)
y_limit = ell_center[1] .* eval_eigenfunction(1)

n_conv = [0, 1, 2, 5, 10, 20]
colors4 = [:gray60, :cornflowerblue, :salmon, :goldenrod, :mediumpurple, :seagreen]

for (i, n) in enumerate(n_conv)
    y = eval_spectral_approximant(n) ./ eigenvalues[1]^n
    lines!(ax4, collect(x_pts), y;
        color=colors4[i], linewidth=1.2, label="n = $n")
end

# Plot the limit curve
lines!(ax4, collect(x_pts), y_limit;
    color=:black, linewidth=2.5, linestyle=:dash, label="ℓ₁(1) v₁(x)")

axislegend(ax4; position=:rt)
save("data/spectral_convergence.pdf", fig4)
println("  → data/spectral_convergence.pdf")

println("\nAll plots generated successfully.")
