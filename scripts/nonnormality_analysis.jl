#!/usr/bin/env julia
# Analyze non-normality of the GKW transfer operator:
#   1. Eigenfunction zero crossings and local extrema vs Markov partition
#   2. Ordered Schur form T_{ij} structure (how modes couple)
#
# Usage:
#   julia --project --startup-file=no scripts/nonnormality_analysis.jl

using CairoMakie, Printf, LinearAlgebra, Serialization

const DATA_DIR = joinpath(@__DIR__, "..", "data")
const NUM_EIGS = 50

# ══════════════════════════════════════════════════════════════════════
# Part 1: Eigenfunction structure vs Markov partition
# ══════════════════════════════════════════════════════════════════════

println("═══ Part 1: Eigenfunction structure vs Markov partition ═══")

# Load eigenvector data (K=1024, 2048-bit precision coefficients)
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

# Load spectral coefficients
coeff_lines = readlines(joinpath(DATA_DIR, "spectral_coefficients_K1024_P2048.txt"))
eigenvalues_spectral = zeros(NUM_EIGS)
for line in coeff_lines
    startswith(line, "#") && continue
    parts = split(line, '\t')
    length(parts) >= 4 || continue
    j = parse(Int, parts[1])
    j > NUM_EIGS && continue
    eigenvalues_spectral[j] = parse(Float64, parts[2])
end

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

# Fine evaluation grid
x_fine = range(0.001, 0.999, length=50000)

# --- Find zero crossings ---
println("Finding zero crossings of eigenfunctions...")
zero_crossings = Vector{Vector{Float64}}(undef, NUM_EIGS)
for j in 1:NUM_EIGS
    coeffs = @view eigenvectors[:, j]
    vals = [eval_poly(coeffs, x) for x in x_fine]
    crossings = Float64[]
    for i in 1:(length(vals)-1)
        if vals[i] * vals[i+1] < 0  # sign change
            # Linear interpolation for zero location
            x0 = x_fine[i] - vals[i] * (x_fine[i+1] - x_fine[i]) / (vals[i+1] - vals[i])
            push!(crossings, x0)
        end
    end
    zero_crossings[j] = crossings
    n_zeros = length(crossings)
    if n_zeros > 0
        @printf("  v_%2d: %3d zero crossings, first at x = %.5f, last at x = %.5f\n",
                j, n_zeros, crossings[1], crossings[end])
    else
        @printf("  v_%2d: no zero crossings (positive eigenfunction)\n", j)
    end
end

# --- Plot 1a: Zero crossings vs Markov partition ---
println("\nGenerating zero crossings plot...")
fig_zeros = Figure(size=(1000, 600), fontsize=12)
ax_zeros = Axis(fig_zeros[1, 1];
    xlabel="Eigenfunction index j",
    ylabel="x (zero crossing location)",
    title="Zero crossings of v_j(x) vs Markov partition boundaries 1/n")

# Plot zero crossings as scatter points
for j in 1:NUM_EIGS
    if !isempty(zero_crossings[j])
        scatter!(ax_zeros, fill(j, length(zero_crossings[j])), zero_crossings[j];
            color=(:steelblue, 0.6), markersize=4)
    end
end

# Markov partition boundaries: x = 1/n
for n in 2:30
    xn = 1.0 / n
    hlines!(ax_zeros, [xn]; color=(:gray50, 0.3), linestyle=:dot, linewidth=0.5)
    if n <= 10
        text!(ax_zeros, NUM_EIGS + 0.5, xn; text="1/$n", fontsize=8, color=:gray40)
    end
end

# Reference curves
lines!(ax_zeros, 2:NUM_EIGS, [1.0/j for j in 2:NUM_EIGS];
    color=:firebrick, linewidth=1.5, linestyle=:dash, label="x = 1/j")
lines!(ax_zeros, 2:NUM_EIGS, [1.0/(j+1) for j in 2:NUM_EIGS];
    color=:darkorange, linewidth=1.5, linestyle=:dash, label="x = 1/(j+1)")

axislegend(ax_zeros; position=:rt)
save(joinpath(DATA_DIR, "eigenfunction_zero_crossings.pdf"), fig_zeros)
println("  → data/eigenfunction_zero_crossings.pdf")

# --- Plot 1b: Number of zeros vs j ---
println("Generating zero count plot...")
n_zeros_vec = [length(zc) for zc in zero_crossings]
fig_nz = Figure(size=(700, 400), fontsize=12)
ax_nz = Axis(fig_nz[1, 1];
    xlabel="Eigenfunction index j",
    ylabel="Number of zero crossings",
    title="Zero crossings count for v_j")
scatter!(ax_nz, 1:NUM_EIGS, n_zeros_vec; color=:steelblue, markersize=6, label="# zeros")
lines!(ax_nz, 1:NUM_EIGS, 0:(NUM_EIGS-1); color=:firebrick, linestyle=:dash, label="j - 1")
axislegend(ax_nz; position=:lt)
save(joinpath(DATA_DIR, "eigenfunction_zero_count.pdf"), fig_nz)
println("  → data/eigenfunction_zero_count.pdf")

# --- Plot 1c: Local extrema within Markov cylinders ---
# For each eigenfunction j, find local extrema (not at boundary)
println("Finding local extrema...")
local_extrema = Vector{Vector{Float64}}(undef, NUM_EIGS)
extrema_vals = Vector{Vector{Float64}}(undef, NUM_EIGS)
for j in 1:NUM_EIGS
    coeffs = @view eigenvectors[:, j]
    vals = [eval_poly(coeffs, x) for x in x_fine]
    locs = Float64[]
    evals = Float64[]
    for i in 2:(length(vals)-1)
        # Local max or min (sign change in derivative approximation)
        if (vals[i] > vals[i-1] && vals[i] > vals[i+1]) ||
           (vals[i] < vals[i-1] && vals[i] < vals[i+1])
            push!(locs, x_fine[i])
            push!(evals, vals[i])
        end
    end
    local_extrema[j] = locs
    extrema_vals[j] = evals
end

# For each j, find the location of the largest local maximum (in absolute value)
println("\nLargest local extremum per eigenfunction:")
fig_lmax = Figure(size=(900, 500), fontsize=12)
ax_lmax = Axis(fig_lmax[1, 1];
    xlabel="Eigenfunction index j",
    ylabel="x (location of largest local extremum)",
    title="Location of dominant local extremum of v_j(x)")

dom_extremum_x = zeros(NUM_EIGS)
for j in 1:NUM_EIGS
    if !isempty(extrema_vals[j])
        idx = argmax(abs.(extrema_vals[j]))
        dom_extremum_x[j] = local_extrema[j][idx]
        @printf("  v_%2d: dominant extremum at x = %.5f  (cylinder n ≈ %d)\n",
                j, dom_extremum_x[j], round(Int, 1.0/dom_extremum_x[j]))
    end
end

scatter!(ax_lmax, 1:NUM_EIGS, dom_extremum_x; color=:steelblue, markersize=8,
         label="argmax |local extremum|")
lines!(ax_lmax, 2:NUM_EIGS, [1.0/j for j in 2:NUM_EIGS];
    color=:firebrick, linewidth=1.5, linestyle=:dash, label="x = 1/j")
lines!(ax_lmax, 2:NUM_EIGS, [1.0/(j+1) for j in 2:NUM_EIGS];
    color=:darkorange, linewidth=1.5, linestyle=:dash, label="x = 1/(j+1)")

axislegend(ax_lmax; position=:rt)
save(joinpath(DATA_DIR, "eigenfunction_local_extrema.pdf"), fig_lmax)
println("  → data/eigenfunction_local_extrema.pdf")

# ══════════════════════════════════════════════════════════════════════
# Part 2: Ordered Schur form — non-normality structure
# ══════════════════════════════════════════════════════════════════════

println("\n═══ Part 2: Ordered Schur form — non-normality ═══")

# Load GenericSchur cache (K=1024, 2048-bit precision)
println("Loading Schur cache (K=1024, P=2048)...")
sd_bf = deserialize(joinpath(DATA_DIR, "bigfloat_schur_K1024_P2048.jls"))
T_bf = sd_bf[:T_bf]
Q_bf = sd_bf[:Q_bf]
n = size(T_bf, 1)
println("  Matrix size: $n × $n  (precision = $(precision(real(T_bf[1,1]))) bits)")

# Convert to Complex{Float64} for ordschur + plotting
T_f64 = Complex{Float64}.(T_bf)
Q_f64 = Complex{Float64}.(Q_bf)
eigenvalues_all = diag(T_f64)
println("  Eigenvalue range: |λ₁| = $(maximum(abs.(eigenvalues_all))), |λ_n| = $(minimum(abs.(eigenvalues_all)))")

# Sort by decreasing |λ|
sorted_idx = sortperm(abs.(eigenvalues_all), rev=true)
println("  Top 10 eigenvalues by magnitude:")
for i in 1:10
    λ = eigenvalues_all[sorted_idx[i]]
    @printf("    λ_%d = %.10f  (|λ| = %.10e)\n", i, real(λ), abs(λ))
end

# Swap adjacent eigenvalues k ↔ k+1 in Schur form via Givens rotation
function swap_schur_1x1!(T::AbstractMatrix, Q::AbstractMatrix, k::Int)
    nn = size(T, 1)
    a, b, c = T[k, k], T[k+1, k+1], T[k, k+1]
    x = (b - a) / c
    nrm = sqrt(one(x) + x * conj(x))
    cs, sn = one(x) / nrm, x / nrm
    for j in 1:nn
        t1, t2 = T[k, j], T[k+1, j]
        T[k, j]   = conj(cs) * t1 + conj(sn) * t2
        T[k+1, j] = -sn * t1 + cs * t2
    end
    for i in 1:nn
        t1, t2 = T[i, k], T[i, k+1]
        T[i, k]   = t1 * cs + t2 * sn
        T[i, k+1] = -t1 * conj(sn) + t2 * cs
    end
    for i in 1:nn
        q1, q2 = Q[i, k], Q[i, k+1]
        Q[i, k]   = q1 * cs + q2 * sn
        Q[i, k+1] = -q1 * conj(sn) + q2 * cs
    end
    T[k+1, k] = zero(eltype(T))
end

# Full sort: move eigenvalues at sorted_idx to positions 1, 2, ..., n
println("Ordering Schur form by decreasing |λ| ($n eigenvalues)...")
T_ord, Q_ord = copy(T_f64), copy(Q_f64)
current_pos = collect(1:n)
for (dest, orig_idx) in enumerate(sorted_idx)
    src = findfirst(==(orig_idx), current_pos)
    for k in (src - 1):-1:dest
        swap_schur_1x1!(T_ord, Q_ord, k)
        current_pos[k], current_pos[k+1] = current_pos[k+1], current_pos[k]
    end
end
for i in 2:n, j in 1:i-1
    T_ord[i, j] = zero(eltype(T_ord))
end
println("  Done. Verifying...")
println("  T_ord[1,1] = $(real(T_ord[1,1]))  (should be λ₁ ≈ 1.0)")
println("  T_ord[2,2] = $(real(T_ord[2,2]))  (should be λ₂)")
println("  Orthogonality defect ‖Q*Q - I‖ = $(opnorm(Q_ord' * Q_ord - I))")

# --- Henrici departure from normality ---
T_offdiag = copy(T_ord)
for i in 1:n
    T_offdiag[i, i] = 0.0
end
henrici = norm(T_offdiag) / norm(T_ord)
println("\n  Henrici departure from normality: ‖T - diag(T)‖_F / ‖T‖_F = $(@sprintf("%.6e", henrici))")

# ── Plot 2a: Heatmap of log₁₀|T_{ij}| for first 80×80 block ──
println("\nGenerating non-normality heatmap...")
N_plot = min(80, n)
T_abs = abs.(T_ord[1:N_plot, 1:N_plot])
T_log = log10.(max.(T_abs, 1e-20))

fig_heat = Figure(size=(800, 700), fontsize=12)
ax_heat = Axis(fig_heat[1, 1];
    xlabel="Column j", ylabel="Row i",
    title="log₁₀|T_{ij}| — Ordered Schur form (K=1024, first $N_plot modes)",
    yreversed=true)
hm = heatmap!(ax_heat, 1:N_plot, 1:N_plot, T_log';
    colormap=:viridis, colorrange=(-18, 0))
Colorbar(fig_heat[1, 2], hm; label="log₁₀|T_{ij}|")
save(joinpath(DATA_DIR, "schur_nonnormality_heatmap.pdf"), fig_heat)
println("  → data/schur_nonnormality_heatmap.pdf")

# ── Plot 2b: Row profiles |T_{i,j}| for selected i ──
println("Generating row profile plots...")
fig_rows = Figure(size=(900, 600), fontsize=12)
ax_rows = Axis(fig_rows[1, 1];
    xlabel="Column j",
    ylabel="|T_{i,j}|",
    title="Off-diagonal Schur coupling: rows i = 1, 2, 5, 10, 20, 50",
    yscale=log10)

row_indices = [1, 2, 5, 10, 20, 50]
colors_row = [:steelblue, :firebrick, :seagreen, :darkorange, :purple4, :goldenrod]

for (ci, i) in enumerate(row_indices)
    j_range = (i+1):min(n, 150)
    vals = abs.(T_ord[i, j_range])
    mask = vals .> 1e-20
    if any(mask)
        scatter!(ax_rows, collect(j_range)[mask], vals[mask];
            color=colors_row[ci], markersize=4, label="i = $i")
    end
end

axislegend(ax_rows; position=:rt, nbanks=2)
save(joinpath(DATA_DIR, "schur_nonnormality_rows.pdf"), fig_rows)
println("  → data/schur_nonnormality_rows.pdf")

# ── Plot 2c: Diagonal decay and off-diagonal row norms ──
println("Generating diagonal + off-diagonal summary...")
fig_diag = Figure(size=(900, 500), fontsize=12)

ax_left = Axis(fig_diag[1, 1];
    xlabel="Index j", ylabel="|λ_j|",
    title="Eigenvalue decay", yscale=log10)
eig_sorted = abs.(diag(T_ord))
scatter!(ax_left, 1:min(NUM_EIGS, n), eig_sorted[1:min(NUM_EIGS, n)];
    color=:steelblue, markersize=5)

ax_right = Axis(fig_diag[1, 2];
    xlabel="Row i", ylabel="‖T_{i, i+1:end}‖₂",
    title="Off-diagonal row norms (coupling strength)", yscale=log10)
row_norms = [norm(T_ord[i, (i+1):n]) for i in 1:min(NUM_EIGS, n)]
scatter!(ax_right, 1:length(row_norms), row_norms;
    color=:firebrick, markersize=5)

save(joinpath(DATA_DIR, "schur_nonnormality_summary.pdf"), fig_diag)
println("  → data/schur_nonnormality_summary.pdf")

println("\nAll analyses complete.")
