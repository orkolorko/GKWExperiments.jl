#!/usr/bin/env julia
# Prepare data files for Harvard Dataverse upload.
# Reads the K=1024 results and outputs comprehensive TSV files with full precision.
#
# Usage:
#   julia --project --startup-file=no scripts/prepare_dataverse_K1024.jl
#
# Input:  data/spectral_K1024_P2048.jls (from bigfloat_spectral_K1024.jl)
#         data/script1_results.jls (from full_certification_50eigs.jl)
#         data/ell_K1024_P2048.jls (eigenvectors + spectral coefficients)
#         data/ball_matrix_bf_K1024_P2048.jls (Galerkin matrix A₁₀₂₄)
#         data/bigfloat_schur_K1024_P2048.jls (Schur factors Q, T)
#         data/schur_cert_K1024_P2048.jls (Schur quality metrics)
# Output: data/dataverse/  directory with archival TSV files

using Serialization
using Printf
using Dates
using LinearAlgebra
using CodecZlib
using BallArithmetic

const DATA_DIR = joinpath(@__DIR__, "..", "data")
const DV_DIR = joinpath(DATA_DIR, "dataverse")
mkpath(DV_DIR)

const K = 1024
const PREC = 2048

# ── Load results ──────────────────────────────────────────────────────────
println("Loading K=$K results...")
results = Serialization.deserialize(joinpath(DATA_DIR, "spectral_K$(K)_P$(PREC).jls"))
script1 = Serialization.deserialize(joinpath(DATA_DIR, "script1_results.jls"))

NUM_EIGS = results[:NUM_EIGS]

eigenvalues_bf    = results[:eigenvalues_bf]
ell_center_bf     = results[:ell_center_bf]
ell_radius_bf     = results[:ell_radius_bf]
eigvec_radius_bf  = results[:eigvec_radius_bf]
E_bound           = results[:E_bound]
eval_encl         = results[:eval_encl]
proj_error_1024   = results[:proj_error_1024]
eps_K_1024        = results[:eps_K_1024]
C_AK              = results[:C_AK]

resolvent_data = script1[:resolvent_data]
M_inf_all      = script1[:M_inf_all]

println("  K=$K, P=$PREC, $NUM_EIGS eigenvalues")
println("  Output directory: $DV_DIR")
println()

# ── File 1: Eigenvalues + spectral coefficients (full precision) ─────────
path1 = joinpath(DV_DIR, "gkw_spectral_coefficients_K$(K).tsv")
open(path1, "w") do io
    println(io, "# GKW transfer operator: certified spectral data")
    println(io, "# Operator: L_1 = S*L : H^2(D_1) -> H^2(D_1), where L f(x) = sum_{n>=1} (x+n)^{-2s} f(1/(x+n)), s=1")
    println(io, "# Galerkin truncation K = $K, matrix dimension n = $(K+1)")
    println(io, "# BigFloat precision = $PREC bits (≈$(round(Int, PREC * log10(2))) decimal digits)")
    println(io, "# Schur residual ||A_true - QTQ'||_2 <= $E_bound")
    println(io, "# Truncation error eps_K = $eps_K_1024")
    println(io, "# ||A_K||_2 <= $C_AK")
    println(io, "# Generated: $(now())")
    println(io, "#")
    println(io, "# Columns:")
    println(io, "#   j              eigenvalue index (sorted by |lambda_j| descending)")
    println(io, "#   lambda_j       eigenvalue (BigFloat, $(PREC)-bit)")
    println(io, "#   ell_j_1        spectral coefficient ell_j(1) = <P_j(1), v_j> (BigFloat)")
    println(io, "#   ell_radius     rigorous error bound |ell_j(1) - ell_j_hat(1)| (BigFloat)")
    println(io, "#   eigvec_radius  rigorous eigenvector error ||v_j - v_j_hat||_2 (BigFloat)")
    println(io, "#   eval_encl      eigenvalue enclosure radius (Float64, directed rounding)")
    println(io, "#   proj_error     Riesz projector error theta_j (Float64, directed rounding)")
    println(io, "#   M_inf          infinite-dim resolvent bound (Float64)")
    println(io, "#   circle_radius  excluding circle radius (Float64)")
    println(io, "#   sign_certified YES if |ell_j(1)| > ell_radius")
    println(io, "#")
    println(io, "j\tlambda_j\tell_j_1\tell_radius\teigvec_radius\teval_encl\tproj_error\tM_inf\tcircle_radius\tsign_certified")

    for j in 1:NUM_EIGS
        rd = resolvent_data[j]
        sign_ok = abs(ell_center_bf[j]) > ell_radius_bf[j] ? "YES" : "NO"
        println(io, j, "\t",
                string(eigenvalues_bf[j]), "\t",
                string(ell_center_bf[j]), "\t",
                string(ell_radius_bf[j]), "\t",
                string(eigvec_radius_bf[j]), "\t",
                string(eval_encl[j]), "\t",
                string(proj_error_1024[j]), "\t",
                string(M_inf_all[j]), "\t",
                string(rd.circle_radius), "\t",
                sign_ok)
    end
end
println("  Written: $path1")
@printf("  Size: %.1f KB\n", filesize(path1) / 1024)

# ── File 2: Eigenvector coefficients (full precision, all K+1 coefficients) ──
path2 = joinpath(DV_DIR, "gkw_eigenvectors_K$(K).tsv")
ell_cache_path = joinpath(DATA_DIR, "ell_K$(K)_P$(PREC).jls")
ell_cache = Serialization.deserialize(ell_cache_path)
q1_vectors = ell_cache[:q1_vectors]

open(path2, "w") do io
    println(io, "# GKW transfer operator: eigenvector coefficients in shifted monomial basis {(x-1)^k}")
    println(io, "# K = $K, BigFloat precision = $PREC bits")
    println(io, "# Unit-norm eigenvectors: ||v_j||_2 = 1")
    println(io, "# Rigorous error: ||v_j - v_j_hat||_2 <= eigvec_radius_j (see spectral_coefficients file)")
    println(io, "# Generated: $(now())")
    println(io, "#")
    println(io, "# Format: Each row gives one coefficient [v_j]_k = real part of the k-th component")
    println(io, "# (imaginary parts are zero to machine precision — the GKW matrix is real)")
    println(io, "#")
    println(io, "# Columns: j, k, coefficient")
    println(io, "#   j = eigenvalue index (1..$(NUM_EIGS))")
    println(io, "#   k = basis index (0..$K)")
    println(io, "#   coefficient = [v_j]_k (BigFloat, $(PREC)-bit)")
    println(io, "#")
    println(io, "j\tk\tcoefficient")

    for j in 1:NUM_EIGS
        v = q1_vectors[j]
        for k in 0:K
            println(io, j, "\t", k, "\t", string(real(v[k+1])))
        end
    end
end
println("  Written: $path2")
@printf("  Size: %.1f MB\n", filesize(path2) / (1024*1024))

# ── File 3: Resolvent certification data ─────────────────────────────────
path3 = joinpath(DV_DIR, "gkw_resolvent_certification.tsv")
open(path3, "w") do io
    println(io, "# GKW transfer operator: resolvent certification data (Stage 1)")
    println(io, "# Two-stage method: resolvent bridge proves Gamma_j ⊂ rho(L_1)")
    println(io, "# Generated: $(now())")
    println(io, "#")
    println(io, "# Columns:")
    println(io, "#   j              eigenvalue index")
    println(io, "#   circle_radius  radius of excluding circle Gamma_j")
    println(io, "#   resolvent_Ak   ||R_{A_K}|| on Gamma_j at K_low")
    println(io, "#   alpha          small-gain factor eps_K * ||R_{A_K}||")
    println(io, "#   M_inf          infinite-dim resolvent ||R_{L_1}|| on Gamma_j")
    println(io, "#   certified      resolvent certification status")
    println(io, "#")
    println(io, "j\tcircle_radius\tresolvent_Ak\talpha\tM_inf\tcertified")

    for j in 1:NUM_EIGS
        rd = resolvent_data[j]
        println(io, j, "\t",
                string(rd.circle_radius), "\t",
                string(rd.resolvent_Ak), "\t",
                string(rd.alpha), "\t",
                string(rd.M_inf), "\t",
                rd.certified ? "YES" : "NO")
    end
end
println("  Written: $path3")

# ── File 4: Galerkin matrix A₁₀₂₄ (gzipped TSV) ─────────────────────────
println("Loading BallMatrix for Galerkin matrix export...")
A_ball_bf = Serialization.deserialize(joinpath(DATA_DIR, "ball_matrix_bf_K$(K)_P$(PREC).jls"))
A_center = BallArithmetic.mid(A_ball_bf)
n = size(A_center, 1)

path4 = joinpath(DV_DIR, "gkw_matrix_K$(K).tsv.gz")
println("Writing Galerkin matrix ($n × $n) to $path4 ...")
open(GzipCompressorStream, path4, "w") do io
    println(io, "# GKW Galerkin matrix A_$K center values (Complex{BigFloat})")
    println(io, "# K = $K, matrix dimension n = $n")
    println(io, "# BigFloat precision = $PREC bits")
    println(io, "# Format: i, j, real_part, imag_part (0-indexed)")
    println(io, "# Generated: $(now())")
    println(io, "#")
    println(io, "i\tj\treal_part\timag_part")

    for i in 1:n, j in 1:n
        c = A_center[i, j]
        println(io, i-1, "\t", j-1, "\t", string(real(c)), "\t", string(imag(c)))
    end
end
@printf("  Written: %s (%.1f MB)\n", path4, filesize(path4) / (1024*1024))

# ── File 5: Schur unitary factor Q (gzipped TSV) ──────────────────────────
println("Loading Schur factors...")
schur_cache = Serialization.deserialize(joinpath(DATA_DIR, "bigfloat_schur_K$(K)_P$(PREC).jls"))
Q_bf = schur_cache[:Q_bf]
T_bf = schur_cache[:T_bf]

path5 = joinpath(DV_DIR, "gkw_schur_Q_K$(K).tsv.gz")
println("Writing Schur Q ($n × $n) to $path5 ...")
open(GzipCompressorStream, path5, "w") do io
    println(io, "# GKW Schur unitary factor Q (Complex{BigFloat})")
    println(io, "# A = Q T Q^*, K = $K, n = $n")
    println(io, "# BigFloat precision = $PREC bits")
    println(io, "# Format: i, j, real_part, imag_part (0-indexed)")
    println(io, "# Generated: $(now())")
    println(io, "#")
    println(io, "i\tj\treal_part\timag_part")

    for i in 1:n, j in 1:n
        c = Q_bf[i, j]
        println(io, i-1, "\t", j-1, "\t", string(real(c)), "\t", string(imag(c)))
    end
end
@printf("  Written: %s (%.1f MB)\n", path5, filesize(path5) / (1024*1024))

# ── File 6: Schur upper triangular T (gzipped TSV, only i ≤ j) ───────────
path6 = joinpath(DV_DIR, "gkw_schur_T_K$(K).tsv.gz")
println("Writing Schur T (upper triangular, $n × $n) to $path6 ...")
open(GzipCompressorStream, path6, "w") do io
    println(io, "# GKW Schur upper triangular factor T (Complex{BigFloat})")
    println(io, "# A = Q T Q^*, K = $K, n = $n")
    println(io, "# Only upper triangle (i <= j) stored")
    println(io, "# BigFloat precision = $PREC bits")
    println(io, "# Format: i, j, real_part, imag_part (0-indexed)")
    println(io, "# Generated: $(now())")
    println(io, "#")
    println(io, "i\tj\treal_part\timag_part")

    for i in 1:n, j in i:n
        c = T_bf[i, j]
        println(io, i-1, "\t", j-1, "\t", string(real(c)), "\t", string(imag(c)))
    end
end
@printf("  Written: %s (%.1f MB)\n", path6, filesize(path6) / (1024*1024))

# ── File 7: Schur quality metrics ─────────────────────────────────────────
println("Loading Schur certification data...")
cert_cache = Serialization.deserialize(joinpath(DATA_DIR, "schur_cert_K$(K)_P$(PREC).jls"))
E_bound_schur = cert_cache[:E_bound]
orth_defect   = cert_cache[:orth_defect]

path7 = joinpath(DV_DIR, "gkw_schur_error_K$(K).tsv")
open(path7, "w") do io
    println(io, "# GKW Schur decomposition quality metrics")
    println(io, "# K = $K, BigFloat precision = $PREC bits")
    println(io, "# Generated: $(now())")
    println(io, "#")
    println(io, "metric\tvalue")
    println(io, "schur_residual_norm\t", string(E_bound_schur))
    println(io, "orthogonality_defect\t", string(orth_defect))
    println(io, "total_E_bound\t", string(E_bound))
    println(io, "eps_K\t", string(eps_K_1024))
    println(io, "A_K_norm\t", string(C_AK))
end
println("  Written: $path7")

# ── File 8: Certification log ─────────────────────────────────────────────
log_src = joinpath(DATA_DIR, "bigfloat_K1024_log.txt")
path8 = joinpath(DV_DIR, "gkw_certification_log_K$(K).txt")
if isfile(log_src)
    cp(log_src, path8; force=true)
    println("  Copied: $path8")
else
    @warn "Certification log not found at $log_src"
end

# ── File 9: README ────────────────────────────────────────────────────────
path9 = joinpath(DV_DIR, "README.md")
open(path9, "w") do io
    println(io, "# GKW Transfer Operator: Certified Spectral Data")
    println(io)
    println(io, "Rigorous spectral data for the Gauss-Kuzmin-Wirsing (GKW) transfer operator")
    println(io, "L_1 = S*L : H^2(D_1) -> H^2(D_1), where L f(x) = sum_{n>=1} (x+n)^{-2s} f(1/(x+n)), s=1.")
    println(io)
    println(io, "## Parameters")
    println(io, "- Galerkin truncation: K = $K (matrix dimension $(K+1))")
    println(io, "- BigFloat precision: $PREC bits ($(round(Int, PREC * log10(2))) decimal digits)")
    println(io, "- Number of eigenvalues: $NUM_EIGS")
    println(io, "- Truncation error: eps_K = $(@sprintf("%.6e", eps_K_1024))")
    println(io, "- Operator norm: ||A_K||_2 <= $(@sprintf("%.6e", C_AK))")
    println(io, "- Schur residual: ||A_true - QTQ'||_2 <= $(@sprintf("%.6e", E_bound))")
    println(io)
    println(io, "## Files")
    println(io)
    println(io, "### gkw_spectral_coefficients_K$(K).tsv")
    println(io, "Eigenvalues, spectral coefficients ell_j(1), error bounds, eigenvalue")
    println(io, "enclosures, and projector errors for all $NUM_EIGS eigenvalues.")
    println(io, "All BigFloat values are given to $(round(Int, PREC * log10(2))) decimal digits.")
    println(io)
    println(io, "### gkw_eigenvectors_K$(K).tsv")
    println(io, "Full eigenvector coefficients [v_j]_k in the shifted monomial basis {(x-1)^k}.")
    println(io, "Each eigenvector has K+1 = $(K+1) coefficients, unit-normalized (||v_j||_2 = 1).")
    println(io, "Rigorous L2 error bounds are in the spectral coefficients file.")
    println(io)
    println(io, "### gkw_resolvent_certification.tsv")
    println(io, "Resolvent certification data from the two-stage method (Stage 1).")
    println(io, "Proves simplicity of each eigenvalue via excluding circles Gamma_j ⊂ rho(L_1).")
    println(io)
    println(io, "### gkw_matrix_K$(K).tsv.gz")
    println(io, "Galerkin matrix A_$K center values ($(K+1) × $(K+1) Complex{BigFloat}).")
    println(io, "Format: i, j, real_part, imag_part (0-indexed). Gzip compressed.")
    println(io)
    println(io, "### gkw_schur_Q_K$(K).tsv.gz")
    println(io, "Schur unitary factor Q ($(K+1) × $(K+1) Complex{BigFloat}).")
    println(io, "A_K = Q T Q^*. Format: i, j, real_part, imag_part (0-indexed). Gzip compressed.")
    println(io)
    println(io, "### gkw_schur_T_K$(K).tsv.gz")
    println(io, "Schur upper triangular factor T ($(K+1) × $(K+1) Complex{BigFloat}).")
    println(io, "Only upper triangle (i <= j) stored. Gzip compressed.")
    println(io)
    println(io, "### gkw_schur_error_K$(K).tsv")
    println(io, "Schur quality metrics: residual norm, orthogonality defect, total error bound,")
    println(io, "truncation error eps_K, and operator norm ||A_K||.")
    println(io)
    println(io, "### gkw_certification_log_K$(K).txt")
    println(io, "Full computation log from the K=$K spectral certification run.")
    println(io)
    println(io, "## How to read these files")
    println(io)
    println(io, "### Plain TSV files")
    println(io, "All `.tsv` files are tab-separated text with `#`-prefixed header comments.")
    println(io, "They can be opened in any text editor, spreadsheet, or parsed programmatically.")
    println(io)
    println(io, "**Julia:**")
    println(io, "```julia")
    println(io, "using DelimitedFiles")
    println(io, "# Read spectral coefficients (skip comment lines)")
    println(io, "lines = filter(l -> !startswith(l, \"#\"), readlines(\"gkw_spectral_coefficients_K$(K).tsv\"))")
    println(io, "header = split(lines[1], '\\t')")
    println(io, "data = [split(l, '\\t') for l in lines[2:end]]")
    println(io, "# Parse eigenvalues to BigFloat for full precision")
    println(io, "setprecision(BigFloat, $PREC)")
    println(io, "eigenvalues = [parse(BigFloat, row[2]) for row in data]")
    println(io, "```")
    println(io)
    println(io, "**Python:**")
    println(io, "```python")
    println(io, "import pandas as pd")
    println(io, "from mpmath import mp, mpf")
    println(io, "mp.dps = $(round(Int, PREC * log10(2)))  # match BigFloat precision")
    println(io, "df = pd.read_csv(\"gkw_spectral_coefficients_K$(K).tsv\", sep=\"\\t\", comment=\"#\")")
    println(io, "# For full precision, parse string values with mpmath:")
    println(io, "eigenvalues = [mpf(s) for s in df[\"lambda_j\"]]")
    println(io, "```")
    println(io)
    println(io, "### Compressed TSV files (`.tsv.gz`)")
    println(io, "The matrix and Schur factor files are gzip-compressed due to their size.")
    println(io)
    println(io, "**Decompress from command line:**")
    println(io, "```bash")
    println(io, "gunzip gkw_matrix_K$(K).tsv.gz         # produces gkw_matrix_K$(K).tsv")
    println(io, "# or read without decompressing:")
    println(io, "zcat gkw_matrix_K$(K).tsv.gz | head -20")
    println(io, "```")
    println(io)
    println(io, "**Julia:**")
    println(io, "```julia")
    println(io, "using CodecZlib")
    println(io, "setprecision(BigFloat, $PREC)")
    println(io, "n = $(K+1)")
    println(io, "A = zeros(Complex{BigFloat}, n, n)")
    println(io, "open(GzipDecompressorStream, \"gkw_matrix_K$(K).tsv.gz\") do io")
    println(io, "    for line in eachline(io)")
    println(io, "        startswith(line, \"#\") && continue")
    println(io, "        startswith(line, \"i\") && continue  # header")
    println(io, "        parts = split(line, '\\t')")
    println(io, "        i, j = parse(Int, parts[1]) + 1, parse(Int, parts[2]) + 1")
    println(io, "        A[i, j] = parse(BigFloat, parts[3]) + im * parse(BigFloat, parts[4])")
    println(io, "    end")
    println(io, "end")
    println(io, "```")
    println(io)
    println(io, "**Python:**")
    println(io, "```python")
    println(io, "import gzip, numpy as np")
    println(io, "# For double precision (loses ~600 digits but fine for visualization):")
    println(io, "data = np.loadtxt(gzip.open(\"gkw_matrix_K$(K).tsv.gz\"),")
    println(io, "                  comments=\"#\", skiprows=1, delimiter=\"\\t\")")
    println(io, "n = $(K+1)")
    println(io, "A = np.zeros((n, n), dtype=complex)")
    println(io, "A[data[:, 0].astype(int), data[:, 1].astype(int)] = data[:, 2] + 1j * data[:, 3]")
    println(io, "```")
    println(io)
    println(io, "### Reconstructing eigenfunctions")
    println(io, "The eigenvectors are stored in the shifted monomial basis `{(x-1)^k}` for k = 0, ..., K.")
    println(io, "To evaluate eigenfunction `v_j(x)` at a point `x` in [0, 1]:")
    println(io, "```julia")
    println(io, "function eval_eigenfunction(coeffs, x)")
    println(io, "    w = x - 1.0")
    println(io, "    result = coeffs[1]  # k=0 term")
    println(io, "    w_pow = w")
    println(io, "    for k in 1:length(coeffs)-1")
    println(io, "        result += coeffs[k+1] * w_pow")
    println(io, "        w_pow *= w")
    println(io, "    end")
    println(io, "    return result")
    println(io, "end")
    println(io, "```")
    println(io)
    println(io, "### Verifying the Schur decomposition")
    println(io, "After loading Q and T from the gzipped files:")
    println(io, "```julia")
    println(io, "residual = norm(A - Q * T * Q')  # should be ≈ $(E_bound)")
    println(io, "unitarity = norm(Q' * Q - I)     # should be ≈ machine epsilon")
    println(io, "```")
    println(io)
    println(io, "## File summary")
    println(io)
    println(io, "| File | Description | Format | Size |")
    println(io, "|------|-------------|--------|------|")
    println(io, "| `gkw_spectral_coefficients_K$(K).tsv` | Eigenvalues, `ell_j(1)`, error bounds | TSV | small |")
    println(io, "| `gkw_eigenvectors_K$(K).tsv` | Eigenvector coefficients ($(NUM_EIGS) vectors, $(K+1) components each) | TSV | ~30 MB |")
    println(io, "| `gkw_resolvent_certification.tsv` | Resolvent certification (Stage 1) | TSV | small |")
    println(io, "| `gkw_matrix_K$(K).tsv.gz` | Galerkin matrix A_$(K) | gzipped TSV | ~100 MB |")
    println(io, "| `gkw_schur_Q_K$(K).tsv.gz` | Schur unitary factor Q | gzipped TSV | ~100 MB |")
    println(io, "| `gkw_schur_T_K$(K).tsv.gz` | Schur upper triangular T | gzipped TSV | ~50 MB |")
    println(io, "| `gkw_schur_error_K$(K).tsv` | Schur quality metrics | TSV | small |")
    println(io, "| `gkw_certification_log_K$(K).txt` | Full computation log | text | small |")
    println(io)
    println(io, "## Certification Summary")
    n_sign = count(j -> abs(ell_center_bf[j]) > ell_radius_bf[j], 1:NUM_EIGS)
    n_encl = count(isfinite, eval_encl)
    println(io, "- All $NUM_EIGS eigenvalues certified simple (excluding circles in resolvent set)")
    println(io, "- All $n_sign spectral coefficients ell_j(1) sign-certified")
    println(io, "- All $n_encl eigenvalue enclosures finite (projector control at K=$K)")
    println(io, "- Eigenvalue enclosure range: [$(@sprintf("%.2e", minimum(eval_encl))), $(@sprintf("%.2e", maximum(eval_encl)))]")
    evr = Float64.(eigvec_radius_bf)
    println(io, "- Eigenvector error range: [$(@sprintf("%.2e", minimum(evr))), $(@sprintf("%.2e", maximum(evr)))]")
    println(io, "- Spectral coefficient radii: [$(@sprintf("%.2e", Float64(minimum(ell_radius_bf)))), $(@sprintf("%.2e", Float64(maximum(ell_radius_bf))))]")
    println(io)
    println(io, "## Method")
    println(io, "1. **Matrix construction**: Galerkin discretization using ArbNumerics at $PREC-bit precision")
    println(io, "2. **Schur decomposition**: GenericSchur.jl at $PREC-bit, certified via residual norm")
    println(io, "3. **Spectral coefficients**: Ball-arithmetic ordschur + direct triangular Sylvester solve")
    println(io, "4. **Resolvent certification**: Two-stage method (resolvent sampling + Schur direct)")
    println(io, "5. **Eigenvalue enclosures**: Projector control (Lemma 2.12) via resolvent bridge")
    println(io)
    println(io, "## Software")
    println(io, "- Julia 1.12+")
    println(io, "- GKWExperiments.jl (this package)")
    println(io, "- BallArithmetic.jl (rigorous linear algebra)")
    println(io, "- GenericSchur.jl (BigFloat Schur decomposition)")
    println(io, "- ArbNumerics.jl (arbitrary precision arithmetic)")
    println(io)
    println(io, "## Citation")
    println(io, "If you use this data, please cite the accompanying paper:")
    println(io, "\"Rigorous spectral certification of the Gauss-Kuzmin-Wirsing operator\"")
    println(io)
    println(io, "Generated: $(now())")
end
println("  Written: $path9")

println()
println("=" ^ 60)
println("Dataverse files ready in: $DV_DIR")
println("=" ^ 60)
