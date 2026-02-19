# GKW Transfer Operator: Certified Spectral Data

Rigorous spectral data for the Gauss-Kuzmin-Wirsing (GKW) transfer operator
L_1 = S*L : H^2(D_1) -> H^2(D_1), where L f(x) = sum_{n>=1} (x+n)^{-2s} f(1/(x+n)), s=1.

## Parameters
- Galerkin truncation: K = 1024 (matrix dimension 1025)
- BigFloat precision: 2048 bits (617 decimal digits)
- Number of eigenvalues: 50
- Truncation error: eps_K = 3.228311e-180
- Operator norm: ||A_K||_2 <= 1.097559e+00
- Schur residual: ||A_true - QTQ'||_2 <= 5.715071e-317

## Files

### gkw_spectral_coefficients_K1024.tsv
Eigenvalues, spectral coefficients ell_j(1), error bounds, eigenvalue
enclosures, and projector errors for all 50 eigenvalues.
All BigFloat values are given to 617 decimal digits.

### gkw_eigenvectors_K1024.tsv
Full eigenvector coefficients [v_j]_k in the shifted monomial basis {(x-1)^k}.
Each eigenvector has K+1 = 1025 coefficients, unit-normalized (||v_j||_2 = 1).
Rigorous L2 error bounds are in the spectral coefficients file.

### gkw_resolvent_certification.tsv
Resolvent certification data from the two-stage method (Stage 1).
Proves simplicity of each eigenvalue via excluding circles Gamma_j ⊂ rho(L_1).

### gkw_matrix_K1024.tsv.gz
Galerkin matrix A_1024 center values (1025 × 1025 Complex{BigFloat}).
Format: i, j, real_part, imag_part (0-indexed). Gzip compressed.

### gkw_schur_Q_K1024.tsv.gz
Schur unitary factor Q (1025 × 1025 Complex{BigFloat}).
A_K = Q T Q^*. Format: i, j, real_part, imag_part (0-indexed). Gzip compressed.

### gkw_schur_T_K1024.tsv.gz
Schur upper triangular factor T (1025 × 1025 Complex{BigFloat}).
Only upper triangle (i <= j) stored. Gzip compressed.

### gkw_schur_error_K1024.tsv
Schur quality metrics: residual norm, orthogonality defect, total error bound,
truncation error eps_K, and operator norm ||A_K||.

### gkw_certification_log_K1024.txt
Full computation log from the K=1024 spectral certification run.

## How to read these files

### Plain TSV files
All `.tsv` files are tab-separated text with `#`-prefixed header comments.
They can be opened in any text editor, spreadsheet, or parsed programmatically.

**Julia:**
```julia
using DelimitedFiles
# Read spectral coefficients (skip comment lines)
lines = filter(l -> !startswith(l, "#"), readlines("gkw_spectral_coefficients_K1024.tsv"))
header = split(lines[1], '\t')
data = [split(l, '\t') for l in lines[2:end]]
# Parse eigenvalues to BigFloat for full precision
setprecision(BigFloat, 2048)
eigenvalues = [parse(BigFloat, row[2]) for row in data]
```

**Python:**
```python
import pandas as pd
from mpmath import mp, mpf
mp.dps = 617  # match BigFloat precision
df = pd.read_csv("gkw_spectral_coefficients_K1024.tsv", sep="\t", comment="#")
# For full precision, parse string values with mpmath:
eigenvalues = [mpf(s) for s in df["lambda_j"]]
```

### Compressed TSV files (`.tsv.gz`)
The matrix and Schur factor files are gzip-compressed due to their size.

**Decompress from command line:**
```bash
gunzip gkw_matrix_K1024.tsv.gz         # produces gkw_matrix_K1024.tsv
# or read without decompressing:
zcat gkw_matrix_K1024.tsv.gz | head -20
```

**Julia:**
```julia
using CodecZlib
setprecision(BigFloat, 2048)
n = 1025
A = zeros(Complex{BigFloat}, n, n)
open(GzipDecompressorStream, "gkw_matrix_K1024.tsv.gz") do io
    for line in eachline(io)
        startswith(line, "#") && continue
        startswith(line, "i") && continue  # header
        parts = split(line, '\t')
        i, j = parse(Int, parts[1]) + 1, parse(Int, parts[2]) + 1
        A[i, j] = parse(BigFloat, parts[3]) + im * parse(BigFloat, parts[4])
    end
end
```

**Python:**
```python
import gzip, numpy as np
# For double precision (loses ~600 digits but fine for visualization):
data = np.loadtxt(gzip.open("gkw_matrix_K1024.tsv.gz"),
                  comments="#", skiprows=1, delimiter="\t")
n = 1025
A = np.zeros((n, n), dtype=complex)
A[data[:, 0].astype(int), data[:, 1].astype(int)] = data[:, 2] + 1j * data[:, 3]
```

### Reconstructing eigenfunctions
The eigenvectors are stored in the shifted monomial basis `{(x-1)^k}` for k = 0, ..., K.
To evaluate eigenfunction `v_j(x)` at a point `x` in [0, 1]:
```julia
function eval_eigenfunction(coeffs, x)
    w = x - 1.0
    result = coeffs[1]  # k=0 term
    w_pow = w
    for k in 1:length(coeffs)-1
        result += coeffs[k+1] * w_pow
        w_pow *= w
    end
    return result
end
```

### Verifying the Schur decomposition
After loading Q and T from the gzipped files:
```julia
residual = norm(A - Q * T * Q')  # should be ≈ 5.715071e-317
unitarity = norm(Q' * Q - I)     # should be ≈ machine epsilon
```

## File summary

| File | Description | Format | Size |
|------|-------------|--------|------|
| `gkw_spectral_coefficients_K1024.tsv` | Eigenvalues, `ell_j(1)`, error bounds | TSV | small |
| `gkw_eigenvectors_K1024.tsv` | Eigenvector coefficients (50 vectors, 1025 components each) | TSV | ~30 MB |
| `gkw_resolvent_certification.tsv` | Resolvent certification (Stage 1) | TSV | small |
| `gkw_matrix_K1024.tsv.gz` | Galerkin matrix A_1024 | gzipped TSV | ~100 MB |
| `gkw_schur_Q_K1024.tsv.gz` | Schur unitary factor Q | gzipped TSV | ~100 MB |
| `gkw_schur_T_K1024.tsv.gz` | Schur upper triangular T | gzipped TSV | ~50 MB |
| `gkw_schur_error_K1024.tsv` | Schur quality metrics | TSV | small |
| `gkw_certification_log_K1024.txt` | Full computation log | text | small |

## Certification Summary
- All 50 eigenvalues certified simple (excluding circles in resolvent set)
- All 50 spectral coefficients ell_j(1) sign-certified
- All 50 eigenvalue enclosures finite (projector control at K=1024)
- Eigenvalue enclosure range: [1.28e-176, 1.41e-119]
- Eigenvector error range: [2.40e-316, 3.23e-289]
- Spectral coefficient radii: [4.52e-316, 1.14e-282]

## Method
1. **Matrix construction**: Galerkin discretization using ArbNumerics at 2048-bit precision
2. **Schur decomposition**: GenericSchur.jl at 2048-bit, certified via residual norm
3. **Spectral coefficients**: Ball-arithmetic ordschur + direct triangular Sylvester solve
4. **Resolvent certification**: Two-stage method (resolvent sampling + Schur direct)
5. **Eigenvalue enclosures**: Projector control (Lemma 2.12) via resolvent bridge

## Software
- Julia 1.12+
- GKWExperiments.jl (this package)
- BallArithmetic.jl (rigorous linear algebra)
- GenericSchur.jl (BigFloat Schur decomposition)
- ArbNumerics.jl (arbitrary precision arithmetic)

## Citation
If you use this data, please cite the accompanying paper:
"Rigorous spectral certification of the Gauss-Kuzmin-Wirsing operator"

Generated: 2026-02-20T05:26:22.214
