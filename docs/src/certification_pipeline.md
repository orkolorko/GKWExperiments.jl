# Certification Pipeline

This page describes the full two-script pipeline that certifies the first 50
eigenvalues of the GKW transfer operator $L_1 : H^2(D_1) \to H^2(D_1)$ and
computes the spectral coefficients $\ell_j(1)$ with rigorous error bounds.
For the discretization of $L_1$ and the definitions of $A_K$, $\varepsilon_K$,
and $C_2$ see [Discretization](discretization.md).

---

## Overview

The pipeline is split into two scripts that communicate through serialized
checkpoints in `data/`:

| Script | File | Outputs |
|--------|------|---------|
| **Script 1** | `scripts/full_certification_50eigs.jl` | Resolvent bounds $M_{\infty,j}$, projector errors, tail resolvent |
| **Script 2** | `scripts/spectral_data_50eigs.jl` | NK radii, $\ell_j(1)$ with error bounds, tail bound |

A third script, `scripts/bigfloat_spectral_K1024.jl`, uses $K = 1024$ and
2048-bit arithmetic to compute tighter spectral coefficients; it loads Script 1's
$M_{\infty,j}$ data and applies a different strategy for $\ell_j(1)$ (see
[K = 1024 Variant](@ref)).

All intermediate results are cached as `.jls` files so that each script can be
interrupted and resumed from any completed phase.

---

## Phase 0 — Constants

Both scripts begin by computing two $K$-independent constants rigorously in Arb
arithmetic:

- **$C_2$**: an operator norm bound for $L_1 : H^2(D_1) \to H^2(D_{3/2})$,
  computed by summing a finite part and bounding the tail via a Hurwitz zeta
  estimate (splitting parameter $N = 5000$, giving $C_2 \lesssim 10.06$).

- **$\varepsilon_K = C_2 \cdot (2/3)^{K+1}$**: the truncation error bound
  $\|L_1 - A_K\|_{H^2(D_1)} \leq \varepsilon_K$ (Corollary 4.1).

The results are converted to rigorous Float64 upper bounds via
`_arb_to_float64_upper`.

---

## Script 1 — Resolvent Certification

### The small-gain condition

The link between the finite matrix $A_K$ and the infinite-dimensional operator
$L_1$ is the **small-gain condition**: if

$$
\alpha = \varepsilon_K \cdot \sup_{z \in \Gamma_j} \|R_{A_K}(z)\| < 1
$$

on a simple closed contour $\Gamma_j$ encircling $\lambda_j$ (and no other
eigenvalue of $A_K$), then $L_1$ has exactly one eigenvalue inside $\Gamma_j$
(Theorem 2.3).  When this holds, the bound

$$
M_{\infty,j} = \frac{\sup_{z \in \Gamma_j}\|R_{A_K}(z)\|}{1 - \alpha}
$$

is a rigorous upper bound on $\sup_{z \in \Gamma_j}\|R_{L_1}(z)\|$.

The resolvent norm $\|R_{A_K}(z)\|$ is bounded at each sample point $z$ by
certifying a lower bound on $\sigma_{\min}(zI - A_K)$ via the `CertifScripts`
subpackage of BallArithmetic.jl (adaptive arcs + Weyl propagation between
samples).

### Phase 1a — Standard resolvent scan at $K = 48$ (eigenvalues 1–20)

For the first 20 eigenvalues (largest in magnitude), the full $(K_{\text{low}}+1)
\times (K_{\text{low}}+1)$ Float64 BallMatrix $A_{48}$ is assembled and a circle
scan is run directly:

```
circle_j = CertificationCircle(λ_j, r_j; samples=256)
cert     = run_certification(A_low, circle_j)
α        = ε_{K_low} * cert.resolvent_original   # RoundUp
M_∞,j    = cert.resolvent_original / (1 - α)     # RoundUp/RoundDown
```

The circle radius $r_j$ is chosen as
$\min(0.01\,|\lambda_j|,\; d_j/3)$ where $d_j$ is the distance from $\lambda_j$
to the nearest other eigenvalue of $A_{48}$.

Phase 1a succeeds for eigenvalues 1–20 because:
- $|\lambda_j|$ is not too small, so $r_j$ can be moderate;
- $\varepsilon_{48} \approx 10^{-10}$ is small enough to close the small-gain
  condition even when $\|R_{A_{48}}\|$ is a few hundred.

### Phase 1b — Schur-direct certification at $K = 256$ (eigenvalues 21–50)

For eigenvalues 21–50, Phase 1a fails because the eigenvalues are small
($|\lambda_j| \sim 10^{-3}$ or smaller), forcing $r_j$ to be tiny and
$\|R_{A_{48}}(z)\| \approx 1/r_j$ to be huge, so $\alpha \geq 1$.

Increasing $K$ to 256 gives $\varepsilon_{256} \approx 10^{-46}$, but running a
full circle scan on the 257×257 BallMatrix can still produce a large resolvent
bound.  Instead, Phase 1b exploits the **natural Schur ordering** to bound the
resolvent via a block structure, avoiding ordschur entirely.

#### Natural block partition

GenericSchur.jl returns eigenvalues in decreasing order of magnitude. For
eigenvalue at Schur position $p$ (i.e., $|\lambda_p| \leq |\lambda_{p-1}|$), the
Schur matrix $T$ of $A_{256}$ splits into three natural blocks without any
reordering:

$$
T = \begin{pmatrix} T_{11} & T_{12} \\ 0 & T_{22} \end{pmatrix},
\qquad
T_{11} \in \mathbb{C}^{(p-1)\times(p-1)},\quad
T_{22} \in \mathbb{C}^{(n-p+1)\times(n-p+1)}.
$$

- $T_{11}$: eigenvalues larger in magnitude than $\lambda_p$ (positions $1, \ldots,
  p-1$).
- $T_{22}$: $\lambda_p$ at position $(1,1)$ followed by all smaller eigenvalues.

On the circle $\Gamma_j$ centered at $\lambda_p$ with radius $r_j$:

- $(zI - T_{22})$ has $\lambda_p$ at distance $r_j$ from $z$, and all other
  $T_{22}$ eigenvalues at distance $\geq$ gap to the next eigenvalue — so
  $\|R_{T_{22}}(z)\|$ is finite and moderate.
- $(zI - T_{11})$ has all eigenvalues of $T_{11}$ separated from $\lambda_p$ by
  construction; a lower bound on $\sigma_{\min}(z_0 I - T_{11})$ at the circle
  centre $z_0 = \lambda_p$ is obtained via **BigFloat SVD + Miyajima
  certification**, and then Weyl propagation gives
  $\sigma_{\min}(zI - T_{11}) \geq \sigma_{\min}(z_0 I - T_{11}) - r_j$.

#### Block resolvent formula

$$
\|R_T(z)\| \leq
\|R_{T_{11}}(z)\| \cdot \bigl(1 + \|T_{12}\|_F \cdot \|R_{T_{22}}(z)\|\bigr)
+ \|R_{T_{22}}(z)\|,
$$

where $\|R_{T_{11}}(z)\| \leq 1/(\sigma_{\min}(z_0 I - T_{11}) - r_j)$ is constant
on $\Gamma_j$, and $\|R_{T_{22}}(z)\|$ is bounded by a CertifScripts scan (with
svdbox fallback) on the smaller matrix $T_{22}$.

#### Schur bridge

Since $A_{256} = Q T Q^*$ (up to residual $\|E\| \leq E_{\text{Schur}}$, bounded
by `compute_schur_and_error`):

$$
\|R_{A_{256}}(z)\| \leq \|Q\| \cdot \|R_T(z)\| \cdot \|Q^{-1}\|,
$$

where $\|Q\|$ and $\|Q^{-1}\|$ are the rigorous bounds returned by
`compute_schur_and_error`.

#### Why no ordschur is needed

The critical insight is that the natural Schur ordering (decreasing $|\lambda|$)
already places $\lambda_p$ at the border of the two blocks.  No Givens swaps are
required.  This avoids the need to track ordschur rounding errors in the resolvent
bound (ordschur error tracking is only needed for $\ell_j(1)$, in Script 2).

### Phase 1c — Fallback

If any eigenvalue in positions 1–20 is not certified by Phase 1a (rare: happens
when the Schur ordering of $A_{48}$ places a near-degenerate eigenvalue at
position $i$ while the true $|\lambda_i| < |\lambda_{i+1}|$ by more than a
threshold), Phase 1c reruns `certify_eigenvalue_schur_direct` at $K = 256$ for
those positions.

### Phase 2 — Transfer bridge and Riesz projector errors

Once $M_{\infty,j}$ is established at $K_{\text{low}}$ (or $K_{\text{high}}$ for
Phase 1b), the resolvent bound is transferred to any higher truncation $K'$ via
the **reverse transfer formula**:

$$
\|R_{A_{K'}}(z)\| \leq \frac{M_{\infty,j}}{1 - M_{\infty,j} \cdot \varepsilon_{K'}}
\quad \text{(valid when } M_{\infty,j} \cdot \varepsilon_{K'} < 1\text{)}.
$$

The **Riesz projector approximation error** (the distance between the
infinite-dimensional Riesz projector $P_{L_1}(\Gamma_j)$ and its finite
approximation $P_{A_{K'}}(\Gamma_j)$) is then bounded by:

$$
\|P_{L_1}(\Gamma_j) - P_{A_{K'}}(\Gamma_j)\| \leq
\frac{|\Gamma_j|}{2\pi} \cdot \frac{\|R_{A_{K'}}(\Gamma_j)\|^2 \cdot \varepsilon_{K'}}
{1 - \|R_{A_{K'}}(\Gamma_j)\| \cdot \varepsilon_{K'}}.
$$

### Phase 3 — Tail resolvent on the separating circle

To bound the contribution of all eigenvalues $\lambda_j$ with $j > 50$, a
separating circle of radius $\rho_{\text{tail}} = (|\lambda_{50}| +
|\lambda_{51}|)/2$ is introduced around the origin.  The resolvent of $L_1$ on
this circle is bounded by the same block formula applied to the full 50/remainder
split of the BigFloat Schur form of $A_{256}$, giving $M_{\infty,\text{tail}}$.

---

## Script 2 — NK Refinement and Spectral Coefficients

### Phase 1 — Newton–Kantorovich certification

For each eigenvalue $j \leq 50$ that passes the small-gain condition, a
Newton–Kantorovich argument gives a much tighter enclosure radius $r_{\mathrm{NK},j}$:

$$
r_{\mathrm{NK},j} = \frac{2y}{(1 - q_0) + \sqrt{(1-q_0)^2 - 4q_k y}}
$$

where $y$ is a residual bound, $q_k$ is a discrete defect, and $q_0$ accounts
for the infinite-dimensional tail ($q_0 = C_2 \cdot \varepsilon_K$).  This is
computed by `certify_eigenpair_nk` with BigFloat arithmetic at $K = 256$.

### Phase 2 — Spectral coefficient $\ell_j(1)$

The spectral coefficient $\ell_j(1)$ is the pairing of the constant function $1$
with the left eigenvector corresponding to $\lambda_j$, i.e.,

$$
\ell_j(1) = e_1^* \Pi_j e_1,
$$

where $\Pi_j$ is the Riesz projector and $e_1$ is the first basis vector (the
constant function in the shifted monomial basis).  The computation proceeds as:

1. **`ordschur_ball`**: Givens rotations with BallArithmetic radius tracking move
   $\lambda_j$ to position $(1,1)$ in the Schur factor $T$, giving reordered
   factors $Q_{\mathrm{ord}}$ and $T_{\mathrm{ord}}$ as `BallMatrix` objects
   (radii propagated rigorously).

2. **Sylvester solve**: the block projector $\Pi_j$ in Schur coordinates has
   the form $P_S = \begin{pmatrix} I & Y \\ 0 & 0 \end{pmatrix}$ where $Y$
   satisfies the Sylvester equation $T_{22}^* Y - Y T_{11}^* = T_{12}^*$ (with
   $Y = -X^*$ and $X$ from `triangular_sylvester_miyajima_enclosure`).

3. **Formula**:
   ```
   q      = conj(Q_ord[1, :])          # = Q_ord^H e₁
   ℓ_j(1) = real(q[1] - dot(X_mid, q[2:end]))
   ```
   Note: `dot(a,b) = aᴴb` in Julia, so `dot(X_mid, q_rest) = Xᴴ q_rest`
   which equals $-Y q_{\mathrm{rest}}$, giving the correct formula.

4. **Error budget**:
   - Sylvester error: componentwise propagation from `BallMatrix` radii of $X$.
   - Projector perturbation: from the resolvent bounds via `spectral_projector_error_bound`.
   - NK eigenvector correction: $\|\Pi_S\|_2 \cdot 2 r_{\mathrm{NK},j}$ where
     $\|\Pi_S\|_2 = \sqrt{1 + \|Y\|_2^2}$.

### Phase 3 — Tail bound

The tail of the spectral expansion is bounded by:

$$
\|R_{50}^{(n)}\|_{H^2(D_1)} \leq \rho_{\mathrm{tail}}^{n+1} \cdot M_{\infty,\mathrm{tail}} \cdot \|Q_{50}(L_1) \cdot 1\|_{H^2},
$$

where $Q_{50}(L_1) = I - \sum_{j=1}^{50} \Pi_j$ is the spectral complement
projector.  The norm $\|Q_{50} \cdot 1\|$ is bounded by computing the Galerkin
approximation

$$
Q_{50}(A_K) e_1 = e_1 - \sum_{j=1}^{50} \ell_j(1) \cdot v_j
$$

(using the certified $\ell_j(1)$ and eigenvectors $v_j$) and adding the projector
approximation errors $\vartheta_j$ from Phase 2.

---

## K = 1024 Variant

The script `scripts/bigfloat_spectral_K1024.jl` refines the spectral data using
$K = 1024$ and 2048-bit precision ($\varepsilon_{1024} \approx 10^{-181}$).
Two key differences from Script 2:

### Direct triangular Sylvester solve (no Miyajima)

The Miyajima enclosure for Sylvester equations fails for $K \geq 513$ because the
condition number of the eigenvector matrix of $T_{22}^*$ grows with the eigenvalue
spread (see `BugReport_MiyajimaSylvesterLargeMatrix.md`).  Instead, $\ell_j(1)$
is computed by a **direct triangular solve**:

$$
z = (T_{22} - \lambda_j I)^{-1} q_{\mathrm{rest}},
\qquad
\ell_j(1) = \operatorname{Re}\!\bigl(q_1 - \textstyle\sum_k [T_{12}]_k \, z_k\bigr).
$$

Here `sum(T12 .* z)` (not `dot`) is used to avoid Julia's implicit conjugation
of the first argument in `dot`.

The solve error (residual $\|(T_{22} - \lambda_j I) z - q_{\mathrm{rest}}\|$) is
$\sim 10^{-309}$ at 2048-bit precision — negligible.  The error budget is
dominated by the **Schur perturbation bound**:

$$
|\hat\ell_j(1) - \ell_j(1)| \leq \varrho_j
$$

where $\varrho_j$ is computed from the ordschur rounding (tracked by
`ordschur_ball` radii) and the Schur decomposition residual $\|A - QTQ^*\|_2$,
using a Neumann-series / resolvent resolvent hybrid bound on the spectral
projector (taking the minimum of the two estimates).

### No NK certification

At $K = 1024$, NK certification (`certify_eigenpair_nk`) cannot be run because
it internally uses the Miyajima Sylvester solve (same failure mode).  Instead,
**eigenvalue enclosures** are obtained by transferring $M_{\infty,j}$ (from
Script 1) to $K = 1024$ and applying the projector error formula:

$$
|\lambda_j^* - \hat\lambda_j| \leq
\frac{\varepsilon_{1024}(1 + \vartheta_j) + 2\|A_{1024}\|_2 \cdot \vartheta_j}{1 - \vartheta_j},
\qquad
\vartheta_j = \frac{|\Gamma_j|}{2\pi} \cdot \frac{\|R_{A_{1024}}\|^2 \varepsilon_{1024}}
{1 - \|R_{A_{1024}}\|\varepsilon_{1024}}.
$$

---

## Summary of Rigorous Bounds Produced

| Quantity | Source | Rigour |
|----------|--------|--------|
| $\varepsilon_K$ | `compute_Δ` (Arb) | Interval arithmetic |
| $\|R_{A_K}(z)\|$ (Phase 1a) | CertifScripts circle scan | Rigorous: `svdbox` + Weyl |
| $\|R_{A_K}(z)\|$ (Phase 1b) | Block formula + Schur bridge | Rigorous: BigFloat SVD + Miyajima on $T_{11}$ |
| $M_{\infty,j}$ | Small-gain bound | Rigorous: `RoundUp`/`RoundDown` |
| $\|P_{L_1} - P_{A_K}\|$ | Riesz projector formula | Rigorous |
| $r_{\mathrm{NK},j}$ | NK discriminant formula | Rigorous: BigFloat |
| $\ell_j(1)$ (K=256) | ordschur\_ball + Miyajima Sylvester | Rigorous: BallMatrix radii |
| $\ell_j(1)$ (K=1024) | ordschur\_ball + triangular solve | Rigorous: Schur perturbation |
| Tail bound | Block resolvent + projector complement | Rigorous |
