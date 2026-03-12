# Discretization of the GKW Transfer Operator

This document describes how the Gauss--Kuzmin--Wirsing (GKW) transfer operator
is discretized into a finite matrix, and how the discretization error is
rigorously controlled.

## The GKW Transfer Operator

The **GKW transfer operator** $L_s$ acts on functions as

$$
(L_s f)(w) = \sum_{n=1}^{\infty} \frac{1}{(w+n)^{2s}} \, f\!\left(\frac{1}{w+n}\right).
$$

For the classical case $s = 1$, this is the transfer operator associated with
the Gauss map $T(x) = \{1/x\}$ (the fractional part of $1/x$) that governs
the continued fraction expansion.  The leading eigenvalue is $\lambda_1 = 1$
(with eigenfunction $1/(1+x)$, the Gauss--Kuzmin density), and the
second eigenvalue $\lambda_2 \approx -0.3036$ is the negative of the
**Wirsing constant**.

Each summand involves a **branch map** $\tau_n(w) = 1/(w+n)$ of the Gauss map,
weighted by the factor $(w+n)^{-2s}$.


## Hardy Space Setting

The operator $L_s$ is studied as a bounded operator on Hardy spaces of
holomorphic functions on disks centered at $w = 1$:

$$
H^2(D_r) = \left\{ f(w) = \sum_{k=0}^{\infty} c_k (w-1)^k \;:\; \|f\|_{(r)}^2 = \sum_{k \geq 0} |c_k|^2 r^{2k} < \infty \right\}
$$

where $D_r = \{w : |w - 1| < r\}$.  The key property is that $L_s$ maps
$H^2(D_1)$ into $H^2(D_{3/2})$ (expanding the domain), and the inclusion
$H^2(D_{3/2}) \hookrightarrow H^2(D_1)$ is a strict contraction, making
$L_s : H^2(D_1) \to H^2(D_1)$ a compact operator.

The operator norm is bounded by (Theorem 3.1 in the reference):

$$
\|L_s\|_{H^2(D_1) \to H^2(D_{3/2})} \leq C_2(N)
= \sum_{n=1}^{N-1} \frac{\sqrt{2n+1}}{(n - 1/2)^2}
+ \sqrt{2 + \frac{2}{N - 1/2}} \cdot \zeta(3/2,\; N - 1/2),
$$

where $\zeta(s, a)$ is the Hurwitz zeta function.  For $N = 1000$ this gives
$C_2 \approx 10.058$.


## Basis and Galerkin Projection

We discretize $L_s$ in the **shifted monomial basis**

$$
\phi_k(w) = (w - 1)^k, \qquad k = 0, 1, \ldots, K.
$$

The Galerkin matrix $A_K$ is the $(K+1) \times (K+1)$ matrix whose entry
$(A_K)_{\ell, k}$ is the $\ell$-th Taylor coefficient of $L_s \phi_k$ at
$w = 1$:

$$
(L_s \phi_k)(w) = \sum_{\ell=0}^{\infty} (A_K)_{\ell, k} \, (w - 1)^\ell.
$$

This is an exact Galerkin projection: the monomial basis is orthogonal in
$H^2(D_r)$ (with weight $r^{2k}$), and the matrix entries are computed
analytically rather than by numerical quadrature.


## Closed-Form Matrix Entries

Each basis function $\phi_k(w) = (w-1)^k$ is expanded using the binomial theorem:

$$
(w - 1)^k = \sum_{j=0}^{k} \binom{k}{j} (-1)^{k-j} w^j.
$$

Applying $L_s$ to $w^j$ and using the relation to the Hurwitz zeta function gives

$$
(L_s \, w^j)(w) = \sum_{n=1}^{\infty} \frac{1}{(w+n)^{2s}} \cdot \frac{1}{(w+n)^j}
= \sum_{n=1}^{\infty} \frac{1}{(w+n)^{2s+j}}.
$$

The $\ell$-th Taylor coefficient at $w = 1$ of $\sum_{n=1}^{\infty}(w+n)^{-(2s+j)}$
involves the Pochhammer symbol (rising factorial) and the Hurwitz zeta evaluated at
$a = 2$:

$$
[(\cdot - 1)^\ell]\; \sum_{n=1}^{\infty} (w + n)^{-(2s+j)}
\;=\; \frac{(-1)^\ell}{\ell!} \, (2s+j)_\ell \; \zeta(2s + j + \ell,\; 2),
$$

where $(z)_\ell = z(z+1)\cdots(z+\ell-1)$ is the Pochhammer symbol and

$$
\zeta(s, a) = \sum_{n=0}^{\infty} \frac{1}{(n + a)^s}
$$

is the Hurwitz zeta function. The shift $a = 2$ (instead of $a = 1$) arises because
the branch maps $\tau_n$ are indexed from $n = 1$, so $(w + n)$ ranges from
$w + 1$ upward, and expanding at $w = 1$ gives $(1 + n)^{-(2s+j)} = (n+1)^{-(2s+j)}$
with $n \geq 1$, i.e., the Hurwitz zeta sum starts at $a = 2$.

Combining the binomial expansion with the Taylor coefficient formula yields
the **closed-form matrix entry**:

$$
\boxed{
(A_K)_{\ell, k} = (-1)^{k+\ell} \sum_{j=0}^{k} (-1)^j \binom{k}{j}
\frac{(2s+j)_\ell}{\ell!} \; \zeta(2s + j + \ell,\; 2).
}
$$


## Implementation: `gkw_matrix_direct_fast`

The function `gkw_matrix_direct_fast` in `src/GKWDiscretization.jl` assembles
$A_K$ using a three-stage algorithm that reduces the complexity from $O(K^4)$
(naive) to $O(K^3)$ Arb multiplications.

### Stage 1: Hurwitz Zeta Cache — $O(K)$ evaluations

The zeta values $\zeta(2s + t, 2)$ for $t = 0, 1, \ldots, 2K$ are precomputed
once and stored in a lookup table `ζtab[t+1]`.  This is the most expensive step
per evaluation (each call goes through libarb's `acb_hurwitz_zeta`), but there
are only $2K + 1$ of them.

### Stage 2: Pochhammer Table — $O(K^2)$ multiplications

The ratios $(2s + j)_\ell / \ell!$ are built via a recurrence rather than
recomputing each Pochhammer from scratch:

$$
R[j, 0] = 1, \qquad R[j, \ell+1] = R[j, \ell] \cdot \frac{2s + j + \ell}{\ell + 1}.
$$

The table `R[j+1, ℓ+1]` stores these values.  To mitigate rounding accumulation
over $K$ recurrence steps, the table is built at $P + 128$ extra bits of
precision and rounded to $P$ bits only when read during assembly.

### Stage 3: Matrix Assembly — $O(K^3)$ multiplications

Each entry is assembled by the inner sum:

$$
(A_K)_{\ell, k} = (-1)^{k+\ell} \sum_{j=0}^{k} (-1)^j \binom{k}{j} \;
R[j+1, \ell+1] \; \texttt{ζtab}[j + \ell + 1].
$$

The columns $k = 0, \ldots, K$ can optionally be assembled in parallel
(`threaded=true`), which is beneficial for $K \geq 256$.

### Rigorous Arithmetic

All computations use the **Arb library** (via ArbNumerics.jl), which tracks
error balls automatically.  Every matrix entry is a rigorous complex ball:
the midpoint is a high-precision approximation and the radius bounds all
rounding and truncation errors.  Typical radii at 512-bit precision are
$\sim 10^{-154}$; at 2048-bit precision, $\sim 10^{-617}$.


## Alternative: DFT-Based Construction

A second method, `build_Ls_matrix_arb`, computes the same matrix via boundary
sampling and discrete Fourier transform.  It uses the identity

$$
(L_s \phi_k)(w) = \sum_{j=0}^{k} (-1)^{k-j} \binom{k}{j} \, \zeta(2s+j,\; w+1)
$$

to evaluate $L_s \phi_k$ at $N$ equally spaced points on the circle $|w - 1| = 1$,
then extracts the first $K + 1$ Taylor coefficients via DFT.  This method is
useful for validation and for cases where $N \gg K$ provides spectral accuracy,
but is typically slower than the direct formula for large $K$.


## Truncation Error Bound

The finite matrix $A_K$ approximates the infinite-dimensional operator $L_s$ with
a rigorously bounded truncation error (Corollary 4.1):

$$
\|L_s - A_K\|_{H^2(D_1)} \leq \varepsilon_K = C_2 \cdot \left(\frac{2}{3}\right)^{K+1}.
$$

The factor $(2/3)^{K+1}$ comes from the compactness of $L_s$: the branch maps
$\tau_n$ map $D_1$ into the smaller disk $D_{2/3}$, so the approximation numbers
of $L_s$ decay exponentially at rate $2/3$.  Representative values:

| $K$ | $\varepsilon_K$ |
|----:|:----------------|
| 64  | $\sim 10^{-12}$ |
| 256 | $\sim 10^{-46}$ |
| 512 | $\sim 10^{-91}$ |
| 1024 | $\sim 10^{-181}$ |
| 2048 | $\sim 10^{-361}$ |

This exponential decay means that even moderate values of $K$ give extremely
accurate approximations of the spectrum.


## From Matrix to Spectrum: The Certification Pipeline

The discretized matrix $A_K$ is the starting point for the full certification
pipeline:

1. **Build $A_K$** via `gkw_matrix_direct_fast` in Arb arithmetic
   (rigorous error balls on every entry).

2. **Convert to BallMatrix** via `BallMatrix(BigFloat, M_arb)` for
   high-precision validated linear algebra (preserving all $P$ bits of
   the Arb midpoint in the BigFloat center).

3. **Schur decomposition** $A_K = Q T Q^*$ via GenericSchur.jl at
   $P$-bit BigFloat precision, with rigorous residual bound
   $\|A_K - Q T Q^*\|_2 \leq E$.

4. **Ordered Schur** to isolate individual eigenvalues, followed by
   direct triangular Sylvester solves for spectral coefficients
   $\ell_j(1) = e_1^* \Pi_j e_1$ (the projection of the constant function
   onto the $j$-th eigenspace).

5. **Resolvent certification** on contours in the complex plane to verify
   the small-gain condition
   $\alpha = \varepsilon_K \cdot \|R_{A_K}(z)\|_{(r)} < 1$, which lifts
   the finite-dimensional spectral data to the infinite-dimensional operator.

6. **Spectral expansion**: the iterate $L_s^n \cdot 1$ is rigorously decomposed as
   $$
   L_s^n \cdot 1 = \sum_{j=1}^{m} \lambda_j^n \, \ell_j(1) \, v_j + R_m^{(n)},
   $$
   where $\|R_m^{(n)}\|_{H^2} \leq C \cdot |\lambda_{m+1}|^n$ is the tail bound.


## Summary of Key Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `gkw_matrix_direct_fast` | `GKWDiscretization` | Build $(K{+}1) \times (K{+}1)$ matrix in $O(K^3)$ |
| `gkw_matrix_direct` | `GKWDiscretization` | Reference $O(K^4)$ implementation |
| `build_Ls_matrix_arb` | `GKWDiscretization` | DFT-based construction for validation |
| `hurwitz_zeta` | `ArbZeta` | Hurwitz $\zeta(s,a)$ via libarb |
| `compute_C2` | `Constants` | Operator norm bound $C_2(N)$ |
| `compute_Δ` | `Constants` | Truncation error $\varepsilon_K = C_2 (2/3)^{K+1}$ |
| `h2_whiten` | `Constants` | $H^2(D_r)$ norm whitening $C_r A C_r^{-1}$ |
