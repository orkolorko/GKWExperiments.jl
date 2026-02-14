# Schur Direct Resolvent Certification

## Summary

`certify_eigenvalue_schur_direct` certifies eigenvalues of the GKW
transfer operator $L_r : H^2(D_r) \to H^2(D_r)$ via its finite
Galerkin approximation $A_K$, using a **direct resolvent bound** in the
original $\lambda$-space. There is **no polynomial**, **no rescaling**,
**no deflation**, and **no ordschur**.

## Mathematical Setup

**Goal**: prove that $\lambda_j \in \sigma(L_r)$ and is simple, with a
rigorous inclusion radius.

**Key inequality** (Lemma 4 of `reference/Gauss.tex`): if
$$
\varepsilon_K \cdot \|(zI - A_K)^{-1}\| < 1 \quad \text{for all } z \in \Gamma,
$$
then every $z$ on the contour $\Gamma$ belongs to $\rho(L_r)$ (resolvent
set of $L_r$). Here $\varepsilon_K = \|L_r - A_K\|_{(r)}$ is the
truncation error from Corollary 4.1.

**Contour**: $\Gamma$ is a circle of radius $\rho$ centered at
$\lambda_j$ (the target eigenvalue). If the small-gain condition
$\alpha = \varepsilon_K \cdot \max_{z \in \Gamma} \|(zI - A_K)^{-1}\| < 1$
holds, then $\Gamma \subset \rho(L_r)$, proving that $L_r$ has exactly
one eigenvalue inside $\Gamma$ (since $A_K$ has exactly one there).

The **eigenvalue inclusion radius** is simply $\rho$ (no backmap needed).

## Computing $\|(zI - A_K)^{-1}\|$ via Block Schur

### Problem
$A_K$ is a 257x257 matrix. The target eigenvalue $|\lambda_j|$ can be as
small as $10^{-20}$, while $\|A_K\| \approx 1$. Direct svdbox on
$(zI - A_K)$ fails because the condition number exceeds $10^{16}$.

### Solution: BigFloat Schur (natural ordering)

1. **Schur decomposition in BigFloat**: $A_K = Q T Q^*$ where $T$ is
   upper triangular with eigenvalues on the diagonal. Using GenericSchur.jl,
   $Q$ is orthogonal to $\sim 10^{-153}$ precision. The Schur residual
   $\|A_K - QTQ^*\|$ is computed rigorously by `compute_schur_and_error`.

2. **Natural ordering**: GenericSchur.jl sorts eigenvalues on the diagonal
   by decreasing magnitude: $|\lambda_1| \geq |\lambda_2| \geq \cdots$.
   This means the block split at position $p$ (the target eigenvalue's
   diagonal position) gives the correct structure **without ordschur**:
   $$
   T = \begin{pmatrix} T_{11} & T_{12} \\ 0 & T_{22} \end{pmatrix}
   $$
   where $T_{11} = T[1\!:\!p\!-\!1,\, 1\!:\!p\!-\!1]$ contains the
   $p-1$ largest eigenvalues, $T_{22} = T[p\!:\!n,\, p\!:\!n]$ contains
   the target and all smaller eigenvalues.

   **Why no ordschur?** ordschur performs Givens rotations in BigFloat,
   introducing rounding errors that are not tracked by
   `compute_schur_and_error`. By using the natural diagonal order, we
   avoid this entirely: the original Schur error bounds apply directly.

3. **Block triangular inversion**: for upper block triangular matrices,
   $$
   (zI - T)^{-1} = \begin{pmatrix}
   (zI - T_{11})^{-1} & (zI - T_{11})^{-1}\, T_{12}\, (zI - T_{22})^{-1} \\
   0 & (zI - T_{22})^{-1}
   \end{pmatrix}
   $$
   This gives the norm bound:
   $$
   \|(zI - T)^{-1}\|_2 \leq \frac{1}{\sigma_{\min}(zI - T_{11})} \left(1 + \frac{\|T_{12}\|_F}{\sigma_{\min}(zI - T_{22})}\right) + \frac{1}{\sigma_{\min}(zI - T_{22})}
   $$

4. **Schur similarity**: $A_K = Q T Q^*$ where $\|Q\|$ and $\|Q^{-1}\|$
   are rigorously bounded by `compute_schur_and_error`:
   $$
   \|(zI - A_K)^{-1}\|_2 \leq \|Q\| \cdot \|(zI - T)^{-1}\|_2 \cdot \|Q^{-1}\|.
   $$

### Why the blocks are well-conditioned

- **$T_{11}$ block** ($k \times k$, eigenvalues $\lambda_1, \ldots, \lambda_{p-1}$):
  all eigenvalues satisfy $|\lambda_i| \gg |\lambda_p|$, so
  $\sigma_{\min}(zI - T_{11})$ is large (distance from $z \approx \lambda_p$ to
  $\sigma(T_{11})$). Computed via **GenericLinearAlgebra's native BigFloat SVD**
  with Miyajima M1 certification.

- **$T_{22}$ block** ($m \times m$, eigenvalues $\lambda_p, \ldots, \lambda_K$):
  all entries are $O(|\lambda_p|)$, so the condition number is moderate.
  Computed via standard Float64 svdbox.

- **$T_{12}$ block**: coupling between the two blocks, with
  $\|T_{12}\|_F \approx 0.07$--$0.13$ (small due to Schur structure).

### Weyl propagation for $T_{11}$

To avoid computing 256 BigFloat SVDs (one per circle sample), we compute
$\sigma_{\min}(z_0 I - T_{11})$ **once** at the circle center $z_0 = \lambda_p$,
then propagate to all circle points using Weyl's perturbation theorem:
$$
\sigma_{\min}(zI - T_{11}) \geq \sigma_{\min}(z_0 I - T_{11}) - |z - z_0|
= \sigma_{\min}(z_0 I - T_{11}) - \rho.
$$
Since $\sigma_{\min}(z_0 I - T_{11}) \gg \rho$, this is very tight.

## Algorithm Steps

```
Input: A_K (Float64 BallMatrix), p (Schur diagonal position of target),
       schur_data_bf (cached BigFloat Schur from compute_schur_and_error)

1. Read target eigenvalue from diag(T)[p]
2. Verify ordering: |T[i,i]| >= |T[p,p]| for all i < p
3. Extract blocks: T_11 = T[1:p-1, 1:p-1], T_12 = T[1:p-1, p:end], T_22 = T[p:end, p:end]
4. Convert T_22 to Float64 BallMatrix (small entries, well-conditioned)
5. Compute ||T_12||_F (BigFloat -> Float64 upper bound)
6. Auto-set circle radius rho = (dist to nearest eigenvalue) / 2
7. sigma_min(z0*I - T_11) via BigFloat SVD (GenericLinearAlgebra + Miyajima M1)
   -> propagate to circle: sigma_11 = sigma_min - rho
8. For each z on circle:
   a. sigma_min(z*I - T_22) via Float64 svdbox
   b. Block formula: res_z = (1/sigma_11)*(1 + ||T_12||/sigma_22) + 1/sigma_22
   c. Track max_resolvent
9. Schur bridge: M_r = ||Q|| * max_resolvent * ||Q^{-1}||
10. Small-gain: alpha = eps_K * M_r < 1 => certified
11. Eigenvalue radius = rho (direct, no backmap)
```

## Error Budget (rigorous)

All error sources are tracked:

| Error source | How it is bounded | Value (K=256) |
|---|---|---|
| Truncation $\varepsilon_K = \|L_r - A_K\|$ | `compute_Δ(K)` (Corollary 4.1) | $5.6 \times 10^{-45}$ |
| Schur residual $\|A_K - QTQ^*\|$ | `compute_schur_and_error` | $\sim 10^{-84}$ |
| $\|Q\|$, $\|Q^{-1}\|$ | `compute_schur_and_error` | $\approx 1.0$ |
| $\sigma_{\min}(z_0 I - T_{11})$ | GenericLinearAlgebra SVD + Miyajima M1 | $10^{-20}$ -- $10^{-9}$ |
| $\sigma_{\min}(zI - T_{22})$ | Float64 svdbox (rigorous enclosure) | $10^{-21}$ -- $10^{-18}$ |
| $\|T_{12}\|_F$ | BigFloat → Float64 upper bound | $0.07$ -- $0.13$ |
| Weyl propagation | $\sigma_{\min}(zI-T_{11}) \geq \sigma_{\min}(z_0 I-T_{11}) - \rho$ | exact inequality |
| ordschur rounding | **none** (ordschur is not used) | $0$ |

## Typical Numbers (K=256)

| Quantity | Value | Notes |
|----------|-------|-------|
| $\varepsilon_K$ | $5.6 \times 10^{-45}$ | Truncation error at K=256 |
| $\|Q\|_2 \cdot \|Q^{-1}\|_2$ | $\approx 1.0$ | Rigorously bounded |
| $\|T_{12}\|_F$ | 0.07--0.13 | Small due to Schur structure |
| $\sigma_{\min}(z_0 I - T_{11})$ | $10^{-20}$--$10^{-9}$ | Distance to nearest T_11 eigenvalue |
| $\sigma_{\min}(zI - T_{22})$ | $10^{-21}$--$10^{-18}$ | Eigenvalue separation in T_22 |
| $M_r$ | $10^{24}$--$10^{30}$ | Resolvent bound |
| $\alpha = \varepsilon_K \cdot M_r$ | $10^{-20}$--$10^{-14}$ | << 1, always certified |

## Key Insight

The approach works because $\varepsilon_K \approx 10^{-45}$ at K=256 is
**astronomically small**. Even with $M_r \sim 10^{30}$ (large resolvent
due to tiny eigenvalue separation), the product $\alpha \sim 10^{-15}$
is still far below 1. This is fundamentally why direct resolvent
certification in the original $\lambda$-space succeeds without any
polynomial manipulation.
