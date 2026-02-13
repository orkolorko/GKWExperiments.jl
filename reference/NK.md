# implementation.md
## Validated eigenpair enclosure for the Gauss transfer operator on a single Hardy space (defect-based NK/Krawczyk)

This document describes an implementation pipeline to **rigorously enclose a simple eigenvalue/eigenvector** of the Gauss transfer operator using a **single-space** Newton–Kantorovich/Krawczyk argument driven by a certified bound on the **defect**
\[
q := \|I - C\,DF\|
\]
rather than a (often pessimistic) global **resolvent bound** on a contour.

The setting is one Hilbert/Banach space:
\[
H := H^2(D_r),\qquad r\in\{1,\tfrac32\},
\]
and one bounded endomorphism \(L_r\in\mathcal B(H)\) (either \(L_1=SL\) or \(L_{3/2}=LR\)), approximated by a finite-rank Galerkin matrix \(A_k\).

---

## 0. Norm conventions and coordinates

### 0.1 Hardy inner product and whitening
On the monomial basis \(m_n(w)=(w-1)^n\), the Gram is diagonal:
\[
S_r = \mathrm{diag}(r^{2n}),\quad C_r = S_r^{1/2} = \mathrm{diag}(r^n).
\]
For a matrix \(M\) written in the monomial basis, the induced \(H^2(D_r)\) operator norm is
\[
\|M\|_{(r)} := \|C_r\,M\,C_r^{-1}\|_2.
\]

**Recommendation:** perform all linear algebra in the *whitened* coordinates
\[
\widetilde M := C_r\,M\,C_r^{-1},
\]
so that \(\|\widetilde M\|_2 = \|M\|_{(r)}\) and the inner product is Euclidean.

### 0.2 Product norm for the eigenpair map
We work on \(X:=\mathbb C\times \mathbb C^N\) and \(Y:=\mathbb C^N\times\mathbb C\) with
\[
\|(z,v)\| := |z| + \|v\|_2,\qquad \|(a,b)\| := \|a\|_2 + |b|.
\]
(Using whitening, this coincides with the \(H^2(D_r)\) norm on the finite-dimensional subspace.)

---

## 1. Discretization of \(L_r\)

### 1.1 Build a finite-rank approximation \(A_k\)
You already have one-sided bounds:
- for \(L_1\): \(A_k = (L_1)_K = \Pi_K\,L_1\) with \(\|L_1-A_k\|_{(1)} \le \varepsilon := C_2(2/3)^{K+1}\),
- for \(L_{3/2}\): \(A_k=(L_{3/2})_k\) with \(\|L_{3/2}-A_k\|_{(3/2)} \le \varepsilon := C_2(2/3)^{k+1}\).

**Output:** monomial-basis matrix `A_mono :: Matrix{T}` (T = Float64/BigFloat/Arb).

### 1.2 Whiten
Compute `C = Diagonal(r .^ (0:N-1))` and set
\[
\widetilde A := C\,A\,C^{-1}.
\]
All norms below are Euclidean 2-norms of whitened objects.

---

## 2. Eigenpair oracle and biorthogonal normalization

### 2.1 Numerical eigenpair
Compute a numerical eigenpair \((\lambda_A, v_A)\) of \(\widetilde A\) (Schur/eig):
- choose the target eigenvalue by modulus/ordering,
- normalize `v_A` to your preferred scale (see §6).

### 2.2 Left eigenvector for the functional \(\ell\)
Compute a left eigenvector \(u_A\) as eigenvector of \(\widetilde A^\ast\) for \(\overline{\lambda_A}\).

Normalize biorthogonally:
\[
u_A^\ast v_A = 1.
\]
Define the linear functional
\[
\ell(v) := u_A^\ast v.
\]

---

## 3. Nonlinear map and Jacobian

### 3.1 Eigenpair map
For a bounded operator \(M\) on \(H\), define
\[
F_M(z,v) := \big(Mv - z v,\ \ell(v)-1\big).
\]

### 3.2 Jacobian in whitened coordinates
At \((z,v)\), the derivative is
\[
DF_M(z,v)[\delta z,\delta v]
=
\big((M-zI)\delta v - \delta z\,v,\ \ell(\delta v)\big).
\]

As a block matrix acting on \((\delta z,\delta v)\in\mathbb C\times\mathbb C^N\):
\[
J_M(z,v)
=
\begin{pmatrix}
-v & M-zI\\
0  & u^\ast
\end{pmatrix}
\quad\in\mathbb C^{(N+1)\times(N+1)}.
\]

**Implementation:** assemble `Jk = J_{A}(λ_A,v_A)` using `A = Ã` and `u = u_A`.

---

## 4. Choose a preconditioner \(C\) and certify the discrete defect \(q_k\)

### 4.1 Preconditioner choices

**Option A (recommended): approximate inverse of the Jacobian).**
Compute a floating approximate inverse
\[
C \approx J_k^{-1}
\]
via LU/QR in high precision.

**Option B (structured inverse).**
Use the explicit inverse formula at a *simple* eigenpair if you can stably implement the reduced inverse on the complement.
(Usually not necessary if Option A is certified well.)

### 4.2 Discrete defect
Compute
\[
R_k := I - C\,J_k,\qquad q_k := \|R_k\|_2.
\]
This is finite-dimensional and should be **small** if \(C\) is a good inverse.

### 4.3 Rigorous upper bound on \(\|R_k\|_2\)
You need a certified upper bound `qk_ub`.

Common strategies:
- **Interval/ball residual:** compute an enclosure of `R_k` in ball arithmetic, then upper bound the spectral norm.
- **Safe bound via \(\infty\)-norm:** use \(\|R\|_2 \le \sqrt{(N+1)}\,\|R\|_\infty\) (coarse but robust).
- **Power-method with enclosure** on \(R^\ast R\) (tighter; more work).
- Any existing certified 2-norm routine you already use (Miyajima-style, Rump-style, etc.).

Also certify
\[
\|C\|_2 \le \mathrm{C\_ub}.
\]

---

## 5. Transfer bounds from \(A_k\) to \(L_r\) (single space, no cross norms)

### 5.1 Operator discretization error
From truncation theory you have
\[
\varepsilon := \|L_r - A_k\|_{(r)} = \|\,\widetilde L_r - \widetilde A\,\|_2
\]
as an *a priori* bound (no need to build \(\widetilde L_r\) explicitly).

### 5.2 Lipschitz facts you exploit
At a fixed point \((z_0,v_0)\):
- **Dependence on the operator:** \(\|DF_{L_r}(z_0,v_0)-DF_{A_k}(z_0,v_0)\| = \|L_r-A_k\| = \varepsilon\).
- **Dependence on \((z,v)\):** for the chosen product norm,
  \[
  \|DF_{L_r}(x)-DF_{L_r}(x_0)\| \le \|x-x_0\|.
  \]

### 5.3 Defect at the base point
Define
\[
q_0 := \|I - C\,DF_{L_r}(x_0)\|.
\]
Then
\[
\boxed{q_0 \ \le\ q_k \ +\ \|C\|\,\varepsilon.}
\]
In validated form:
\[
q0\_ub := qk\_ub + C\_ub*\varepsilon.
\]

### 5.4 Residual size
At the discrete eigenpair oracle \(x_0=(\lambda_A,v_A)\),
\[
F_{L_r}(x_0) = ((L_r-A_k)v_A,\ 0).
\]
Thus
\[
\boxed{
y := \|C F_{L_r}(x_0)\|
\ \le\ \|C\|\,\varepsilon\,\|v_A\|_2.
}
\]
Validated:
\[
y\_ub := C\_ub * \varepsilon * \|v_A\|_2.
\]

---

## 6. Newton–Kantorovich/Krawczyk radius from a quadratic (small ball)

Define the Krawczyk/Newton map:
\[
K(x)=x - C\,F_{L_r}(x).
\]
On a ball \(B(x_0,r)\), the derivative bound satisfies
\[
\sup_{x\in B(x_0,r)}\|DK(x)\|
= \sup_{x\in B}\|I - C\,DF_{L_r}(x)\|
\le q_0 + \|C\|\,r.
\]
Let
\[
q(r) := q_0 + \|C\|\,r.
\]

### 6.1 Conditions
You want:
1. **Contraction:** \(q(r) < 1\),
2. **Invariance:** \(y \le (1-q(r))\,r\).

Using \(q(r)=q_0+\|C\|r\), invariance is equivalent to
\[
\|C\|\,r^2 + (q_0-1)\,r + y \le 0.
\]
This holds iff the discriminant is nonnegative:
\[
\boxed{(1-q_0)^2 \ \ge\ 4\,\|C\|\,y.}
\]

### 6.2 Explicit radius
Assuming \(q_0<1\) and the discriminant condition,
\[
\boxed{
r
=
\frac{(1-q_0)-\sqrt{(1-q_0)^2 - 4\|C\|y}}{2\|C\|}.
}
\]
Then \(K\) maps \(B(x_0,r)\) into itself and is a contraction there, hence there is a **unique**
true eigenpair \(x_\ast=(\lambda_\ast,v_\ast)\) with \(\ell(v_\ast)=1\) in that ball.

### 6.3 Practical scaling tip
If you normalize \(v_A\) so that \(\|v_A\|_2 \approx 1\), then \(y\) shrinks proportionally.
(Your constraint \(\ell(v)=1\) fixes the “phase/scale” in the Newton problem; the *numerical* size of \(v_A\) still affects the residual bound \(y\).)

---

## 7. End-to-end algorithm (pseudocode)

```text
Input: r ∈ {1, 3/2}, dimension N, truncation index k (or K), bound C2, target eigen-index j
Output: certified ball B((λ_A,v_A), r_NK) containing unique true eigenpair

1. Build finite-rank matrix A_mono approximating L_r on span{(w-1)^n}_{n=0}^{N-1}.
2. Whiten: Ã = C_r * A_mono * C_r^{-1}.
3. Compute numerical right eigenpair (λ_A, v_A) of Ã.
4. Compute numerical left eigenvector u_A of Ã* for conj(λ_A).
5. Normalize u_A* v_A = 1. Define ℓ(v)=u_A* v.

6. Assemble Jacobian Jk = [[-v_A, (Ã-λ_A I)]; [0, u_A*]].
7. Choose preconditioner C ≈ inv(Jk) (floating).
8. Certify:
     qk_ub ≥ ||I - C*Jk||_2
     C_ub  ≥ ||C||_2

9. Set ε = C2*(2/3)^(k+1)  (or K+1 depending on L1/L3/2 construction).
10. Compute:
     q0_ub = qk_ub + C_ub*ε
     y_ub  = C_ub*ε*||v_A||_2

11. Check:
     q0_ub < 1
     (1 - q0_ub)^2 ≥ 4*C_ub*y_ub
12. Set:
     r_NK = ((1-q0_ub) - sqrt((1-q0_ub)^2 - 4*C_ub*y_ub)) / (2*C_ub)

Return: enclosure ball radius r_NK and eigenpair oracle (λ_A, v_A).
