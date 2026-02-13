#!/usr/bin/env julia
# Generate LaTeX tables for spectral expansion from data/spectral_expansion.txt
# New format: ℓ_j(1) coefficients + eigenvector coefficients (50 per eigenvalue)

using Printf

# Parse the data file
ell_data = Dict{Int, NamedTuple{(:lambda, :ell_c, :ell_r, :norm_P1, :proj_norm, :idemp, :sep, :nk_rad),
                                 Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64}}}()
eigvec_coeffs = Dict{Int, Vector{Tuple{Int, Float64, Float64}}}()  # j => [(k, center, nk_radius), ...]

current_section = :none
for line in eachline("data/spectral_expansion.txt")
    stripped = strip(line)
    if startswith(stripped, '#')
        if occursin("SECTION: ell_j", stripped)
            global current_section = :ell
        elseif occursin("SECTION: eigenvector", stripped)
            global current_section = :eigvec
        end
        continue
    end
    isempty(stripped) && continue

    parts = split(stripped, '\t')
    if current_section == :ell && length(parts) >= 9
        j = parse(Int, parts[1])
        ell_data[j] = (lambda = parse(Float64, parts[2]),
                       ell_c = parse(Float64, parts[3]),
                       ell_r = parse(Float64, parts[4]),
                       norm_P1 = parse(Float64, parts[5]),
                       proj_norm = parse(Float64, parts[6]),
                       idemp = parse(Float64, parts[7]),
                       sep = parse(Float64, parts[8]),
                       nk_rad = parse(Float64, parts[9]))
    elseif current_section == :eigvec && length(parts) >= 5
        j = parse(Int, parts[1])
        k = parse(Int, parts[3])
        c = parse(Float64, parts[4])
        r = parse(Float64, parts[5])
        if !haskey(eigvec_coeffs, j)
            eigvec_coeffs[j] = Tuple{Int, Float64, Float64}[]
        end
        push!(eigvec_coeffs[j], (k, c, r))
    end
end

io = IOBuffer()

# Section header
println(io, raw"""
\clearpage
\section*{Rigorous Spectral Expansion}

Since all $20$ eigenvalues are certified to be simple (Theorem~\ref{thm:two-stage}),
each eigenvalue $\lambda_j$ has a rank-one Riesz projector
$P_j = v_j \otimes \ell_j$,
where $v_j$ is the right eigenvector ($\|v_j\|_2 = 1$) and $\ell_j$ the left (dual) eigenvector,
normalized so that $\ell_j(v_j) = 1$.
The spectral expansion of $L^n$ applied to the constant function $\mathbf{1}$ is
\begin{equation}
\label{eq:spectral-expansion}
L^n \mathbf{1} = \sum_{j=1}^{N} \lambda_j^n \, \ell_j(\mathbf{1}) \, v_j + R_N(n),
\end{equation}
where $\ell_j(\mathbf{1}) = \langle P_j(\mathbf{1}), v_j \rangle$ is the spectral coefficient
and $R_N(n)$ is bounded by $|\lambda_{N+1}|^n$ times the tail projector norm.

\medskip\noindent
\textbf{Computation of $\ell_j(\mathbf{1})$.}
The Riesz projector $P_j$ is computed rigorously at $K = 256$ using Sylvester-based
Schur enclosure (ball arithmetic).
The right eigenvector $v_j$ is the Schur eigenvector with $\|v_j\|_2 = 1$,
enclosed by the Newton--Kantorovich certification with radius $r_{\mathrm{NK}}$
(Table~\ref{tab:two-stage-results}).
The spectral coefficient is obtained via the rigorous inner product
$\ell_j(\mathbf{1}) = \langle P_j(\mathbf{1}), \hat{v}_j \rangle \pm \delta$,
where $\delta$ accounts for both the ball arithmetic radius and the NK
eigenvector error:
$\delta \leq r_{\mathrm{ball}} + \|P_j(\mathbf{1})\|_2 \cdot 2 r_{\mathrm{NK}} / (1 - r_{\mathrm{NK}})$.
""")

# Summary table with ℓ_j(1)
println(io, raw"""
\begin{table}[ht]
\centering
\caption{Rigorous spectral expansion coefficients $\ell_j(\mathbf{1})$ and projector diagnostics at $K = 256$.
The spectral coefficient $\ell_j(\mathbf{1})$ is computed as $\langle P_j(\mathbf{1}), v_j \rangle$
with certified error bounds.}
\label{tab:spectral-coefficients}
\begin{tabular}{rlllcc}
\toprule
$j$ & $\hat\lambda_j$ & $\ell_j(\mathbf{1})$ & $\pm\;\delta$ & $\|P_j\|$ & $\|P_j^2 - P_j\|$ \\
\midrule""")

for j in 1:20
    d = ell_data[j]
    λ_str = @sprintf("%.10f", d.lambda)
    ell_str = @sprintf("%+.10e", d.ell_c)
    r_str = @sprintf("%.2e", d.ell_r)
    p_str = @sprintf("%.4f", d.proj_norm)
    i_str = @sprintf("%.2e", d.idemp)
    println(io, "$j & \$$λ_str\$ & \$$ell_str\$ & \$$r_str\$ & \$$p_str\$ & \$$i_str\$ \\\\")
end

println(io, raw"""\bottomrule
\end{tabular}
\end{table}""")

# ── Eigenfunction figures ──────────────────────────────────────────────────
println(io, raw"""
\clearpage
\subsection*{Eigenfunctions}

The following figures show the eigenfunctions $v_j(x)$ evaluated on $[0.01, 0.99]$
in the shifted monomial basis $\{(x-1)^k\}_{k=0}^{K}$, with $K = 256$.

\begin{figure}[ht]
\centering
\includegraphics[width=\textwidth]{eigenfunctions_1_6.pdf}
\caption{Leading eigenfunctions $v_1, \ldots, v_6$. The invariant density $v_1$ (top left)
is positive; higher eigenfunctions oscillate with increasing frequency.
Each panel shows $\lambda_j$ and $\ell_j(\mathbf{1})$.}
\label{fig:eigenfunctions-1-6}
\end{figure}

\begin{figure}[ht]
\centering
\includegraphics[width=\textwidth]{eigenfunctions_7_12.pdf}
\caption{Higher eigenfunctions $v_7, \ldots, v_{12}$.
The corresponding eigenvalues $|\lambda_j|$ are of order $10^{-3}$--$10^{-5}$;
these modes decay rapidly under iteration.}
\label{fig:eigenfunctions-7-12}
\end{figure}

\begin{figure}[ht]
\centering
\includegraphics[width=0.85\textwidth]{eigenfunctions_7_12_overlay.pdf}
\caption{Higher eigenfunctions $v_7, \ldots, v_{12}$ overlaid on the same axes,
showing their relative magnitudes and the progressive decay of amplitude with
increasing index.}
\label{fig:eigenfunctions-7-12-overlay}
\end{figure}
""")

# ── Spectral approximant figures ──────────────────────────────────────────
println(io, raw"""
\begin{figure}[ht]
\centering
\includegraphics[width=0.85\textwidth]{spectral_approximant.pdf}
\caption{Spectral approximant $S_{20}(n, x) = \sum_{j=1}^{20} \lambda_j^n \, \ell_j(\mathbf{1}) \, v_j(x)$
for $n = 0, 1, 2, 5, 10$.  At $n = 0$ the sum reconstructs the constant function~$\mathbf{1}$;
at $n = 1$ it approximates the transfer operator applied to~$\mathbf{1}$;
for large~$n$ it converges to $\lambda_1^n \, \ell_1(\mathbf{1}) \, v_1(x)$.}
\label{fig:spectral-approximant}
\end{figure}

\begin{figure}[ht]
\centering
\includegraphics[width=0.85\textwidth]{spectral_convergence.pdf}
\caption{Convergence of $S_{20}(n, x) / \lambda_1^n$ to the invariant density
$\ell_1(\mathbf{1}) \, v_1(x)$ (dashed black).
The spectral gap $|\lambda_2 / \lambda_1| \approx 0.304$ governs the rate of convergence.}
\label{fig:spectral-convergence}
\end{figure}
""")

# Eigenvector coefficient tables (50 coefficients, 2 columns of 25)
println(io, raw"""
\clearpage
\subsection*{Eigenvector Coefficients $[v_j]_k$}

The following tables list the first $50$ coefficients of each unit-norm
eigenvector $v_j$ in the shifted monomial basis $\{(w-1)^k\}_{k=0}^{K}$.
Each coefficient is known to within $\pm\, r_{\mathrm{NK}}$
(Table~\ref{tab:two-stage-results}), where $r_{\mathrm{NK}}$
is the Newton--Kantorovich eigenpair enclosure radius.
""")

for j in 1:20
    d = ell_data[j]
    coeffs = sort(eigvec_coeffs[j], by=x->x[1])

    λ_str = @sprintf("%.6e", d.lambda)
    ell_str = @sprintf("%+.6e", d.ell_c)
    nk_str = @sprintf("%.2e", d.nk_rad)

    println(io, """
\\paragraph{\$j = $j\$: \$\\lambda_{$j} = $λ_str\$, \\quad \$\\ell_{$j}(\\mathbf{1}) = $ell_str\$, \\quad \$r_{\\mathrm{NK}} = $nk_str\$}
{\\footnotesize
\\begin{tabular}{r@{\\;\\;}l@{\\quad}r@{\\;\\;}l}
\\toprule
\$k\$ & \$[v_{$j}]_k\$ & \$k\$ & \$[v_{$j}]_k\$ \\\\
\\midrule""")

    nrows = 25
    for row in 0:nrows-1
        k1 = row
        k2 = row + nrows
        c1 = coeffs[k1+1][2]
        c2 = coeffs[k2+1][2]
        c1_str = @sprintf("%+.10e", c1)
        c2_str = @sprintf("%+.10e", c2)
        println(io, "$k1 & \$$c1_str\$ & $k2 & \$$c2_str\$ \\\\")
    end

    println(io, raw"""\bottomrule
\end{tabular}
}
""")
end

# ────────────────────────────────────────────────────────────────────────────
# Tail bound section (from data/tail_bound.txt)
# ────────────────────────────────────────────────────────────────────────────

if isfile("data/tail_bound.txt")
    # Parse tail bound data
    tail = Dict{String, Float64}()
    for line in eachline("data/tail_bound.txt")
        stripped = strip(line)
        startswith(stripped, '#') && continue
        isempty(stripped) && continue
        parts = split(stripped, '\t')
        length(parts) >= 2 || continue
        tail[parts[1]] = parse(Float64, parts[2])
    end

    ρ_eff = tail["rho_eff"]
    M_inf = tail["M_inf"]
    eps_K_tail = tail["eps_K"]
    alpha_tail = tail["alpha"]
    λ_N = tail["lambda_20"]
    λ_Np1 = tail["lambda_21"]
    resolvent_Ak = tail["resolvent_Ak"]
    circle_r = tail["circle_radius"]
    norm_Q_1 = get(tail, "norm_Q_N_1", 1.0)
    prefactor = get(tail, "prefactor", M_inf)

    println(io, raw"""
\clearpage
\subsection*{Tail Bound for the Spectral Remainder}

The spectral expansion~\eqref{eq:spectral-expansion} has remainder
\[
R_N(n) \;=\; L^n \mathbf{1} - \sum_{j=1}^{N} \lambda_j^n \, \ell_j(\mathbf{1}) \, v_j.
\]
We bound $\|R_N(n)\|_2$ by combining the Cauchy integral formula with a
direct computation of the \emph{tail projection} $Q_N \mathbf{1}$, where
$Q_N = I - \sum_{j=1}^N P_j$ is the tail spectral projector.

\medskip\noindent
\textbf{Factored tail bound.}
Since $Q_N$ commutes with $L_r$ (Riesz projectors commute with the operator,
regardless of normality), we have the factorization
\[
R_N(n) = Q_N L^n \mathbf{1} = (Q_N L^n)(Q_N \mathbf{1}),
\]
using only $Q_N^2 = Q_N$ (Riesz idempotency) and $L$-invariance of $\operatorname{range}(Q_N)$.
The Cauchy integral on a circle $\Gamma = \{|z| = \rho\}$ in the eigenvalue gap
$|\lambda_{N+1}| < \rho < |\lambda_N|$ bounds the operator factor:
\[
\|Q_N L^n\| = \left\|\frac{1}{2\pi i} \oint_\Gamma z^n R_{L_r}(z)\,dz\right\|
\leq \rho^{n+1} \cdot M_\infty,
\]
where $M_\infty = \sup_{z\in\Gamma} \|R_{L_r}(z)\|$ is obtained via the resolvent bridge.
The tail projection $\|Q_N \mathbf{1}\|_2 = \|\mathbf{1} - \sum_{j=1}^N P_j \mathbf{1}\|_2$
is computed directly from the rigorous Riesz projectors at $K=256$.

The combined bound is:
\[
\boxed{\|R_N(n)\|_2 \leq \rho^{n+1} \cdot M_\infty \cdot \|Q_N \mathbf{1}\|_2.}
\]""")

    # Tail bound parameter table
    ρ_str = @sprintf("%.6e", ρ_eff)
    M_str = @sprintf("%.4e", M_inf)
    eps_str = @sprintf("%.4e", eps_K_tail)
    α_str = @sprintf("%.4e", alpha_tail)
    res_str = @sprintf("%.4e", resolvent_Ak)
    cr_str = @sprintf("%.6e", circle_r)
    λN_str = @sprintf("%.6e", abs(λ_N))
    λNp1_str = @sprintf("%.6e", abs(λ_Np1))
    q1_str = @sprintf("%.6e", norm_Q_1)
    pf_str = @sprintf("%.4e", prefactor)

    println(io, """
\\begin{table}[ht]
\\centering
\\caption{Tail bound parameters for the spectral remainder \$R_{20}(n)\$.
Resolvent certified at \$K=64\$; tail projection \$\\|Q_{20}\\mathbf{1}\\|\$ computed at \$K=256\$.}
\\label{tab:tail-bound}
\\begin{tabular}{ll}
\\toprule
Parameter & Value \\\\
\\midrule
\$|\\lambda_{20}|\$ & \$$λN_str\$ \\\\
\$|\\lambda_{21}|\$ & \$$λNp1_str\$ \\\\
Circle radius \$\\rho\$ & \$$cr_str\$ \\\\
\$\\|R_{A_K}(z)\\|\$ on \$\\Gamma\$ & \$$res_str\$ \\\\
\$\\varepsilon_K\$ (at \$K=64\$) & \$$eps_str\$ \\\\
Small-gain \$\\alpha = \\varepsilon_K \\cdot \\|R_{A_K}\\|\$ & \$$α_str\$ \\\\
\$M_\\infty = \\|R_{L_r}\\|\$ on \$\\Gamma\$ & \$$M_str\$ \\\\
\$\\|Q_{20} \\mathbf{1}\\|_2\$ & \$$q1_str\$ \\\\
Prefactor \$C = M_\\infty \\cdot \\|Q_{20}\\mathbf{1}\\|\$ & \$$pf_str\$ \\\\
\\bottomrule
\\end{tabular}
\\end{table}""")

    # Sample tail bounds
    println(io, raw"""

\begin{table}[ht]
\centering
\caption{Rigorous upper bounds on $\|R_{20}(n)\|_2 \leq \rho^{n+1} \cdot M_\infty \cdot \|Q_{20}\mathbf{1}\|_2$.}
\label{tab:tail-bounds-sample}
\begin{tabular}{rl}
\toprule
$n$ & $\|R_{20}(n)\|_2 \leq$ \\
\midrule""")

    for nn in [1, 2, 5, 10, 20, 50]
        bound = ρ_eff^(nn + 1) * prefactor
        if bound == 0.0
            log_bound = (nn + 1) * log10(ρ_eff) + log10(prefactor)
            exp_str = @sprintf("%d", floor(Int, log_bound))
            println(io, "$nn & \$< 10^{$exp_str}\$ \\\\")
        else
            b_str = @sprintf("%.4e", bound)
            println(io, "$nn & \$$b_str\$ \\\\")
        end
    end

    println(io, raw"""\bottomrule
\end{tabular}
\end{table}""")
else
    @warn "data/tail_bound.txt not found, skipping tail bound section"
end

content = String(take!(io))

# Read existing base file (two-stage results without spectral section)
existing = read("data/two_stage_results.tex", String)

# Remove any existing spectral section (from \clearpage\n\section*{Rigorous Spectral to end of doc)
# and insert the new one
idx = findfirst("\\clearpage\n\\section*{Rigorous Spectral", existing)
if idx !== nothing
    # Remove old spectral section
    base = existing[1:idx.start-1]
    # Find the \end{document} that should remain
    new_content = base * content * "\n\\end{document}\n"
else
    # Insert before \end{document}
    new_content = replace(existing,
        "\n\\end{document}" => "\n" * content * "\n\\end{document}")
end

# Ensure \usepackage{graphicx} is in the preamble
if !occursin("\\usepackage{graphicx}", new_content)
    new_content = replace(new_content,
        "\\usepackage{booktabs}" => "\\usepackage{booktabs}\n\\usepackage{graphicx}")
end

write("data/two_stage_results.tex", new_content)
println("Updated data/two_stage_results.tex with spectral expansion tables")
