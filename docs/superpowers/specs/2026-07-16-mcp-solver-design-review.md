# Review of 2026-07-16-mcp-solver-design.md

I have carefully reviewed your design for the pure-Python, JAX-based PATH solver. The plan demonstrates a strong theoretical understanding of the 1993 PATH algorithm and the mathematics of mixed complementarity problems. 

However, translating this 30-year-old algorithmic design into a modern pure-Python/JAX ecosystem—especially targeting 5000-variable CGE models—exposes several critical weaknesses. Here is a skeptical breakdown of where this implementation is most likely to fail or hit severe bottlenecks:

### 1. The Stage 1 Domain Hazard (Fischer-Burmeister vs. CGE Models)
The design posits the Stage 1 Semismooth Newton solver as an easy-to-build "reference oracle." **This solver is highly likely to fail immediately on real CGE models due to domain errors.**
* **The Problem:** CGE models heavily feature equations that are mathematically undefined for negative numbers (e.g., CES/CET functions involving $x^{\sigma-1}$, or $\log(x)$). 
* **The Weakness:** Stage 2's normal map evaluates $f$ exclusively at projected points $\pi_B(x)$, keeping inputs safely within bounds (e.g., $z \ge 0$). However, the Stage 1 Fischer-Burmeister formulation $\Phi(z) = 0$ is unconstrained. The line search algorithm will routinely propose Newton steps $z_k + \alpha d$ that violate bounds (e.g., negative prices). JAX will return `NaN` arrays, the merit function $\Psi$ becomes `NaN`, and the line search will shrink to zero and stall.
* **Verdict:** Stage 1 will not work as a reliable fallback for real CGE models unless you implement a damped/interior method, which defeats the "fast to build" objective.

### 2. JAX Memory Exhaustion (`jacfwd` at N=5000)
The assumption that dense JAX Jacobians will scale to 5000 variables without issue is dangerous.
* **The Problem:** JAX's `jacfwd` computes the Jacobian using forward-mode AD by pushing a `vmap` (batching) over the input dimension $N$. For $N=5000$, JAX will instantiate a batch dimension of 5000 for *every intermediate array* in the CGE evaluation graph. 
* **The Weakness:** CGE models are computationally dense. A batch size of 5000 on a large compute graph will almost certainly cause a catastrophic Out-Of-Memory (OOM) error on standard hardware, completely locking up the solver before the first iteration finishes.
* **Verdict:** You cannot rely on naive dense `jacfwd` for $N > 1000$. You will need to implement chunked/sequential Jacobian generation (e.g., using `jax.lax.map` over the basis vectors, or a chunked `vmap`) to trade time for memory, or bite the bullet on coloring/sparse AD earlier than planned.

### 3. Numerical Annihilation in Levenberg-Marquardt
In Stage 1, the fallback step is explicitly defined as solving the normal equations: $(H^T H + \mu I) d = -H^T \Phi$.
* **The Problem:** CGE models are notorious for being wildly poorly scaled (e.g., quantities in billions, prices around 1.0), leading to highly ill-conditioned Jacobians.
* **The Weakness:** Explicitly forming $H^T H$ *squares* the condition number of $H$. If your CGE Jacobian has a condition number of $10^8$ (very common), $H^T H$ will have a condition number of $10^{16}$, completely destroying all significant digits in float64 precision. The fallback step will yield pure numerical noise.
* **Verdict:** You must avoid explicitly forming $H^T H$. Instead, solve the equivalent regularized least-squares problem directly using a QR factorization or SVD: $\min_d \left\| \begin{bmatrix} H \\ \sqrt{\mu}I \end{bmatrix} d - \begin{bmatrix} -\Phi \\ 0 \end{bmatrix} \right\|_2$.

### 4. Pure Python LU Product-Form Updates
Algorithm PATH requires Lemke-style complementary pivoting, which the design correctly notes needs rank-1 updates to avoid $O(N^3)$ refactorization per pivot.
* **The Problem:** SciPy/NumPy has *no built-in dense rank-1 LU update routines* (like Bartels-Golub or Forrest-Tomlin). 
* **The Weakness:** You will be forced to maintain the Product Form of the Inverse (PFI) using eta-matrices, likely stored in pure Python lists. While computationally $O(N^2)$, the Python interpreter overhead of looping through up to 50 eta-vectors for every forward/backward solve (which happens multiple times per pivot for ratio tests) might become a noticeable bottleneck. Furthermore, PFI without sophisticated numerical stabilization is notoriously unstable compared to direct LU updates. If the basis becomes ill-conditioned, the accumulated roundoff error over 50 pivots might cause the path to diverge.
* **Verdict:** Ensure you have strict accuracy monitoring (checking $\|Bz - b\|$) and dynamically trigger refactorizations earlier than 50 pivots if accuracy degrades.

### 5. Lexicographic Ratio Tests vs. PFI
The design specifies a "lexicographic-style ratio test" for resolving degeneracy.
* **The Problem:** True lexicographic pivoting requires inspecting rows of the inverse basis matrix $B^{-1}$. 
* **The Weakness:** When using product-form updates (eta-matrices), you do not have $B^{-1}$ explicitly available. Extracting a single row of $B^{-1}$ requires a backward solve against a unit vector ($O(N^2)$). In highly degenerate CGE models, resolving a tie might require checking multiple rows, resulting in multiple $O(N^2)$ solves per pivot purely for tie-breaking.
* **Verdict:** Consider using a small, deterministic random perturbation of the bounds (a standard, robust alternative to lexicographic rules) to resolve degeneracy. It is vastly cheaper computationally when using implicit inverses and perfectly sufficient for empirical economic models.

### Summary Recommendation
The algorithmic architecture is structurally sound, but the "pure Python + JAX" constraint creates severe friction for both the numerical linear algebra and AD scaling. I recommend:
1. **Ditching Stage 1** as a reference solver. It will likely waste your time due to out-of-bounds evaluation errors on economics equations. Use a trivial known-good Python solver on a 5-variable problem to test your PATH pivoting logic instead.
2. **Chunking the AD:** Write a wrapper around `jacfwd` that batches the $N$ directions in chunks of ~100-250 to strictly control memory.
3. **Using Perturbation:** Avoid lexicographic pivoting to keep your $O(N^2)$ Python pivot loop as lightweight as possible.
