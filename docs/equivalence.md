# Aggregate Computing ↔ Graph Neural Networks: A Formal Isomorphism

**Abstract.**
We establish a formal isomorphism between Aggregate Computing (AC) programs and
Message-Passing Neural Networks (MPNNs). We show that the four core AC combinators
— `rep`, `nbr`, `branch`, and `mux` — correspond precisely to per-node recurrent
cells, MPNN aggregate steps, dynamic subgraph partitioning with state reset, and
pointwise gating, respectively. The composition `rep` ∘ `nbr` is isomorphic to a
weight-tied Recurrent MPNN, where $T$ rounds of the AC program correspond to $T$
applications of a single shared-weight GNN layer. We provide formal definitions,
state and prove the correspondence theorems with proof sketches, characterise
differentiability through soft relaxations of discrete operators, and validate
the framework against a reference implementation.

---

## 1. Introduction

Aggregate Computing (AC) [Viroli et al., 2019; Beal et al., 2015] is a
paradigm for programming collective adaptive systems. Programs are written
against a set of *combinators* — `rep`, `nbr`, `branch`, `mux` — that
abstract over the distributed execution on a network of devices. Graph Neural
Networks (GNNs) [Gilmer et al., 2017; Scarselli et al., 2009] are a family
of neural architectures that operate on graph-structured data via message
passing. Despite arising from different communities, the two frameworks share
a deep structural similarity: both process information on a graph through
local, iterative, neighbor-to-neighbor communication.

**Contributions.** We make this connection precise:

1. We formalise AC execution as an event structure and show its isomorphism
   to a space-time unrolled graph (Proposition 1).
2. We prove five construct-level isomorphisms (Theorems 1–5), mapping each
   AC combinator to its GNN counterpart.
3. We characterise the differentiability of the resulting pipeline through
   soft relaxations of `min`, `max`, and edge masking (Propositions 2–4).
4. We provide a reference implementation whose API is aligned with the
   formal objects introduced here (Table 1).

**Organisation.** Section 2 introduces notation, assumptions, and
definitions. Section 3 formalises event-structure semantics. Section 4
states and proves the main isomorphism theorems. Section 5 treats
differentiability. Section 6 discusses fixed-point convergence.
Section 7 presents a worked example (the gradient algorithm).
Section 8 describes the implementation architecture and provides
the formal-to-code mapping. Section 9 concludes with future directions.

### 1.1 Related Work

**Aggregate Computing.** The field calculus [Viroli et al., 2019] provides
the theoretical foundation; its core combinators (`rep`, `nbr`, `branch`)
have well-studied denotational semantics based on computational fields.

**Message-Passing Neural Networks.** The MPNN framework [Gilmer et al.,
2017] unifies many GNN architectures (GCN, GAT, GraphSAGE) under a
common message-pass-then-update scheme. Recurrent variants [Li et al.,
2016] share weights across layers, mirroring iterative algorithms.

**Implicit layers and fixed points.** Deep Equilibrium Models [Bai et al.,
2019] and implicit GNNs [Gu et al., 2020] define the output as the fixed
point of a contractive map, connecting to the convergence behaviour of AC
self-stabilising algorithms.

---

## 2. Preliminaries

### 2.1 Network Model

**Assumption 1** (Synchronous execution). *All devices fire in lock-step
at discrete rounds $t = 0, 1, 2, \ldots$ Every device executes the same
program, observes the exports of its neighbours from the previous round,
and broadcasts its own export at the end of the round.*

> *Remark.* In the general AC model devices fire asynchronously. The
> synchronous simplification adopted here is standard in the GNN
> literature, where layers are applied to all nodes simultaneously.
> Extending the results to the fully asynchronous case — which requires
> modelling stale messages and per-node clocks — is left as future work.

**Definition 1** (Network graph). *A* network graph *is an undirected
graph $G = (V, E)$ where $V = \{1, \ldots, N\}$ is the set of devices
(nodes) and $E \subseteq V \times V$ is the set of communication links
(edges). Self-loops $(i, i) \in E$ are permitted. The open neighbourhood
of node $i$ is $\mathcal{N}(i) = \{j \mid (j, i) \in E\}$.*

**Definition 2** (Computational field). *A* computational field *over
$G$ is a mapping $\phi : V \to \mathbb{R}^d$ that assigns a
$d$-dimensional feature vector to every device. We write $\phi_i \in
\mathbb{R}^d$ for the value at device $i$. When $d = 1$, we identify
$\phi$ with a vector in $\mathbb{R}^N$.*

### 2.2 Aggregate Computing Combinators

We now define the four core AC combinators. Each definition states the
mathematical semantics; the corresponding implementation signatures are
collected in Table 1 (Section 8).

**Definition 3** (rep — temporal evolution). *Let $\text{name}$ be a
unique state identifier, let $s^{(0)} \in \mathbb{R}^d$ be an initial
value, and let $f : \mathbb{R}^d \to \mathbb{R}^d$ be an update function.
The combinator* $\operatorname{rep}(\text{name},\; s^{(0)},\; f)$
*defines a per-node recurrent state:*

$$s_i^{(t)} = f\!\left(s_i^{(t-1)}\right), \qquad s_i^{(0)} = s^{(0)}, \qquad \forall\, i \in V,\; t \geq 1.$$

*The identifier $\text{name}$ is used by the runtime to store and retrieve
the state across rounds.*

> *Remark.* The update function $f$ may itself invoke other combinators
> (notably `nbr`), in which case the recurrence incorporates neighbourhood
> information (see Theorem 3).

**Definition 4** (nbr — neighbourhood aggregation). *Let $\phi$ be a
computational field (the* expression*), let $\varphi : \mathbb{R}^d \to
\mathbb{R}^{d'}$ be an optional message transform (defaulting to the
identity), and let $\bigoplus$ be a permutation-invariant aggregation
operator. The combinator* $\operatorname{nbr}(\phi,\; \bigoplus)$
*computes, at each device $i$:*

$$m_i = \bigoplus_{j \in \mathcal{N}(i)} \varphi(\phi_j)$$

*where $\bigoplus \in \{\operatorname{sum}, \operatorname{mean},
\min, \max\}$ or a learned aggregator (see Section 5). Optionally, each
device may also broadcast $\phi_i$ under a named* tag *for use by
other combinators in the same round.*

**Definition 5** (branch — domain restriction with isolation). *Let
$c \in \{0, 1\}^N$ be a Boolean condition field and let $p_{\top},
p_{\bot}$ be two sub-programs. The combinator*
$\operatorname{branch}(c,\; p_{\top},\; p_{\bot})$ *partitions the
graph and executes each sub-program in isolation:*

1. *Define the partition edge sets:*
$$E_k = \{(i, j) \in E \mid c_i = k \wedge c_j = k\}, \quad k \in \{0, 1\}.$$

2. *Execute $p_{\top}$ on the induced subgraph $G_1 = (V_1, E_1)$ and
   $p_{\bot}$ on $G_0 = (V_0, E_0)$, where $V_k = \{i \in V \mid c_i = k\}$.*

3. *Combine:*
$$y_i = \begin{cases} p_{\top}(x_i,\, G_1) & \text{if } c_i = 1, \\ p_{\bot}(x_i,\, G_0) & \text{if } c_i = 0. \end{cases}$$

*Critical properties:*
- *Messages do not cross partitions.*
- *When a device switches branches ($c_i^{(t)} \neq c_i^{(t-1)}$), its
  named* rep *states are reset to their initial values (see Definition 11).*

**Definition 6** (mux — pointwise conditional selection). *Let $c \in
[0, 1]^N$ be a condition field and let $e_1, e_2$ be two expressions.
The combinator* $\operatorname{mux}(c,\; e_1,\; e_2)$ *evaluates both
expressions on the full graph and selects per node:*

$$y_i = \begin{cases} (e_1)_i & \text{if } c_i \geq 0.5, \\ (e_2)_i & \text{if } c_i < 0.5. \end{cases}$$

*Unlike* branch, *no topology change occurs and no state is reset.
Both $e_1$ and $e_2$ observe the complete neighbourhood.*

### 2.3 Message-Passing Neural Networks

**Definition 7** (MPNN). *A Message-Passing Neural Network [Gilmer et
al., 2017] on a graph $G = (V, E)$ computes, at each layer $\ell$:*

$$h_i^{(\ell)} = \gamma^{(\ell)}\!\left(h_i^{(\ell-1)},\; \bigoplus_{j \in \mathcal{N}(i)} \phi^{(\ell)}\!\left(h_i^{(\ell-1)},\, h_j^{(\ell-1)}\right)\right)$$

*where $\phi^{(\ell)}$ is the message function, $\bigoplus$ is a
permutation-invariant aggregation, and $\gamma^{(\ell)}$ is the update
function.*

**Definition 8** (Recurrent MPNN). *A Recurrent MPNN is an MPNN in which
all layers share the same parameters: $\phi^{(\ell)} = \phi_\theta$ and
$\gamma^{(\ell)} = \gamma_\theta$ for all $\ell$. Running $T$ layers is
equivalent to applying a single parametric map $T$ times:*

$$H^{(t+1)} = \operatorname{MPNN}_\theta(H^{(t)},\, G), \qquad t = 0, \ldots, T - 1.$$

### 2.4 The Space-Time Unrolled Graph

**Definition 9** (Space-time unrolled graph). *Given a network graph
$G = (V, E)$ and a time horizon $T$, the* space-time unrolled graph
*is $\hat{G} = (\hat{V}, \hat{E})$ where:*

$$\hat{V} = \{(i, t) \mid i \in V,\; t \in \{0, \ldots, T\}\}$$

$$\hat{E} = \hat{E}_{\mathrm{temp}} \cup \hat{E}_{\mathrm{spat}}$$

*with:*
- *Temporal edges (from* rep*):* $\hat{E}_{\mathrm{temp}} = \{((i, t),\, (i, t+1)) \mid i \in V,\; 0 \leq t < T\}$
- *Spatial edges (from* nbr*):* $\hat{E}_{\mathrm{spat}} = \{((j, t),\, (i, t+1)) \mid (j, i) \in E,\; 0 \leq t < T\}$

---

## 3. Event-Structure Semantics

**Definition 10** (AC event structure). *An AC execution over $G = (V, E)$
for $T$ rounds induces an event structure $\mathcal{E} = (Ev, \to,
\leadsto, \#)$ where:*
- *Events:* $Ev = \{e_{i,t} \mid i \in V,\; t \in \{0, \ldots, T\}\}$
- *Causality:* $e_{i,t} \to e_{i,t+1}$ *(device $i$ at round $t$ causes
  its own round $t + 1$)*
- *Communication:* $e_{j,t} \leadsto e_{i,t+1}$ *iff $(j, i) \in E$
  (device $j$'s export at round $t$ is available to neighbour $i$ at
  round $t + 1$)*
- *Conflict:* $e_{i,t} \mathrel{\#} e_{j,t}$ *when devices $i$ and $j$
  are in different branch partitions at round $t$*

**Definition 11** (State reset on branch switch). *Let
$\operatorname{branch}(c, p_{\top}, p_{\bot})$ be active with
condition field $c^{(t)}$ at round $t$. A device $i$ is said to have*
switched *if $c_i^{(t)} \neq c_i^{(t-1)}$. For every switched device,
each named* rep *state $s$ with initial value $s^{(0)}$ is reset:*

$$s_i^{(t)} = \begin{cases} s^{(0)} & \text{if } c_i^{(t)} \neq c_i^{(t-1)}, \\ f(s_i^{(t-1)}) & \text{otherwise}. \end{cases}$$

*At $t = 0$ no device is considered switched.*

**Proposition 1** (Space-time isomorphism). *The event structure
$\mathcal{E}$ is isomorphic to the space-time unrolled graph $\hat{G}$
via the bijection $\psi : e_{i,t} \mapsto (i, t)$, which maps:*
- *$\to$ to $\hat{E}_{\mathrm{temp}}$,*
- *$\leadsto$ to $\hat{E}_{\mathrm{spat}}$,*
- *$\#$ to the absence of edges between conflicting events.*

*Proof sketch.* The bijection $\psi$ is immediate from the definitions.
Causality $e_{i,t} \to e_{i,t+1}$ maps to the temporal edge
$((i,t), (i,t+1)) \in \hat{E}_{\mathrm{temp}}$. Communication
$e_{j,t} \leadsto e_{i,t+1}$ maps to the spatial edge
$((j,t), (i,t+1)) \in \hat{E}_{\mathrm{spat}}$. Conflict removes edges
between nodes in distinct branch partitions at the same round, which
corresponds to the edge masking in Definition 5. $\square$

---

## 4. Construct-Level Isomorphisms

This section contains the main results. Each theorem establishes a
correspondence between an AC combinator and a GNN operation.

### 4.1 rep ↔ Per-Node Recurrent Cell

**Theorem 1** (rep–RNN isomorphism). *The AC combinator*
$\operatorname{rep}(\text{name},\; s^{(0)},\; f)$ *is isomorphic to a
per-node recurrent neural network cell with no neighbourhood input:*

$$h_i^{(t)} = f_\theta(h_i^{(t-1)}), \qquad h_i^{(0)} = h_0,$$

*where $h_0 = s^{(0)}$ and $f_\theta = f$. The isomorphism maps temporal
causality $e_{i,t} \to e_{i,t+1}$ to the recurrent connection
$h_i^{(t-1)} \to h_i^{(t)}$.*

*Proof sketch.* Both constructs define the same recurrence relation on a
per-node state with identical initial conditions. The state at node $i$
at time $t$ depends only on its own state at time $t - 1$, with no
spatial interaction. On the space-time graph $\hat{G}$, this computation
uses only the temporal edges $\hat{E}_{\mathrm{temp}}$. $\square$

> *Remark.* When $f$ invokes `nbr` internally, the recurrence
> incorporates neighbourhood information. This composed case is
> covered by Theorem 3.

### 4.2 nbr ↔ MPNN Aggregate Step

**Theorem 2** (nbr–MPNN isomorphism). *The AC combinator*
$\operatorname{nbr}(\phi,\; \bigoplus)$ *with optional message transform
$\varphi$ is isomorphic to the aggregate step of an MPNN:*

$$m_i = \bigoplus_{j \in \mathcal{N}(i)} \varphi(\phi_j).$$

*Given a sparse edge representation* $\texttt{edge\_index} \in
\mathbb{Z}^{2 \times |E|}$ *with source indices in row 0 and target
indices in row 1, this is computed as:*

$$m = \operatorname{scatter}\!\left(\varphi\!\left(\phi[\texttt{edge\_index}[0]]\right),\; \texttt{edge\_index}[1],\; \bigoplus\right).$$

*The isomorphism maps the communication relation $e_{j,t} \leadsto
e_{i,t+1}$ to the message-passing edges.*

*Proof sketch.* The `nbr` combinator exports $\phi_j$ from each device
$j$ and delivers it to all neighbours $i \in \mathcal{N}^{-1}(j)$. The
aggregation function $\bigoplus$ is applied per target node. This is
exactly the scatter-based MPNN aggregate step over the spatial edges
$\hat{E}_{\mathrm{spat}}$ of the space-time graph. $\square$

### 4.3 rep ∘ nbr ↔ Recurrent MPNN (Central Result)

**Theorem 3** (rep ∘ nbr – Recurrent MPNN isomorphism). *The composition*

$$\operatorname{rep}\!\left(\text{name},\; s^{(0)},\; s \mapsto g\!\left(s,\; \operatorname{nbr}(\varphi(s),\; \bigoplus)\right)\right)$$

*is isomorphic to a Recurrent MPNN (Definition 8) with update function
$\gamma_\theta = g$ and message function $\phi_\theta = \varphi$:*

$$h_i^{(t)} = \gamma_\theta\!\left(h_i^{(t-1)},\; \bigoplus_{j \in \mathcal{N}(i)} \phi_\theta(h_j^{(t-1)})\right), \qquad h_i^{(0)} = s^{(0)}.$$

*Running $T$ AC rounds is isomorphic to applying a $T$-layer weight-tied
GNN on the space-time graph $\hat{G}$:*

$$H^{(T)} = \underbrace{\operatorname{MPNN}_\theta \circ \cdots \circ \operatorname{MPNN}_\theta}_{T \text{ times}}(H^{(0)},\, G).$$

*Proof sketch.* At each round $t$, every device $i$ reads its own state
$s_i^{(t-1)}$ (via `rep`, along temporal edges) and the states
$\{s_j^{(t-1)}\}_{j \in \mathcal{N}(i)}$ (via `nbr`, along spatial
edges). It then applies $g$ to produce $s_i^{(t)}$. This is precisely
the MPNN update equation. Weight-tying follows from the fact that all
devices run the *same* program at every round — the functions $g$ and
$\varphi$ do not change with $t$. $\square$

**Corollary 1.** *$T$ rounds of an AC program composed of*
$\operatorname{rep}$ *and* $\operatorname{nbr}$ *compute the same
function as a $T$-layer GNN with shared weights on $\hat{G}$.*

### 4.4 branch ↔ Dynamic Subgraph GNN with State Reset

**Theorem 4** (branch–subgraph isomorphism). *The AC combinator*
$\operatorname{branch}(c, p_{\top}, p_{\bot})$ *is isomorphic to running
two independent MPNNs on dynamically induced subgraphs, with state reset
for devices that switch partitions.*

*Formally, given condition field $c^{(t)} \in \{0, 1\}^N$:*

**(a) Partition.**
$$E_k^{(t)} = \{(i, j) \in E \mid c_i^{(t)} = k \wedge c_j^{(t)} = k\}, \qquad k \in \{0, 1\}$$

**(b) Isolated execution.**
$$y_i^{(t)} = \begin{cases} p_{\top}(x_i,\, G_1^{(t)}) & \text{if } c_i^{(t)} = 1, \\ p_{\bot}(x_i,\, G_0^{(t)}) & \text{if } c_i^{(t)} = 0, \end{cases}$$
*where $G_k^{(t)} = (V_k^{(t)}, E_k^{(t)})$.*

**(c) State reset.** *For every device $i$ that switched branches
($c_i^{(t)} \neq c_i^{(t-1)}$), each named state $s$ in the reset set
$\mathcal{R}$ is reinitialised: $s_i^{(t)} \leftarrow s^{(0)}$
(Definition 11).*

*The isomorphism maps the conflict relation $\#$ to edge removal: events
in different partitions cannot exchange messages.*

*Proof sketch.* The branch combinator partitions $V$ by $c$ and restricts
communication to within-partition edges. On $\hat{G}$, this removes all
spatial edges $((j, t), (i, t+1))$ where $c_j^{(t)} \neq c_i^{(t)}$,
which is exactly the conflict relation. Each partition then executes an
independent MPNN on its induced subgraph. The state-reset rule
(Definition 11) ensures that a device entering a new partition starts
with a fresh state, matching AC semantics where entering a branch begins
a new computation. $\square$

### 4.5 mux ↔ Pointwise Gating

**Theorem 5** (mux–gating isomorphism). *The AC combinator*
$\operatorname{mux}(c, e_1, e_2)$ *is isomorphic to element-wise gating:*

$$y_i = \begin{cases} (e_1)_i & \text{if } c_i \geq 0.5, \\ (e_2)_i & \text{if } c_i < 0.5. \end{cases}$$

*Both $e_1$ and $e_2$ are evaluated on the full graph $G$ with no
topology modification. The event structure contains no conflict: all
events communicate regardless of $c$.*

*Proof sketch.* Unlike `branch`, `mux` does not partition the edge set.
Both sub-expressions see the complete neighbourhood $\mathcal{N}(i)$.
The output is a per-node selection, identical to the gating mechanism
in GRU cells or mixture-of-experts architectures. No edge is removed
from $\hat{G}$, so no conflict arises in $\mathcal{E}$. $\square$

### 4.6 Summary

| AC Combinator | GNN Operation | Event Relation | Theorem |
|---|---|---|---|
| $\operatorname{rep}(\text{name}, s^{(0)}, f)$ | Per-node RNN cell | Causality ($\to$) | 1 |
| $\operatorname{nbr}(\phi, \bigoplus)$ | MPNN aggregate step | Communication ($\leadsto$) | 2 |
| $\operatorname{rep} \circ \operatorname{nbr}$ | Recurrent MPNN (weight-tied) | $\to$ and $\leadsto$ | 3 |
| $\operatorname{branch}(c, p_\top, p_\bot)$ | Dynamic subgraph + state reset | Conflict ($\#$) | 4 |
| $\operatorname{mux}(c, e_1, e_2)$ | Pointwise gating | No conflict | 5 |

---

## 5. Differentiability

The discrete operators in the AC combinators (`min`, `max`, Boolean edge
masking) are non-differentiable at certain points. We introduce smooth
relaxations that enable end-to-end gradient-based optimisation.

### 5.1 Soft Aggregation Operators

**Definition 12** (Soft-min). *For values $x_1, \ldots, x_K \in
\mathbb{R}$ and temperature $\tau > 0$:*

$$\operatorname{softmin}_\tau(x_1, \ldots, x_K) = -\tau \log \sum_{k=1}^{K} \exp(-x_k / \tau).$$

*For numerical stability, the implementation uses the log-sum-exp trick:*

$$\operatorname{softmin}_\tau(\mathbf{x}) = -\tau \left(M + \log \sum_{k=1}^{K} \exp\!\left(\frac{-x_k}{\tau} - M\right)\right), \quad M = \max_k\!\left(\frac{-x_k}{\tau}\right).$$

**Definition 13** (Soft-max). *The soft-max is defined by symmetry:*
$$\operatorname{softmax}_\tau(x_1, \ldots, x_K) = -\operatorname{softmin}_\tau(-x_1, \ldots, -x_K).$$

**Proposition 2** (Convergence of soft-min). *For any finite set of values
$\{x_k\}$:*
$$\lim_{\tau \to 0^+} \operatorname{softmin}_\tau(x_1, \ldots, x_K) = \min_k x_k.$$

*Proof sketch.* As $\tau \to 0^+$, the exponential $\exp(-x_k / \tau)$ is
dominated by the term with the smallest $x_k$. The log-sum-exp reduces to
the maximum of $\{-x_k / \tau\}$, and scaling by $-\tau$ recovers the
minimum. $\square$

> *Remark.* In hard mode ($\tau$ not applicable), the implementation uses
> `scatter_reduce` with `reduce="amin"` / `"amax"`, which has a well-defined
> subgradient.

### 5.2 Soft Edge Masking for Branch

**Definition 14** (Soft partition mask). *For condition field $c \in
[0, 1]^N$, temperature $\tau > 0$, and a target partition $k \in
\{0, 1\}$, the soft edge weight for an edge $(j, i) \in E$ is:*

*For partition $k = 1$ (true branch):*
$$w_{ji}^{(\top)} = \sigma\!\left(\tau\,(c_j - 0.5)\right) \cdot \sigma\!\left(\tau\,(c_i - 0.5)\right)$$

*For partition $k = 0$ (false branch):*
$$w_{ji}^{(\bot)} = \sigma\!\left(\tau\,(0.5 - c_j)\right) \cdot \sigma\!\left(\tau\,(0.5 - c_i)\right)$$

*where $\sigma$ is the sigmoid function.*

**Proposition 3** (Convergence of soft mask). *As $\tau \to \infty$:*
$$w_{ji}^{(\top)} \to \mathbb{1}[c_j \geq 0.5] \cdot \mathbb{1}[c_i \geq 0.5], \qquad w_{ji}^{(\bot)} \to \mathbb{1}[c_j < 0.5] \cdot \mathbb{1}[c_i < 0.5].$$

*Proof sketch.* $\sigma(\tau \cdot x) \to \mathbb{1}[x > 0]$ as $\tau
\to \infty$. $\square$

> *Remark.* An alternative formulation uses the same-partition
> probability $P_{\text{same}} = c_j c_i + (1 - c_j)(1 - c_i)$ and
> masks via $\sigma(\tau(P_{\text{same}} - 0.5))$. This is used in the
> general `mask_edges` variant (see Table 1).

### 5.3 End-to-End Differentiability

**Proposition 4** (Differentiability). *An AC program composed of*
$\operatorname{rep}$, $\operatorname{nbr}$, $\operatorname{branch}$,
*and* $\operatorname{mux}$ *— with aggregations in soft mode and
branch masking in soft mode — defines a computation graph that is
differentiable with respect to:*

1. *All learnable parameters $\theta$ in the update, message, and gating
   functions.*
2. *Edge weights $w$ appearing in message expressions.*
3. *Initial states $s^{(0)}$.*
4. *Temperature parameters $\tau$ governing soft operators.*

*Proof sketch.* Each combinator maps to a composition of differentiable
PyTorch operations: `scatter_add`, `exp`, `log`, `sigmoid`, `torch.where`.
The soft-min (Definition 12) is differentiable for all finite inputs and
$\tau > 0$. The soft partition mask (Definition 14) is differentiable in
$c$ and $\tau$. The `mux` combinator is a linear interpolation
(Definition 6) and therefore differentiable. Back-propagation through the
$T$-round unrolled computation graph follows standard automatic
differentiation (back-propagation through time). $\square$

---

## 6. Fixed-Point Convergence

**Definition 15** (Fixed point of a Recurrent MPNN). *A state configuration
$H^* \in \mathbb{R}^{N \times d}$ is a* fixed point *of the Recurrent MPNN
$\operatorname{MPNN}_\theta$ if:*

$$H^* = \operatorname{MPNN}_\theta(H^*,\, G).$$

> *Remark.* Many AC algorithms (e.g., gradient, distance estimation) are
> designed to converge to a fixed point. This corresponds to the *implicit
> layer* formulation in the GNN literature [Bai et al., 2019; Gu et al.,
> 2020], where the output is defined as the fixed point of a contractive
> map rather than as the result of a fixed number of iterations.
> Convergence guarantees depend on the contractive properties of
> $\operatorname{MPNN}_\theta$; a detailed analysis is beyond the scope
> of this work.

---

## 7. Worked Example: The Gradient Algorithm

### 7.1 AC Program

The *gradient* (multi-hop distance estimation) is a classic AC building
block. Using the combinators from Section 2:

$$\operatorname{rep}\!\left(\texttt{"dist"},\; \infty,\; d \mapsto \operatorname{mux}\!\left(\text{source},\; 0,\; \operatorname{nbr}(d + w,\; \min)\right)\right)$$

At each round $t$, device $i$ computes:

$$d_i^{(t)} = \begin{cases} 0 & \text{if } \text{source}_i = 1, \\ \displaystyle\min_{j \in \mathcal{N}(i)} \left(d_j^{(t-1)} + w\right) & \text{otherwise}, \end{cases}$$

with $d_i^{(0)} = \infty$.

### 7.2 GNN Equivalent

By Theorem 3, this is isomorphic to a **Bellman–Ford MPNN** with
$\min$-aggregation:

$$h_i^{(t)} = \text{source}_i \cdot 0 + (1 - \text{source}_i) \cdot \min_{j \in \mathcal{N}(i)}\!\left(h_j^{(t-1)} + w\right).$$

With $w = 1$ (unit hop cost), after $T \geq \operatorname{diam}(G)$
rounds the field converges to the shortest-path distance from the source
set.

### 7.3 Differentiable Variants

**Learnable edge weight.** Making $w$ a learnable parameter
$w \in \mathbb{R}$ and training via MSE loss against ground-truth
distances allows the system to discover the correct edge cost
(converging from an arbitrary initialisation to $w \approx 1$).

**Attention aggregator.** Replacing the built-in $\min$ with a learned
GAT-style aggregator of the form

$$\alpha_{ji} = \operatorname{softmax}_{j \in \mathcal{N}(i)}\!\left(\frac{-\operatorname{LeakyReLU}(a \cdot m_j + b)}{\tau}\right), \qquad \hat{m}_i = \sum_{j \in \mathcal{N}(i)} \alpha_{ji}\, m_j,$$

yields a Graph Attention Network fused with Bellman–Ford. When trained,
the attention sharpens ($\tau \to 0$, $a > 0$) to approximate the hard
minimum.

### 7.4 Convergence

After $T$ rounds, the gradient field satisfies:

$$d_i^{(T)} = \min_{\text{path } P : \text{source} \to i,\; |P| \leq T} \sum_{(u,v) \in P} w,$$

i.e., the shortest-path distance using at most $T$ hops. For a connected
graph with diameter $D$, $T \geq D$ rounds suffice for exact convergence.

---

## 8. Implementation Architecture

The reference implementation is organised in four layers:

```
┌──────────────────────────────────────────────────────────┐
│  DSL Layer  (dsl.py)                                     │
│  rep(), nbr(), branch(), mux(), field, const(), mid()    │
│  AggregateContext, DeviceContext                          │
├──────────────────────────────────────────────────────────┤
│  Module Layer  (layers.py)                               │
│  RepLayer, NbrLayer, BranchLayer, MuxLayer               │
├──────────────────────────────────────────────────────────┤
│  Functional Layer  (functional.py)                       │
│  scatter_aggr, mask_edges, mask_edges_for_partition      │
│  soft_where, _scatter_softmin, _scatter_softmax          │
├──────────────────────────────────────────────────────────┤
│  Core Layer  (core.py)                                   │
│  RoundContext, StateManager                              │
└──────────────────────────────────────────────────────────┘
```

All layers accept and produce `torch.Tensor` and support `autograd` for
end-to-end differentiation.

### Table 1. Formal-to-Code Mapping

| Formal Object | Code Entity | Signature / Key Parameters |
|---|---|---|
| **Definition 1** — Network graph $G$ | `AggregateContext` | `(edge_index: Tensor[2,E], num_nodes: int)` |
| **Definition 2** — Computational field $\phi$ | `Tensor [N]` or `Tensor [N, d]` | Standard PyTorch tensor |
| Field constructors ($0, 1, \infty$) | `field.of(v)`, `field.zeros()`, `field.ones()`, `field.inf()` | Context-aware; infer $N$ from active context |
| **Definition 3** — `rep` | `rep(name, init, fn)` | `name: str`, `init: float\|Tensor`, `fn: Tensor → Tensor` |
| rep state storage | `StateManager.get_or_init(name, init_val)` | Stores `Tensor [N, *d]` keyed by `name` |
| **Definition 4** — `nbr` | `nbr(expr, aggr, mode, tau, ...)` | `aggr: str\|Callable`, `mode: "hard"\|"soft"`, `tau: float` |
| Message transform $\varphi$ | `NbrLayer(transform_fn=...)` | `transform_fn: Tensor → Tensor` (optional) |
| Named export (tag) | `nbr(..., tag="name")` | Stores expression in `ctx.exports[tag]` |
| Scatter aggregation $\bigoplus$ | `scatter_aggr(src, index, N, aggr, mode, tau)` | Dispatches to sum/mean/min/max or custom callable |
| **Definition 5** — `branch` | `branch(cond, p_T, p_F, branch_name, reset_states, mode, tau)` | `reset_states: dict[str, float\|Tensor]` |
| Branch-switch detection | `StateManager.track_branch(name, cond)` | Returns `Tensor [N]` bool (switched nodes) |
| State reset | `StateManager.reset_states_for_nodes(mask, init_map)` | Resets named states where `mask=True` |
| Partition edge masking | `mask_edges_for_partition(edge_index, cond, partition, mode, tau)` | Returns `(edge_index, edge_weight)` |
| **Definition 6** — `mux` | `mux(cond, if_true, if_false)` | `cond: Tensor [N]`, both branches evaluated on full graph |
| Pointwise selection | `soft_where(cond, x, y)` | `torch.where(c >= 0.5, x, y)` |
| **Definition 7** — MPNN layer | `NbrLayer` + `RepLayer` composed | `NbrLayer(aggr, transform_fn, mode, tau)` + `RepLayer(name, init, update_fn)` |
| **Definition 8** — Recurrent MPNN | Weight-tied loop over rounds | `for t in range(T): with ctx.round(): ...` |
| **Definition 9** — Space-time graph $\hat{G}$ | Implicit in the $T$-round loop | Temporal edges = `rep` state passing; spatial edges = `edge_index` |
| **Definition 11** — State reset rule | `BranchLayer.forward()` | Calls `track_branch` then `reset_states_for_nodes` |
| **Definition 12** — Soft-min | `_scatter_softmin(src, index, N, tau, fill)` | Log-sum-exp stabilised (Definition 12) |
| **Definition 13** — Soft-max | `_scatter_softmax(src, index, N, tau, fill)` | Via negation: $-\operatorname{softmin}(-x)$ |
| **Definition 14** — Soft partition mask | `mask_edges_for_partition(..., mode="soft", tau)` | Per-partition sigmoid weights |
| Same-partition mask (alternative) | `mask_edges(edge_index, cond, mode, tau)` | $\sigma(\tau(P_{\text{same}} - 0.5))$ |
| Local device execution | `DeviceContext(num_neighbors)` | Star graph: node 0 = self, 1…K = neighbours |

---

## 9. Conclusion

We have established a formal isomorphism between the four core Aggregate
Computing combinators and standard GNN operations. The central result
(Theorem 3) shows that the composition `rep` ∘ `nbr` is isomorphic to a
Recurrent MPNN with weight-tying, where $T$ AC rounds correspond to $T$
applications of a shared GNN layer on a space-time unrolled graph. The
remaining combinators map to domain restriction with state reset (`branch`,
Theorem 4) and pointwise gating (`mux`, Theorem 5). Smooth relaxations of
discrete operators (soft-min, soft edge masking) make the entire pipeline
differentiable, enabling gradient-based learning of AC program parameters.

**Future work.** Three directions are immediate: (i) extending the
isomorphism to the asynchronous AC execution model, which requires
modelling stale messages and per-node clocks; (ii) analysing the
expressive power of AC programs through the lens of the Weisfeiler–Leman
graph isomorphism hierarchy; and (iii) exploiting the fixed-point
structure (Section 6) via implicit differentiation for memory-efficient
training of long-horizon AC programs.

---

## References

- Bai, S., Kolter, J. Z., & Koltun, V. (2019). Deep Equilibrium Models. *NeurIPS*.
- Beal, J., Pianini, D., & Viroli, M. (2015). Aggregate Programming for the Internet of Things. *IEEE Computer*, 48(9), 22–30.
- Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural Message Passing for Quantum Chemistry. *ICML*.
- Gu, F., Chang, H., Zhu, W., Sojoudi, S., & El Ghaoui, L. (2020). Implicit Graph Neural Networks. *NeurIPS*.
- Li, Y., Tarlow, D., Brockschmidt, M., & Zemel, R. (2016). Gated Graph Sequence Neural Networks. *ICLR*.
- Scarselli, F., Gori, M., Tsoi, A. C., Hagenbuchner, M., & Monfardini, G. (2009). The Graph Neural Network Model. *IEEE Transactions on Neural Networks*, 20(1), 61–80.
- Viroli, M., Beal, J., Damiani, F., Audrito, G., Casadei, R., & Pianini, D. (2019). From distributed coordination to field calculus and aggregate computing. *Journal of Logical and Algebraic Methods in Programming*, 109.
