# Space-Fluid regions as a learned SCR program

This experiment asks whether learning the parameters of an explicitly coordinated
collective program improves the tradeoff between reconstruction error and the
number of samplers (regions), compared with the hand-designed program, black-box
search and a pure neural network. Parameters are learned offline and frozen for
evaluation. The same `program(runtime)` runs in a batched `AggregateContext` and in
independent `DeviceRuntime` instances. Boids remains a separate, unchanged experiment.

## Commands and artifacts

From the repository root, after `uv sync --extra cpu --extra decentralized`:

```bash
python -m examples.seams space-fluid --profile smoke --stage all \
  --out generated/seams/space-fluid-smoke
python -m examples.seams space-fluid --profile compact-cpu --stage all \
  --budget-seconds 21600 --out generated/seams/space-fluid-compact

# Resume the same configuration and sources; each invocation has a new time allowance.
python -m examples.seams space-fluid --profile compact-cpu --stage train \
  --out generated/seams/space-fluid-compact
# Stages in order: train, insights, evaluate, report.

# Zone clustering and the hotspot scaling study (resumable, own output directories).
python -m examples.seams space-fluid-clusters --profile compact-cpu
python -m examples.seams space-fluid-hotspot --profile compact-cpu

# Existing standalone Boids command and checkpoints remain supported.
python -m examples.seams boids --profile paper-cpu \
  --out generated/seams/boids-paper-cpu
```

`uv run python` can replace `python`. `scripts/run_visual_campaign.py` delegates to
these commands and accepts `--suite space-fluid|boids`. The check/render scripts
inspect and render Space-Fluid runs.

Source and data hashes prevent incompatible resumes; use a new directory after
changing code or configuration. A kernel-owned file lock prevents concurrent runs
or renderers from writing the same directory. `last.pt` contains model, optimizer,
update index, best model and history. Mini-batches and search candidates come from
named deterministic streams, so interruption does not change their order. `best.pt`
is selected by the **hard validation objective**, including update zero. A deadline
is checked between atomic units. No profile is reduced on timeout. Exit code 2
means incomplete work; `progress.json`, `plan.json`, stage status files and
`REPORT.md` identify what remains.

| Artifact | Contents |
|---|---|
| `manifest.json`, `plan.json` | Configuration, software/data hashes, panel definitions and expected work |
| `data/bank.pt` | Reproducible training and validation episodes, including simulator truth |
| `checkpoints/<method>/seed*/` | Last/best checkpoints, hard validation history, gradient faithfulness |
| `insights.json`, `insights.pt` | Gradient saliency, horizons, block attribution, learned components |
| `episodes/` | Per-episode metrics, raw per-round curves, synchronous equivalence checks |
| `replays/` | Selected spatial output traces, observations, truth and topology |
| `results.csv`, `results.json` | Completed results, never invented entries for pending jobs |
| `summaries.json`, `paired.json` | Panel statistics and paired seed/episode intervals |
| `figures/`, `animations/` | PNG/PDF plots (Pareto, generalisation, insights); shared-trajectory GIFs |
| `timing.json` | CPU forward/backward and executor pilot timings |

## Coordination program: Space-Fluid as composed SCR blocks

The program is a composition of `diffield` library blocks that mirror ScaFi,
FCPP and Collektive. Every block runs unchanged on a central graph and on
independent devices:

```python
neighbours = nbr(obs)
metric = link_map(self.metric, scatter_range(), neighbours, obs)  # one metric for every block
mean = gather_avg(neighbours, include_self=True)
variance = (gather_avg(neighbours * neighbours, include_self=True) - mean.square()).clamp(0)
strength = self.strength(obs, mean, variance, priority)

election = bounded_election(strength, radius=1.0, metric=metric)          # S
with aligned_on(election.leader, weight=election.confidence):             # one region per leader
    source = election.leader == mid()
    potential = distance_to(source, metric)                               # G
    total, count = converge_cast(potential, (obs, 1.0), add, metric)      # C
    estimate, stamp = broadcast(source, (total / count, time), metric)
```

- **`nbr(x)`** gives the neighbours' previous-round value of any field, exchanged
  as a state slot. Devices exchange only state, so this is the device-safe way
  to read neighbours.
- **`bounded_election`** is Pianini et al., ACSOS 2022, Alg. 1, which is
  Space-Fluid's Fig. 6. It keeps the lexicographic minimum of
  `(-strength, distance, leader ID)` among candidacies within `radius`. It
  returns the leader, the distance, whether the device leads, and the confidence
  of following its leader.
- **`aligned_on(key)`** partitions the enclosed blocks, like Collektive
  `alignedOn`, FCPP `split` and ScaFi `align`. A link exists only when the
  sender's published key equals the receiver's key. A device that changes
  region restarts the enclosed blocks.
- **`distance_to`** is the ScaFi G / Collektive `distanceTo` block.
- **`converge_cast`** is ScaFi C / Collektive `convergeCast`, run causally. A
  device's parent is its lower-potential neighbour that minimises
  `potential + metric`, with ties going to the lowest ID.
- **`broadcast`** gradient-casts along the metric.

Strength options follow the paper: local value, neighbourhood mean and
neighbourhood variance, plus a fixed-weight random priority. Zero weights give
the random-priority election. Discrete decisions compare values quantised to
`1e-5`, so float rounding that depends on batch shape cannot flip a choice
between central and device runs.

Each block adds a round of propagation, so regions settle more slowly than in a
single fused iterate. On a static field, the leader counts sum to the number of
devices (`collected_fraction = 1`).

Execution, asynchrony, fault handling and the cost function's symmetry are as
follows. Synchronous execution exchanges published previous states over current
links at a barrier. Async execution starts with one execution on every device,
then uses randomly ordered independent activations with probability 0.5. Only an
actual send updates a neighbour's mailbox. Paused nodes retain state; partitions,
rejoins and geometry changes never reset the program. Connectivity is a
convergence objective; fragmentation is measured on the current graph.

## Methods

Let `d` be link length, `a,b` training-normalized observations and `delta=|a-b|`.
The cost is `1e-4 + w_space*d + w_signal*delta + w_product*d*delta`, radius fixed
to one. The strength is `priority + z . v` with `z` the normalized value,
neighbourhood mean and neighbourhood variance. The neural variants add shared
`3-16-16-1` tanh MLPs: a cost multiplier `exp(tanh(.))` and an additive strength
term, both exactly inert at initialization. The paper's `min(eps, delta)` clip is
not modelled.

| Method | Specification |
|---|---|
| Fixed spatial | Cost `(4,0,0)`, random priority |
| Fixed combined | Cost `(2,1,1)`, random priority (paired reference) |
| Fixed value | Cost `(2,1,1)`, strength `(4,0,0)`: the paper's "value" option |
| Fixed variance | Cost `(2,1,1)`, strength `(0,0,-4)`: homogeneous neighbourhoods lead |
| Parameter search | Combined cost scaled log-uniformly within `exp(+-2)`, strength weights uniform in `[-4,4]`; 48 candidates, hard validation selection (hotspot: see below) |
| CEM (hotspot only) | Cross-entropy method over the same box: diagonal Gaussian, population 24, 6 elites |
| Learned parametric | Three positive cost weights and three strength weights, Adam |
| Neurosymbolic | Same plus the cost and strength MLPs, Adam |
| Recurrent GNN | MPNN + GRU (edge MLP on both states and range, mean aggregation), same rounds, neighbours and signals; its state has the SCR wire width (17 floats per link); Adam |
| Central K-means | Independent fits to `(x,y,normalized observation)` for several K; cluster means reconstruct |

**Fairness of the GNN.** It runs in the same batched, synchronous and asynchronous
executors with identical per-link bytes, uses no IDs, and is trained on the same
objective, batches, updates and hard-validation selection. Its learning rate
(0.03) is the best hard-validation choice among 0.003, 0.01 and 0.03 at equal
updates. It outputs one value per device: every device is its own sampler, so its
leader fraction is one and the penalty is a constant. It is therefore a
reconstruction reference with N samplers, not a compressed partition; its raw
NRMSE is reported as is. K-means has instantaneous global observations and no
distributed cost model.

## Training and the gradient estimator

For each `lambda in {0.3, 1.0, 3.0}` (main 0.3), minimize

```
MSE(regional estimate, simulator truth) / sigma_train^2 + lambda * active leader fraction
```

The grid was calibrated on validation data only: at 0.1 "every device samples"
ties this objective. Only training and evaluation access truth and global
statistics. Training batches combine four sequences as disjoint graphs; full
backpropagation through rounds preserves state and message paths. Adam (rate
0.005, 400 updates), gradients clipped at norm 5, weights bounded to `[-8,8]`.

Training runs the same composition under `with with_mode("soft", tau=T)`. Each
block then uses its own relaxation:
- **`bounded_election`** uses the relaxed "first admissible candidate in
  lexicographic order". For fixed incoming candidates:

  ```
  P(c) ~ admit(c) * prod_j (1 - admit(j) * sigmoid(margin(c, j) / T)),
  admit = sigmoid((radius - distance) / T)   (own candidacy: 1)
  ```

  The margin compares strengths, or distances when strengths are equal. As
  `T -> 0` this is exactly the hard choice. Keys, distances and the election
  indicator become expectations. The leader ID stays hard, and `confidence` is
  the probability of following the chosen leader.
- **`aligned_on`** keeps the hard partition. Additive blocks weight each link by
  the confidence of both ends (`membership()`).
- **`converge_cast`** scales a child's subtree by the probability of its chosen
  parent.
- **`distance_to`** and **`broadcast`** use soft minima.

After the region, `estimate = follow(election, estimate)` reads the estimate as
a member of the region the device would follow:
- **Hard mode:** the identity.
- **Soft mode:** it mixes the neighbours' estimates on links whose relayed
  candidacy names another leader, weighted by the election's support for that
  candidacy.

Without `follow`, the hard partition hides the signal "this device would be
better off in the next region". The relaxed error gradient was then about 20x
too weak (reconstruction) or wrong-signed (clustering), and gradient learning
lost to random search.

Both studies were calibrated on validation data against hard central
differences (step 0.15 on the six named weights):
- **Leader term, `T=0.1`:** its gradient agrees with the differences.
- **Error term, `T=0.07`:** it keeps the right sign; sharper temperatures explode
  through time.
- **Error gain:** the error slope is still about 3x weaker than the hard
  difference (least squares 3.1 in both studies), so training weights the error
  term by `error_gain = 3`.

Selection and every reported number use the true hard objective.

Evaluation, selection and every reported number use the hard program.
`sensitivity.json` reports, for each named weight:
- the surrogate derivative;
- the hard central difference;
- their sign agreement;
- the hard change after descent steps along the surrogate gradient.

## Gradient insights

The `insights` stage reads the gradient of trained models on fixed Gaussian and
ring test episodes. It reports:
- **Saliency** of every reading, by role: leader, region boundary, interior.
- **Temporal horizon** of late estimates, SCR versus GNN.
- **Gradient reaching each block's output:**
  - S, the election's confidence;
  - G, the potential;
  - C, the collected total;
  - B, the broadcast estimate.

  This shows the gradient flowing through every block of the composition.
- **Gradient share** between the strength and metric parameters.

A hard check perturbs the 10% most salient readings by 0.1 sigma and compares
them with the same number of random readings.

## Hotspot: when the gradient matters

With six global weights the black-box search and Adam reach the same program:
the tuned behaviour has few effective degrees of freedom. `hotspot.py` asks what
happens when a requirement needs more of them and the program gets more knobs.

**Requirement.** The reconstruction study with an alarm: errors where the truth
exceeds `0.5` weigh `1 + gain * sigmoid((truth - 0.5) / 0.05)`, gain in
{0, 3, 9, 27}. Gain 0 is the main-study objective. With a gain the best program
wants fine regions on hotspots and coarse ones elsewhere.

**Knobs.** `EdgeMetric(knots=K)` turns the three cost weights into a
piecewise-linear curve of the link's level: one weight triple per knot, knots
evenly spaced from mean - 1 std to mean + 3 std, linear interpolation in between.
`K = 1` is the main-study program (6 tuned scalars with the strength); K in
{1, 2, 4, 8, 16, 32, 64} gives 6 to 195. The neurosymbolic program (712) and
the GNN (4929) are the neural ends. The SCR composition is unchanged.

**Optimizers at equal compute.** Adam (400 updates), random search and CEM
(diagonal Gaussian over the search box, population 24, 6 elites) each get about
the same wall-clock: 528 hard validation evaluations take as long as 400 Adam
updates. Hyperparameters were chosen on validation in a seed-0 pilot (gain 9):
Adam rate 0.1 (parametric) and 0.03 (neural) among 0.005/0.03/0.1, the GNN keeps
0.03; a search box of `exp(+-4)` around the default cost and `[-6, 6]` strength,
wide enough to hold the pilot Adam solutions. Seeds 0 and 1, 24 held-out test
episodes; a gain-9 run with three seeds is archived in
`generated/seams/space-fluid-hotspot-3seeds`.

**Results** (hard weighted test objective, lower is better; `REPORT.md` and
`gain*/REPORT.md` give intervals and paired differences):

| Gain | Adam K=1 | Adam best K>1 | CEM K=1 / K=64 | Search K=1 / K=64 | Neural | GNN | Fixed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.082 | 0.080 (K=8) | 0.082 / 0.094 | 0.085 / 0.106 | 0.085 | 0.300 | 0.106 |
| 3 | 0.101 | 0.091 (K=16) | 0.105 / 0.114 | 0.101 / 0.178 | 0.100 | 0.300 | 0.152 |
| 9 | 0.118 | 0.100 (K=2) | 0.118 / 0.132 | 0.120 / 0.181 | 0.100 | 0.300 | 0.242 |
| 27 | 0.137 | 0.111 (K=2) | 0.139 / 0.142 | 0.138 / 0.242 | 0.117 | 0.302 | 0.513 |

- **Sanity check, `K = 1`.** At every gain search and CEM equal Adam within
  0.004 (paired intervals contain zero except CEM at gain 3, +0.004). The surrogate gradient through
  the whole program reaches the black-box optimum.
- **The knobs pay off when the requirement is heterogeneous.** At gain 0 more
  knots change nothing (K=8 against K=1: -0.001 [-0.004, +0.001]); at gain 9
  K=2 gives -0.018 [-0.030, -0.010], at gain 27 -0.026 [-0.033, -0.019]. Two
  knots, one for the background and one for hotspots, carry most of it.
- **Only the gradient keeps its quality as the program grows.** Adam is flat in
  K up to 195 scalars (one outlier, gain 27 K=32). Random search degrades from
  K=4 (+0.02 to +0.12 against Adam at K >= 8). CEM stays within 0.01 of Adam up to
  K=8 and then falls behind (K=64: +0.012 to +0.029; significant at gains 0, 3
  and 9).
- **Neural ends.** The neurosymbolic program matches the best parametric curve.
  The GNN reconstructs almost perfectly (NRMSE below threshold 0.005-0.018) but
  every device is its own sampler, so it pays the full leader term (0.3).
- **Insight.** `gain9/figures/hotspot-curves`: Adam learns the same shape in
  every seed, a low range weight below the threshold and a high one above it:
  coarse regions in the background, fine regions on hotspots. Black-box curves
  differ from seed to seed. The bottom row is the descent direction per knot at
  the common start, from one backward pass. `hotspot-leaders`: more knots move
  samplers from the background onto hotspots.
- Synchronous central/device equivalence passed for the largest parametric
  program and the GNN at every gain (8/8).

## Simulator and compact CPU protocol

Training families are constant fields, Gaussian peaks, mixtures and ellipses.
Movement, deformation, emergence/disappearance and mixture splitting/merging vary
throughout an episode. Rings and curved fronts are reserved for test. Train,
validation and test use independent named streams and disjoint Gaussian width
intervals `[0.07,0.14)`, `[0.145,0.17)` and `[0.175,0.23)`. Thus held-out geometry is
also an extrapolation test. Other continuous geometry parameters use independent
draws. The separated intervals refer to base widths; deformation can make their instantaneous widths overlap. Node locations are static; the phenomenon moves.

Training uses perturbed grids. Test additionally uses an uneven density
(70% of nodes in a concentrated patch). Communication is symmetric 6-nearest-neighbor
connectivity. This controls local degree across sizes; increasing node count
changes spatial resolution rather than keeping a fixed physical radio range.
Faults start at one third and restore at two thirds of the episode: extra observation
30% undirected link loss, 20% paused nodes, or a vertical partition/rejoin.
Perception is noise-free: devices read the phenomenon exactly. Data, priorities
and activation schedules are shared between methods. Training/validation cycle
through conditions by episode index; data generation is independent of model seed.

| Protocol quantity | `compact-cpu` | `smoke` |
|---|---:|---:|
| Training seeds | 3 | 1 |
| Adam updates / batch sequences | 400 / 4 | 2 / 2 |
| Training nodes / rounds | 64 / 48 | 16 / 12 |
| Train / validation episodes per family | 16 / 4 | 2 / 1 |
| Test episodes per family and condition | 4 | 1 |
| Evaluation rounds | 96 | 18 |
| Transfer size / layout | 256 / uneven | 32 / uneven |
| DeviceRuntime | 64 nodes; clean and partition; Gaussian and ring; 2 episodes | 16 nodes; 1 episode |
| Search candidates | 48 | 3 |
| K-means K | 4, 8, 16 | 2, 4 |

The main panel crosses all six families with the four conditions. Tradeoff values
other than the main one are evaluated only on main/clean (the Pareto panel).
The distributed panel evaluates fixed-combined and the learners at the main
tradeoff; synchronous runs are an equivalence check at seed 0, asynchronous runs
use all seeds. Fixed methods and K-means do not depend on the training seed; their
seed-0 result is shared. `plan.json` contains the exact inventory (5264 jobs).

## Measurements and interpretation

Primary metrics are training-normalized reconstruction error versus region count,
Space-Fluid's between-region spread of means `sigma(mu_s)` and mean within-region
spread `mu(sigma_s)`, mean region size, collected fraction, fragmentation,
assignment stability, evidence age and numeric message bytes. All rounds count.
Recovery means NRMSE <= 0.2 for five consecutive rounds after a fault or its
restoration.

Statistical comparisons form paired differences on shared episodes, then resample
training seeds and episodes in a two-way clustered bootstrap (2,000 draws), without
multiplicity correction. With three seeds and four episodes per cell, per-family
intervals are wide; headline tables pool panels. More regions alone can reduce
error: the conclusion depends on the error-versus-regions tradeoff, held-out shapes,
transfer and distributed results. Negative results and update-zero selections
remain in the report.

## Relation to prior work and SEAMS

[Self-organising Coordination Regions](https://dl.ifip.org/IFIP-TC6/hal-02365498v1)
([author record](https://apice.unibo.it/bin/view/Publication/SelfOrganisingRegionsCoordination2019))
provides the pattern implemented here: sparse leader choice, a potential toward
each leader, collection to the leader and propagation of its decision. The
service is spatial reconstruction; the learned parts are the link cost and the
leader strength inside that fixed structure.

[Sensing-driven clustering](https://link.springer.com/article/10.1007/s11721-022-00215-y)
studies field-based groups that respond to sensed values and spatial dynamics.
Space-Fluid supplies this implementation's bounded cumulative-distance election.
The learned-cost comparison tests what offline learning adds while keeping that
coordination structure explicit.

[Superpixel Sampling Networks](https://openaccess.thecvf.com/content_ECCV_2018/html/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.html)
learn representations for task-dependent differentiable image grouping.
[Differentiable Clustering with Perturbed Spanning Forests](https://arxiv.org/abs/2305.16358)
uses differentiable combinatorial clustering through perturbed forest constructions.
They motivate learning representations and selection surrogates, but neither is
implemented as a baseline here, and this relaxed recurrence is not their
estimator. Our claimed contribution is a hypothesis to test: learning a metric
inside a specified local collective program, with actual delayed message execution.

The [SEAMS 2027 research criteria](https://conf.researchr.org/track/seams-2027/seams-2027-research-track)
include novelty, relevance, soundness, presentation and verifiability. Service
quality, reconfiguration, recovery and execution cost connect this experiment to
self-adaptation; explicit protocols, limitations, provenance and raw outputs support
verifiability. They do not themselves establish novelty or a successful result.

## Verification and unchanged Boids

```bash
python -m pytest tests/test_space_fluid.py -q
python -m pytest tests/test_boids.py tests/test_boids_flocking.py \
  tests/decentralized/test_mesa_equivalence.py -q
python -m pytest tests -q
```

The second command preserves the 38-test Boids/equivalence regression. Shared
random streams, paired bootstrap, atomic artifacts and directory guards are
independent utilities. Boids dynamics, learning, checkpoint tensors and command
arguments are preserved; only imports were decoupled from the retired sensor
suite. `boids --source-run` can still read historical Boids traces and checkpoints.
