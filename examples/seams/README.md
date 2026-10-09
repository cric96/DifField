# Space-Fluid regions as a learned SCR program

This experiment asks whether learning the parameters of an explicitly coordinated
collective program improves the tradeoff between reconstruction error and the
number of samplers (regions), compared with the hand-designed program it starts
from, a hybrid whose link metric is a GNN over the neighbourhood, and a pure neural
network. Parameters are learned offline and frozen for
evaluation. The same `program(runtime)` runs in a batched `AggregateContext` and in
independent `DeviceRuntime` instances. Boids remains a separate, unchanged experiment.

## Commands and artifacts

From the repository root, after `uv sync --extra cpu --extra decentralized`:

```bash
# Whole comparison: the three campaigns in parallel, then SUMMARY.md with the figures.
scripts/run_seams_comparison.sh generated/seams/comparison
PROFILE=smoke scripts/run_seams_comparison.sh generated/seams/comparison-smoke

# Single campaigns (resumable; --device cuda trains on the GPU, evaluation and
# DeviceRuntime stay on the CPU).
python -m examples.seams space-fluid --profile compact-cpu --device cuda --stage all \
  --out generated/seams/comparison/main          # stages: train, insights, evaluate, report
python -m examples.seams space-fluid-scenarios --profile compact-cpu --device cuda \
  --out generated/seams/comparison/scenarios
python -m examples.seams space-fluid-clusters --profile compact-cpu --device cuda \
  --out generated/seams/comparison/clusters
python -m examples.seams space-fluid-summary --out generated/seams/comparison

# Existing standalone Boids command and checkpoints remain supported.
python -m examples.seams boids --profile paper-cpu \
  --out generated/seams/boids-paper-cpu
```

`uv run python` can replace `python`; the GPU needs `uv sync --extra cuda`. `scripts/run_visual_campaign.py` delegates to
these commands and accepts `--suite space-fluid|boids`. The check/render scripts
inspect and render Space-Fluid runs.

Source and data hashes prevent incompatible resumes; use a new directory after
changing code or configuration. A kernel-owned file lock prevents concurrent runs
or renderers from writing the same directory. `last.pt` contains model, optimizer,
update index, best model and history. Mini-batches (smoke profile only) come from
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
neighbourhood mean and neighbourhood variance. The hybrid keeps every SCR block and
multiplies the cost of link `(i,j)` by `exp(tanh(g(z_i,z_j) + g(z_j,z_i)))`, where `z`
is a 2-float embedding from one message-passing layer over the normalized
(value, mean, variance) of the device and its neighbours. `g` has a zero last layer,
so the hybrid starts exactly as the parametric program. The paper's `min(eps, delta)` clip is
not modelled.

| Method | Specification |
|---|---|
| Fixed spatial | Cost `(4,0,0)`, random priority |
| Fixed combined | Cost `(2,1,1)`, random priority (paired reference) |
| Fixed value | Cost `(2,1,1)`, strength `(4,0,0)`: the paper's "value" option |
| Fixed variance | Cost `(2,1,1)`, strength `(0,0,-4)`: homogeneous neighbourhoods lead |
| Learned parametric | Three positive cost weights and three strength weights, Adam |
| Hybrid | Same plus the GNN link-metric modulator (529 parameters, 5 more floats per link), Adam |
| Recurrent GNN | MPNN + GRU (edge MLP on both states and range, mean aggregation), same rounds, neighbours and signals; its state has the SCR wire width (17 floats per link); Adam |
| Central K-means | Independent fits to `(x,y,normalized observation)` for several K; cluster means reconstruct |
| Central GNN (clustering only) | 4-layer message passing over the whole current graph, every round, to 16 slots; a cluster is a connected component of neighbours in one slot; same objective, no K |

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
statistics. Every update uses all 64 training sequences as disjoint graphs (full
batch, so updates are deterministic); full backpropagation through rounds preserves
state and message paths. Adam (400 updates, rate 0.03, the hybrid's
network 0.01, cosine decay to 10%), gradients clipped at norm 5, weights bounded to
`[-8,8]`. The surrogate keeps hard leader IDs and partitions, so even a full-batch
loss is only piecewise smooth. With full batches the parametric program is
deterministic: its seeds coincide. The rates were chosen in a 150-update, seed-0
validation pilot: 0.1 reaches lower objectives but diverges at `lambda = 3`; 0.03 is
smooth at every tradeoff, and the hybrid's network at 0.01 removes its partition jumps.

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
too weak (reconstruction) or wrong-signed (clustering).

Both studies were calibrated on validation data against hard central
differences (step 0.15 on the six named weights):
- **Leader term, `T=0.1`:** its gradient agrees with the differences.
- **Error term, `T=0.07`:** it keeps the right sign; sharper temperatures explode
  through time.
- **Error gain:** the error slope is still about 3x weaker than the hard
  difference (least squares 3.1 in both studies), so training weights the error
  term by `error_gain = 3`.

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

## Static and moving phenomenon (`space-fluid-scenarios`)

The reconstruction task with the phenomenon frozen (`static`) or moving as above.
Fixed-combined, parametric, hybrid and the GNN are trained in each scenario, tested
on both (clean, batched) and, in their own scenario, on asynchronous devices under
link loss, paused nodes, a partition and permanent crashes.
`figures/scenarios-stabilization` shows, round by round on clean episodes, NRMSE,
regions, elected leaders, regions minus leaders and the share of devices keeping
their leader: with a frozen phenomenon the partition settles and every region has
exactly one leader; with a moving one it follows the field.

## Clustering of the phenomenon (`space-fluid-clusters`)

Zones dominated by one Gaussian with distinct levels and blurred boundaries, static or
drifting. Every learned method minimises the same objective from the phenomenon,
region-mean error + 0.3 x regions / N, without labels; ARI against the zones is
measured only at evaluation. The SCR programs form contiguous regions by construction;
the central GNN sees the whole graph every round and its clusters are connected
components of neighbours sharing a slot. Its training propagates the probability of
sharing a slot along the likeliest path (a sum over paths saturates and stops the
gradient), which is exact for one-hot slots. K-means and Ward are central references
given the true K. Reported: ARI, cluster count, leaders versus regions, fragmentation
and stability, batched and on asynchronous devices under faults.

## Comparison (`space-fluid-summary`)

`SUMMARY.md` and `figures/compare-*` put the three studies side by side: paired NRMSE
difference to the initial configuration and samplers per generalisation panel; the
objective across static and moving training/test and fault recovery; the
stabilization figure; ARI per method and condition; hard validation over training for
every learned method.

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
| Adam updates / batch sequences | 400 / 64 (full) | 2 / 2 |
| Training nodes / rounds | 64 / 48 | 16 / 12 |
| Train / validation episodes per family | 16 / 4 | 2 / 1 |
| Test episodes per family and condition | 4 | 1 |
| Evaluation rounds | 96 | 18 |
| Transfer size / layout | 256 / uneven | 32 / uneven |
| DeviceRuntime | 64 nodes; clean and partition; Gaussian and ring; 2 episodes | 16 nodes; 1 episode |
| K-means K | 4, 8, 16 | 2, 4 |

The main panel crosses all six families with the four conditions. Tradeoff values
other than the main one are evaluated only on main/clean (the Pareto panel).
The distributed panel evaluates fixed-combined and the learners at the main
tradeoff; synchronous runs are an equivalence check at seed 0, asynchronous runs
use all seeds. Fixed methods and K-means do not depend on the training seed; their
seed-0 result is shared. `plan.json` contains the exact inventory (4672 jobs).

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
