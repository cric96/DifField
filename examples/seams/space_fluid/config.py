"""Versioned protocols. Deadlines leave work pending; they never shrink a panel."""

from dataclasses import asdict, dataclass

TRAIN_FAMILIES = ("constant", "gaussian", "mixture", "ellipse")
TEST_FAMILIES = (*TRAIN_FAMILIES, "ring", "front")
CONDITIONS = ("clean", "link_loss", "node_stop", "partition")
LAYOUTS = ("jittered", "uniform", "uneven")
LEARNERS = ("parametric", "hybrid", "gnn")
FIXED = ("spatial", "combined", "value", "variance")


@dataclass(frozen=True)
class RegionConfig:
    profile: str
    seeds: tuple[int, ...] = (0, 1, 2)
    updates: int = 400
    batch_size: int = 64  # >= training episodes: full batch, deterministic updates
    nodes: int = 64
    train_rounds: int = 48
    eval_rounds: int = 96
    train_episodes: int = 16  # per training family
    validation_episodes: int = 4  # per training family
    test_episodes: int = 4  # per family/condition/layout
    transfer_nodes: tuple[int, ...] = (256,)
    topology_layouts: tuple[str, ...] = ("uneven",)
    distributed_nodes: tuple[int, ...] = (64,)
    distributed_conditions: tuple[str, ...] = ("clean", "partition")
    distributed_families: tuple[str, ...] = ("gaussian", "ring")
    distributed_episodes: int = 2  # fixed family/index pairs per condition/size
    # Calibrated on validation only: at 0.1 "every device samples" ties the objective.
    lambdas: tuple[float, ...] = (0.3, 1.0, 3.0)
    main_lambda: float = 0.3  # every panel; other values only on main/clean (Pareto)
    k_values: tuple[int, ...] = (4, 8, 16)
    activation_probabilities: tuple[float, ...] = (0.5,)
    # Calibrated against hard central differences on validation data, for the
    # composed program in both studies: the leader slope agrees at T=0.1; the error
    # slope keeps its sign at T=0.07 (sharper explodes) but is ~3x weaker (least
    # squares 3.1 in both studies), so the error term is weighted by 3.
    error_temperature: float = 0.07
    leader_temperature: float = 0.1
    error_gain: float = 3.0
    validate_every: int = 20
    learning_rate: float = 0.03  # full-batch pilot, seed 0, validation: stable at every lambda
    network_learning_rate: float = 0.01  # hybrid GNN modulator
    final_rate_fraction: float = 0.1  # cosine decay of every learning rate
    gnn_learning_rate: float = 0.03  # best hard validation among 0.003/0.01/0.03, full batch
    degree: int = 6
    threads: int = 1
    device: str = "cpu"  # training only; evaluation and DeviceRuntime stay on CPU
    data_seed: int = 20260930
    recovery_threshold: float = 0.2
    recovery_rounds: int = 5
    schema_version: int = 3

    def to_dict(self):
        return asdict(self)


def protocol(profile: str) -> RegionConfig:
    if profile == "compact-cpu":
        return RegionConfig(profile)
    if profile == "smoke":
        return RegionConfig(
            profile,
            seeds=(0,),
            updates=2,
            batch_size=2,
            nodes=16,
            train_rounds=12,
            eval_rounds=18,
            train_episodes=2,
            validation_episodes=1,
            test_episodes=1,
            transfer_nodes=(32,),
            distributed_nodes=(16,),
            distributed_episodes=1,
            validate_every=1,
            k_values=(2, 4),
        )
    raise ValueError(f"Unknown profile: {profile}")
