"""Trace handling and teacher-forced supervision for boids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import torch
from ..domain.teacher import teacher_rollout_from_specs

if TYPE_CHECKING:
    from ..domain.specs import LearnableBoidsSpec, ModelSpec, SimulationSpec, TeacherDynamics


@dataclass(frozen=True)
class BoidsTrace:
    """Captured boids simulation trace for imitation learning."""

    positions0: torch.Tensor
    velocities0: torch.Tensor
    pos_seq: torch.Tensor
    vel_seq: torch.Tensor
    preclip_vel_seq: torch.Tensor
    metadata: dict[str, Any]

    @property
    def rounds(self) -> int:
        return int(self.pos_seq.shape[0])

    def sliced(self, rounds: int) -> BoidsTrace:
        """Return a slice of the trace up to the specified number of rounds."""
        active_rounds = max(0, min(int(rounds), self.rounds))
        metadata = dict(self.metadata)
        metadata["active_rounds"] = active_rounds
        return BoidsTrace(
            positions0=self.positions0,
            velocities0=self.velocities0,
            pos_seq=self.pos_seq[:active_rounds],
            vel_seq=self.vel_seq[:active_rounds],
            preclip_vel_seq=self.preclip_vel_seq[:active_rounds],
            metadata=metadata,
        )


def trace_metadata_from_specs(
    *,
    seed: int,
    simulation: "SimulationSpec",
    teacher: "TeacherDynamics",
    model: "ModelSpec",
) -> dict[str, Any]:
    """Generate metadata matching a simulation specification."""
    return {
        "seed": int(seed),
        "rounds": int(simulation.rounds),
        "num_nodes": int(simulation.num_nodes),
        "radius": float(simulation.radius),
        "sep": float(simulation.sep),
        "dt": float(simulation.dt),
        "init_velocity_scale": float(simulation.init_velocity_scale),
        "init_connectivity": str(model.init_connectivity),
        "init_k_neighbors": int(model.init_k_neighbors),
        "init_min_degree": int(model.init_min_degree),
        "teacher_w_sep": float(teacher.w_sep),
        "teacher_w_align": float(teacher.w_align),
        "teacher_w_cohesion": float(teacher.w_cohesion),
        "teacher_damping": float(teacher.damping),
        "teacher_max_speed": float(teacher.max_speed),
        "source": "teacher",
    }


def trace_file_path(trace_dir: Path, *, seed: int, rounds: int) -> Path:
    """Standard naming convention for trace files."""
    return trace_dir / f"seed{seed}_h{rounds}.pt"


@torch.no_grad()
def teacher_trace_from_specs(
    *,
    seed: int,
    positions0: torch.Tensor,
    velocities0: torch.Tensor,
    simulation: "SimulationSpec",
    teacher: "TeacherDynamics",
    model: "ModelSpec",
) -> BoidsTrace:
    """Generate a boids trace using teacher parameters."""
    pos_seq, vel_seq, preclip_vel_seq = teacher_rollout_from_specs(
        positions0=positions0,
        velocities0=velocities0,
        rounds=simulation.rounds,
        simulation=simulation,
        teacher=teacher,
        model=model,
        return_preclip=True,
    )
    return BoidsTrace(
        positions0=positions0,
        velocities0=velocities0,
        pos_seq=pos_seq,
        vel_seq=vel_seq,
        preclip_vel_seq=preclip_vel_seq,
        metadata=trace_metadata_from_specs(
            seed=seed, simulation=simulation, teacher=teacher, model=model
        ),
    )


def save_boids_trace(trace: BoidsTrace, path: Path) -> Path:
    """Serialize a boids trace to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "positions0": trace.positions0.detach().cpu(),
            "velocities0": trace.velocities0.detach().cpu(),
            "pos_seq": trace.pos_seq.detach().cpu(),
            "vel_seq": trace.vel_seq.detach().cpu(),
            "preclip_vel_seq": trace.preclip_vel_seq.detach().cpu(),
            "metadata": dict(trace.metadata),
        },
        path,
    )
    return path


def load_boids_trace(
    path: Path,
    *,
    device: torch.device | None = None,
    expected_metadata: dict[str, Any] | None = None,
) -> BoidsTrace:
    """Load a serialized boids trace from disk with optional validation."""
    payload = torch.load(path, map_location="cpu")
    metadata = dict(payload["metadata"])
    if expected_metadata is not None:
        _validate_trace_metadata(metadata, expected_metadata)

    def move(tensor: torch.Tensor) -> torch.Tensor:
        return tensor if device is None else tensor.to(device)

    return BoidsTrace(
        positions0=move(payload["positions0"]),
        velocities0=move(payload["velocities0"]),
        pos_seq=move(payload["pos_seq"]),
        vel_seq=move(payload["vel_seq"]),
        preclip_vel_seq=move(payload["preclip_vel_seq"]),
        metadata=metadata,
    )


def _metadata_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return abs(float(actual) - expected) <= 1e-9
    return actual == expected


def _validate_trace_metadata(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if not _metadata_matches(actual_value, expected_value):
            raise ValueError(
                f"replay trace metadata mismatch for {key}: expected {expected_value!r}, found {actual_value!r}"
            )


def build_supervision_traces(
    initial_conditions: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    spec: "LearnableBoidsSpec",
) -> list[BoidsTrace]:
    """Orchestrate trace generation/loading for a training run."""
    traces: list[BoidsTrace] = []
    trace_dir = spec.training.replay_trace_dir

    for idx, (positions0, velocities0) in enumerate(initial_conditions):
        seed = spec.seed + idx * 1337
        expected_metadata = trace_metadata_from_specs(
            seed=seed,
            simulation=spec.simulation,
            teacher=spec.teacher,
            model=spec.model,
        )
        path = (
            None
            if trace_dir is None
            else trace_file_path(trace_dir, seed=seed, rounds=spec.simulation.rounds)
        )

        if spec.training.supervision_mode == "replay":
            if path is None:
                raise ValueError("replay supervision requires replay_trace_dir")
            if path.exists():
                trace = load_boids_trace(
                    path,
                    device=spec.simulation.device,
                    expected_metadata=expected_metadata,
                )
            else:
                trace = teacher_trace_from_specs(
                    seed=seed,
                    positions0=positions0,
                    velocities0=velocities0,
                    simulation=spec.simulation,
                    teacher=spec.teacher,
                    model=spec.model,
                )
                save_boids_trace(trace, path)
                trace = load_boids_trace(
                    path,
                    device=spec.simulation.device,
                    expected_metadata=expected_metadata,
                )
        else:
            trace = teacher_trace_from_specs(
                seed=seed,
                positions0=positions0,
                velocities0=velocities0,
                simulation=spec.simulation,
                teacher=spec.teacher,
                model=spec.model,
            )
            if spec.training.save_replay_traces and path is not None:
                save_boids_trace(trace, path)

        traces.append(trace)

    return traces
