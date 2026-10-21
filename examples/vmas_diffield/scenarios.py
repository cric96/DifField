"""Scenario metadata and observation wire-format constants for VMAS."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioSpec:
    has_goal: bool
    field_terms: tuple[str, ...]
    primary_metric: str
    n_agents: int
    smooth_collision: bool
    sense_kind: str | None = None
    vmas_name: str | None = None
    n_knowers: int | None = None


# Fixed VMAS observation formats; this pipeline uses the scenario defaults.
DISCOVERY_LIDAR_RANGE = 0.35
SAMPLING_AGENT_LIDAR_RAYS = 12

SCENARIO_SPEC: dict[str, ScenarioSpec] = {
    "flocking": ScenarioSpec(
        True,
        ("separation", "alignment", "cohesion", "goal", "follow", "lead_dir"),
        "order",
        6,
        True,
    ),
    "flocking_beacon": ScenarioSpec(
        True,
        ("separation", "alignment", "cohesion", "goal", "follow", "lead_dir"),
        "goal_prox",
        6,
        True,
        vmas_name="flocking",
        n_knowers=1,
    ),
    "navigation": ScenarioSpec(
        True, ("separation", "goal", "brake", "avoid"), "on_goal_frac", 6, True
    ),
    "discovery": ScenarioSpec(
        False,
        ("separation", "sense", "recruit", "explore", "disperse"),
        "coverage",
        6,
        True,
        sense_kind="lidar",
    ),
    "sampling": ScenarioSpec(
        False,
        ("separation", "sense", "social", "explore", "disperse"),
        "coverage",
        4,
        True,
        sense_kind="grid",
    ),
}
