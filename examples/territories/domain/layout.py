"""Domain entities for territory formation."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TerritoryLayout:
    """Description of the physical and logical layout of a territory experiment."""

    scenario_preset: str
    sink_positions: tuple[tuple[int, int], ...]
    sink_indices: tuple[int, ...]
    sink_fields: tuple[torch.Tensor, ...]
    sink_mask: torch.Tensor
    sink_coordinate_field: torch.Tensor
    demand: torch.Tensor
    risk: torch.Tensor
    node_features: torch.Tensor

    @property
    def num_sinks(self) -> int:
        return len(self.sink_indices)


@dataclass(frozen=True)
class TerritoryOutputs:
    """Results of a territory formation program rollout."""

    potential: torch.Tensor
    owner_coords: torch.Tensor
    hard_owner: torch.Tensor
    collected_load: torch.Tensor
    collected_risk: torch.Tensor
    sink_loads: torch.Tensor
    sink_risks: torch.Tensor
