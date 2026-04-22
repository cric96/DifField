"""Per-node state management for aggregate programs."""

from __future__ import annotations

import torch
from torch import Tensor


class StateManager:
    """Manages per-node recurrent states for iterate, with branch-switch reset."""

    def __init__(self, num_nodes: int, device: torch.device | str = "cpu") -> None:
        self.num_nodes = num_nodes
        self.device = torch.device(device)
        self._states: dict[str, Tensor] = {}
        self._branch_prev: dict[str, Tensor] = {}

    def _infer_device(self) -> torch.device:
        if self._states:
            first_state = next(iter(self._states.values()))
            return first_state.device
        return self.device

    def get_or_init(self, init_val: Tensor, *, name: str) -> Tensor:
        """Retrieve existing state or initialize it."""
        if name not in self._states:
            target_device = self._infer_device()
            if init_val.shape[0] != self.num_nodes:
                raise ValueError(
                    f"Expected first dimension {self.num_nodes}, got {tuple(init_val.shape)}",
                )
            self._states[name] = init_val.clone().to(target_device)
        return self._states[name]

    def update(self, new_val: Tensor, *, name: str) -> None:
        """Store updated state (called after each round for an iterate)."""
        self._states[name] = new_val

    def track_branch(self, branch_name: str, cond: Tensor) -> Tensor:
        """Track branch assignment and return a mask of nodes that switched."""
        cond_bool = cond.bool() if cond.dtype != torch.bool else cond
        if branch_name not in self._branch_prev:
            self._branch_prev[branch_name] = cond_bool.clone()
            return torch.zeros(
                self.num_nodes, dtype=torch.bool, device=cond_bool.device
            )

        prev = self._branch_prev[branch_name]
        switched = prev != cond_bool
        self._branch_prev[branch_name] = cond_bool.clone()
        return switched

    def reset_states_for_nodes(self, mask: Tensor, init_map: dict[str, Tensor]) -> None:
        """Reset iterate states for nodes indicated by *mask*."""
        if not mask.any():
            return

        for name, init_val in init_map.items():
            if name not in self._states:
                continue

            current_state = self._states[name]
            broadcast_mask = mask.unsqueeze(-1) if current_state.dim() > 1 else mask
            reset_val = (
                init_val.expand_as(current_state) if init_val.dim() == 0 else init_val
            )
            self._states[name] = torch.where(broadcast_mask, reset_val, current_state)

    def reset(self) -> None:
        """Clear all states and branch tracking."""
        self._states.clear()
        self._branch_prev.clear()

    def snapshot(self) -> dict[str, Tensor]:
        """Return a deep copy of all current states."""
        return {key: value.clone() for key, value in self._states.items()}

    def restore(self, snapshot: dict[str, Tensor]) -> None:
        """Restore states from a snapshot."""
        self._states = {key: value.clone() for key, value in snapshot.items()}

    def get_state(self, *, name: str) -> Tensor | None:
        """Get a specific state by name, returning None if not found."""
        return self._states.get(name)

    def keys(self) -> list[str]:
        """Return a list of all state keys."""
        return list(self._states.keys())
