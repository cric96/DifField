"""Core context and state management for aggregate computation rounds.

Provides RoundContext (graph + round state) and StateManager (per-node state
tracking with branch-switch reset logic).
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import Tensor

from .pyg_backend import maybe_make_data


class RoundContext:
    """Holds the graph topology and per-round execution state.

    Attributes
    ----------
    edge_index : Tensor [2, E]
        Sparse edge list (source, target).
    edge_weight : Tensor [E]
        Edge metric / cost for each edge.
    message_weight : Tensor [E] | None
        Optional multiplicative transport weight for generic neighborhood
        aggregation. Used internally by constructs such as soft branch.
    num_nodes : int
        Number of nodes in the graph.
    round_num : int
        Current round number (0-indexed).
    state : StateManager
        Manages per-node recurrent states and branch tracking.
    """

    def __init__(self, edge_index: Tensor, num_nodes: int, edge_weight: Tensor | None = None) -> None:
        self.edge_index = edge_index
        self.edge_weight = edge_weight if edge_weight is not None else \
                           torch.ones(edge_index.shape[1], device=edge_index.device)
        self.message_weight: Tensor | None = None
        self.num_nodes = num_nodes
        # Optional PyG projection used by the new backend when available.
        self.data = maybe_make_data(self.edge_index, self.num_nodes, self.edge_weight)
        self.round_num = 0
        self.state = StateManager(num_nodes, device=edge_index.device)
        self.exports: dict[str, Tensor] = {}
        self._neighbor_message_overrides: dict[str, Tensor] = {}

    @contextmanager
    def round(self):
        """Context manager for a single round of aggregate execution."""
        self.exports.clear()
        self._neighbor_message_overrides.clear()
        yield self
        self.round_num += 1

    def reset(self) -> None:
        """Reset everything for a fresh execution."""
        self.round_num = 0
        self.state.reset()
        self.exports.clear()
        self._neighbor_message_overrides.clear()
        self.message_weight = None


class StateManager:
    """Manages per-node recurrent states for rep, with branch-switch reset.

    Each state is identified by a string name and stores a Tensor of shape
    [num_nodes, *feature_dims].
    """

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

    # ---- rep state ----

    def get_or_init(self, name: str, init_val: Tensor | float) -> Tensor:
        """Retrieve existing state or initialize it."""
        if name not in self._states:
            if isinstance(init_val, Tensor):
                assert init_val.shape[0] == self.num_nodes
                target_device = self._infer_device()
                self._states[name] = init_val.clone().to(target_device)
            else:
                target_device = self._infer_device()
                self._states[name] = torch.full(
                    (self.num_nodes,), init_val, dtype=torch.float32, device=target_device
                )
        return self._states[name]

    def update(self, name: str, new_val: Tensor) -> None:
        """Store updated state (called after each round for a rep)."""
        self._states[name] = new_val

    # ---- branch reset ----

    def track_branch(self, branch_name: str, cond: Tensor) -> Tensor:
        """Track branch assignment and return a mask of nodes that switched.

        Parameters
        ----------
        branch_name : str
            Identifier for this branch construct (to support nested branches).
        cond : Tensor [N] bool or float

        Returns
        -------
        switched : Tensor [N] bool
            True for nodes whose branch assignment changed since last call.
        """
        cond_bool = cond.bool() if cond.dtype != torch.bool else cond
        if branch_name not in self._branch_prev:
            # First round: no switch
            self._branch_prev[branch_name] = cond_bool.clone()
            return torch.zeros(self.num_nodes, dtype=torch.bool, device=cond_bool.device)

        prev = self._branch_prev[branch_name]
        switched = prev != cond_bool
        self._branch_prev[branch_name] = cond_bool.clone()
        return switched

    def reset_states_for_nodes(self, mask: Tensor, init_map: dict[str, Tensor | float]) -> None:
        """Reset rep states for nodes indicated by *mask*.

        Parameters
        ----------
        mask : Tensor [N] bool
            True for nodes that need state reset.
        init_map : dict
            Mapping from state name to the initial value to reset to.
        """
        if not mask.any():
            return
        for name, init_val in init_map.items():
            if name not in self._states:
                continue
            current_state = self._states[name]

            # Broadcast mask to match state shape: [N] → [N, 1, …]
            broadcast_mask = (
                mask.unsqueeze(-1) if current_state.dim() > 1 else mask
            )

            if isinstance(init_val, Tensor):
                # Scalar tensors need expanding; shaped tensors are used as-is
                reset_val = (
                    init_val.expand_as(current_state)
                    if init_val.dim() == 0
                    else init_val
                )
            else:
                reset_val = torch.full_like(current_state, init_val)

            self._states[name] = torch.where(
                broadcast_mask, reset_val, current_state,
            )

    def reset(self) -> None:
        """Clear all states and branch tracking."""
        self._states.clear()
        self._branch_prev.clear()
