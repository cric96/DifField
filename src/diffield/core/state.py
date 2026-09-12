"""Per-node state management for aggregate programs."""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import torch
from torch import Tensor

from ..constants import CONDITION_THRESHOLD
from .alignment import SEP, path_has_label, resolve_label


class StateManager:
    """Per-node recurrent state for ``iterate``, keyed by alignment path."""

    def __init__(self, num_nodes: int, device: torch.device | str = "cpu") -> None:
        self.num_nodes = num_nodes
        self.device = torch.device(device)
        self._states: dict[str, Tensor] = {}
        self._inits: dict[str, Tensor] = {}
        self._aligned: dict[str, Tensor] = {}

    def _infer_device(self) -> torch.device:
        if self._states:
            first_state = next(iter(self._states.values()))
            return first_state.device
        return self.device

    def resolve(self, query: str) -> str:
        """Resolve a bare label to the full alignment path that carries it."""
        return resolve_label(query, self._states)

    def resolve_all(self, query: str) -> list[str]:
        """Every slot matching *query*, exactly or by label.

        A label used in both arms of a ``branch`` matches both slots.  Injecting
        a neighbour's exported value writes to all of them: each partition reads
        its own slot and is restricted to its own nodes afterwards.
        """
        if query in self._states:
            return [query]
        return [key for key in self._states if path_has_label(key, query)]

    def resolve_for_node(self, query: str, node: int) -> str:
        """Resolve *query* to the slot that *node* is aligned with.

        A label used in both arms of a ``branch`` matches two slots, but any
        single node is in only one partition, so from that node's point of view
        the label is unambiguous.  Falls back to plain resolution when the
        restriction does not narrow it down.
        """
        if query in self._states:
            return query
        candidates = [key for key in self._states if path_has_label(key, query)]
        if len(candidates) > 1:
            aligned = [
                key
                for key in candidates
                if self._is_aligned_at(key, node)
            ]
            if len(aligned) == 1:
                return aligned[0]
        return resolve_label(query, self._states)

    def _is_aligned_at(self, key: str, node: int) -> bool:
        mask = self._aligned.get(key)
        if mask is None:
            return True
        return bool(mask.reshape(-1)[node].item() >= CONDITION_THRESHOLD)

    def get_or_init(self, init_val: Tensor, *, name: str) -> Tensor:
        """Retrieve existing state or initialize it.

        The initializer is recorded so that :meth:`restrict` can return a slot
        to its initial value for nodes that are not aligned with it.
        """
        self._inits[name] = init_val
        self._aligned.pop(name, None)
        if name not in self._states:
            target_device = self._infer_device()
            if init_val.shape[0] != self.num_nodes:
                raise ValueError(
                    f"Expected first dimension {self.num_nodes}, got {tuple(init_val.shape)}",
                )
            self._states[name] = init_val.clone().to(target_device)
        return self._states[name]

    def update(self, new_val: Tensor, *, name: str) -> None:
        """Store updated state (called after each round for an iterate).

        A bare label is resolved to the alignment path that carries it, so
        callers holding a user-facing name keep working.  Creating a brand new
        slot from an unresolvable name warns, since it usually means a typo that
        would otherwise silently fork a second slot.
        """
        if name not in self._states:
            try:
                name = self.resolve(name)
            except KeyError:
                if not name.startswith(SEP):
                    warnings.warn(
                        f"creating a new state slot for {name!r}, which matches "
                        f"no existing alignment path; known: {sorted(self._states)}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        self._states[name] = new_val

    def restrict(
        self,
        keys: Iterable[str],
        *,
        keep: Tensor,
        mode: str | None = None,
        tau: float | None = None,
    ) -> None:
        """Return *keys* to their initial value for nodes outside *keep*.

        This realises the availability mask of the field-calculus store: a node
        that is not aligned with an occurrence reads that occurrence's
        initializer, rather than a value some other partition computed for it.
        :func:`field_where` is used rather than ``torch.where`` so that soft mode
        stays differentiable through the restriction.
        """
        from ..functional import field_where

        for key in keys:
            state = self._states.get(key)
            init = self._inits.get(key)
            if state is None or init is None:
                continue
            init_like = init if init.shape == state.shape else init.expand_as(state)
            self._states[key] = field_where(keep, state, init_like, mode=mode, tau=tau)
            self._aligned[key] = keep

    def reset(self) -> None:
        """Clear all states and branch tracking."""
        self._states.clear()
        self._inits.clear()
        self._aligned.clear()

    def snapshot(self) -> dict[str, Tensor]:
        """Return a deep copy of all current states."""
        return {key: value.clone() for key, value in self._states.items()}

    def restore(self, snapshot: dict[str, Tensor]) -> None:
        """Restore states from a snapshot."""
        self._states = {key: value.clone() for key, value in snapshot.items()}

    def get_state(self, *, name: str) -> Tensor | None:
        """Get a state by full alignment path or by bare label.

        Returns ``None`` when nothing matches.  An *ambiguous* label raises
        :exc:`AlignmentError` rather than silently picking one slot.
        """
        if name in self._states:
            return self._states[name]
        try:
            return self._states[self.resolve(name)]
        except KeyError:
            return None

    def keys(self) -> list[str]:
        """Return a list of all state keys."""
        return list(self._states.keys())

    def initializers(self) -> dict[str, Tensor]:
        """The initial value recorded for each slot by its last ``get_or_init``.

        A caller rebuilding a store at a different ``num_nodes`` needs these to
        fill the rows it has no value for, so that an unheard-from node reads as
        its initialiser rather than as someone else's state.
        """
        return dict(self._inits)
