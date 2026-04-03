"""Named constants used throughout the aggregate-GNN library.

Centralises magic numbers so that every module references
the same well-documented value.
"""

# ── Condition thresholds ────────────────────────────────────────────────

CONDITION_THRESHOLD: float = 0.5
"""Threshold for interpreting a continuous condition as boolean.

Used in :func:`soft_where`, :class:`BranchLayer` state merging,
and anywhere ``c >= CONDITION_THRESHOLD`` selects the "true" branch.
"""

BROADCAST_NEAR_ZERO: float = 0.05
"""When ``broadcast()`` receives a float mask, values ≤ this threshold
are treated as *True* (i.e. root nodes).  This heuristic lets a
gradient field (whose source is 0.0) be used directly as a mask.
"""

# ── Temperature defaults ────────────────────────────────────────────────

DEFAULT_TAU_SOFT_AGGR: float = 1.0
"""Default temperature τ for soft min/max aggregation in :class:`NbrLayer`.
Lower τ → sharper (closer to hard min/max).
"""

DEFAULT_TAU_BRANCH: float = 10.0
"""Default temperature τ for sigmoid edge masking in :class:`BranchLayer`.
Higher τ → sharper partition boundary.
"""

# ── Aggregation fill values ─────────────────────────────────────────────

FILL_VALUE_MIN: float = float("inf")
"""Fill value for min-aggregation: isolated nodes default to +∞."""

FILL_VALUE_MAX: float = float("-inf")
"""Fill value for max-aggregation: isolated nodes default to -∞."""

FILL_VALUE_DEFAULT: float = 0.0
"""Fill value for sum/mean aggregation and custom aggregators."""

# ── Numerical stability ────────────────────────────────────────────────

LOG_EPSILON: float = 1e-30
"""Clamp floor before ``log()`` in the logsumexp trick inside
:func:`_scatter_softmin`, preventing ``log(0) = -inf``.
"""
