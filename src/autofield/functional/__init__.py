"""Low-level differentiable operations for autofield."""

from .aggregation import scatter_aggr
from .conditionals import soft_where
from .folding import scatter_binary_fold, scatter_min_by_first
from .masking import mask_edges, mask_edges_for_partition

__all__ = [
    "scatter_aggr",
    "scatter_binary_fold",
    "scatter_min_by_first",
    "mask_edges",
    "mask_edges_for_partition",
    "soft_where",
]