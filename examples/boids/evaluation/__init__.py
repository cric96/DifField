"""Boids evaluation logic: per-seed metrics and aggregations."""

from .evaluator import evaluate_seed, mean_eval_metrics

__all__ = ["evaluate_seed", "mean_eval_metrics"]
