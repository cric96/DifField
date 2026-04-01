"""Gradient example family."""

from .attention import main as attention_main
from .fixed import main as fixed_main
from .large import main as large_main
from .learnable import main as learnable_main
from .local import main as local_main
from .models import GradientModel, LearnableMovingGradient
from .moving_nodes import main as moving_nodes_main
from .moving_nodes_learnable import main as moving_nodes_learnable_main
from .moving_source import main as moving_source_main

__all__ = [
	"GradientModel",
	"LearnableMovingGradient",
	"attention_main",
	"fixed_main",
	"large_main",
	"learnable_main",
	"local_main",
	"moving_nodes_learnable_main",
	"moving_nodes_main",
	"moving_source_main",
]