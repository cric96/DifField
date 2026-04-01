"""Channel example family."""

from .core import CHANNEL_THRESHOLD, channel_body
from .large import main as large_main
from .small import build_scenario, main as small_main
from .specs import ChannelProgramSpec, GridSpec, LargeChannelSpec, SmallChannelSpec
from .viz import (
	plot_channel_evolution,
	plot_channel_final_fields,
	plot_channel_large_evolution,
	plot_channel_large_final,
	plot_channel_large_setup,
	plot_channel_overlay,
	plot_channel_setup,
)
from .workflow import LargeChannelWorkflow, SmallChannelWorkflow

__all__ = [
	"build_scenario",
	"CHANNEL_THRESHOLD",
	"ChannelProgramSpec",
	"channel_body",
	"GridSpec",
	"large_main",
	"LargeChannelSpec",
	"LargeChannelWorkflow",
	"plot_channel_evolution",
	"plot_channel_final_fields",
	"plot_channel_large_evolution",
	"plot_channel_large_final",
	"plot_channel_large_setup",
	"plot_channel_overlay",
	"plot_channel_setup",
	"SmallChannelSpec",
	"SmallChannelWorkflow",
	"small_main",
]