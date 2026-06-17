"""Shared helpers for DeviceContext tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from torch import Tensor

from diffield.dsl import DeviceContext, branch, iterate

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from diffield import AggregateContext

ABS_TOL = 1e-6

GLOBAL_SYNC_ROUNDS = 4

ONE_NEIGHBOR = 1
TWO_NEIGHBORS = 2
THREE_NEIGHBORS = 3

GRADIENT_STATE = "dist"
GRADIENT_EXPORT = "_grad_dist"
STATE_NAME = "state"
OUTER_STATE = "outer"
COUNTER_STATE = "counter"
BRANCH_STATE = "branch_state"
X_STATE = "x"
Y_STATE = "y"

UNIT_VALUE = 1.0
FALSE_BRANCH_VALUE = 10.0
MUX_FALLBACK_VALUE = 100.0
POLLUTED_INIT = 1000.0
VERY_LARGE_INIT = 10000.0
BRANCH_THRESHOLD = 1.5

LINE_NEIGHBOR_INDICES = ((1,), (0, 2), (1,))
DENSE_PAIR_NEIGHBOR_INDICES = ((1, 1, 1), (0, 0, 0))


def values(*items: float) -> Tensor:
    return torch.tensor(items, dtype=torch.float32)


def iterated(value: float, count: int) -> list[float]:
    return [value] * count


def _neighbor_list(values: Sequence[float] | Tensor) -> list[float]:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.dim() == 0:
        return [float(tensor.item())]
    return tensor.flatten().tolist()


def maybe_exports(
    name: str, values: Sequence[float] | Tensor | None
) -> dict[str, list[float] | Tensor] | None:
    if values is None:
        return None
    return cast("dict[str, list[float] | Tensor]", {name: _neighbor_list(values)})


def maybe_export_map(
    values_by_name: Mapping[str, Sequence[float] | Tensor] | None,
) -> dict[str, list[float] | Tensor] | None:
    if values_by_name is None:
        return None
    return cast(
        "dict[str, list[float] | Tensor]",
        {name: _neighbor_list(vals) for name, vals in values_by_name.items()},
    )


def device_value(device: DeviceContext, tensor: Tensor) -> float:
    return device.result(tensor).item()


def unwrap_device_output(device: DeviceContext, value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(unwrap_device_output(device, item) for item in value)
    if isinstance(value, Tensor):
        return device_value(device, value)
    return value


def run_round(
    device: DeviceContext,
    program: Callable[[], Any],
    *,
    neighbor_exports: Mapping[str, Sequence[float] | Tensor] | None = None,
    neighbor_messages: Mapping[str, Sequence[float] | Tensor] | None = None,
    neighbor_ranges: float | list[float] | Tensor | None = None,
) -> Any:
    with device.round(
        neighbor_exports=maybe_export_map(neighbor_exports),
        neighbor_messages=maybe_export_map(neighbor_messages),
        neighbor_ranges=neighbor_ranges,
    ):
        return unwrap_device_output(device, program())


def run_named_iterate(
    device: DeviceContext,
    *,
    state_name: str,
    update: Callable[[Tensor], Tensor],
    init: float = 0.0,
    neighbor_values: Sequence[float] | Tensor | None = None,
    neighbor_ranges: float | list[float] | Tensor | None = None,
) -> float:
    return run_round(
        device,
        lambda: iterate(device.local_field(init), update, name=state_name),
        neighbor_exports=maybe_exports(state_name, neighbor_values),
        neighbor_ranges=neighbor_ranges,
    )


def run_branched_iterate(
    device: DeviceContext,
    *,
    condition: Tensor,
    state_name: str,
    true_update: Callable[[Tensor], Tensor],
    false_update: Callable[[Tensor], Tensor],
    true_init: float = 0.0,
    false_init: float = 0.0,
    neighbor_values: Sequence[float] | Tensor | None = None,
) -> float:
    return run_round(
        device,
        lambda: branch(
            condition,
            lambda: iterate(device.local_field(true_init), true_update, name=state_name),
            lambda: iterate(device.local_field(false_init), false_update, name=state_name),
        ),
        neighbor_exports=maybe_exports(state_name, neighbor_values),
    )


def collect_round_outputs(
    ctx: AggregateContext,
    rounds: int,
    program: Callable[[], Tensor],
) -> list[Tensor]:
    outputs: list[Tensor] = []
    for _ in range(rounds):
        with ctx.round():
            output = program()
        outputs.append(output.detach().clone())
    return outputs


def assert_float_close(
    actual: float, expected: float, *, atol: float = ABS_TOL
) -> None:
    assert actual == pytest.approx(expected, abs=atol)


def assert_tuple_close(
    actual: tuple[float, ...],
    expected: tuple[float, ...],
    *,
    atol: float = ABS_TOL,
) -> None:
    assert actual == pytest.approx(expected, abs=atol)


class DeviceNetwork:
    """Coordinate snapshots and neighbor exports for local-device tests."""

    def __init__(self, *devices: DeviceContext):
        self.devices = devices

    def snapshot(self, *state_names: str) -> list[dict[str, Any]]:
        return [
            {name: device.get_state(name) for name in state_names}
            for device in self.devices
        ]

    def values_for(
        self,
        snapshots: list[dict[str, Any]],
        state_name: str,
        neighbor_indices: Sequence[int],
    ) -> list[Any]:
        return [snapshots[idx][state_name] for idx in neighbor_indices]

    def exports_for(
        self,
        snapshots: list[dict[str, Any]],
        neighbor_indices: Sequence[int],
    ) -> dict[str, list[Any]]:
        first_snapshot = snapshots[0]
        return {
            name: self.values_for(snapshots, name, neighbor_indices)
            for name in first_snapshot
        }

    def exports(
        self,
        snapshots: list[dict[str, Any]],
        *neighbor_indices_by_device: Sequence[int],
    ) -> list[dict[str, list[Any]]]:
        return [
            self.exports_for(snapshots, neighbor_indices)
            for neighbor_indices in neighbor_indices_by_device
        ]
