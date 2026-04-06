"""Parameterized cost models for multi-sink territory formation."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .specs import TerritoryModelSpec
except ImportError:
    from specs import TerritoryModelSpec


def _softplus_param(raw: torch.Tensor, minimum: float = 1e-3) -> torch.Tensor:
    return torch.nn.functional.softplus(raw) + minimum


def _inverse_softplus_target(value: float, minimum: float = 1e-3) -> float:
    adjusted = max(float(value) - minimum, 1e-6)
    return float(torch.log(torch.expm1(torch.tensor(adjusted, dtype=torch.float32))).item())


class LocalSurchargeNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.net(node_features).squeeze(-1))


class LearnableTerritoryModel(nn.Module):
    def __init__(
        self,
        *,
        mode: str = "scalars",
        hidden_dim: int = 32,
        init_range_weight_target: float = 0.60,
        init_risk_weight_target: float = 0.30,
        init_assignment_tau_target: float = 0.80,
        init_surcharge_weight_target: float = 0.10,
    ) -> None:
        super().__init__()
        if mode not in {"scalars", "hybrid"}:
            raise ValueError(f"Unsupported territory model mode: {mode}")
        self.mode = mode
        self.range_weight_raw = nn.Parameter(torch.tensor(_inverse_softplus_target(init_range_weight_target)))
        self.risk_weight_raw = nn.Parameter(torch.tensor(_inverse_softplus_target(init_risk_weight_target)))
        self.assignment_tau_raw = nn.Parameter(torch.tensor(_inverse_softplus_target(init_assignment_tau_target, minimum=0.02)))
        if self.mode == "hybrid":
            self.surcharge_weight_raw = nn.Parameter(torch.tensor(_inverse_softplus_target(init_surcharge_weight_target)))
            self.local_scorer = LocalSurchargeNetwork(hidden_dim=hidden_dim)
        else:
            self.local_scorer = None

    @classmethod
    def from_spec(cls, spec: TerritoryModelSpec) -> "LearnableTerritoryModel":
        return cls(
            mode=spec.mode,
            hidden_dim=spec.hidden_dim,
            init_range_weight_target=spec.init_range_weight_target,
            init_risk_weight_target=spec.init_risk_weight_target,
            init_assignment_tau_target=spec.init_assignment_tau_target,
            init_surcharge_weight_target=spec.init_surcharge_weight_target,
        )

    @property
    def range_weight(self) -> torch.Tensor:
        return _softplus_param(self.range_weight_raw)

    @property
    def risk_weight(self) -> torch.Tensor:
        return _softplus_param(self.risk_weight_raw)

    @property
    def assignment_tau(self) -> torch.Tensor:
        return _softplus_param(self.assignment_tau_raw, minimum=0.02)

    @property
    def surcharge_weight(self) -> torch.Tensor:
        if self.local_scorer is None:
            return self.range_weight.new_tensor(0.0)
        return _softplus_param(self.surcharge_weight_raw)

    def local_correction(self, node_features: torch.Tensor) -> torch.Tensor:
        if self.local_scorer is None:
            return torch.zeros(node_features.shape[0], device=node_features.device, dtype=node_features.dtype)
        return self.surcharge_weight * self.local_scorer(node_features)

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = [
            self.range_weight_raw,
            self.risk_weight_raw,
            self.assignment_tau_raw,
        ]
        if self.local_scorer is not None:
            params.append(self.surcharge_weight_raw)
            params.extend(list(self.local_scorer.parameters()))
        return params

    def current_parameters(self) -> dict[str, float]:
        return {
            "range_weight": float(self.range_weight.detach().item()),
            "risk_weight": float(self.risk_weight.detach().item()),
            "assignment_tau": float(self.assignment_tau.detach().item()),
            "surcharge_weight": float(self.surcharge_weight.detach().item()),
        }