#!/usr/bin/env python3
"""Gradient with a LEARNABLE ATTENTION AGGREGATOR.

Instead of using the built-in ``min`` aggregation, this example passes
a custom differentiable aggregator to ``nbr``.  The aggregator learns
**which neighbour to attend to** using a GAT-style attention mechanism.
"""

import sys
import argparse
sys.path.insert(0, "src")

import torch
import torch.nn as nn
import numpy as np

from aggregate_gnn import GridScenario, SimulationEngine, rep, mux, nbr
from aggregate_gnn.dsl import field
from aggregate_gnn.utils import get_grid_distances


# ── Constants ───────────────────────────────────────────────────────────

LEAKY_RELU_SLOPE = 0.2
MASKED_LOGIT = -1e9          # logit for non-finite messages (→ zero attention)
ATTENTION_DENOM_EPS = 1e-8   # prevents division by zero in softmax denominator


# ── Custom aggregator ───────────────────────────────────────────────────

class AttentionMinAggr(nn.Module):
    """Learnable soft-min via GAT-style attention."""
    def __init__(self, tau_init: float = 1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self._log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())

    @property
    def tau(self):
        return self._log_tau.exp()

    def forward(self, msg: torch.Tensor, index: torch.Tensor,
                num_nodes: int) -> torch.Tensor:
        tau = self.tau
        finite = msg.isfinite()
        safe_msg = torch.where(finite, msg, torch.zeros_like(msg))

        score = torch.nn.functional.leaky_relu(
            self.a * safe_msg + self.b, LEAKY_RELU_SLOPE)
        neg_score = -score / tau
        neg_score = torch.where(finite, neg_score,
                                torch.full_like(neg_score, MASKED_LOGIT))

        max_s = neg_score.new_full((num_nodes,), MASKED_LOGIT)
        max_s.scatter_reduce_(0, index, neg_score, reduce="amax",
                              include_self=True)
        exp_s = (neg_score - max_s[index]).exp()
        sum_exp = neg_score.new_zeros(num_nodes)
        sum_exp.scatter_add_(0, index, exp_s)
        alpha = exp_s / sum_exp[index].clamp(min=ATTENTION_DENOM_EPS)

        out = safe_msg.new_zeros(num_nodes)
        out.scatter_add_(0, index, alpha * safe_msg)

        has_finite = safe_msg.new_zeros(num_nodes)
        has_finite.scatter_add_(0, index, finite.float())
        out = torch.where(has_finite > 0, out, torch.tensor(float("inf")))
        return out


# ── Model ───────────────────────────────────────────────────────────────

class AttentionGradientModel(nn.Module):
    def __init__(self, scenario, T):
        super().__init__()
        self.scenario = scenario
        self.T = T
        self.w = nn.Parameter(torch.tensor(1.5))
        self.attn_aggr = AttentionMinAggr(tau_init=1.0)

    def forward(self, source):
        engine = SimulationEngine.from_scenario(self.scenario)
        w = self.w
        attn = self.attn_aggr

        def program(_runtime):
            return rep("dist", float("inf"), lambda d:
                mux(source, field.of(0.0), nbr(d + w, aggr=attn))
            )

        d, _ = engine.run(rounds=self.T, program=program, signals={"source": source})
        return d


def parse_args():
    parser = argparse.ArgumentParser(description="Attention Gradient AC program")
    parser.add_argument("--rows", type=int, default=5, help="Grid rows")
    parser.add_argument("--cols", type=int, default=5, help="Grid columns")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    return parser.parse_args()


def setup_data(args):
    scenario = GridScenario(args.rows, args.cols, connectivity=4)
    source = torch.zeros(scenario.num_nodes)
    source[0] = 1.0
    target = get_grid_distances(args.rows, args.cols, src_r=0, src_c=0, connectivity=4)
    return scenario, source, target


def train_model(args, model, source, target):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("=== Gradient with Learnable Attention Aggregator ===")
    print(f"Grid: {args.rows}×{args.cols}  Rounds: {model.T}")
    print(f"AC:   rep(inf)(d => mux(src, 0, nbr(d + w, aggr=attn_min)))")
    print(f"GNN:  GAT-style Bellman-Ford MPNN")
    print(f"Params: w (hop cost), a/b (attention), τ (temperature)")
    print(f"  Initial w={model.w.item():.2f}, "
          f"τ={model.attn_aggr.tau.item():.2f}, "
          f"a={model.attn_aggr.a.item():.2f}")
    print()

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        pred = model(source)
        mask = pred.isfinite()
        loss = nn.functional.mse_loss(pred[mask], target[mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(pred.view(args.rows, args.cols).detach().numpy().round(2))
            tau = model.attn_aggr.tau.item()
            a = model.attn_aggr.a.item()
            print(f"Epoch {epoch+1:3d}  loss={loss.item():8.4f}  "
                  f"w={model.w.item():.4f}  τ={tau:.4f}  a={a:.4f}")

    print()
    print(f"Final:  w={model.w.item():.4f}  (target: 1.0)")
    print(f"        τ={model.attn_aggr.tau.item():.4f}  "
          f"(lower = sharper ≈ hard min)")
    print(f"        a={model.attn_aggr.a.item():.4f}  "
          f"(positive = attend to low values)")


def main():
    args = parse_args()
    scenario, source, target = setup_data(args)
    T = args.rows + args.cols
    
    model = AttentionGradientModel(scenario, T)
    train_model(args, model, source, target)

    with torch.no_grad():
        pred = model(source)
    print()
    print("Predicted:")
    print(pred.view(args.rows, args.cols).numpy().round(2))


if __name__ == "__main__":
    main()
