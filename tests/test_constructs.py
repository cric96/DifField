"""Tests for aggregate-GNN constructs."""

import sys
sys.path.insert(0, "src")

import pytest
import torch

from aggregate_gnn import (
    AggregateContext,
    EventSchedule,
    GridScenario,
    ScheduledEvent,
    SimulationEngine,
    SnapshotRecorder,
    branch,
    const,
    gradient,
    mux,
    nbr,
    nbrRange,
    rep,
)
from aggregate_gnn.layers import RepLayer, NbrLayer
from aggregate_gnn.functional import scatter_aggr, scatter_min_by_first, mask_edges, soft_where
from aggregate_gnn.dsl import field, DeviceContext
from aggregate_gnn.utils import make_grid_graph


# ===== Helpers =====

def triangle_graph():
    """Triangle: 0-1, 1-2, 0-2 (undirected)."""
    edge_index = torch.tensor([
        [0, 1, 1, 2, 0, 2],
        [1, 0, 2, 1, 2, 0],
    ], dtype=torch.long)
    return edge_index, 3


def line_graph():
    """Line: 0-1-2-3 (undirected)."""
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ], dtype=torch.long)
    return edge_index, 4


# ===== scatter_aggr tests =====

class TestScatterAggr:
    def test_sum(self):
        src = torch.tensor([1.0, 2.0, 3.0])
        index = torch.tensor([0, 0, 1])
        out = scatter_aggr(src, index, 2, aggr="sum")
        assert torch.allclose(out, torch.tensor([3.0, 3.0]))

    def test_mean(self):
        src = torch.tensor([1.0, 3.0, 5.0])
        index = torch.tensor([0, 0, 1])
        out = scatter_aggr(src, index, 2, aggr="mean")
        assert torch.allclose(out, torch.tensor([2.0, 5.0]))

    def test_min_hard(self):
        src = torch.tensor([3.0, 1.0, 5.0])
        index = torch.tensor([0, 0, 1])
        out = scatter_aggr(src, index, 2, aggr="min", mode="hard", fill_value=float("inf"))
        assert torch.allclose(out, torch.tensor([1.0, 5.0]))

    def test_min_soft(self):
        src = torch.tensor([3.0, 1.0, 5.0])
        index = torch.tensor([0, 0, 1])
        out = scatter_aggr(src, index, 2, aggr="min", mode="soft", tau=0.01, fill_value=float("inf"))
        assert torch.allclose(out, torch.tensor([1.0, 5.0]), atol=0.05)

    def test_max_hard(self):
        src = torch.tensor([3.0, 1.0, 5.0])
        index = torch.tensor([0, 0, 1])
        out = scatter_aggr(src, index, 2, aggr="max", mode="hard", fill_value=float("-inf"))
        assert torch.allclose(out, torch.tensor([3.0, 5.0]))

    def test_min_by_first_hard_keeps_last_equal_minimum(self):
        src = torch.tensor(
            [
                [2.0, 20.0],
                [1.0, 10.0],
                [1.0, 30.0],
                [5.0, 50.0],
            ]
        )
        index = torch.tensor([0, 0, 0, 1])
        fill_row = torch.tensor(
            [
                [float("inf"), -1.0],
                [float("inf"), -1.0],
            ]
        )

        out = scatter_min_by_first(src, index, 2, mode="hard", fill_row=fill_row)

        assert torch.allclose(out, torch.tensor([[1.0, 30.0], [5.0, 50.0]]))


# ===== rep tests =====

class TestRep:
    def test_accumulates(self):
        """rep with f(x) = x + 1 should produce 1, 2, 3, ..."""
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        results = []
        for t in range(5):
            with ctx.round():
                val = rep("counter", 0.0, lambda s: s + 1)
            results.append(val[0].item())
        assert results == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_per_node_state(self):
        """Each node maintains independent state."""
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        # Init each node to its ID
        init = torch.arange(n, dtype=torch.float32)
        for t in range(3):
            with ctx.round():
                val = rep("state", init, lambda s: s + 1)
        expected = init + 3
        assert torch.allclose(val, expected)

    def test_nested_rep_multiple_nbr(self):
        """Check what happens with nested rep and multiple nbr."""
        edge_index, n = line_graph() # 0-1-2-3
        ctx = AggregateContext(edge_index, n)
        
        results = []
        for t in range(3):
            with ctx.round():
                val = rep("outer", torch.zeros(n), lambda outer_s:
                    rep("inner", torch.zeros(n), lambda inner_s:
                        nbr(outer_s, aggr="sum") + nbr(inner_s, aggr="sum") + 1.0
                    )
                )
            results.append(val.clone())
            
        # Round 0:
        # outer_s = 0, inner_s = 0
        # nbr(0) + nbr(0) + 1.0 = 1.0 -> inner becomes 1.0, outer becomes 1.0
        #
        # Round 1:
        # outer_s = 1.0, inner_s = 1.0
        # nbr(outer_s) -> node 0 gets from 1 (1.0). Node 1 gets from 0, 2 (2.0)
        # nbr(inner_s) -> node 0 gets from 1 (1.0). Node 1 gets from 0, 2 (2.0)
        
        # Let's verify the exact state progression
        # Round 0: outer_s=0, inner_s=0 -> inner becomes 1.0, outer becomes 1.0
        assert torch.allclose(results[0], torch.tensor([1.0, 1.0, 1.0, 1.0]))
        
        # Round 1: outer_s=1.0, inner_s=1.0
        # nbr(outer_s) -> [1.0, 2.0, 2.0, 1.0]
        # nbr(inner_s) -> [1.0, 2.0, 2.0, 1.0]
        # inner becomes [1+1+1, 2+2+1, 2+2+1, 1+1+1] = [3.0, 5.0, 5.0, 3.0]
        assert torch.allclose(results[1], torch.tensor([3.0, 5.0, 5.0, 3.0]))
        
        # Round 2: outer_s=[3,5,5,3], inner_s=[3,5,5,3]
        # nbr(outer_s) -> [5.0, 8.0, 8.0, 5.0]
        # nbr(inner_s) -> [5.0, 8.0, 8.0, 5.0]
        # inner becomes [5+5+1, 8+8+1, 8+8+1, 5+5+1] = [11.0, 17.0, 17.0, 11.0]
        assert torch.allclose(results[2], torch.tensor([11.0, 17.0, 17.0, 11.0]))


# ===== nbr tests =====

class TestNbr:
    def test_sum_triangle(self):
        """On a triangle, nbr(x, sum) should sum neighbor values."""
        edge_index, n = triangle_graph()
        ctx = AggregateContext(edge_index, n)
        x = torch.tensor([1.0, 2.0, 3.0])
        with ctx.round():
            m = nbr(x, aggr="sum")
        # Node 0: receives from 1,2 → 2+3=5
        # Node 1: receives from 0,2 → 1+3=4
        # Node 2: receives from 0,1 → 1+2=3
        assert torch.allclose(m, torch.tensor([5.0, 4.0, 3.0]))

    def test_min_triangle(self):
        edge_index, n = triangle_graph()
        ctx = AggregateContext(edge_index, n)
        x = torch.tensor([10.0, 2.0, 5.0])
        with ctx.round():
            m = nbr(x, aggr="min")
        # Node 0: min(2, 5) = 2
        # Node 1: min(10, 5) = 5
        # Node 2: min(10, 2) = 2
        assert torch.allclose(m, torch.tensor([2.0, 5.0, 2.0]))

    def test_ignores_context_edge_weight_by_default(self):
        edge_index, n = triangle_graph()
        edge_weight = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        ctx = AggregateContext(edge_index, n, edge_weight=edge_weight)
        x = torch.tensor([1.0, 2.0, 3.0])

        with ctx.round():
            m = nbr(x, aggr="sum")

        assert torch.allclose(m, torch.tensor([5.0, 4.0, 3.0]))

    def test_explicit_edge_weight_still_scales_messages(self):
        edge_index, n = triangle_graph()
        ctx = AggregateContext(edge_index, n)
        x = torch.tensor([1.0, 2.0, 3.0])
        message_weight = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        with ctx.round():
            m = nbr(x, aggr="sum", edge_weight=message_weight)

        assert torch.allclose(m, torch.tensor([27.0, 17.0, 14.0]))

    def test_nbr_range_supports_weighted_shortest_paths(self):
        edge_index, _ = triangle_graph()
        edge_weight = torch.tensor([2.0, 2.0, 2.0, 2.0, 10.0, 10.0])
        source = torch.tensor([1.0, 0.0, 0.0])
        ctx = AggregateContext(edge_index, 3, edge_weight=edge_weight)

        for _ in range(4):
            with ctx.round():
                dist = rep(
                    "weighted_dist",
                    float("inf"),
                    lambda dist_old: mux(source, field.of(0.0), nbr(dist_old + nbrRange(), aggr="min")),
                )

        assert torch.allclose(dist, torch.tensor([0.0, 2.0, 4.0]))


# ===== branch tests =====

class TestBranch:
    def test_isolation(self):
        """Nodes in different branches should not exchange messages."""
        # Line graph: 0-1-2-3
        # cond: [T, T, F, F] → branch isolates {0,1} from {2,3}
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        cond = torch.tensor([True, True, False, False])
        x = torch.tensor([1.0, 2.0, 10.0, 20.0])

        with ctx.round():
            result = branch(
                cond,
                lambda: nbr(x, aggr="sum"),    # true-branch: sums within {0,1}
                lambda: nbr(x, aggr="sum"),    # false-branch: sums within {2,3}
            )
        # Node 0: receives from 1 → 2.0
        # Node 1: receives from 0 → 1.0
        # Node 2: receives from 3 → 20.0
        # Node 3: receives from 2 → 10.0
        assert torch.allclose(result, torch.tensor([2.0, 1.0, 20.0, 10.0]))

    def test_different_nbr_in_branches(self):
        """Different branches can perform different neighbor aggregations."""
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        cond = torch.tensor([True, True, False, False])
        x = torch.tensor([1.0, 2.0, 30.0, 10.0])

        with ctx.round():
            result = branch(
                cond,
                lambda: nbr(x, aggr="sum"),    # true-branch: sums within {0,1}
                lambda: nbr(x, aggr="min"),    # false-branch: mins within {2,3}
            )
        # Node 0: receives from 1 → sum([2.0]) = 2.0
        # Node 1: receives from 0 → sum([1.0]) = 1.0
        # Node 2: receives from 3 → min([10.0]) = 10.0
        # Node 3: receives from 2 → min([30.0]) = 30.0
        assert torch.allclose(result, torch.tensor([2.0, 1.0, 10.0, 30.0]))

    def test_state_reset_on_switch(self):
        """When a node switches branch, rep state should reset."""
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)

        # Round 1: all nodes in true branch, rep accumulates
        cond1 = torch.tensor([True, True, True, True])
        with ctx.round():
            branch(
                cond1,
                lambda: rep("val", 0.0, lambda s: s + 1),
                lambda: rep("val", 0.0, lambda s: s + 1),
                reset_states={"val": 0.0},
            )
        # state should be 1.0 for all
        assert torch.allclose(ctx._ctx.state._states["val"], torch.tensor([1.0, 1.0, 1.0, 1.0]))

        # Round 2: node 3 switches to false branch
        cond2 = torch.tensor([True, True, True, False])
        with ctx.round():
            branch(
                cond2,
                lambda: rep("val", 0.0, lambda s: s + 1),
                lambda: rep("val", 0.0, lambda s: s + 1),
                reset_states={"val": 0.0},
            )
        # Nodes 0,1,2 didn't switch → state = 2.0
        # Node 3 switched → reset to 0.0, then +1 = 1.0
        assert torch.allclose(ctx._ctx.state._states["val"], torch.tensor([2.0, 2.0, 2.0, 1.0]))

    def test_branch_rep_nbr_nested(self):
        """Test nesting of branch, rep, and nbr."""
        edge_index, n = line_graph() # 0-1-2-3
        ctx = AggregateContext(edge_index, n)
        cond = torch.tensor([True, True, False, False])
        
        results = []
        for t in range(3):
            with ctx.round():
                val = branch(
                    cond,
                    lambda: rep("true_val", torch.zeros(n), lambda s: nbr(s, aggr="sum") + 1.0),
                    lambda: rep("false_val", torch.zeros(n), lambda s: nbr(s, aggr="max") + 10.0),
                )
            results.append(val.clone())
            
        # Round 0:
        # True branch (nodes 0,1): nbr(0, sum) + 1 = 1.0
        # False branch (nodes 2,3): nbr(0, max) + 10 = 10.0
        assert torch.allclose(results[0], torch.tensor([1.0, 1.0, 10.0, 10.0]))
        
        # Round 1:
        # True branch (nodes 0,1):
        # node 0 receives from 1 (1.0), node 1 receives from 0 (1.0) -> sum is 1.0 -> + 1.0 = 2.0
        # False branch (nodes 2,3):
        # node 2 receives from 3 (10.0), node 3 receives from 2 (10.0) -> max is 10.0 -> + 10.0 = 20.0
        assert torch.allclose(results[1], torch.tensor([2.0, 2.0, 20.0, 20.0]))
        
        # Round 2:
        # True branch: nbr sum is 2.0 -> + 1.0 = 3.0
        # False branch: nbr max is 20.0 -> + 10.0 = 30.0
        assert torch.allclose(results[2], torch.tensor([3.0, 3.0, 30.0, 30.0]))


# ===== mux tests =====

class TestMux:
    def test_no_isolation(self):
        """mux should NOT isolate communication between branches."""
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        cond = torch.tensor([True, True, False, False])
        x = torch.tensor([1.0, 2.0, 10.0, 20.0])

        with ctx.round():
            result = mux(
                cond.float(),
                lambda: nbr(x, aggr="sum"),  # true: full topology sum
                lambda: nbr(x, aggr="sum"),  # false: full topology sum
            )
        # Both branches see ALL edges (no masking)
        # Node 0: receives from 1 → 2.0
        # Node 1: receives from 0, 2 → 1 + 10 = 11.0
        # Node 2: receives from 1, 3 → 2 + 20 = 22.0
        # Node 3: receives from 2 → 10.0
        assert torch.allclose(result, torch.tensor([2.0, 11.0, 22.0, 10.0]))

    def test_selection(self):
        """mux selects different value per node based on condition."""
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        cond = torch.tensor([1.0, 1.0, 0.0, 0.0])

        with ctx.round():
            result = mux(
                cond,
                torch.tensor([10.0, 20.0, 30.0, 40.0]),
                torch.tensor([100.0, 200.0, 300.0, 400.0]),
            )
        assert torch.allclose(result, torch.tensor([10.0, 20.0, 300.0, 400.0]))


# ===== Gradient algorithm tests =====

class TestGradient:
    def test_gradient_fixed(self):
        """Gradient with w=1 should produce Manhattan distances on a grid."""
        rows, cols = 5, 5
        edge_index, n = make_grid_graph(rows, cols)

        source = torch.zeros(n)
        source[0] = 1.0
        w = torch.tensor(1.0)

        ctx = AggregateContext(edge_index, n)
        T = rows + cols
        for t in range(T):
            with ctx.round():
                d = rep("dist", float("inf"), lambda d:
                    mux(source, torch.zeros(n), nbr(d + w, aggr="min")))

        expected = torch.zeros(n)
        for r in range(rows):
            for c in range(cols):
                expected[r * cols + c] = float(r + c)

        assert torch.allclose(d, expected)

    def test_differentiability(self):
        """Backward pass should work through rep+nbr+mux composition."""
        rows, cols = 3, 3
        edge_index, n = make_grid_graph(rows, cols)

        source = torch.zeros(n)
        source[0] = 1.0
        w = torch.tensor(1.0, requires_grad=True)

        ctx = AggregateContext(edge_index, n)
        for t in range(6):
            with ctx.round():
                d = rep("dist", float("inf"), lambda d:
                    mux(source, torch.zeros(n), nbr(d + w, aggr="min")))

        loss = d[d.isfinite()].sum()
        loss.backward()
        assert w.grad is not None
        assert w.grad.item() != 0.0

    def test_convenience_gradient_uses_edge_weight_by_default(self):
        edge_index, _ = triangle_graph()
        edge_weight = torch.tensor([2.0, 2.0, 2.0, 2.0, 10.0, 10.0])
        source = torch.tensor([1.0, 0.0, 0.0])
        ctx = AggregateContext(edge_index, 3, edge_weight=edge_weight)

        for _ in range(4):
            with ctx.round():
                d = gradient(source, name="weighted")

        assert torch.allclose(d, torch.tensor([0.0, 2.0, 4.0]))

    def test_nested_rep_multiple_nbr_differentiability(self):
        """Backward pass should work through nested rep and multiple nbr."""
        rows, cols = 3, 3
        edge_index, n = make_grid_graph(rows, cols)

        w1 = torch.tensor(0.5, requires_grad=True)
        w2 = torch.tensor(0.5, requires_grad=True)

        ctx = AggregateContext(edge_index, n)
        for t in range(3):
            with ctx.round():
                val = rep("outer", torch.zeros(n), lambda outer_s:
                    rep("inner", torch.zeros(n), lambda inner_s:
                        nbr(outer_s * w1, aggr="sum") + nbr(inner_s * w2, aggr="sum") + 1.0
                    )
                )

        loss = val.sum()
        loss.backward()
        
        assert w1.grad is not None
        assert w2.grad is not None
        assert w1.grad.item() != 0.0
        assert w2.grad.item() != 0.0


# ===== soft_where tests =====

class TestSoftWhere:
    def test_basic(self):
        cond = torch.tensor([1.0, 0.0, 1.0])
        x = torch.tensor([10.0, 20.0, 30.0])
        y = torch.tensor([100.0, 200.0, 300.0])
        result = soft_where(cond, x, y)
        assert torch.allclose(result, torch.tensor([10.0, 200.0, 30.0]))

    def test_boolean_cond(self):
        cond = torch.tensor([True, False])
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        result = soft_where(cond, x, y)
        assert torch.allclose(result, torch.tensor([1.0, 4.0]))


# ===== mask_edges tests =====

class TestMaskEdges:
    def test_hard_mask(self):
        # Line: 0-1-2-3, cond: [T, T, F, F]
        edge_index, _ = line_graph()
        cond = torch.tensor([True, True, False, False])
        ei_out, ew = mask_edges(edge_index, cond, mode="hard")
        # Should keep: 0→1, 1→0, 2→3, 3→2
        assert ei_out.shape[1] == 4


# ===== Layers inside DSL tests =====

class TestLayersInDSL:
    def test_nbr_layer_in_dsl(self):
        """NbrLayer should work inside a DSL `with ctx.round():` block."""
        edge_index, n = triangle_graph()
        ctx = AggregateContext(edge_index, n)
        nbr_layer = NbrLayer(aggr="sum")
        x = torch.tensor([1.0, 2.0, 3.0])
        with ctx.round():
            m = nbr_layer(x)  # no explicit ctx — picks up DSL context
        assert torch.allclose(m, torch.tensor([5.0, 4.0, 3.0]))

    def test_rep_layer_in_dsl(self):
        """RepLayer (1-arg update_fn) should work inside a DSL block."""
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        rep_layer = RepLayer("counter", 0.0, lambda s: s + 1)
        results = []
        for t in range(3):
            with ctx.round():
                val = rep_layer(torch.zeros(n))  # no explicit ctx
            results.append(val[0].item())
        assert results == [1.0, 2.0, 3.0]

    def test_layers_compose_with_dsl(self):
        """Mix nn.Module layers and DSL functions in the same round."""
        edge_index, n = triangle_graph()
        ctx = AggregateContext(edge_index, n)
        nbr_layer = NbrLayer(aggr="min", fill_value=float("inf"))
        x = torch.tensor([10.0, 2.0, 5.0])
        with ctx.round():
            # Use NbrLayer, then combine with DSL mux
            messages = nbr_layer(x)
            result = mux(
                torch.tensor([1.0, 0.0, 1.0]),
                messages,
                torch.zeros(n),
            )
        # Node 0 (cond=1): nbr min = min(2,5) = 2
        # Node 1 (cond=0): 0
        # Node 2 (cond=1): nbr min = min(10,2) = 2
        assert torch.allclose(result, torch.tensor([2.0, 0.0, 2.0]))

    def test_gradient_with_layers(self):
        """Gradient algorithm using NbrLayer inside DSL rep()."""
        edge_index = torch.tensor([
            [0, 1, 1, 2, 0, 1, 2],
            [1, 0, 2, 1, 0, 1, 2],
        ], dtype=torch.long)  # line 0-1-2 + self-loops
        n = 3
        source = torch.tensor([1.0, 0.0, 0.0])
        nbr_min = NbrLayer(aggr="min")

        ctx = AggregateContext(edge_index, n)
        for t in range(4):
            with ctx.round():
                d = rep("d", float("inf"), lambda s:
                    mux(source, torch.zeros(n), nbr_min(s + 1)))
        assert torch.allclose(d, torch.tensor([0.0, 1.0, 2.0]))


# ===== field constructors tests =====

class TestField:
    def test_of(self):
        edge_index, n = triangle_graph()
        ctx = AggregateContext(edge_index, n)
        with ctx.round():
            f = field.of(3.14)
        assert f.shape == (n,)
        assert torch.allclose(f, torch.tensor([3.14, 3.14, 3.14]))

    def test_zeros_ones_inf(self):
        edge_index, n = triangle_graph()
        ctx = AggregateContext(edge_index, n)
        with ctx.round():
            z = field.zeros()
            o = field.ones()
            i = field.inf()
        assert torch.allclose(z, torch.zeros(n))
        assert torch.allclose(o, torch.ones(n))
        assert (i == float("inf")).all()

    def test_in_gradient_program(self):
        """field.of(0.0) replaces torch.zeros(n) inside a program."""
        edge_index = torch.tensor([
            [0, 1, 1, 2, 0, 1, 2],
            [1, 0, 2, 1, 0, 1, 2],
        ], dtype=torch.long)
        n = 3
        source = torch.tensor([1.0, 0.0, 0.0])
        w = torch.tensor(1.0)

        ctx = AggregateContext(edge_index, n)
        for t in range(4):
            with ctx.round():
                d = rep("dist", float("inf"), lambda d:
                    mux(source, field.of(0.0), nbr(d + w, aggr="min")))
        assert torch.allclose(d, torch.tensor([0.0, 1.0, 2.0]))


# ===== DeviceContext (local execution) tests =====

class TestDeviceContext:
    def test_local_field(self):
        device = DeviceContext(num_neighbors=3)
        f = device.local_field(own=1.0, nbr=0.0)
        assert f.shape == (4,)
        assert f[0] == 1.0
        assert (f[1:] == 0.0).all()

    def test_result(self):
        t = torch.tensor([42.0, 1.0, 2.0])
        assert DeviceContext.result(t) == 42.0

    def test_local_matches_global(self):
        """Local execution for one device matches full-graph execution."""
        # Line: 0-1-2 with self-loops
        edge_index = torch.tensor([
            [0, 1, 1, 2, 0, 1, 2],
            [1, 0, 2, 1, 0, 1, 2],
        ], dtype=torch.long)
        n = 3
        source = torch.tensor([1.0, 0.0, 0.0])
        w = torch.tensor(1.0)

        # --- global ---
        ctx = AggregateContext(edge_index, n)
        global_states = []
        for t in range(4):
            with ctx.round():
                d = rep("dist", float("inf"), lambda d:
                    mux(source, field.of(0.0), nbr(d + w, aggr="min")))
            global_states.append(d.detach().clone())

        # --- local for device 2 (distance = 2) ---
        device_id = 2
        nbrs = [1]  # only neighbour (excluding self-loop)
        device = DeviceContext(num_neighbors=1)
        source_local = device.local_field(
            own=source[device_id].item(), nbr=0.0,
        )
        for t in range(4):
            nbr_exports = None if t == 0 else {
                "dist": [global_states[t - 1][j].item() for j in nbrs],
            }
            with device.round(neighbor_exports=nbr_exports):
                d_local = rep("dist", float("inf"), lambda d:
                    mux(source_local, field.of(0.0), nbr(d + w, aggr="min")))

        assert abs(device.result(d_local).item()
                   - global_states[-1][device_id].item()) < 1e-6

    def test_isolated_device(self):
        """A source device with zero neighbours always gets 0."""
        device = DeviceContext(num_neighbors=0)
        source_local = device.local_field(own=1.0)  # this IS the source
        w = torch.tensor(1.0)
        for t in range(3):
            with device.round():
                d = rep("dist", float("inf"), lambda d:
                    mux(source_local, field.of(0.0), nbr(d + w, aggr="min")))
        # source=1 for this device, so mux selects field.of(0) → 0
        assert device.result(d).item() == 0.0

    def test_neighbor_ranges_available_locally(self):
        device = DeviceContext(num_neighbors=2)

        with device.round(neighbor_ranges=[1.5, 2.5]):
            ranges = nbr(nbrRange(), aggr="sum")

        assert abs(device.result(ranges).item() - 4.0) < 1e-6

    def test_local_weighted_gradient_matches_global(self):
        edge_index = torch.tensor([
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ], dtype=torch.long)
        edge_weight = torch.tensor([2.0, 2.0, 3.0, 3.0])
        source = torch.tensor([1.0, 0.0, 0.0])

        ctx = AggregateContext(edge_index, 3, edge_weight=edge_weight)
        global_states = []
        for _ in range(4):
            with ctx.round():
                d = gradient(source, name="dist")
            global_states.append(d.detach().clone())

        device = DeviceContext(num_neighbors=1, self_loop=False)
        source_local = device.local_field(own=0.0, nbr=0.0)
        for round_idx in range(4):
            neighbor_exports = None if round_idx == 0 else {
                "_grad_dist": [global_states[round_idx - 1][1].item()],
            }
            with device.round(neighbor_exports=neighbor_exports, neighbor_ranges=[3.0]):
                d_local = gradient(source_local, name="dist")

        assert abs(device.result(d_local).item() - global_states[-1][2].item()) < 1e-6

    def test_decentralized_branching(self):
        """Decentralized branch on A--B--C matches centralized (with self-loops)."""
        # Graph: A -- B -- C (+ self-loops from DeviceContext)
        # Condition: A(True), B(True), C(False)
        # True branch:  rep("state", 0, s → nbr(s, sum) + 1)
        # False branch: rep("state", 0, s → nbr(s, max) + 10)
        devA = DeviceContext(num_neighbors=1)
        devB = DeviceContext(num_neighbors=2)
        devC = DeviceContext(num_neighbors=1)

        condA = devA.local_field(own=1.0, nbr=1.0)
        condB = devB.local_field(own=1.0, nbr=[1.0, 0.0])
        condC = devC.local_field(own=0.0, nbr=1.0)

        def run_branching(dev, cond, nbr_states):
            exports = None if nbr_states is None else {"state": nbr_states}
            with dev.round(neighbor_exports=exports):
                val = branch(
                    cond,
                    lambda: rep("state", dev.local_field(0.0),
                                lambda s: nbr(s, aggr="sum") + 1.0),
                    lambda: rep("state", dev.local_field(0.0),
                                lambda s: nbr(s, aggr="max") + 10.0),
                )
            return dev.result(val).item()

        # Round 1 (states start at 0 → aggregation on zeros)
        rA1 = run_branching(devA, condA, None)
        rB1 = run_branching(devB, condB, None)
        rC1 = run_branching(devC, condC, None)
        assert rA1 == 1.0   # sum(0,0)+1  (self + nbr, both 0)
        assert rB1 == 1.0   # sum(0,0)+1
        assert rC1 == 10.0  # max(0)+10   (self-loop only)

        # Round 2: inject previous rep-states as neighbour exports
        rA2 = run_branching(devA, condA, [rB1])
        rB2 = run_branching(devB, condB, [rA1, rC1])
        rC2 = run_branching(devC, condC, [rB1])
        assert rA2 == 3.0   # sum(self=1, B=1)+1 = 3
        assert rB2 == 3.0   # sum(self=1, A=1)+1 = 3  (C masked out)
        assert rC2 == 20.0  # max(self=10)+10 = 20    (B masked out)

        # Round 3
        rA3 = run_branching(devA, condA, [rB2])
        rB3 = run_branching(devB, condB, [rA2, rC2])
        rC3 = run_branching(devC, condC, [rB2])
        assert rA3 == 7.0   # sum(self=3, B=3)+1 = 7
        assert rB3 == 7.0   # sum(self=3, A=3)+1 = 7
        assert rC3 == 30.0  # max(self=20)+10 = 30

    def test_decentralized_multiple_assignments_nested(self):
        """Decentralized execution with mux-wrapped rep + branch matches centralized.

        Three devices on a line: dev0 -- dev1 -- dev2 (+ self-loops).
        Uses get_state() to inject the neighbour's raw rep state (not the
        mux output) into neighbor_exports — the correct protocol.
        """
        dev0 = DeviceContext(num_neighbors=1)
        dev1 = DeviceContext(num_neighbors=2)
        dev2 = DeviceContext(num_neighbors=1)

        src0 = dev0.local_field(own=1.0, nbr=0.0)           # dev0 is source
        src1 = dev1.local_field(own=0.0, nbr=[1.0, 0.0])    # dev1 knows dev0=src
        src2 = dev2.local_field(own=0.0, nbr=0.0)            # dev2 not source

        def run_device(dev, src, exports):
            with dev.round(neighbor_exports=exports):
                x = rep("x", dev.local_field(0.0), lambda s: s + 1.0)
                y = mux(
                    src,
                    rep("y", dev.local_field(0.0),
                        lambda s: nbr(s + x, aggr="sum")),
                    dev.local_field(100.0),
                )
                cond = x > 1.5
                z = branch(
                    cond,
                    lambda: rep("z", dev.local_field(0.0),
                                lambda s: s + y + 10.0),
                    lambda: dev.local_field(-1.0),
                )
            return (dev.result(x).item(),
                    dev.result(y).item(),
                    dev.result(z).item())

        def snapshot_states(*devs):
            """Snapshot rep states from all devices before running a round."""
            return [{"x": d.get_state("x"), "y": d.get_state("y")} for d in devs]

        def make_exports(snap, nbr_indices):
            """Build neighbor_exports from a snapshot for given neighbour indices."""
            return {
                "x": [snap[i]["x"] for i in nbr_indices],
                "y": [snap[i]["y"] for i in nbr_indices],
            }

        # Round 1 — no exports
        r0 = run_device(dev0, src0, None)
        r1 = run_device(dev1, src1, None)
        r2 = run_device(dev2, src2, None)
        assert r0 == (1.0, 2.0, -1.0)
        assert r1 == (1.0, 100.0, -1.0)
        assert r2 == (1.0, 100.0, -1.0)

        # Round 2 — snapshot states from R1, then run all devices
        snap = snapshot_states(dev0, dev1, dev2)
        r0 = run_device(dev0, src0, make_exports(snap, [1]))
        r1 = run_device(dev1, src1, make_exports(snap, [0, 2]))
        r2 = run_device(dev2, src2, make_exports(snap, [1]))
        assert r0 == (2.0, 9.0, 31.0)
        assert r1 == (2.0, 100.0, 220.0)
        assert r2 == (2.0, 100.0, 220.0)

        # Round 3 — snapshot states from R2, then run all devices
        snap = snapshot_states(dev0, dev1, dev2)
        r0 = run_device(dev0, src0, make_exports(snap, [1]))
        r1 = run_device(dev1, src1, make_exports(snap, [0, 2]))
        r2 = run_device(dev2, src2, make_exports(snap, [1]))
        assert r0 == (3.0, 28.0, 69.0)
        assert r1 == (3.0, 100.0, 330.0)
        assert r2 == (3.0, 100.0, 330.0)


# ===== Composition tests =====

class TestComposition:
    def test_mux_rep_branch_composition(self):
        """Test top-level mux -> rep -> branch -> nbr."""
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        
        cond_mux = torch.tensor([True, True, False, False])
        cond_branch = torch.tensor([True, True, True, False])
        
        results = []
        for t in range(3):
            with ctx.round():
                val = mux(
                    cond_mux,
                    rep("outer_rep", torch.zeros(n), lambda outer_s:
                        branch(
                            cond_branch,
                            lambda: nbr(outer_s, aggr="sum") + 1.0,
                            lambda: nbr(outer_s, aggr="max") + 10.0
                        )
                    ),
                    rep("other_rep", torch.zeros(n), lambda s: nbr(s, aggr="sum") + 100.0)
                )
            results.append(val.clone())

        assert torch.allclose(results[0], torch.tensor([1.0, 1.0, 100.0, 100.0]))
        assert torch.allclose(results[1], torch.tensor([2.0, 3.0, 300.0, 200.0]))
        assert torch.allclose(results[2], torch.tensor([4.0, 5.0, 600.0, 400.0]))

    def test_rep_branch_rep_composition(self):
        """Test top-level rep -> branch -> rep -> nbr."""
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        
        cond = torch.tensor([True, True, False, False])
        
        results = []
        for t in range(3):
            with ctx.round():
                val = rep("outer", torch.zeros(n), lambda outer_s:
                    branch(
                        cond,
                        lambda: rep("inner_true", torch.zeros(n), lambda inner_s:
                            nbr(outer_s, aggr="sum") + nbr(inner_s, aggr="sum") + 1.0
                        ),
                        lambda: rep("inner_false", torch.zeros(n), lambda inner_s:
                            nbr(outer_s, aggr="max") + nbr(inner_s, aggr="max") + 10.0
                        )
                    )
                )
            results.append(val.clone())
            
        assert torch.allclose(results[0], torch.tensor([1.0, 1.0, 10.0, 10.0]))
        assert torch.allclose(results[1], torch.tensor([3.0, 3.0, 30.0, 30.0]))
        assert torch.allclose(results[2], torch.tensor([7.0, 7.0, 70.0, 70.0]))


    def test_multiple_assignments_composition(self):
        """Test multiple assignments mixing rep, branch, and nbr."""
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        cond = torch.tensor([True, True, False, False])
        
        results_x = []
        results_y = []
        results_z = []
        
        for t in range(3):
            with ctx.round():
                # top level rep
                x = rep("x_rep", torch.zeros(n), lambda s: s + 1.0)
                
                # branch that uses x
                y = branch(
                    cond,
                    lambda: rep("y_true", torch.zeros(n), lambda s: nbr(s, aggr="sum") + x),
                    lambda: rep("y_false", torch.zeros(n), lambda s: nbr(s, aggr="max") + x * 2)
                )
                
                # another variable that combines both
                z = rep("z_rep", torch.zeros(n), lambda s: nbr(s + y, aggr="sum"))
                
            results_x.append(x.clone())
            results_y.append(y.clone())
            results_z.append(z.clone())

        # Round 0
        assert torch.allclose(results_x[0], torch.tensor([1.0, 1.0, 1.0, 1.0]))
        assert torch.allclose(results_y[0], torch.tensor([1.0, 1.0, 2.0, 2.0]))
        assert torch.allclose(results_z[0], torch.tensor([1.0, 3.0, 3.0, 2.0]))
        
        # Round 1
        assert torch.allclose(results_x[1], torch.tensor([2.0, 2.0, 2.0, 2.0]))
        assert torch.allclose(results_y[1], torch.tensor([3.0, 3.0, 6.0, 6.0]))
        assert torch.allclose(results_z[1], torch.tensor([6.0, 13.0, 14.0, 9.0]))
        
        # Round 2
        assert torch.allclose(results_x[2], torch.tensor([3.0, 3.0, 3.0, 3.0]))
        assert torch.allclose(results_y[2], torch.tensor([6.0, 6.0, 12.0, 12.0]))
        assert torch.allclose(results_z[2], torch.tensor([19.0, 38.0, 40.0, 26.0]))


# ===== Simulation framework tests =====

class TestSimulationFramework:
    def test_event_schedule_moves_source(self):
        scenario = GridScenario(3, 3, connectivity=4)
        engine = SimulationEngine.from_scenario(scenario)

        source = scenario.marker(0, 0)

        def move_to_center(runtime):
            runtime.signals["source"] = scenario.marker(1, 1)
            runtime.metadata["source_pos"] = (1, 1)

        schedule = EventSchedule([
            ScheduledEvent(round_idx=2, callback=move_to_center, name="move_center")
        ])
        recorder = SnapshotRecorder(state_fields=["dist"], capture_output=True, record_rounds={1, 2, 5})

        def program(runtime):
            src = runtime.signals["source"]
            return rep("dist", float("inf"), lambda d: mux(src, field.of(0.0), nbr(d + 1.0, aggr="min")))

        output, runtime = engine.run(
            rounds=6,
            program=program,
            signals={"source": source},
            metadata={"source_pos": (0, 0)},
            schedule=schedule,
            recorder=recorder,
        )

        center_idx = scenario.pos_to_idx(1, 1)
        assert runtime.metadata["source_pos"] == (1, 1)
        assert recorder.records[2]["output"][center_idx].item() == 0.0
        assert output[center_idx].item() == 0.0

    def test_step_api_and_recording(self):
        scenario = GridScenario(2, 2, connectivity=4)
        engine = SimulationEngine.from_scenario(scenario)
        runtime = engine.init_runtime(signals={"source": scenario.marker(0, 0)})
        recorder = SnapshotRecorder(state_fields=["dist"], capture_output=True, record_rounds={0, 1})

        def program(rt):
            src = rt.signals["source"]
            return rep("dist", float("inf"), lambda d: mux(src, field.of(0.0), nbr(d + 1.0, aggr="min")))

        out0 = engine.step(runtime=runtime, program=program, recorder=recorder)
        out1 = engine.step(runtime=runtime, program=program, recorder=recorder)

        assert runtime.round_idx == 2
        assert 0 in recorder.records and 1 in recorder.records
        assert torch.allclose(out0, recorder.records[0]["output"])
        assert torch.allclose(out1, recorder.records[1]["output"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
