"""DIFFIELD-in-VMAS experiments: differentiable controllers trained by SHAC and imitation."""

from shared.plotting.style import ROLE_LABEL

# Relabel the shared plotting roles for THIS pipeline only (process-local; the
# shared defaults keep serving the other example scripts unchanged): here the
# policy spectrum is program-with-hand-θ → learned-θ → GNN-gated-θ → pure GNN.
ROLE_LABEL.update({
    "expert": "program (hand θ)",
    "parametric": "program (learned θ)",
    "hybrid": "hybrid (GNN-gated program)",
})
