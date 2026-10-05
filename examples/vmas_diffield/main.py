#!/usr/bin/env python3
"""DIFFIELD aggregate programs as controllers in VMAS — SHAC RL + imitation.

Trains per-scenario aggregate programs (``programs.FieldProgram``, written in
the DIFFIELD DSL: election / gradient / broadcast / collect / gossip) inside
the VMAS differentiable simulator on 5 scenarios (flocking, flocking_beacon,
navigation, discovery, sampling) by:
  * SHAC: analytic-gradient RL backpropagating the VMAS reward (plus a
    differentiable soft-collision term) through the physics into the policy;
  * imitation: behaviour-cloning the hand-weighted program (parameter recovery
    + the depth-expressivity study with ``neural_dK`` students).

Policies: ``expert`` (hand-weighted program, no training), ``parametric``
(learned static program weights), ``hybrid`` (program weights controlled
per-agent/per-step by a neural gate network), ``neural`` (black-box GNN
baseline), ``hybrid_res`` (opt-in ablation: program + free force residual).

Extra evaluations per scenario: zero-shot scale generalization
(``--gen-scales``) and self-healing under mid-episode comm dropout / beacon
swap (``--perturb-drop``).

Outputs under ``<out-root>/vmas-<scenario>-<mode>/``: JSON + plots (reward /
metric curves, actor/critic-loss + field-grad training diagnostics, weight
trajectories, gate heatmaps, recovery bars, final bars, scale-gen,
self-healing; all mean ± 95% CI over seeds) + GIFs.

Examples:
  uv run python examples/vmas_diffield/main.py --mode shac
  uv run python examples/vmas_diffield/main.py --mode imitation --scenarios flocking,navigation
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from shared.metrics import aggregate, aggregate_curve  # noqa: E402
from shared.plotting import (  # noqa: E402
    apply_paper_style,
    color_of,
    export_moving_gif,
    label_of,
    panel_label,
)
from shared.plotting import band as _band  # noqa: E402
from shared.plotting import savefig as _savefig  # noqa: E402
from vmas_diffield.policies import build_policy  # noqa: E402
from vmas_diffield.scenarios import SCENARIO_SPEC  # noqa: E402
from vmas_diffield.train import (  # noqa: E402
    evaluate,
    evaluate_perturbed,
    make_program,
    train_imitation,
    train_shac,
)
from vmas_diffield.vmas_env import (  # noqa: E402
    env_action_range,
    make_diff_env,
    obs_to_perception,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_paper_style()
except ImportError:
    plt = None

# Colour/marker/linestyle roles are fixed in shared.plotting.style (ROLE_COLOR
# etc.) and looked up by policy kind via color_of/label_of, so every figure in
# this pipeline uses the same identity mapping.
# vmas_diffield/__init__.py relabels a few roles for this pipeline only.


def _curve_plot(runs_by_kind, attr, *, ylabel, out_path):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for kind, runs in runs_by_kind.items():
        means, cis = aggregate_curve([getattr(r, attr) for r in runs])
        _band(ax, runs[0].eval_steps[: len(means)], means, cis, role=kind)
    ax.set_xlabel("SHAC update")
    ax.set_ylabel(ylabel)
    ax.legend()
    _savefig(fig, out_path)


def _shac_loss_plot(runs_by_kind, out_path):
    """Training diagnostics: (a) actor loss, (b) critic TD loss, (c) pre-clip
    gradient norm on the interpretable field weights. Shows *that* and *how*
    each trained policy learns (the expert has no curves and is skipped);
    panel (c) is where e.g. navigation's brake gradient visibly turns on once
    agents start reaching their goals."""
    if plt is None:
        return
    panels = [
        ("actor_loss_curve", "actor loss (↓)", False),
        ("critic_loss_curve", "critic TD loss", True),
        ("grad_norm_field", "field-weight |grad| (pre-clip)", True),
    ]
    trained = {
        k: runs for k, runs in runs_by_kind.items()
        if getattr(runs[0], "actor_loss_curve", None)
    }
    if not trained:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6), squeeze=False)
    for ax, (attr, ylabel, log), tag in zip(axes[0], panels, "abc", strict=True):
        drew = False
        for kind, runs in trained.items():
            curves = [getattr(r, attr, []) for r in runs]
            curves = [c for c in curves if c]
            if not curves:
                continue  # e.g. field-grad panel for the pure neural policy
            means, cis = aggregate_curve(curves)
            if log:
                means = [max(m, 1e-12) for m in means]
            _band(ax, list(range(len(means))), means, cis, role=kind)
            drew = True
        if log and drew:
            ax.set_yscale("log")
        ax.set_xlabel("SHAC update")
        ax.set_ylabel(ylabel)
        if drew:
            ax.legend(fontsize=8)
        panel_label(ax, tag)
    _savefig(fig, out_path)


def _weight_traj_plot(runs_by_kind, out_path):
    if plt is None:
        return
    kinds = [k for k in runs_by_kind if runs_by_kind[k][0].weight_traj]
    if not kinds:
        return
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.6 * len(kinds), 3.8), squeeze=False)
    # Weight terms within a policy's panel are a small, fixed within-panel set
    # (not cross-figure identities), so a compact local qualitative cycle is
    # fine here; the cross-figure identity channel is the panel role/tag.
    term_colors = [
        "#2a78d6", "#e34948", "#1baf7a", "#eda100", "#8a5fd1", "#5f9ea0",
    ]
    for ax, kind, tag in zip(axes[0], kinds, "abcdefgh", strict=False):
        runs = runs_by_kind[kind]
        keys = list(runs[0].weight_traj[0].keys())
        x = runs[0].eval_steps[: len(runs[0].weight_traj)]
        for key, tcolor in zip(keys, term_colors, strict=False):
            curves = [[wt[key] for wt in r.weight_traj] for r in runs]
            means, cis = aggregate_curve(curves)
            lo = [m - c for m, c in zip(means, cis, strict=True)]
            hi = [m + c for m, c in zip(means, cis, strict=True)]
            ax.plot(x[: len(means)], means, color=tcolor, linewidth=1.8, label=key)
            ax.fill_between(x[: len(means)], lo, hi, color=tcolor, alpha=0.18, linewidth=0)
        ax.set_xlabel("SHAC update")
        ax.set_ylabel(f"{label_of(kind)}: softplus(weight)")
        ax.legend(fontsize=8)
        panel_label(ax, tag)
    _savefig(fig, out_path)


def _final_bars(runs_by_kind, primary, out_path):
    if plt is None:
        return
    keys = [("reward", "reward (↑)"), ("primary", f"{primary} (↑)")]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 3.8), squeeze=False)
    kinds = list(runs_by_kind)
    for ax, (key, ylabel), tag in zip(axes[0], keys, "ab", strict=True):
        means = [aggregate([r.final[key] for r in runs_by_kind[k]])["mean"] for k in kinds]
        cis = [aggregate([r.final[key] for r in runs_by_kind[k]])["ci"] for k in kinds]
        ax.bar(
            [label_of(k) for k in kinds],
            means,
            yerr=cis,
            capsize=4,
            color=[color_of(k) for k in kinds],
            width=0.6,
            error_kw={"linewidth": 1.2, "ecolor": "#52514e"},
        )
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(True, axis="y", alpha=0.5)
        ax.grid(False, axis="x")
        panel_label(ax, tag)
    _savefig(fig, out_path)


def _recovery_bars(results_by_kind, expert, out_path):
    if plt is None:
        return
    kinds = [k for k in results_by_kind if results_by_kind[k][0].recovered]
    if not kinds:
        return
    terms = list(results_by_kind[kinds[0]][0].recovered.keys())
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.6 * len(kinds), 3.8), squeeze=False)
    x = range(len(terms))
    for ax, kind, tag in zip(axes[0], kinds, "abcdefgh", strict=False):
        res = results_by_kind[kind]
        means = [aggregate([r.recovered[t] for r in res])["mean"] for t in terms]
        cis = [aggregate([r.recovered[t] for r in res])["ci"] for t in terms]
        ax.bar(
            x,
            means,
            0.5,
            yerr=cis,
            capsize=4,
            color=color_of(kind),
            error_kw={"linewidth": 1.2, "ecolor": "#52514e"},
            label=label_of(kind) + " (recovered)",
        )
        ax.scatter(
            list(x),
            [expert[t] for t in terms],
            color=color_of("expert"),
            marker="_",
            s=500,
            linewidths=2.2,
            label=label_of("expert"),
            zorder=3,
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(terms, rotation=20)
        ax.set_ylabel(f"{label_of(kind)} recovery")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.5)
        ax.grid(False, axis="x")
        panel_label(ax, tag)
    _savefig(fig, out_path)


def _loss_plot(results_by_kind, out_path):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for kind, res in results_by_kind.items():
        means, cis = aggregate_curve([r.loss_curve for r in res])
        # The program students recover the teacher to machine precision, so
        # their MSE underflows to literal 0.0 — which a log axis silently
        # drops, making exactly the policies that learn BEST vanish from the
        # plot. Floor at 1e-12 so "hit the numerical floor" stays visible.
        means = [max(m, 1e-12) for m in means]
        _band(ax, list(range(1, len(means) + 1)), means, cis, role=kind)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("behaviour-cloning MSE")
    ax.legend()
    ax.grid(True, which="both", alpha=0.5)
    _savefig(fig, out_path)


def _save_gif(pos_trace, out_path, title, fps):
    if plt is None or not pos_trace:
        return
    rounds = len(pos_trace)
    norm = [(p * 0.4 + 0.5).clamp(0, 1) for p in pos_trace]
    speed = [torch.zeros(norm[0].shape[0])] + [
        (norm[r] - norm[r - 1]).norm(dim=-1) for r in range(1, rounds)
    ]
    export_moving_gif(
        positions_by_round={r: norm[r] for r in range(rounds)},
        values_by_round={r: speed[r] for r in range(rounds)},
        source_idx=0,
        output_path=str(out_path),
        title=title,
        fps=fps,
        show_links=False,
        show_source=False,
    )


def _best(runs):
    return max(runs, key=lambda r: r.final["reward"])


def run_scale_generalization(scenario, args, runs_by_kind, out_dir):
    """Zero-shot transfer of trained weights to unseen swarm sizes.

    The central "aggregate computing" claim: a field program's weights are
    per-agent/per-edge quantities, so the same trained policy should run
    unchanged on more or fewer agents. This evaluates every already-trained
    policy (no retraining, no gradient steps) at ``args.gen_scales`` multiples
    of the scenario's base agent count and reports how the primary metric
    degrades with scale — the sharpest test of "did it learn a *program*, or
    did it memorise statistics of one fixed graph size".
    """
    if not args.gen_scales:
        return
    spec = SCENARIO_SPEC[scenario]
    base_n = spec.n_agents
    # Hard-capped at 2x: VMAS's ScenarioUtils.find_random_pos_for_entity spawns
    # entities via an *unbounded* rejection-sampling while-loop (no retry cap, just
    # a warning past 50k tries) in a fixed-size arena. Packing e.g. 3x discovery's
    # agents plus its 7 fixed targets pushes placement past feasibility and the
    # loop spins forever instead of raising — this hung a full run for ~30 CPU
    # minutes before being killed. 2x stays comfortably packable everywhere.
    n_list = sorted({max(2, min(round(base_n * s), base_n * 2)) for s in args.gen_scales})
    device = torch.device(args.device)
    curve = {kind: [] for kind in runs_by_kind}
    n_big = n_list[-1]
    best_big_trace = {}  # kind -> pos_trace, for a demo GIF at the largest N tested
    for n_agents in n_list:
        env = make_diff_env(
            scenario,
            num_envs=args.gen_num_envs,
            n_agents=n_agents,
            device=device,
            max_steps=args.episode_len + 2,
        )
        u_range = env_action_range(env)
        for kind, runs in runs_by_kind.items():
            vals = []
            candidates = []  # (reward, trace) — only populated at n_big
            for r in runs:
                obs = env.reset()
                p0 = obs_to_perception(obs, scenario=scenario, radius=args.radius)
                policy = build_policy(kind, p0, spec, scenario, hidden=args.hidden).to(device)
                policy.load_state_dict(r.policy_state)
                policy.eval()
                program = make_program(scenario, args)
                ev, trace, _ = evaluate(
                    env,
                    policy,
                    program,
                    scenario=scenario,
                    args=args,
                    u_range=u_range,
                    steps=args.episode_len,
                    record_trace=(n_agents == n_big and not args.no_render),
                )
                vals.append(ev["primary"])
                if n_agents == n_big and not args.no_render:
                    candidates.append((ev["reward"], trace))
                del policy
            curve[kind].append(aggregate(vals))
            if candidates:
                best_big_trace[kind] = max(candidates, key=lambda c: c[0])[1]
        _free_env(env)

    _scale_plot(n_list, base_n, curve, spec.primary_metric, out_dir / "scale_generalization.png")
    if n_big > base_n:
        # Visual "more agents" stress test: the same zero-shot-transferred policy,
        # rendered at the largest tested swarm size, so scale-invariance is
        # something you can *see* and not just a number in scale_generalization.png.
        for kind, trace in best_big_trace.items():
            _save_gif(
                trace,
                out_dir / f"scale_gen_{kind}_n{n_big}.gif",
                f"SHAC {scenario} @ N={n_big} (zero-shot, trained N={base_n}) — {label_of(kind)}",
                args.gif_fps,
            )
    return {"n_agents": n_list, "trained_n": base_n, "results": curve}


def _scale_plot(n_list, base_n, curve, primary, out_path):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for kind, points in curve.items():
        means = [pt["mean"] for pt in points]
        cis = [pt["ci"] for pt in points]
        ax.errorbar(
            n_list, means, yerr=cis, marker="o", capsize=3, color=color_of(kind),
            label=label_of(kind),
        )
    ax.axvline(base_n, color="#9a9890", linestyle=":", linewidth=1.0)
    ax.set_xlabel(f"agent count at evaluation (zero-shot, trained at N={base_n})")
    ax.set_ylabel(f"{primary} (↑)")
    ax.legend()
    _savefig(fig, out_path)


def _gate_heatmap(scenario, runs_by_kind, out_path):
    """How the hybrid's controller modulates the program: mean gate per term
    over the final evaluation episode (a), and the per-agent gate trace of the
    most-modulated term (b). Gates == 1 means "the plain program"."""
    if plt is None:
        return
    runs = runs_by_kind.get("hybrid")
    if not runs or not runs[0].gate_trace:
        return
    best = _best(runs)
    gates = torch.stack(best.gate_trace)  # [T, n_agents, K]
    terms = SCENARIO_SPEC[scenario].field_terms
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    term_colors = ["#2a78d6", "#e34948", "#1baf7a", "#eda100", "#8a5fd1", "#5f9ea0"]
    mean_g = gates.mean(dim=1)  # [T, K]
    for k, (term, tcolor) in enumerate(zip(terms, term_colors, strict=False)):
        axes[0].plot(mean_g[:, k], color=tcolor, linewidth=1.8, label=term)
    axes[0].axhline(1.0, color="#9a9890", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("episode step")
    axes[0].set_ylabel("mean gate (1 = plain program)")
    axes[0].legend(fontsize=8)
    panel_label(axes[0], "a")
    var_k = int(gates.var(dim=(0, 1)).argmax().item())
    im = axes[1].imshow(
        gates[:, :, var_k].T, aspect="auto", cmap="coolwarm", vmin=0.0, vmax=2.0,
        interpolation="nearest",
    )
    axes[1].set_xlabel("episode step")
    axes[1].set_ylabel(f"agent — gate({terms[var_k]})")
    fig.colorbar(im, ax=axes[1], shrink=0.85)
    panel_label(axes[1], "b")
    _savefig(fig, out_path)


def run_self_healing(scenario, args, runs_by_kind, out_dir):
    """Eval-only self-* probe: per-step primary-metric series while the comm
    graph degrades mid-episode (sustained i.i.d. edge dropout), plus — for
    partial-obs scenarios — a beacon swap that forces re-election. The
    program's persistent fields must re-converge on the fly; the same
    perturbed graph is what the neural baseline's message passing sees."""
    spec = SCENARIO_SPEC[scenario]
    cases: dict[str, dict] = {f"drop{dr:g}": {"drop_rate": dr} for dr in args.perturb_drop}
    if spec.n_knowers is not None:
        cases["knower_swap"] = {"swap_knower": True}
    if not cases:
        return None
    device = torch.device(args.device)
    env = make_diff_env(
        scenario, num_envs=args.gen_num_envs, n_agents=spec.n_agents,
        device=device, max_steps=args.episode_len + 2,
    )
    u_range = env_action_range(env)
    out = {}
    for kind, runs in runs_by_kind.items():
        best = _best(runs)
        obs = env.reset()
        p0 = obs_to_perception(obs, scenario=scenario, radius=args.radius)
        policy = build_policy(kind, p0, spec, scenario, hidden=args.hidden).to(device)
        policy.load_state_dict(best.policy_state)
        policy.eval()
        series_by_case = {}
        for label, kw in [("baseline", {}), *cases.items()]:
            program = make_program(scenario, args)
            series_by_case[label] = evaluate_perturbed(
                env, policy, program, scenario=scenario, args=args, u_range=u_range,
                steps=args.episode_len, seed=0, **kw,
            )
        out[kind] = series_by_case
        del policy
    _free_env(env)
    _self_healing_plot(
        out, list(cases), spec.primary_metric, args.episode_len // 2,
        out_dir / "self_healing.png",
    )
    return out


def _self_healing_plot(out, case_labels, primary, onset, out_path):
    if plt is None or not out:
        return
    fig, axes = plt.subplots(
        1, len(case_labels), figsize=(4.6 * len(case_labels), 3.6), squeeze=False
    )
    for ax, case, tag in zip(axes[0], case_labels, "abcdefgh", strict=False):
        for kind, cases in out.items():
            base = cases["baseline"]["primary"]
            pert = cases[case]["primary"]
            ax.plot(base, color=color_of(kind), linewidth=1.0, alpha=0.35)
            ax.plot(pert, color=color_of(kind), linewidth=1.8, label=label_of(kind))
        ax.axvline(onset, color="#52514e", linestyle="--", linewidth=1.0)
        ax.text(
            0.985, 0.03, f"{case} from dashed line\n(thin = unperturbed)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#52514e",
        )
        ax.set_xlabel("episode step")
        ax.set_ylabel(f"{primary} (↑)")
        ax.legend(fontsize=8)
        panel_label(ax, tag)
    _savefig(fig, out_path)


def policies_for(_scenario, requested):
    """All policy kinds run on every scenario (the per-scenario program is
    supplied by ``FieldProgram``; there is no scenario-gated policy anymore)."""
    return list(requested)


def run_shac(scenario, args, out_dir):
    spec = SCENARIO_SPEC[scenario]
    kinds = policies_for(scenario, args.policy_list)
    # One env per scenario, reused across policies/seeds (avoids GPU memory
    # accumulation from creating many VMAS worlds, which have circular refs).
    env = make_diff_env(
        scenario,
        num_envs=args.num_envs,
        n_agents=spec.n_agents,
        device=torch.device(args.device),
        max_steps=args.episode_len + args.horizon + 2,
    )
    # Per-(policy, seed) checkpoints make the sweep crash-resumable: a rare
    # native segfault (observed: heap corruption detonating inside numpy's
    # RandomState during VMAS's per-step local_seed swap, sampling only, ~once
    # per few hundred update-cycles) then costs one unit's progress instead of
    # the scenario — the launcher simply retries and completed units reload.
    ck_dir = out_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    runs_by_kind = {}
    for kind in kinds:
        print(f"\n=== SHAC {scenario} / {kind} ===")
        runs = []
        for s in args.seed_list:
            ck = ck_dir / f"shac_{kind}_seed{s}.pt"
            if ck.exists():
                print(f"  [resume] {scenario}/{kind} seed={s} from checkpoint")
                runs.append(torch.load(ck, weights_only=False))
                continue
            r = train_shac(scenario, kind, seed=s, args=args, env=env)
            torch.save(r, ck)
            runs.append(r)
        runs_by_kind[kind] = runs

    _curve_plot(
        runs_by_kind,
        "eval_reward",
        ylabel="mean per-step reward",
        out_path=out_dir / "shac_reward_curve.png",
    )
    _curve_plot(
        runs_by_kind,
        "eval_metric",
        ylabel=spec.primary_metric,
        out_path=out_dir / "shac_metric_curve.png",
    )
    _shac_loss_plot(runs_by_kind, out_dir / "shac_loss_curve.png")
    _weight_traj_plot(runs_by_kind, out_dir / "shac_weight_trajectories.png")
    _final_bars(runs_by_kind, spec.primary_metric, out_dir / "shac_final_bars.png")
    _gate_heatmap(scenario, runs_by_kind, out_dir / "gate_heatmap.png")
    scale_gen = run_scale_generalization(scenario, args, runs_by_kind, out_dir)
    self_healing = run_self_healing(scenario, args, runs_by_kind, out_dir)
    if not args.no_render:
        for kind, runs in runs_by_kind.items():
            _save_gif(
                _best(runs).pos_trace,
                out_dir / f"shac_{kind}.gif",
                f"SHAC {scenario} — {label_of(kind)}",
                args.gif_fps,
            )

    summary = {
        "scenario": scenario,
        "mode": "shac",
        "seeds": args.seed_list,
        "primary_metric": spec.primary_metric,
        "results": {
            kind: {
                "final_reward": aggregate([r.final["reward"] for r in runs]),
                "final_primary": aggregate([r.final["primary"] for r in runs]),
                "weights": runs[0].weights,
                "n_params": runs[0].n_params,
            }
            for kind, runs in runs_by_kind.items()
        },
        "scale_generalization": scale_gen,
        "self_healing": self_healing,
    }
    _dump(summary, out_dir)
    print(f"\n=== SHAC {scenario} summary (mean ± 95% CI) ===")
    for kind, runs in runs_by_kind.items():
        rew = aggregate([r.final["reward"] for r in runs])
        met = aggregate([r.final["primary"] for r in runs])
        print(
            f"  {label_of(kind):>26}: reward={rew['mean']:.3f}±{rew['ci']:.3f}  "
            f"{spec.primary_metric}={met['mean']:.3f}±{met['ci']:.3f}  "
            f"params={runs[0].n_params}  w={runs[0].weights}"
        )
    _free_env(env)


def _free_env(env):
    del env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def run_imitation(scenario, args, out_dir):
    spec = SCENARIO_SPEC[scenario]
    kinds = args.imit_policy_list
    env = make_diff_env(
        scenario,
        num_envs=args.num_envs,
        n_agents=spec.n_agents,
        device=torch.device(args.device),
        max_steps=args.bc_rollout + 2,
    )
    ck_dir = out_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    results_by_kind = {}
    for kind in kinds:
        print(f"\n=== Imitation {scenario} / {kind} ===")
        results = []
        for s in args.seed_list:
            ck = ck_dir / f"imit_{kind}_seed{s}.pt"
            if ck.exists():
                print(f"  [resume] {scenario}/{kind} seed={s} from checkpoint")
                results.append(torch.load(ck, weights_only=False))
                continue
            r = train_imitation(scenario, kind, seed=s, args=args, env=env)
            torch.save(r, ck)
            results.append(r)
        results_by_kind[kind] = results
    expert = results_by_kind[kinds[0]][0].expert
    _loss_plot(results_by_kind, out_dir / "imit_loss_curve.png")
    _recovery_bars(results_by_kind, expert, out_dir / "imit_recovery_bars.png")
    if not args.no_render:
        for kind, res in results_by_kind.items():
            best = min(res, key=lambda r: r.final_loss)
            _save_gif(
                best.pos_trace,
                out_dir / f"imit_{kind}.gif",
                f"Imitation {scenario} — {label_of(kind)}",
                args.gif_fps,
            )
    summary = {
        "scenario": scenario,
        "mode": "imitation",
        "seeds": args.seed_list,
        "expert": expert,
        "results": {
            kind: {
                "final_loss": aggregate([r.final_loss for r in res]),
                "recovered": {t: aggregate([r.recovered[t] for r in res]) for t in res[0].recovered}
                if res[0].recovered
                else None,
            }
            for kind, res in results_by_kind.items()
        },
    }
    _dump(summary, out_dir)
    print(f"\n=== Imitation {scenario} (expert={expert}) ===")
    for kind, res in results_by_kind.items():
        loss = aggregate([r.final_loss for r in res])
        line = f"  {label_of(kind):>26}: BC_loss={loss['mean']:.5f}±{loss['ci']:.5f}"
        if res[0].recovered:
            rec = {t: aggregate([r.recovered[t] for r in res])["mean"] for t in res[0].recovered}
            line += "  recovered=" + ", ".join(f"{t}={rec[t]:.2f}" for t in rec)
        print(line)
    _free_env(env)


def _dump(summary, out_dir):
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _add_program_args(p) -> None:
    # Aggregate-program knobs (programs.FieldProgram): grain of the S-block
    # election (hop metric: agents follow a leader within grain/2 hops, and a
    # vanished leader is forgotten in ~grain rounds), softmax temperature of
    # the gradient-descent read-out, and how many no-grad warm-up rounds to
    # run on the initial graph at each rollout reset (fields otherwise
    # converge over the first ~diameter physics steps — authentic, but it
    # eats into short SHAC horizons).
    p.add_argument("--elect-grain", type=float, default=16.0)
    p.add_argument("--desc-tau", type=float, default=0.15)
    p.add_argument("--program-warmup", type=int, default=4)
    # Hybrid gate regulariser: pulls gates toward 1 (= the plain program) so
    # the controller only deviates where deviation earns reward.
    p.add_argument("--gate-l2", type=float, default=3e-3)
    # Self-healing probe: sustained comm dropout rates applied from mid-episode
    # (empty string disables). Partial-obs scenarios additionally get a
    # knower-swap case automatically.
    p.add_argument("--perturb-drop", type=str, default="0.3,0.6")


def _add_reward_args(p) -> None:
    p.add_argument("--smooth-collision-weight", type=float, default=2.0)
    p.add_argument("--min-dist", type=float, default=0.15)
    # Shaped SHAC reward for `flocking` (order + cohesion + goal-tracking - separation);
    # replaces VMAS's built-in reward, which only shapes pairwise inter-agent distance
    # and ignores heading/target (see train.flocking_shaped_reward).
    p.add_argument("--flock-w-align", type=float, default=1.2)
    p.add_argument("--flock-w-cohesion", type=float, default=0.6)
    p.add_argument("--flock-w-separation", type=float, default=1.0)
    p.add_argument("--flock-w-goal", type=float, default=0.8)
    # flocking_beacon overrides the goal weight: tracking the unseen target IS
    # the task there (see train.shac_reward).
    p.add_argument("--beacon-w-goal", type=float, default=2.5)
    p.add_argument("--flock-cohesion-k", type=float, default=3.0)
    p.add_argument("--flock-cohesion-temp", type=float, default=0.05)
    # Dense shaping added to discovery's sparse covering reward: pulls each agent
    # toward its nearest uncovered target (see train.discovery_shaped_reward).
    p.add_argument("--disc-w-approach", type=float, default=0.3)
    # Navigation arrival reward (see train.navigation_shaped_reward): a wide
    # `approach` bump gives a cross-arena gradient, a sharp `hold` bump (scale ~
    # the 0.1 on-goal radius) makes parking on the goal the highest-paying state.
    # Replaces VMAS's net-progress reward, which pays to cross the goal disc at
    # speed as much as to sit on it and makes SHAC delete the brake.
    p.add_argument("--nav-w-approach", type=float, default=1.0)
    p.add_argument("--nav-approach-scale", type=float, default=0.6)
    p.add_argument("--nav-w-hold", type=float, default=1.5)
    p.add_argument("--nav-hold-scale", type=float, default=0.12)
    # How early the navigation PD controller starts braking (see brake_term):
    # tight enough to cruise to the edge of the on-goal disc, then stop hard.
    p.add_argument("--nav-arrive-radius", type=float, default=0.12)


def parse_args():
    p = argparse.ArgumentParser(description="DIFFIELD-in-VMAS: SHAC + imitation")
    p.add_argument("--mode", choices=["shac", "imitation"], default="shac")
    p.add_argument(
        "--scenarios", type=str,
        default="flocking,flocking_beacon,navigation,discovery,sampling",
    )
    p.add_argument("--policies", type=str, default="expert,parametric,hybrid,neural")
    # Imitation students: interpretable recoveries + the depth-expressivity
    # series (how deep must a feed-forward GNN be to clone the recurrent
    # multi-hop program?).
    p.add_argument(
        "--imit-policies", type=str, default="parametric,hybrid,neural,neural_d2,neural_d3"
    )
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-envs", type=int, default=96)
    p.add_argument("--updates", type=int, default=150)
    p.add_argument("--horizon", type=int, default=12)
    # 60 (not 40): with terminal speed ~0.4 the first ~20 steps of an episode are
    # pure travel from random spawns, which at 40 steps dominates every per-step
    # metric (navigation's on_goal_frac ceiling was ~0.3 for ANY policy) and
    # compresses the differences the evaluation is supposed to resolve.
    p.add_argument("--episode-len", type=int, default=60)
    p.add_argument("--eval-every", type=int, default=10)
    # Average the last N eval snapshots for the reported "final" scalar. A wider
    # window was tried to average out the neural policy's eval oscillation but
    # was reverted: it drags DOWN the program policies, whose reward is still
    # rising at the end, by averaging in the approach (measured flocking
    # parametric 1.49->1.38 at window 5) — under-reporting their converged value.
    p.add_argument("--final-window", type=int, default=3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--tau", type=float, default=0.2)
    # 3e-2, not 1e-2: Adam's per-update step on the raw field weights is
    # lr-bound, and the flat softplus(0.3) init sits ~3.5 raw units below
    # navigation's productive PD regime (w_goal ~2-4) — at 1e-2 that distance
    # alone eats ~350 updates, i.e. more than the whole budget (measured:
    # w_goal crawls 0.3->0.51 in 60 updates while its |grad| is a healthy
    # ~0.2-0.6). At 3e-2 the same run reaches the expert's reward within ~100
    # updates with the interpretable weights settling cleanly (few-parameter
    # policies tolerate the aggressive LR).
    p.add_argument("--actor-lr-field", type=float, default=3e-2)
    p.add_argument("--actor-lr-neural", type=float, default=3e-3)
    # Fraction of updates over which the phase-uniform window cap ramps from 0
    # (spawn-anchored, the reliable bootstrap distribution) to the full episode
    # (late phases enter the gradient); see train.py's module docstring. 0
    # disables the ramp (full-range prefix from update 0).
    p.add_argument("--phase-ramp-frac", type=float, default=0.33)
    # Hybrid residual: penalises ||residual||^2 so it stays a correction, not a
    # second unconstrained policy (see train.train_shac).
    p.add_argument("--residual-l2", type=float, default=3e-3)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--critic-epochs", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--act-scale", type=float, default=1.0)
    p.add_argument("--explore-frac-start", type=float, default=0.4)
    p.add_argument("--explore-frac-end", type=float, default=0.05)
    p.add_argument("--radius", type=float, default=0.5)
    p.add_argument("--sep", type=float, default=0.2)
    _add_program_args(p)
    _add_reward_args(p)
    p.add_argument("--bc-rollout", type=int, default=40)
    p.add_argument("--bc-epochs", type=int, default=120)
    p.add_argument("--bc-lr", type=float, default=0.02)
    # Cosine-decay the imitation LR to this fraction of its start by the last
    # epoch: the deep-GNN students blow out of their minimum late in training at
    # a flat aggressive LR (see train.train_imitation); decaying lets them
    # settle without slowing the initial descent.
    p.add_argument("--bc-lr-min-frac", type=float, default=0.05)
    # Hybrid only: its neural gate trains at bc_lr * this fraction (slower than
    # the field weights) — see train.train_imitation for why the flat-LR
    # combination gets stuck.
    p.add_argument("--bc-lr-neural-frac", type=float, default=0.25)
    p.add_argument("--gif-fps", type=int, default=8)
    p.add_argument("--no-render", action="store_true")
    # Zero-shot scale-generalization: re-evaluate trained SHAC policies (no
    # retraining) at these multiples of the scenario's training agent count.
    p.add_argument("--gen-scales", type=str, default="1,1.5,2")
    p.add_argument("--gen-num-envs", type=int, default=32)
    p.add_argument("--out-root", type=str, default="generated")
    return p.parse_args()


def main():
    args = parse_args()
    # Run the autograd backward pass single-threaded. VMAS rollouts build large
    # cross-module CUDA graphs; the multithreaded engine's per-device stream
    # handoff intermittently raises "Expected stream_.device_type() == CUDA"
    # in a backward worker thread. Single-threaded backward avoids that race at
    # a negligible cost for these small (16-env) graphs.
    torch.autograd.set_multithreading_enabled(False)
    args.scenario_list = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    args.policy_list = [s.strip() for s in args.policies.split(",") if s.strip()]
    args.imit_policy_list = [s.strip() for s in args.imit_policies.split(",") if s.strip()]
    args.seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.gen_scales = [float(s) for s in args.gen_scales.split(",") if s.strip()]
    args.perturb_drop = [float(s) for s in args.perturb_drop.split(",") if s.strip()]
    print(
        f"mode={args.mode} scenarios={args.scenario_list} policies={args.policy_list} "
        f"seeds={args.seed_list} device={args.device}"
    )
    for scenario in args.scenario_list:
        out_dir = Path(args.out_root) / f"vmas-{scenario}-{args.mode}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.mode == "shac":
            run_shac(scenario, args, out_dir)
        else:
            run_imitation(scenario, args, out_dir)
        print(f"Artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
