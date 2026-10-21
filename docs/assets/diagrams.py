"""Draw the conceptual figures of the docs as SVG, in a light and in a dark theme.

Each figure is written to ``docs/assets/<name>.svg`` and ``docs/assets/<name>-dark.svg``;
the pages pick one with ``<picture>`` and ``prefers-color-scheme``:

    uv run python docs/assets/diagrams.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from pathlib import Path

ASSETS = Path(__file__).resolve().parent
SANS = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', SFMono-Regular, Menlo, Consolas, monospace"
# Distances along the colours of the logo: near is blue, far is orange; white text on all.
RAMP = ("#1f5fa8", "#1f7f74", "#3f7f2f", "#b45309", "#9a3412")
# Arrowheads are 10 px whatever the stroke, and reach this far past the end of their stroke,
# so that the stroke ends inside the head instead of poking out of its tip.
ARROW_TIP = 3


@dataclass(frozen=True)
class Theme:
    """The colours of a figure on a page of one theme, and the suffix of its file."""

    suffix: str
    text: str
    muted: str
    card: str
    border: str
    edge: str
    blue: str
    green: str
    orange: str
    tint: str  # behind a highlighted element


LIGHT = Theme(
    suffix="", text="#1f2328", muted="#59636e", card="#f6f8fa", border="#d1d9e0",
    edge="#afb8c1", blue="#1f5fa8", green="#2f7d4f", orange="#b45309", tint="#fff4e5",
)
DARK = Theme(
    suffix="-dark", text="#f0f6fc", muted="#9198a1", card="#151b23", border="#3d444d",
    edge="#57606a", blue="#6ea8fe", green="#7ee2a8", orange="#f5a35c", tint="#2d2116",
)


class Svg:
    """A figure under construction: shapes are appended in drawing order."""

    def __init__(self, width: int, height: int, theme: Theme) -> None:
        self.width, self.height, self.theme = width, height, theme
        self.parts: list[str] = []

    def add(self, element: str) -> None:
        """Append raw SVG markup."""
        self.parts.append(element)

    def card(self, x: float, y: float, w: float, h: float, *, fill: str | None = None,
             stroke: str | None = None, dashed: bool = False, radius: float = 14) -> None:
        """A rounded panel."""
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill or self.theme.card}" stroke="{stroke or self.theme.border}" '
            f'stroke-width="1.5"{dash}/>'
        )

    def text(self, x: float, y: float, content: str, *, size: float = 14,
             color: str | None = None, weight: int = 400, anchor: str = "start",
             mono: bool = False) -> None:
        """A line of text, escaped unless it is already made of tspans."""
        body = content if content.startswith("<tspan") else escape(content)
        self.add(
            f'<text x="{x}" y="{y}" font-family="{MONO if mono else SANS}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color or self.theme.text}" '
            f'text-anchor="{anchor}" xml:space="preserve">{body}</text>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, *, color: str | None = None,
             width: float = 1.5, arrow: bool = False, dashed: bool = False) -> None:
        """A straight segment, with the tip of an arrowhead at its end if asked."""
        marker = ""
        if arrow:
            marker = f' marker-end="url(#{self._marker(color or self.theme.edge)})"'
            x1, y1, x2, y2 = shorten(x1, y1, x2, y2, 0, ARROW_TIP)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color or self.theme.edge}" '
            f'stroke-width="{width}" stroke-linecap="round"{dash}{marker}/>'
        )

    def path(self, d: str, *, color: str, width: float = 2, dashed: bool = False) -> None:
        """A path ending in an arrowhead, whose tip lies ARROW_TIP px past the end of d."""
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        self.add(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round"{dash} marker-end="url(#{self._marker(color)})"/>'
        )

    def node(self, x: float, y: float, label: str, fill: str, *, r: float = 15,
             ring: str | None = None) -> None:
        """A device: a filled disc with its value in white."""
        stroke = ring or self.theme.card
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{3 if ring else 2}"/>')
        self.text(x, y + 4.5, label, size=13, color="#ffffff", weight=700, anchor="middle")

    def chip(self, x: float, y: float, label: str, color: str) -> None:
        """A small tag on a link, centred on (x, y)."""
        width = 14 + 8.4 * len(label)
        self.add(f'<rect x="{x - width / 2}" y="{y - 12}" width="{width}" height="24" rx="12" '
                 f'fill="{self.theme.card}" stroke="{color}" stroke-width="1.5"/>')
        self.text(x, y + 5, label, size=14, color=color, weight=700, anchor="middle", mono=True)

    def _marker(self, color: str) -> str:
        return "arrow-" + color.lstrip("#")

    def render(self) -> str:
        """The finished document."""
        colors = {part.split("url(#arrow-")[1].split(")")[0]
                  for part in self.parts if "url(#arrow-" in part}
        markers = "".join(
            f'<marker id="arrow-{color}" viewBox="0 0 10 10" refX="{10 - ARROW_TIP}" refY="5" '
            f'markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" '
            f'orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="#{color}"/></marker>'
            for color in sorted(colors)
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}">'
            f"<defs>{markers}</defs>{''.join(self.parts)}</svg>\n"
        )


def shorten(x1: float, y1: float, x2: float, y2: float, start: float,
            end: float) -> tuple[float, float, float, float]:
    """The segment from (x1, y1) to (x2, y2), less ``start`` px at its start, ``end`` at its end."""
    length = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    return x1 + ux * start, y1 + uy * start, x2 - ux * end, y2 - uy * end


def code(*tokens: tuple[str, str | None]) -> str:
    """A line of code as tspans: each token is (text, colour or None for the default)."""
    return "".join(
        f'<tspan fill="{color}">{escape(text)}</tspan>' if color else
        f"<tspan>{escape(text)}</tspan>"
        for text, color in tokens
    )


def program(theme: Theme) -> Svg:
    """The idea in three steps: write one program, run it as message passing, learn through it."""
    svg = Svg(960, 340, theme)
    blue, green, orange = theme.blue, theme.green, theme.orange
    headings = ("1. Write one program", "2. Every node runs it", "3. Gradients flow back")
    for index, heading in enumerate(headings):
        x = 10 + index * 318
        svg.card(x, 10, 302, 320)
        svg.text(x + 151, 44, heading, size=17, weight=700, anchor="middle",
                 color=(blue, green, orange)[index])

    lines = [
        code(("iterate", blue), ("(", None)),
        code(("  field.inf(),", None)),
        code(("  lambda", theme.muted), (" d: ", None), ("mux", blue), ("(", None)),
        code(("    source, field.of(0.0),", None)),
        code(("    ", None), ("gather_min", blue), ("(", None)),
        code(("      ", None), ("scatter", blue), ("(d)", None)),
        code(("      + ", None), ("scatter_range", blue), ("()", None)),
        code(("    ),", None)),
        code(("  ),", None)),
        code((")", None)),
    ]
    for row, line in enumerate(lines):
        svg.text(36, 82 + row * 21, line, size=13.5, mono=True)
    svg.text(161, 314, "the distance to the nearest source", size=13, color=theme.muted,
             anchor="middle")

    # A small network: the value of each device is its distance from the source.
    nodes = {
        "s": (372, 120, "0"), "a": (440, 82, "1"), "b": (446, 168, "1"),
        "c": (516, 122, "2"), "d": (520, 214, "2"), "e": (586, 170, "3"),
        "f": (590, 262, "3"), "g": (380, 230, "1"),
    }
    links = [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("b", "d"), ("c", "e"),
             ("d", "e"), ("d", "f"), ("s", "g"), ("g", "d"), ("e", "f")]
    for first, second in links:
        (x1, y1, _), (x2, y2, _) = nodes[first], nodes[second]
        svg.line(x1, y1, x2, y2)
    for sender in ("b", "e", "g"):
        (x1, y1, _), (x2, y2, _) = nodes[sender], nodes["d"]
        svg.line(*shorten(x1, y1, x2, y2, 17, 19), color=green, width=2.5, arrow=True)
    for key, (x, y, label) in nodes.items():
        svg.node(x, y, label, RAMP[int(label)], ring=green if key == "d" else None)
    svg.text(479, 300, "values cross the links, each node", size=13, color=theme.muted,
             anchor="middle")
    svg.text(479, 318, "keeps the least: a round of message passing", size=13,
             color=theme.muted, anchor="middle")

    # Rounds unroll in time; the loss at the end reaches the parameters of every round.
    for row, label in enumerate(("round 1", "round 2", "⋮", "round T")):
        y = 70 + row * 48
        if label == "⋮":
            svg.text(770, y + 22, label, size=18, color=theme.muted, anchor="middle")
            continue
        svg.card(712, y, 116, 34, fill=theme.tint, stroke=theme.border, radius=8)
        svg.text(770, y + 22, label, size=14, weight=600, anchor="middle")
        svg.line(690, 248, 711, y + 17, color=theme.muted, width=1)
    for y1, y2 in ((105, 117), (153, 165), (201, 213)):
        svg.line(770, y1, 770, y2, color=blue, width=2, arrow=True)
    svg.card(712, 262, 116, 34, fill=theme.card, stroke=orange, radius=8)
    svg.text(770, 284, "loss", size=14, weight=700, anchor="middle", color=orange)
    svg.line(770, 249, 770, 261, color=blue, width=2, arrow=True)
    # The length of the route ends on a dash, so the arrowhead stays attached to the line.
    svg.path("M 830 279 H 852 A 12 12 0 0 0 864 267 V 99 A 12 12 0 0 0 852 87 H 834",
             color=orange, width=2.2, dashed=True)
    svg.text(906, 190, "∂loss/∂θ", size=14, weight=700, color=orange, anchor="middle")
    svg.card(656, 232, 34, 32, fill=theme.card, stroke=theme.muted, radius=16)
    svg.text(673, 253, "θ", size=16, weight=700, anchor="middle", color=theme.muted)
    svg.text(797, 318, "θ: link costs, weights, networks", size=13, color=theme.muted,
             anchor="middle")
    return svg


def round_anatomy(theme: Theme) -> Svg:
    """One round of the distance program: a node field, a link field, a node field again."""
    svg = Svg(960, 342, theme)
    blue, green, orange = theme.blue, theme.green, theme.orange
    graph = {"s": (0, 40), "b": (78, 30), "c": (150, 78), "d": (98, 128)}
    links = [("s", "b"), ("b", "c"), ("b", "d"), ("c", "d")]
    stages = (
        ("node field  d  [N]", "what each node holds", ("0", "1", "2", "∞")),
        ("link field  [E]", "scatter(d) + scatter_range()", None),
        ("node field  [N]", "gather_min, then mux at the source", ("0", "1", "2", "2")),
    )
    for index, (title, caption, values) in enumerate(stages):
        x0 = 10 + index * 318
        svg.card(x0, 10, 290, 250)
        svg.text(x0 + 145, 42, title, size=15, weight=700, anchor="middle", mono=True,
                 color=(blue, green, orange)[index])
        svg.text(x0 + 145, 244, caption, size=13, color=theme.muted, anchor="middle",
                 mono=index == 1)
        ox, oy = x0 + 70, 80
        for first, second in links:
            (x1, y1), (x2, y2) = graph[first], graph[second]
            svg.line(ox + x1, oy + y1, ox + x2, oy + y2,
                     color=theme.edge if index != 1 else theme.border)
        if index == 1:
            # The values the links into d carry: the distance of the sender, plus one.
            for sender, value, side in (("b", "2", 1), ("c", "3", -1)):
                (x1, y1), (x2, y2) = graph[sender], graph["d"]
                svg.line(*shorten(ox + x1, oy + y1, ox + x2, oy + y2, 17, 19), color=green,
                         width=2.5, arrow=True)
                # Beside the link, on the side away from the other one.
                length = math.hypot(x2 - x1, y2 - y1)
                nx, ny = side * (y1 - y2) / length, side * (x2 - x1) / length
                svg.chip(ox + (x1 + x2) / 2 + 18 * nx, oy + (y1 + y2) / 2 + 18 * ny, value, green)
        for position, (key, (x, y)) in enumerate(graph.items()):
            if index == 1:
                svg.add(f'<circle cx="{ox + x}" cy="{oy + y}" r="15" fill="{theme.card}" '
                        f'stroke="{green if key == "d" else theme.edge}" stroke-width="2"/>')
                svg.text(ox + x, oy + y + 4.5, key, size=13, weight=700, anchor="middle",
                         color=theme.muted)
                continue
            value = values[position]
            fill = RAMP[min(int(value), 4)] if value != "∞" else theme.edge
            ring = orange if index == 2 and key == "d" else None
            svg.node(ox + x, oy + y, value, fill, ring=ring)
    for x in (300, 618):
        svg.line(x + 4, 135, x + 24, 135, color=theme.muted, width=2, arrow=True)
    svg.path("M 790 262 V 288 A 12 12 0 0 1 778 300 H 172 A 12 12 0 0 1 160 288 V 265",
             color=blue, width=2)
    svg.text(475, 330, "iterate keeps the result: the next round starts from it", size=13.5,
             color=blue, anchor="middle", weight=600)
    return svg


def architecture(theme: Theme) -> Svg:
    """The packages of diffield, from the programs down to the runtime."""
    svg = Svg(960, 420, theme)
    blue, green, orange = theme.blue, theme.green, theme.orange
    svg.card(10, 10, 600, 56, fill="none", dashed=True)
    svg.text(30, 36, "programs", size=15, weight=700)
    svg.text(30, 56, "your code, and the experiments in examples/ (moving to a repository of "
             "their own)", size=13, color=theme.muted)
    stack = (
        ("diffield.dsl",
         "iterate · scatter · gather · branch · mux · gradient · broadcast · elect"),
        ("diffield.layers", "the primitives as torch.nn.Module: IterateLayer, GatherLayer, …"),
        ("diffield.functional", "scatter_aggr, masks, folds: exact in hard mode, smooth in soft"),
        ("diffield.core", "AggregateContext · RoundContext · StateManager · alignment"),
    )
    for row, (name, content) in enumerate(stack):
        y = 96 + row * 78
        svg.card(10, y, 600, 60)
        svg.text(30, y + 26, name, size=15, weight=700, mono=True, color=blue)
        svg.text(30, y + 47, content, size=13, color=theme.muted)
        svg.line(310, y - 28 if row == 0 else y - 16, 310, y - 2, color=theme.muted, width=1.8,
                 arrow=True)
    side = (
        (96, "diffield.sim", green,
         ("scenarios, SimulationEngine,", "events, recording:", "rounds over the whole graph")),
        (252, "diffield.decentralized", orange,
         ("the same program, one device", "at a time on Mesa,", "talking by messages")),
    )
    for y, name, color, rows in side:
        svg.card(650, y, 300, 138)
        svg.text(670, y + 28, name, size=15, weight=700, mono=True, color=color)
        for index, row in enumerate(rows):
            svg.text(670, y + 56 + index * 20, row, size=13, color=theme.muted)
    svg.path("M 649 165 H 628 V 350 H 615", color=green, width=2)
    svg.path("M 649 321 H 638 V 372 H 615", color=orange, width=2)
    svg.text(310, 412, "dsl, sim and decentralized build on core, never the other way round",
             size=13, color=theme.muted, anchor="middle")
    return svg


def execution(theme: Theme) -> Svg:
    """The same program, batched over the graph or run device by device, gives the same field."""
    svg = Svg(960, 320, theme)
    blue, orange = theme.blue, theme.orange
    graph = {
        0: (60, 70), 1: (140, 50), 2: (120, 140), 3: (210, 110),
        4: (200, 200), 5: (290, 170), 6: (90, 220), 7: (300, 70),
    }
    links = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (2, 6), (6, 4),
             (3, 7), (5, 7)]
    distance = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 2, 7: 3}
    titles = (
        ("Batched", "SimulationEngine", blue),
        ("Device by device", "run_decentralized", orange),
    )
    for index, (title, api, color) in enumerate(titles):
        x0 = 10 + index * 520
        svg.card(x0, 10, 420, 300)
        svg.text(x0 + 24, 42, title, size=17, weight=700, color=color)
        svg.text(x0 + 396, 42, api, size=13, mono=True, color=theme.muted, anchor="end")
        ox, oy = x0 + 40, 40
        focus = {2} | {b for a, b in links if a == 2} | {a for a, b in links if b == 2}
        for first, second in links:
            (x1, y1), (x2, y2) = graph[first], graph[second]
            in_star = index == 1 and 2 in (first, second)
            svg.line(ox + x1, oy + y1, ox + x2, oy + y2, color=orange if in_star else theme.edge,
                     width=2.5 if in_star else 1.5, dashed=index == 1 and not in_star)
        for key, (x, y) in graph.items():
            faded = index == 1 and key not in focus
            fill = theme.edge if faded else RAMP[distance[key]]
            svg.node(ox + x, oy + y, str(distance[key]), fill,
                     ring=orange if index == 1 and key == 2 else None)
        caption = (
            "one tensor operation per round, over all N nodes"
            if index == 0 else "each device sees its neighbours' last messages"
        )
        svg.text(x0 + 210, 296, caption, size=13, color=theme.muted, anchor="middle")
    svg.text(480, 152, "=", size=44, weight=700, anchor="middle", color=theme.text)
    svg.text(480, 182, "same field,", size=13, color=theme.muted, anchor="middle")
    svg.text(480, 200, "node for node", size=13, color=theme.muted, anchor="middle")
    return svg


def inside_round(theme: Theme) -> Svg:
    """Inside a round: each primitive is a layer over a tensor operation on the round's context."""
    svg = Svg(960, 400, theme)
    blue, green, orange = theme.blue, theme.green, theme.orange
    headings = (
        (95, "program"), (335, "primitive and layer"), (615, "what it runs on"), (865, "the round"),
    )
    for x, heading in headings:
        svg.text(x, 32, heading, size=14, weight=700, color=theme.muted, anchor="middle")
    rows = (
        ("iterate(init, fn)", "IterateLayer", "StateManager", "a slot per place in the program"),
        ("gather(scatter(x))", "GatherLayer", "scatter_aggr", "fold the messages of each node"),
        ("branch(cond, …)", "BranchLayer", "mask_edges_for_partition",
         "a frame per side, no link across"),
        ("mux(cond, a, b)", "MuxLayer", "field_where", "pick, or blend when soft"),
    )
    tops = [50 + row * 78 for row in range(len(rows))]
    for x, title, lines, color in (
        (10, "program(runtime)", ("called once", "per round,", "for every node"), blue),
        (780, "RoundContext", ("links, ranges,", "state, alignment", "of this round"), orange),
    ):
        svg.card(x, 50, 170, tops[-1] + 60 - 50)
        svg.text(x + 85, 150, title, size=14, weight=700, mono=True, color=color, anchor="middle")
        for index, line in enumerate(lines):
            svg.text(x + 85, 178 + index * 20, line, size=13, color=theme.muted, anchor="middle")
    for y, (primitive, layer, operation, role) in zip(tops, rows, strict=True):
        svg.card(220, y, 230, 60)
        svg.text(236, y + 26, primitive, size=14, weight=700, mono=True, color=blue)
        svg.text(236, y + 47, layer, size=13, color=theme.muted, mono=True)
        svg.card(490, y, 250, 60)
        svg.text(506, y + 26, operation, size=14, weight=700, mono=True, color=green)
        svg.text(506, y + 47, role, size=13, color=theme.muted)
        for x1, x2 in ((180, 220), (450, 490), (740, 780)):
            svg.line(x1 + 2, y + 30, x2 - 2, y + 30, color=theme.muted, width=1.8, arrow=True)
    svg.text(480, 386, "hard mode: exact folds and choices · soft mode: the same calls, relaxed "
             "by a temperature", size=13, color=theme.muted, anchor="middle")
    return svg


def across_rounds(theme: Theme) -> Svg:
    """Across rounds: the engine applies the events, moves the links, runs a round and records."""
    svg = Svg(960, 330, theme)
    blue, green, orange = theme.blue, theme.green, theme.orange
    steps = (
        (10, 30, "1  events", "EventSchedule", green,
         ("callbacks before given rounds: change", "the signals, move devices")),
        (550, 30, "2  topology", "Scenario", green,
         ("hands its links and ranges to", "AggregateContext.update_topology")),
        (550, 200, "3  round", "AggregateContext.round()", blue,
         ("runs program(runtime): the state of the", "last round in, the new state out")),
        (10, 200, "4  record", "SnapshotRecorder", orange,
         ("keeps states, exports and outputs", "at the rounds you ask for")),
    )  # fmt: skip
    for x, y, title, owner, color, lines in steps:
        svg.card(x, y, 400, 100)
        svg.text(x + 20, y + 30, title, size=15, weight=700, color=color)
        svg.text(x + 380, y + 30, owner, size=13, mono=True, color=theme.muted, anchor="end")
        for index, line in enumerate(lines):
            svg.text(x + 20, y + 58 + index * 20, line, size=13, color=theme.muted)
    svg.line(412, 80, 548, 80, color=theme.muted, width=1.8, arrow=True)
    svg.line(750, 132, 750, 198, color=theme.muted, width=1.8, arrow=True)
    svg.line(548, 250, 412, 250, color=theme.muted, width=1.8, arrow=True)
    svg.line(210, 198, 210, 132, color=theme.muted, width=1.8, arrow=True)
    svg.text(222, 170, "next round", size=13, color=theme.muted)
    svg.text(480, 160, "SimulationEngine.run(rounds)", size=14, weight=700, mono=True,
             anchor="middle")
    svg.text(480, 182, "the state stays in the context", size=13, color=theme.muted,
             anchor="middle")
    svg.text(480, 320, "engine.ctx.reset() starts again from round zero", size=13,
             color=theme.muted, anchor="middle")
    return svg


def alignment(theme: Theme) -> Svg:
    """The state of a program lives at its places in the evaluation tree, named by paths."""
    svg = Svg(960, 310, theme)
    blue, orange = theme.blue, theme.orange
    svg.card(10, 10, 370, 290)
    lines = [
        code(("def", theme.muted), (" program(runtime):", None)),
        code(("    d = ", None), ("gradient", blue), ('(source, name="dist")', None)),
        code(("    total = ", None), ("branch", blue), ("(", None)),
        code(("        ~obstacle,", None)),
        code(("        lambda", theme.muted), (": ", None), ("collect_cast", blue),
             ("(d, …),", None)),
        code(("        lambda", theme.muted), (": field.zeros(),", None)),
        code(('        branch_name="obstacle")', None)),
        code(("    hops = ", None), ("iterate", blue), ("(field.zeros(), step)", None)),
    ]
    for row, line in enumerate(lines):
        svg.text(28, 52 + row * 24, line, size=13, mono=True)
    svg.text(195, 284, "the program, as written", size=13, color=theme.muted, anchor="middle")
    svg.line(382, 150, 418, 150, color=theme.muted, width=1.8, arrow=True)

    def chip_width(label: str) -> float:
        return 14 + 8.4 * len(label)

    tree = {  # name: (x, y, label, parent, holds state)
        "root": (440, 150, "/", None, False),
        "gradient": (545, 60, "gradient:dist", "root", False),
        "it-gradient": (690, 60, "it#0", "gradient", True),
        "branch": (545, 150, "br:obstacle", "root", False),
        "true": (690, 118, "T#0", "branch", False),
        "collect": (805, 118, "collect_cast#0", "true", False),
        "it-collect": (915, 118, "it#0", "collect", True),
        "false": (690, 182, "F#0", "branch", False),
        "it-hops": (545, 240, "it#0", "root", True),
    }
    for x, y, label, parent, _ in tree.values():
        if parent is not None:
            px, py, plabel, _, _ = tree[parent]
            svg.line(px + chip_width(plabel) / 2, py, x - chip_width(label) / 2, y,
                     color=theme.edge, width=1.5)
    for x, y, label, _, state in tree.values():
        svg.chip(x, y, label, orange if state else blue)
    svg.text(915, 146, "state", size=12, color=orange, anchor="middle")
    svg.text(690, 88, "state", size=12, color=orange, anchor="middle")
    svg.text(545, 268, "state", size=12, color=orange, anchor="middle")
    svg.text(690, 210, "no state", size=12, color=theme.muted, anchor="middle")
    svg.text(680, 296, code(("/br:obstacle/T#0/collect_cast#0/it#0", orange)), size=13,
             mono=True, anchor="middle")
    return svg


FIGURES = {
    "program": program,
    "round": round_anatomy,
    "architecture": architecture,
    "execution": execution,
    "inside-round": inside_round,
    "across-rounds": across_rounds,
    "alignment": alignment,
}


def main() -> None:
    """Write every figure in both themes."""
    for name, draw in FIGURES.items():
        for theme in (LIGHT, DARK):
            (ASSETS / f"{name}{theme.suffix}.svg").write_text(draw(theme).render())


if __name__ == "__main__":
    main()
