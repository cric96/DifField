"""Alignment: evaluation-tree paths identifying each syntactic occurrence.

Field calculus identifies every occurrence of a stateful construct by its
position in the *evaluation tree*, not by its position in the source text.  This
module implements that identity as a ``/``-joined path of tokens, each of the
form ``kind#index`` or ``kind:label``:

    /gradient:dist/it#0/gt#0
    /br:obstacle/T/collect_cast#0/it#0
    /it#2

``index`` is an occurrence counter for that *kind* within the current frame, so
uniqueness is structural: a counter cannot collide with itself.  Two calls to the
same library block therefore land in distinct frames automatically, and the same
logical program written at two source locations yields the *same* path — which is
what lets a local replay line up with a global run.

Paths contain no filenames, line numbers, hashes or object ids, so they are
byte-identical across processes and usable directly as message keys.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager

from .stack import context_stack

SEP = "/"

_CHECK_MODES = ("off", "warn", "error")
_check_mode = "warn"


class AlignmentError(RuntimeError):
    """Raised when two constructs in the same scope claim the same identity."""


def set_alignment_check(mode: str) -> None:
    """Configure round-to-round alignment drift reporting.

    ``"off"`` disables the check, ``"warn"`` (default) reports added/removed
    keys once per round, ``"error"`` raises.  Drift means the program's
    evaluation tree changed shape between rounds — usually a Python-level
    ``if`` reshuffling occurrence counters, which silently re-keys state.
    """
    global _check_mode
    if mode not in _CHECK_MODES:
        raise ValueError(f"mode must be one of {_CHECK_MODES}, got {mode!r}")
    _check_mode = mode


def get_alignment_check() -> str:
    """Return the current alignment drift check mode."""
    return _check_mode


def split_path(key: str) -> list[str]:
    """Split an alignment path into its tokens, dropping the leading empty one."""
    return [segment for segment in key.split(SEP) if segment]


def path_has_label(key: str, label: str) -> bool:
    """Return whether any token of *key* carries ``:label``."""
    needle = ":" + label
    return any(segment.endswith(needle) for segment in split_path(key))


class _Frame:
    """One level of the evaluation tree: a prefix plus per-kind counters."""

    __slots__ = ("counters", "minted", "prefix")

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix  # already ends with SEP
        self.counters: dict[str, int] = {}
        self.minted: list[str] = []

    def token(self, kind: str, label: str | None) -> str:
        """Mint the next token for *kind*, or a stable labelled token."""
        if label is not None:
            # Labelled tokens carry no index so that adding an unlabelled
            # sibling never renames a state the user checkpoints by name.
            return f"{kind}:{label}"
        index = self.counters.get(kind, 0)
        self.counters[kind] = index + 1
        return f"{kind}#{index}"


class Aligner:
    """Maintains the alignment path for one execution context."""

    def __init__(self) -> None:
        self._frames: list[_Frame] = [_Frame(SEP)]
        self._seen: set[str] = set()
        self._previous: set[str] | None = None

    # ------------------------------------------------------------------
    # path construction
    # ------------------------------------------------------------------
    @property
    def path(self) -> str:
        """The current frame's prefix."""
        return self._frames[-1].prefix

    def key(self, kind: str, label: str | None = None) -> str:
        """Mint a leaf key for a construct occurrence in the current frame."""
        top = self._frames[-1]
        key = top.prefix + top.token(kind, label)
        if key in self._seen:
            raise AlignmentError(
                f"duplicate alignment key {key!r}: two constructs in the same "
                f"scope use the label {label!r}. Give them distinct names."
            )
        self._seen.add(key)
        for frame in self._frames:
            frame.minted.append(key)
        return key

    @contextmanager
    def scope(self, kind: str, label: str | None = None) -> Generator[str, None, None]:
        """Mint a key *and* open a child frame under it (used by ``iterate``)."""
        key = self.key(kind, label)
        self._frames.append(_Frame(key + SEP))
        try:
            yield key
        finally:
            self._pop()

    @contextmanager
    def frame(
        self, kind: str, label: str | None = None
    ) -> Generator[_Frame, None, None]:
        """Open a frame without minting a key (functions, branch partitions)."""
        parent = self._frames[-1]
        child = _Frame(parent.prefix + parent.token(kind, label) + SEP)
        self._frames.append(child)
        try:
            yield child
        finally:
            self._pop()

    def _pop(self) -> None:
        child = self._frames.pop()
        self._frames[-1].minted.extend(child.minted)

    # ------------------------------------------------------------------
    # round lifecycle
    # ------------------------------------------------------------------
    def minted(self) -> set[str]:
        """Keys minted since the last reset."""
        return set(self._frames[0].minted)

    def begin_round(self) -> None:
        """Clear the path so the next round mints the same keys again."""
        self._frames = [_Frame(SEP)]
        self._seen = set()

    def end_round(self) -> None:
        """Compare this round's evaluation tree against the previous one."""
        self._report_drift(self._seen)
        self._previous = set(self._seen)

    def reset(self) -> None:
        """Forget the path and the drift baseline entirely."""
        self.begin_round()
        self._previous = None

    def _report_drift(self, current: set[str]) -> None:
        if _check_mode == "off" or self._previous is None:
            return
        added = sorted(current - self._previous)
        removed = sorted(self._previous - current)
        if not added and not removed:
            return
        message = (
            "alignment drift between rounds: the program's evaluation tree "
            "changed shape, so persistent state was re-keyed. "
            f"added={added} removed={removed}"
        )
        if _check_mode == "error":
            raise AlignmentError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=4)


def current_aligner() -> Aligner | None:
    """Return the active context's aligner, or ``None`` outside a round."""
    stack = context_stack()
    return stack[-1].align if stack else None


def aggregate(
    fn: Callable | None = None, *, kind: str | None = None, label_arg: str = "name"
) -> Callable:
    """Mark a function as an aggregate block so its body gets its own frame.

    Two calls to a decorated block occupy distinct frames, so the states inside
    them never alias.  Correctness does not depend on the decorator — occurrence
    counters already keep sibling calls disjoint — but a frame keeps the keys
    stable when the caller changes, and keeps them readable.

    ``label_arg`` names the keyword whose value labels the frame (``name`` by
    default), so ``gradient(src, name="dist")`` yields ``/gradient:dist/...``.
    """

    def decorate(func: Callable) -> Callable:
        tag = kind or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            aligner = current_aligner()
            if aligner is None:  # callable outside a round; nothing to align
                return func(*args, **kwargs)
            with aligner.frame(tag, kwargs.get(label_arg)):
                return func(*args, **kwargs)

        wrapper.__aggregate__ = True  # type: ignore[attr-defined]
        return wrapper

    return decorate(fn) if fn is not None else decorate


def resolve_label(query: str, keys: Iterable[str]) -> str:
    """Resolve a bare label to a full alignment path among *keys*.

    An exact match wins.  Otherwise a query matches a key when some token of
    that key carries the label, and an ambiguous query raises.
    """
    known = list(keys)
    if query in known:
        return query
    hits = [key for key in known if path_has_label(key, query)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise KeyError(f"no entry matching {query!r}; known: {sorted(known)}")
    raise AlignmentError(
        f"{query!r} is ambiguous, it matches {sorted(hits)}. "
        "Use the full alignment path."
    )


class AlignedDict(dict):
    """A dict whose lookups fall back to alignment-label resolution.

    Exports and neighbour message overrides are keyed by full alignment path,
    but users refer to them by the label they wrote (``"dist_src"``).  A miss on
    the exact key therefore retries as a label.
    """

    def _resolved(self, key):
        if not isinstance(key, str):
            raise KeyError(key)
        return resolve_label(key, self.keys())

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        return dict.__getitem__(self, self._resolved(key))

    def __contains__(self, key) -> bool:
        if dict.__contains__(self, key):
            return True
        try:
            self._resolved(key)
        except (KeyError, AlignmentError):
            return False
        return True

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, AlignmentError):
            return default
