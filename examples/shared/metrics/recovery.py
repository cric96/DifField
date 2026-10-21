"""How long a run takes to return within tolerance after an event."""

import math


def recovery(errors, start, threshold=0.2, consecutive=5):
    """The rounds from ``start`` to the first of ``consecutive`` rounds with errors in tolerance.

    Args:
        errors: One error per round; ``None`` and non-finite errors are never in tolerance.
        start: The round of the event.
        threshold: The largest error in tolerance.
        consecutive: How many rounds in a row recovery needs.

    Returns:
        ``status`` (``recovered`` if the run was in tolerance for ``consecutive`` rounds before
        the event, ``attained`` if not, ``no_recovery`` if it never gets there: a censored run),
        ``rounds`` (``None`` when censored) and ``baseline_compliant``.
    """
    compliant = [e is not None and math.isfinite(e) and e <= threshold for e in errors]
    before = compliant[max(0, start - consecutive) : start]
    baseline = len(before) == consecutive and all(before)
    for t in range(start, len(errors) - consecutive + 1):
        if all(compliant[t : t + consecutive]):
            return {
                "status": "recovered" if baseline else "attained",
                "rounds": t - start,
                "baseline_compliant": baseline,
            }
    return {"status": "no_recovery", "rounds": None, "baseline_compliant": baseline}
