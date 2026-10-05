"""Guard independent input and output artifact directories."""


def separate(source, out):
    source, out = source.resolve(), out.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    out.mkdir(parents=True, exist_ok=True)
