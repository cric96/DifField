"""Stable named random streams shared by independent experiments."""

import hashlib

import torch


def seed_for(seed: int, *labels) -> int:
    key = "/".join(map(str, (seed, *labels))).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**31 - 1)


def rng(seed: int, *labels) -> torch.Generator:
    return torch.Generator().manual_seed(seed_for(seed, *labels))
