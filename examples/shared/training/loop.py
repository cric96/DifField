"""Checkpoints of trained models."""

import torch


def frozen(path, build):
    """The model of a checkpoint, in eval mode and without gradients.

    Args:
        path: A checkpoint holding the weights under ``model``.
        build: Builds the untrained model from the checkpoint.
    """
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    model = build(checkpoint)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.requires_grad_(False)
    return model
