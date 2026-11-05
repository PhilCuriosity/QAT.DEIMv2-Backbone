"""Export compatibility patches for the DEIMv2 backbone/encoder graph."""
from __future__ import annotations

import torch
from torch import nn

_INSTALLED = False


def install_export_compat() -> None:
    """Make RepNCSPELAN exportable without changing its math.

    The upstream generator expression in ``RepNCSPELAN4/5.forward`` makes
    Dynamo feed the original tensor to ``cv3`` instead of the ``cv2`` output.
    Express the same split/cat sequence explicitly.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from engine.deim.hybrid_encoder import RepNCSPELAN4, RepNCSPELAN5

    def explicit_forward(module: nn.Module, x: torch.Tensor):
        first, second = module.cv1(x).split((module.c, module.c), 1)
        third = module.cv2(second)
        fourth = module.cv3(third)
        return module.cv4(torch.cat((first, second, third, fourth), 1))

    RepNCSPELAN4.forward = explicit_forward
    RepNCSPELAN5.forward = explicit_forward
    _INSTALLED = True


def install_torchvision_compat() -> None:
    """Bridge the project's torchvision-0.20 transform hooks to 0.21."""
    from engine.data.transforms._transforms import ConvertBoxes, ConvertPILImage, PadToSize

    for cls in (PadToSize, ConvertBoxes, ConvertPILImage):
        if "_transform" in cls.__dict__:
            cls.transform = cls._transform
        if "_get_params" in cls.__dict__:
            cls.make_params = cls._get_params
