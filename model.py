from functools import partial

import torch

from torch import Tensor

from torch.nn import (
    Module,
    ModuleList,
    Conv2d,
    Sigmoid,
    SiLU,
    Upsample,
    MaxPool2d
)

from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from huggingface_hub import PyTorchModelHubMixin


class PixelCrush(Module, PyTorchModelHubMixin):
    AVAILABLE_DOWNSCALE_RATIOS = {0.5, 0.25, 0.125}

    AVAILABLE_HIDDEN_RATIOS = {1, 2, 4}

    def __init__(
        self,
        downscale_ratio: float,
        num_channels: int,
        hidden_ratio: int,
        num_encoder_layers: int,
    ):
        super().__init__()

        if downscale_ratio not in self.AVAILABLE_DOWNSCALE_RATIOS:
            raise ValueError(
                f"Downscale ratio must be either 0.5, 0.25, or 0.125, {downscale_ratio} given."
            )

        self.skip = Upsample(scale_factor=downscale_ratio, mode="bicubic")

        self.encoder = Encoder(num_channels, hidden_ratio, num_encoder_layers)

        self.decoder = SuperpixelConv2d(num_channels, downscale_ratio)

        self.downscale_ratio = downscale_ratio

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def add_weight_norms(self) -> None:
        """Add weight normalization to all Conv2d layers in the model."""

        for module in self.modules():
            if isinstance(module, Conv2d):
                weight_norm(module)

    def remove_weight_norms(self) -> None:
        """Remove weight normalization parameterization."""

        for module in self.modules():
            if isinstance(module, Conv2d) and hasattr(module, "parametrizations"):
                params = [name for name in module.parametrizations.keys()]

                for name in params:
                    remove_parametrizations(module, name)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        s = self.skip.forward(x)

        z = self.encoder.forward(x)
        z = self.decoder.forward(z)

        z = s + z  # Global residual connection

        return z, s

    @torch.no_grad()
    def test_compare(self, x: Tensor) -> Tensor:
        z, s = self.forward(x)

        z = torch.clamp(z, 0, 1)
        s = torch.clamp(s, 0, 1)

        return z, s
    
    @torch.no_grad()
    def downscale(self, x: Tensor) -> Tensor:
        z, _ = self.forward(x)

        z = torch.clamp(z, 0, 1)

        return z


class Encoder(Module):
    """A low-resolution subnetwork employing a deep stack of encoder blocks."""

    def __init__(self, num_channels: int, hidden_ratio: int, num_layers: int):
        super().__init__()

        assert num_layers > 0, "Number of layers must be greater than 0."

        self.stem = Conv2d(3, num_channels, kernel_size=3, padding=1)

        self.body = ModuleList(
            [
                EncoderBlock(num_channels, hidden_ratio)
                for _ in range(num_layers)
            ]
        )

        self.checkpoint = lambda layer, x: layer(x)

    def enable_activation_checkpointing(self) -> None:
        """
        Instead of memorizing the activations of the forward pass, recompute them
        at every encoder block.
        """

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    def forward(self, x: Tensor) -> Tensor:
        z = self.stem.forward(x)

        for layer in self.body:
            z = self.checkpoint(layer, z)

        return z


class EncoderBlock(Module):
    """A single encoder block consisting of two stages and a residual connection."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        self.stage1 = PixelAttention(num_channels)
        self.stage2 = InvertedBottleneck(num_channels, hidden_ratio)

    def forward(self, x: Tensor) -> Tensor:
        z = self.stage1.forward(x)
        z = self.stage2.forward(z)

        z = x + z  # Local residual connection

        return z


class PixelAttention(Module):
    """An element-wise spatial attention module with depth-wise convolutions."""

    def __init__(self, num_channels: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."

        self.conv = Conv2d(
            num_channels, num_channels, kernel_size=11, padding=5, groups=num_channels
        )

        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv.forward(x)
        z = self.sigmoid.forward(z)

        z = z * x

        return z


class InvertedBottleneck(Module):
    """A wide non-linear activation block with 3x3 convolutions."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."
        assert hidden_ratio in {1, 2, 4}, "Hidden ratio must be either 1, 2, or 4."

        hidden_channels = hidden_ratio * num_channels

        self.conv1 = Conv2d(num_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(hidden_channels, num_channels, kernel_size=3, padding=1)

        self.silu = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv1.forward(x)
        z = self.silu.forward(z)
        z = self.conv2.forward(z)

        return z

class SuperpixelConv2d(Module):
    """A low-resolution decoder with super-pixel convolution."""

    def __init__(self, in_channels: int, downscale_ratio: float):
        super().__init__()

        assert downscale_ratio in {0.5, 0.25, 0.125}, "Upscale ratio must be either 0.5, 0.25, or 0.125."

        self.conv = Conv2d(in_channels, 3, kernel_size=3, padding=1)

        kernel_size = int(1 / downscale_ratio)

        self.pool = MaxPool2d(kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv.forward(x)
        z = self.pool.forward(z)

        return z