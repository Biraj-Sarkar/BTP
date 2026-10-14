"""Segmentation architectures the trained extractor can use.

One function, `build(arch, encoder, pretrained)`, is the single place a model is
constructed. Both the trainer and the read-only loader call it, so a bundle can
never be served by a different architecture than the one that produced it: the
bundle records `arch`, and the loader rebuilds from that string.

Three architectures, all encoder-decoder with skip connections and all sharing
the same ImageNet-pretrained encoder, so a comparison between them is a
comparison of decoders rather than of backbones:

`unet`
    The original. The decoder upsamples and concatenates the encoder feature
    map at each scale.

`unetplusplus`
    Nested dense skips. Instead of passing an encoder feature map straight
    across, it is routed through a grid of intermediate blocks so that the
    decoder sees features that have already been partly upsampled. The stated
    motivation is that a raw deep feature map and a freshly upsampled one are
    semantically far apart, and concatenating them asks the decoder to bridge
    that gap on its own.

`attention`
    Attention U-Net (Oktay et al., 2018). Each skip connection passes through a
    gate that scores every spatial position using the coarser decoder feature
    map as context, then multiplies the skip by that score. The decoder can
    therefore suppress parts of the skip it does not want — relevant here,
    because the skip connections carry high-resolution edges from eyelashes and
    lid margin that sit immediately beside the tissue being segmented.

    `segmentation_models_pytorch` has no Attention U-Net, so the gates and the
    decoder are implemented below against its encoder. Its `scse` decoder
    attention is a different mechanism (it recalibrates features *after*
    concatenation rather than gating the skip before it) and is available as
    `unet-scse` for completeness.

Abbreviations
-------------
smp      `segmentation_models_pytorch`
ReLU     rectified linear unit
BN       batch normalisation
"""

from __future__ import annotations

from typing import List

ARCHITECTURES = ("unet", "unetplusplus", "attention", "unet-scse")


def build(arch: str = "unet", encoder: str = "efficientnet-b0", pretrained: bool = True):
    """Construct an untrained model. `pretrained` loads ImageNet encoder weights.

    Kept free of any file I/O: the trainer writes parameters, the loader reads
    them, and both get their architecture from here.
    """

    import segmentation_models_pytorch as smp

    weights = "imagenet" if pretrained else None
    if arch == "unet":
        return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1)
    if arch == "unetplusplus":
        return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1)
    if arch == "unet-scse":
        return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1,
                        decoder_attention_type="scse")
    if arch == "attention":
        return AttentionUnet(encoder_name=encoder, encoder_weights=weights)
    raise ValueError(f"Unknown architecture {arch!r}; choose from {ARCHITECTURES}")


# ---------------------------------------------------------------------------
# Attention U-Net


def _torch_modules():
    import torch
    import torch.nn as nn
    return torch, nn


class _AttentionGate:
    """Created via `_make_attention_gate` so torch is imported lazily."""


def _make_classes():
    """Build the Attention U-Net classes once torch is importable.

    Defined inside a function so that importing this module costs nothing on the
    classical path, which has no torch.
    """

    torch, nn = _torch_modules()
    import torch.nn.functional as F

    class AttentionGate(nn.Module):
        """Oktay et al. additive attention.

        `gate` is the coarser decoder feature map, `skip` the encoder feature
        map being carried across. Both are projected to a common width, summed,
        passed through ReLU and a 1-channel convolution, and squashed to [0, 1].
        The result is one weight per spatial position, which multiplies the skip.
        A position the decoder has no use for is scaled toward zero rather than
        concatenated at full strength.
        """

        def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
            super().__init__()
            self.theta = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
            self.phi = nn.Conv2d(gate_channels, inter_channels, 1, bias=True)
            self.psi = nn.Conv2d(inter_channels, 1, 1, bias=True)

        def forward(self, skip, gate):
            if gate.shape[-2:] != skip.shape[-2:]:
                gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            attention = torch.sigmoid(self.psi(F.relu(self.theta(skip) + self.phi(gate))))
            return skip * attention

    class DecoderBlock(nn.Module):
        """Upsample, gate the skip, concatenate, then two 3x3 convolutions."""

        def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
            super().__init__()
            self.gate = AttentionGate(in_channels, skip_channels, max(skip_channels // 2, 1)) \
                if skip_channels > 0 else None
            total = in_channels + skip_channels
            self.block = nn.Sequential(
                nn.Conv2d(total, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            )

        def forward(self, x, skip=None):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if skip is not None and self.gate is not None:
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, self.gate(skip, x)], dim=1)
            return self.block(x)

    class AttentionUnetImpl(nn.Module):
        """smp encoder, attention-gated decoder, 1-channel logit output.

        Decoder widths match smp's U-Net defaults so that a difference against
        the `unet` baseline is the attention gates and not decoder capacity.
        """

        def __init__(self, encoder_name: str = "efficientnet-b0",
                     encoder_weights: str | None = "imagenet",
                     decoder_channels: List[int] = (256, 128, 64, 32, 16)):
            super().__init__()
            import segmentation_models_pytorch as smp
            from segmentation_models_pytorch.encoders import get_encoder

            self.encoder = get_encoder(encoder_name, in_channels=3, depth=5, weights=encoder_weights)
            encoder_channels = list(self.encoder.out_channels)      # [C0 .. C5], C0 is the input stem
            # smp lists channels shallow -> deep; the decoder consumes them reversed.
            reversed_channels = encoder_channels[::-1]              # deepest first
            head_channels = reversed_channels[0]
            skips = reversed_channels[1:]                           # what each block concatenates
            in_ch = [head_channels] + list(decoder_channels[:-1])
            self.blocks = nn.ModuleList([
                DecoderBlock(i, s, o) for i, s, o in zip(in_ch, skips, decoder_channels)
            ])
            self.head = nn.Conv2d(decoder_channels[-1], 1, 3, padding=1)
            self.name = f"attention-unet-{encoder_name}"

        def forward(self, x):
            features = self.encoder(x)[::-1]                        # deepest first
            out, skips = features[0], features[1:]
            for i, block in enumerate(self.blocks):
                out = block(out, skips[i] if i < len(skips) else None)
            return self.head(out)

    return AttentionUnetImpl


_IMPL = None


def AttentionUnet(encoder_name: str = "efficientnet-b0", encoder_weights: str | None = "imagenet"):
    global _IMPL
    if _IMPL is None:
        _IMPL = _make_classes()
    return _IMPL(encoder_name=encoder_name, encoder_weights=encoder_weights)
