from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def passthrough(x, **kwargs):
    return x


def get_logit_fn(name):
    if name == "dice":
        out = F.softmax
    elif name == "nll":
        out = F.log_softmax
    elif name is None or name == "none":
        out = passthrough
    else:
        valid_names = ("dice", "nll", "none")
        raise ValueError(
            f'Loss must be one of ({", ".join(f"`{valid_name}`" for valid_name in valid_names)}, None), `{name}` was passed.'
        )
    return out


def get_activation(name, n_channels):
    if name == "elu":
        out = nn.ELU(inplace=True)
    elif name == "relu":
        out = nn.ReLU(inplace=True)
    elif name == "prelu":
        out = nn.PReLU(n_channels)
    else:
        valid_names = ("elu", "relu", "prelu")
        raise ValueError(
            f'Activation must be one of {", ".join(f"`{valid_name}`" for valid_name in valid_names)}, `{name}` was passed.'
        )
    return out


class VNet(nn.Module):
    """
    A PyTorch implementation of segmentation model described in F. Milletari, N. Navab and S.A. Ahmadi's
    "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
    (https://arxiv.org/pdf/1606.04797.pdf)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        wf: int,
        batch_norm: bool,
        activation: str,
        loss: str,
        kaiming_normal: bool,
    ):
        super().__init__()

        down_blocks = []
        down_in_channels = in_channels
        for i in range(depth - 1):
            if i > 0:
                down_conv_out_channels = None
                down_out_channels = 2 * down_in_channels
            else:
                down_conv_out_channels = in_channels * (2**wf)
                down_out_channels = 2 * down_conv_out_channels

            down_block = VNetDownBlock(
                in_channels=down_in_channels,
                conv_out_channels=down_conv_out_channels,
                out_channels=down_out_channels,
                n_convs=min(i + 1, 3),
                batch_norm=batch_norm,
                activation=activation,
                kaiming_normal=kaiming_normal,
            )
            down_blocks.append(down_block)
            down_in_channels = down_out_channels
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        up_in_channels = down_out_channels
        for i in range(depth - 1):
            if i > 0:
                up_out_channels = up_in_channels // 4
                up_conv_out_channels = up_in_channels // 2
            else:
                up_out_channels = up_in_channels // 2
                up_conv_out_channels = None
            up_block = VNetUpBlock(
                in_channels=up_in_channels,
                conv_out_channels=up_conv_out_channels,
                out_channels=up_out_channels,
                n_convs=min(depth - i + 1, 3),
                batch_norm=batch_norm,
                activation=activation,
                kaiming_normal=kaiming_normal,
            )

            up_in_channels = 2 * up_out_channels
            up_blocks.append(up_block)

        self.up_blocks = nn.ModuleList(up_blocks)

        self.final_conv_block = VNetOutputBlock(
            in_channels=2 * up_out_channels,
            conv_out_channels=up_out_channels,
            out_channels=out_channels,
            batch_norm=batch_norm,
            activation=activation,
            loss=loss,
            kaiming_normal=kaiming_normal,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bridges = []
        for i, down_block in enumerate(self.down_blocks):
            x = down_block(x)
            x_bridges.append(x)
            x = down_block.pool(x)
        for i, up_block in enumerate(self.up_blocks):
            x_bridge = x_bridges.pop(-1) if i > 0 else None
            x = up_block(x, x_bridge)

        return self.final_conv_block(x, x_bridges.pop(-1))


class VNetOutputBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_out_channels: int,
        out_channels: int,
        batch_norm: bool,
        activation: str,
        loss: str,
        kaiming_normal: bool,
    ):
        super().__init__()
        self.conv_block = VNetNConvBlock(
            in_channels=in_channels,
            out_channels=conv_out_channels,
            n_convs=1,
            batch_norm=batch_norm,
            activation=activation,
            kaiming_normal=kaiming_normal,
        )
        conv = nn.Conv3d(
            in_channels=conv_out_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        if kaiming_normal:
            nn.init.kaiming_normal_(conv.weight)
        self.final_conv = nn.Sequential(
            conv, get_activation(activation, out_channels)
        )
        self.logit_fn = get_logit_fn(loss)

    def forward(self, x: torch.Tensor, x_bridge: torch.Tensor) -> torch.Tensor:
        x_cat = torch.cat([x, x_bridge], dim=1)
        x = self.conv_block(x_cat, x)
        x = self.final_conv(x)
        return self.logit_fn(x, dim=1)


class VNetDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_convs: int,
        batch_norm: bool,
        activation: str,
        kaiming_normal: bool,
        conv_out_channels: Optional[int] = None,
    ):
        super().__init__()
        if conv_out_channels is None:
            conv_out_channels = in_channels
        self.conv_block = VNetNConvBlock(
            n_convs=n_convs,
            in_channels=in_channels,
            out_channels=conv_out_channels,
            batch_norm=batch_norm,
            activation=activation,
            kaiming_normal=kaiming_normal,
        )
        conv = nn.Conv3d(
            in_channels=conv_out_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        if kaiming_normal:
            nn.init.kaiming_normal_(conv.weight)
        self.pool = nn.Sequential(
            conv, get_activation(activation, out_channels)
        )

    def forward(self, x):
        x = self.conv_block(x, x)
        return x


class VNetUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_convs: int,
        batch_norm: bool,
        activation: str,
        kaiming_normal: bool,
        conv_out_channels: Optional[int] = None,
    ):
        super().__init__()
        if conv_out_channels is None:
            conv_out_channels = in_channels
        self.conv_block = VNetNConvBlock(
            in_channels=in_channels,
            out_channels=conv_out_channels,
            n_convs=n_convs,
            batch_norm=batch_norm,
            activation=activation,
            kaiming_normal=kaiming_normal,
        )
        conv_t = nn.ConvTranspose3d(
            in_channels=conv_out_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )
        if kaiming_normal:
            nn.init.kaiming_normal_(conv_t.weight)
        self.up_conv = nn.Sequential(
            conv_t, get_activation(activation, out_channels)
        )

    def forward(
        self, x: torch.Tensor, x_bridge: Optional[torch.Tensor] = None
    ):
        x_cat = torch.cat([x, x_bridge], dim=1) if x_bridge is not None else x
        x = self.conv_block(x_cat, x)
        return self.up_conv(x)


class VNetNConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_convs: int,
        batch_norm: bool,
        activation: str,
        kaiming_normal: bool,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        layers = []
        for _ in range(n_convs):
            conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=5,
                padding=2,
            )
            if kaiming_normal:
                nn.init.kaiming_normal_(conv.weight)
            layer = [conv, get_activation(activation, out_channels)]
            if batch_norm:
                layer.append(nn.BatchNorm3d(out_channels))
            in_channels = out_channels
            layers.extend(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, x_bridge: torch.Tensor):
        return self.layers(x) + x_bridge
