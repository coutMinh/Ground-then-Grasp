# SPDX-License-Identifier: MIT
# Copyright (c) 2024 An Vuong
# Adapted from Fsoft-AIC/LGD at 501904dc5c30c8fad58d90f7e4f49b4095f11eb0.
# See licenses/LGD-MIT.txt and THIRD_PARTY.md.
# Adaptation: RGB default, standalone nn.Module, package-local residual block.
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in
