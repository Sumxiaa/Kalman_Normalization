"""
ResNet with Kalman Normalization statistics passing mechanism

This module implements the state input generation and residual blocks
with statistics passing as described in the Kalman Normalization paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .kalman_norm import KalmanNorm


class StateInputGenerator(nn.Module):
    """
    Generate pseudo-input following N(mean, var) distribution
    Used to predict next layer's statistics, ensuring gradient propagation

    Corresponds to TensorFlow's state_input() function
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        shape: Tuple[int, ...],
        mean: torch.Tensor,
        var: torch.Tensor,
        split_num: int
    ) -> torch.Tensor:
        """
        Args:
            shape: Target shape [1, C*split_num, H, W] (PyTorch format)
            mean: Mean [C*split_num]
            var: Variance [C*split_num]
            split_num: Split number

        Returns:
            statistic_input: [split_num, C, H, W]
        """
        # Generate standard normal distribution
        z = torch.randn(shape, device=mean.device, dtype=mean.dtype)

        # Transform: x = sqrt(var) * z + mean
        # mean and var need reshape to [1, C*split_num, 1, 1]
        mean_reshaped = mean.view(1, -1, 1, 1)
        var_reshaped = var.view(1, -1, 1, 1)

        statistic_input = z * var_reshaped.sqrt() + mean_reshaped
        statistic_input = F.relu(statistic_input)

        # Convert [1, C*split_num, H, W] to [split_num, C, H, W]
        C_total = statistic_input.shape[1]
        C = C_total // split_num
        H, W = statistic_input.shape[2], statistic_input.shape[3]

        statistic_input = statistic_input.view(1, C, split_num, H, W)
        statistic_input = statistic_input.permute(2, 1, 0, 3, 4).squeeze(2)  # [split_num, C, H, W]

        return statistic_input


class KNResidualBlock(nn.Module):
    """
    ResNet residual block with statistics passing (full version, aligned with paper)

    Implementation key points:
    1. Use StateInputGenerator to generate pseudo-input
    2. Share convolution weights to process pseudo-input
    3. Statistics also have residual connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        split_num: int = 64,
        p_rate: float = 0.9,
        first: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.split_num = split_num
        self.first = first

        # Normalization layers
        if not first:
            self.kn = KalmanNorm(in_channels, split_num, p_rate)
        self.kn1 = KalmanNorm(out_channels, split_num, p_rate)

        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)

        # State input generator
        self.state_gen = StateInputGenerator()

        # Shortcut
        self.need_proj = (stride != 1) or (in_channels != out_channels)
        if self.need_proj:
            self.pool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        m: Optional[torch.Tensor],
        v: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input [N, C_in, H, W]
            m: Previous mean [C_in * split_num]
            v: Previous var [C_in * split_num]

        Returns:
            out: Output [N, C_out, H', W']
            m_out: Output mean [C_out * split_num]
            v_out: Output var [C_out * split_num]
        """
        identity = x
        N, C_in, H, W = x.shape

        # --- Block 1: KN + ReLU (skip for first block) ---
        if self.first:
            b1, m1, v1 = x, m, v
        else:
            b1, m1, v1 = self.kn(x, m, v)
            b1 = F.relu(b1)

        # --- Generate state input for conv1 ---
        if m1 is not None and v1 is not None:
            C_concat = C_in * self.split_num
            state_shape = (1, C_concat, H, W)
            state_input = self.state_gen(state_shape, m1, v1, self.split_num)

            # --- Conv1 (shared weights for main path and state path) ---
            c1 = self.conv1(b1)

            # Apply same convolution to state_input
            with torch.no_grad():  # No gradient computation for state path
                state_input = self.conv1(state_input)

            # Compute predicted statistics from state_input
            state_concat = state_input.view(
                1, self.out_channels, self.split_num,
                state_input.shape[2], state_input.shape[3]
            ).permute(0, 2, 1, 3, 4).reshape(
                1, self.out_channels * self.split_num,
                state_input.shape[2], state_input.shape[3]
            )
            m1_pred = state_concat.mean(dim=[0, 2, 3])
            v1_pred = state_concat.var(dim=[0, 2, 3], unbiased=False)
        else:
            c1 = self.conv1(b1)
            m1_pred, v1_pred = None, None

        # --- KN1 + ReLU ---
        c1, m2, v2 = self.kn1(c1, m1_pred, v1_pred)
        c1 = F.relu(c1)

        # --- Generate state input for conv2 ---
        if m2 is not None and v2 is not None:
            _, _, H2, W2 = c1.shape
            state_shape2 = (1, self.out_channels * self.split_num, H2, W2)
            state_input2 = self.state_gen(state_shape2, m2, v2, self.split_num)

            # --- Conv2 ---
            c2 = self.conv2(c1)

            with torch.no_grad():
                state_input2 = self.conv2(state_input2)

            state_concat2 = state_input2.view(
                1, self.out_channels, self.split_num,
                state_input2.shape[2], state_input2.shape[3]
            ).permute(0, 2, 1, 3, 4).reshape(
                1, self.out_channels * self.split_num,
                state_input2.shape[2], state_input2.shape[3]
            )
            m3 = state_concat2.mean(dim=[0, 2, 3])
            v3 = state_concat2.var(dim=[0, 2, 3], unbiased=False)
        else:
            c2 = self.conv2(c1)
            m3, v3 = m2, v2

        # --- Shortcut ---
        if self.need_proj:
            identity = self.pool(identity)
            # Zero-padding for channel increase
            if self.out_channels > self.in_channels:
                pad_c = self.out_channels - self.in_channels
                identity = F.pad(identity, (0, 0, 0, 0, pad_c // 2, pad_c // 2))

            # Also need to pad statistics
            if m is not None and self.out_channels > self.in_channels:
                pad_stat = (self.out_channels - self.in_channels) * self.split_num // 2
                m = F.pad(m, (pad_stat, pad_stat))
                v = F.pad(v, (pad_stat, pad_stat))

        # --- Residual connection (both features and statistics) ---
        out = c2 + identity

        # Statistics also have residual connection
        if m is not None and m3 is not None:
            m_out = m + m3
            v_out = v + v3
        else:
            m_out = m3
            v_out = v3

        return out, m_out, v_out


class KNResNet(nn.Module):
    """
    ResNet with Kalman Normalization

    A simple ResNet architecture using KalmanNorm for demonstration.
    """

    def __init__(
        self,
        num_classes: int = 10,
        split_num: int = 64,
        p_rate: float = 0.9
    ):
        super().__init__()
        self.split_num = split_num

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.kn1 = KalmanNorm(16, split_num, p_rate)

        # Residual blocks
        self.block1 = KNResidualBlock(16, 16, 1, split_num, p_rate, first=True)
        self.block2 = KNResidualBlock(16, 32, 2, split_num, p_rate)
        self.block3 = KNResidualBlock(32, 64, 2, split_num, p_rate)

        # Output layers
        self.kn_out = KalmanNorm(64, split_num, p_rate)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv + KN
        x = self.conv1(x)
        x, m, v = self.kn1(x, None, None)
        x = F.relu(x)

        # Residual blocks
        x, m, v = self.block1(x, m, v)
        x, m, v = self.block2(x, m, v)
        x, m, v = self.block3(x, m, v)

        # Output
        x, _, _ = self.kn_out(x, m, v)
        x = F.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
