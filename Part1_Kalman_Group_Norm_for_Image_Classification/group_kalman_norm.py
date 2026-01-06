"""
Group Kalman Normalization for PyTorch

设计目标：
- 在通道分组的基础上做归一化（类似 GroupNorm），弱化 batch size 依赖
- 对每一层自己的组统计做“时间上的 Kalman/EMA 平滑”，提高估计稳定性
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class GroupKalmanNorm(nn.Module):
    """
    Group Kalman Normalization - combines GroupNorm-style grouping
    with Kalman/EMA smoothing of group-wise statistics over time.

    Args:
        num_features: Number of channels (C)
        num_groups: Number of groups (G), default=4
        p_rate: Kalman filtering factor ∈ [0,1]，越大越信历史统计
        eps: Small constant for numerical stability
        affine: Whether to use learnable gamma and beta

    重要设计点：
    - 归一化本身是「按组」进行：每个样本、每个 group 在 (C/G, H, W) 上做标准化
    - 同时维护每层自己的 running_mean / running_var（形状 [G]）
      并通过 p_rate 将当前 batch 的统计与历史统计做融合
    - 不再依赖跨层的 pre_mean / pre_var 做 Kalman 更新（避免耦合导致训练变慢）
      但为了接口兼容，仍然返回当前 batch 的 group 级统计 (mean, var) 供上层使用
    """

    def __init__(
        self,
        num_features: int,
        num_groups: int = 4,
        p_rate: float = 0.9,
        eps: float = 1e-5,
        affine: bool = True
    ):
        super().__init__()
        assert num_features % num_groups == 0, \
            f"num_features {num_features} must be divisible by num_groups {num_groups}"

        self.num_features = num_features
        self.num_groups = num_groups
        self.p_rate = p_rate  # Kalman / EMA 融合系数
        self.eps = eps
        self.affine = affine
        self.channels_per_group = num_features // num_groups

        if affine:
            self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        # 每层自己的 running stats（在时间维度上平滑）
        self.register_buffer("running_mean", torch.zeros(num_groups))
        self.register_buffer("running_var", torch.ones(num_groups))
        # 这里用一个简单的 EMA，这个 momentum 独立于 p_rate
        self.momentum = 1.0 - self.p_rate  # 你也可以单独调一个参数

    def forward(
        self,
        x: torch.Tensor,
        pre_mean: Optional[torch.Tensor] = None,
        pre_var: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [N, C, H, W]
            pre_mean: 保留接口，为了兼容（当前实现不使用）
            pre_var:  保留接口，为了兼容（当前实现不使用）

        Returns:
            output: Normalized tensor [N, C, H, W]
            mean: 当前 batch 的 group 级别均值 [G]（可选择传给下一层做 summary）
            var: 当前 batch 的 group 级别方差 [G]
        """
        N, C, H, W = x.shape
        G = self.num_groups

        # [N, C, H, W] -> [N, G, C/G, H, W]
        x_grouped = x.view(N, G, C // G, H, W)

        # --- 1. 先做 GN-style 统计 ---
        # 对每个样本、每个 group，在 (C/G, H, W) 上统计：
        # mean_sample: [N, G, 1, 1, 1]
        mean_sample = x_grouped.mean(dim=[2, 3, 4], keepdim=True)
        var_sample = x_grouped.var(dim=[2, 3, 4], keepdim=True, unbiased=False)

        # 再在 batch 维上平均，得到 group 级别的统计 [G]
        batch_mean = mean_sample.mean(dim=0).view(G)  # [G]
        batch_var = var_sample.mean(dim=0).view(G)    # [G]

        # --- 2. Kalman / EMA 更新每层自己的 running stats（时间维度） ---
        if self.training:
            with torch.no_grad():
                # 简单 EMA：running <- (1-m)*running + m*batch
                m = self.momentum
                self.running_mean.mul_(1.0 - m).add_(m * batch_mean)
                self.running_var.mul_(1.0 - m).add_(m * batch_var)

            # Kalman 式融合：当前用于归一化的统计
            # 这里可以理解为：在 “running”（历史） 和 “当前 batch” 之间再次平滑一次
            kalman_mean = self.p_rate * self.running_mean + (1.0 - self.p_rate) * batch_mean
            kalman_var = self.p_rate * self.running_var + (1.0 - self.p_rate) * batch_var

            mean_used = kalman_mean
            var_used = kalman_var
        else:
            # eval 时直接使用 running stats（类似 BatchNorm 的行为）
            mean_used = self.running_mean
            var_used = self.running_var

        # --- 3. 用 [G] 级别的 mean/var 做归一化 ---
        mean_broadcast = mean_used.view(1, G, 1, 1, 1)  # [1,G,1,1,1]
        var_broadcast = var_used.view(1, G, 1, 1, 1)

        x_norm = (x_grouped - mean_broadcast) / (var_broadcast + self.eps).sqrt()

        # [N, G, C/G, H, W] -> [N, C, H, W]
        x_norm = x_norm.view(N, C, H, W)

        # Affine
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta

        # 为了和之前接口兼容，这里返回当前 batch 的 group 级统计（不返回 running）
        return x_norm, batch_mean.detach(), batch_var.detach()

    def extra_repr(self) -> str:
        return (
            f'{self.num_features}, num_groups={self.num_groups}, '
            f'p_rate={self.p_rate}, eps={self.eps}, affine={self.affine}'
        )
