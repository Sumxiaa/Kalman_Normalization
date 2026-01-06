"""
Group Normalization for PyTorch

Paper: Group Normalization (https://arxiv.org/abs/1803.08494)

This is a standalone implementation for consistency with GroupKalmanNorm interface.
For production use, consider using torch.nn.GroupNorm directly.
"""

import torch
import torch.nn as nn


class GroupNorm(nn.Module):
    """
    Group Normalization for PyTorch

    Paper: Group Normalization (https://arxiv.org/abs/1803.08494)

    This implementation is functionally equivalent to PyTorch's nn.GroupNorm,
    implemented separately to:
    1. Maintain consistent interface with GroupKalmanNorm
    2. Facilitate understanding and debugging

    Args:
        num_features: Number of channels (C)
        num_groups: Number of groups (G), default=4
        eps: Small constant for numerical stability
        affine: Whether to use learnable gamma and beta

    Note:
        - GN does not depend on batch size, training and inference behavior are identical
        - No running_mean/running_var needed
    """

    def __init__(
        self,
        num_features: int,
        num_groups: int = 4,
        eps: float = 1e-5,
        affine: bool = True
    ):
        super().__init__()
        assert num_features % num_groups == 0, \
            f"num_features {num_features} must be divisible by num_groups {num_groups}"

        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine
        self.channels_per_group = num_features // num_groups

        # [O2] Only create affine parameters, no moving stats
        if affine:
            # shape: [1, C, 1, 1] for easy broadcasting
            self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [N, C, H, W]

        Returns:
            output: Normalized tensor [N, C, H, W]
        """
        N, C, H, W = x.shape
        G = self.num_groups

        # Reshape: [N, C, H, W] -> [N, G, C/G, H, W]
        x = x.view(N, G, C // G, H, W)

        # Compute group-wise statistics (over C/G, H, W dimensions)
        mean = x.mean(dim=[2, 3, 4], keepdim=True)  # [N, G, 1, 1, 1]
        var = x.var(dim=[2, 3, 4], keepdim=True, unbiased=False)

        # Normalize
        x = (x - mean) / (var + self.eps).sqrt()

        # Restore shape: [N, G, C/G, H, W] -> [N, C, H, W]
        x = x.view(N, C, H, W)

        # Affine transformation
        if self.affine:
            x = x * self.gamma + self.beta

        return x

    def extra_repr(self) -> str:
        return (
            f'{self.num_features}, num_groups={self.num_groups}, '
            f'eps={self.eps}, affine={self.affine}'
        )
