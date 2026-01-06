"""
PyTorch Kalman Normalization

A PyTorch implementation of Kalman Normalization based on the paper:
"Batch Kalman Normalization: Towards Training Deep Neural Networks with Micro-Batches"
(https://arxiv.org/abs/1802.03133)

This implementation includes bug fixes and optimizations based on code review feedback.
"""

from .kalman_norm import KalmanNorm, KalmanNormSimple
from .group_norm import GroupNorm
from .group_kalman_norm import GroupKalmanNorm

__all__ = [
    'KalmanNorm',
    'KalmanNormSimple',
    'GroupNorm',
    'GroupKalmanNorm',
]

__version__ = '1.0.0'
