"""
Custom feature extractors for Stable Baselines 3.

These networks replace the default MLP with architectures better suited
for time-series trading data.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class LargeMLPExtractor(BaseFeaturesExtractor):
    """
    Larger MLP with residual connections and layer normalization.

    Architecture:
        Input -> Linear(256) -> LayerNorm -> ReLU -> Dropout
              -> Linear(256) + residual -> LayerNorm -> ReLU -> Dropout
              -> Linear(128) -> LayerNorm -> ReLU
              -> Output (128-dim)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.block2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.block3 = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.block1(observations)
        x = self.block2(x) + x  # residual connection
        x = self.block3(x)
        return x


class AttentionExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using self-attention over the observation window.

    Reshapes the flat observation into (window_size, feature_dim) and applies
    multi-head self-attention, then pools and projects to the output dimension.

    Parameters
    ----------
    observation_space : gym.spaces.Box
    features_dim : int
        Output feature dimension (default: 128).
    window_size : int
        Number of time steps in the observation window.
    n_heads : int
        Number of attention heads (default: 4).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        window_size: int = 10,
        n_heads: int = 4,
    ):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]
        # Last 2 values are position and tick_ratio
        self.window_size = window_size
        self.feature_dim = (input_dim - 2) // window_size
        self.extra_dim = 2  # position + tick_ratio

        # Project features to attention dimension
        attn_dim = 64
        self.input_proj = nn.Linear(self.feature_dim, attn_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(attn_dim)

        # Combine attended features with extra info
        self.output_proj = nn.Sequential(
            nn.Linear(attn_dim + self.extra_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Split: window features vs extra (position, tick_ratio)
        window_flat = observations[:, :-2]
        extra = observations[:, -2:]

        # Reshape to (batch, window_size, feature_dim)
        window = window_flat.view(batch_size, self.window_size, self.feature_dim)

        # Self-attention
        x = self.input_proj(window)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(attn_out + x)  # residual + norm

        # Mean pooling over time
        x = x.mean(dim=1)

        # Combine with position info
        x = torch.cat([x, extra], dim=1)
        x = self.output_proj(x)

        return x
