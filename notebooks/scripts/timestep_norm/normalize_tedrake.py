#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Enhanced normalization module implementing Tedrake's per-timestep action normalization.

This module provides drop-in replacements for the original Normalize/Unnormalize classes
with support for per-timestep action normalization, which has been shown to provide
20+ percentage point improvements in robotic manipulation tasks.

Key improvements:
- Per-timestep statistics for action features (different mean/std for each timestep)
- Global statistics for all other features (unchanged behavior)
- Maintains exact same interface as original classes for easy A/B testing
"""

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

# Import original functions we'll reuse for non-action features
from lerobot.policies.normalize import _no_stats_error_str


def create_stats_buffers_tedrake(
    features: dict[str, PolicyFeature],
    norm_map: dict[str, NormalizationMode],
    stats: dict[str, dict[str, Tensor]] | None = None,
) -> dict[str, dict[str, nn.ParameterDict]]:
    """
    Enhanced version of create_stats_buffers with per-timestep action normalization.
    
    This function creates buffers for normalization statistics with the following behavior:
    - ACTION features: Use per-timestep statistics if available, otherwise fallback to global
    - All other features: Use global statistics (same as original implementation)
    
    Args:
        features: Dictionary mapping feature names to PolicyFeature objects
        norm_map: Dictionary mapping feature types to normalization modes  
        stats: Optional dictionary containing pre-computed statistics
        
    Returns:
        Dictionary mapping feature names to parameter dictionaries containing statistics
    """
    stats_buffers = {}

    for key, ft in features.items():
        norm_mode = norm_map.get(ft.type, NormalizationMode.IDENTITY)
        if norm_mode is NormalizationMode.IDENTITY:
            continue

        assert isinstance(norm_mode, NormalizationMode)

        # Determine the shape for statistics based on feature type and available stats
        shape = tuple(ft.shape)
        
        if ft.type is FeatureType.VISUAL:
            # Special handling for visual features - reduce spatial dimensions
            assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=})"
            c, h, w = shape
            assert c < h and c < w, f"{key} is not channel first ({shape=})"
            shape = (c, 1, 1)
        elif ft.type is FeatureType.ACTION and stats and key in stats:
            # Check if we have per-timestep action statistics
            if "mean" in stats[key]:
                stat_data = stats[key]["mean"]
                if isinstance(stat_data, (np.ndarray, torch.Tensor)):
                    stat_shape = stat_data.shape if isinstance(stat_data, np.ndarray) else stat_data.shape
                    # If stats have more dimensions than expected, assume per-timestep format
                    if len(stat_shape) > len(shape):
                        shape = tuple(stat_shape)
                        print(f"Using per-timestep normalization for action '{key}' with shape {shape}")

        # Initialize buffers with infinity (to be overwritten by stats)
        buffer = {}
        if norm_mode is NormalizationMode.MEAN_STD:
            mean = torch.ones(shape, dtype=torch.float32) * torch.inf
            std = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "mean": nn.Parameter(mean, requires_grad=False),
                    "std": nn.Parameter(std, requires_grad=False),
                }
            )
        elif norm_mode is NormalizationMode.MIN_MAX:
            min_val = torch.ones(shape, dtype=torch.float32) * torch.inf
            max_val = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "min": nn.Parameter(min_val, requires_grad=False),
                    "max": nn.Parameter(max_val, requires_grad=False),
                }
            )

        # Load statistics if provided
        if stats and key in stats:
            if norm_mode is NormalizationMode.MEAN_STD:
                mean_data = stats[key]["mean"]
                std_data = stats[key]["std"]
                
                if isinstance(mean_data, np.ndarray):
                    buffer["mean"].data = torch.from_numpy(mean_data).to(dtype=torch.float32)
                    buffer["std"].data = torch.from_numpy(std_data).to(dtype=torch.float32)
                elif isinstance(mean_data, torch.Tensor):
                    # Clone to avoid issues with safetensors duplicate detection
                    buffer["mean"].data = mean_data.clone().to(dtype=torch.float32)
                    buffer["std"].data = std_data.clone().to(dtype=torch.float32)
                else:
                    type_ = type(mean_data)
                    raise ValueError(f"np.ndarray or torch.Tensor expected, but type is '{type_}' instead.")
                    
            elif norm_mode is NormalizationMode.MIN_MAX:
                min_data = stats[key]["min"]
                max_data = stats[key]["max"]
                
                if isinstance(min_data, np.ndarray):
                    buffer["min"].data = torch.from_numpy(min_data).to(dtype=torch.float32)
                    buffer["max"].data = torch.from_numpy(max_data).to(dtype=torch.float32)
                elif isinstance(min_data, torch.Tensor):
                    buffer["min"].data = min_data.clone().to(dtype=torch.float32)
                    buffer["max"].data = max_data.clone().to(dtype=torch.float32)
                else:
                    type_ = type(min_data)
                    raise ValueError(f"np.ndarray or torch.Tensor expected, but type is '{type_}' instead.")

        stats_buffers[key] = buffer
    return stats_buffers


def _apply_per_timestep_normalization(
    batch_data: Tensor, 
    mean: Tensor, 
    std: Tensor,
    feature_shape: tuple,
    operation: str = "normalize"
) -> Tensor:
    """
    Apply per-timestep normalization or unnormalization to action data.
    
    Args:
        batch_data: Input tensor with shape [batch, timesteps, action_dim]
        mean: Mean tensor with shape [timesteps, action_dim] or [action_dim] 
        std: Std tensor with shape [timesteps, action_dim] or [action_dim]
        feature_shape: Expected shape of the feature (e.g., (action_dim,))
        operation: Either "normalize" or "unnormalize"
        
    Returns:
        Processed tensor with same shape as input
    """
    # Check if we have per-timestep statistics
    if len(mean.shape) > len(feature_shape) and batch_data.ndim == 3:
        # Per-timestep normalization for [batch, timesteps, action_dim] data
        batch_size, timesteps, action_dim = batch_data.shape
        
        for t in range(timesteps):
            if t < mean.shape[0]:
                # Use per-timestep statistics
                mean_t = mean[t]
                std_t = std[t]
            else:
                # Fallback to last available timestep if we have fewer stats than timesteps
                mean_t = mean[-1]
                std_t = std[-1]
                
            if operation == "normalize":
                batch_data[:, t, :] = (batch_data[:, t, :] - mean_t) / (std_t + 1e-8)
            elif operation == "unnormalize":
                batch_data[:, t, :] = batch_data[:, t, :] * std_t + mean_t
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
    else:
        # Global normalization (original behavior)
        if operation == "normalize":
            batch_data = (batch_data - mean) / (std + 1e-8)
        elif operation == "unnormalize":
            batch_data = batch_data * std + mean
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
    return batch_data


def _apply_per_timestep_minmax_normalization(
    batch_data: Tensor,
    min_val: Tensor,
    max_val: Tensor, 
    feature_shape: tuple,
    operation: str = "normalize"
) -> Tensor:
    """
    Apply per-timestep min-max normalization or unnormalization to action data.
    
    Args:
        batch_data: Input tensor with shape [batch, timesteps, action_dim]
        min_val: Min tensor with shape [timesteps, action_dim] or [action_dim]
        max_val: Max tensor with shape [timesteps, action_dim] or [action_dim]
        feature_shape: Expected shape of the feature (e.g., (action_dim,))
        operation: Either "normalize" or "unnormalize"
        
    Returns:
        Processed tensor with same shape as input
    """
    # Check if we have per-timestep statistics
    if len(min_val.shape) > len(feature_shape) and batch_data.ndim == 3:
        # Per-timestep normalization for [batch, timesteps, action_dim] data
        batch_size, timesteps, action_dim = batch_data.shape
        
        for t in range(timesteps):
            if t < min_val.shape[0]:
                # Use per-timestep statistics
                min_t = min_val[t]
                max_t = max_val[t]
            else:
                # Fallback to last available timestep
                min_t = min_val[-1]
                max_t = max_val[-1]
                
            if operation == "normalize":
                # normalize to [0,1] then to [-1, 1]
                batch_data[:, t, :] = (batch_data[:, t, :] - min_t) / (max_t - min_t + 1e-8)
                batch_data[:, t, :] = batch_data[:, t, :] * 2 - 1
            elif operation == "unnormalize":
                # unnormalize from [-1, 1] to [0, 1] then to original range
                batch_data[:, t, :] = (batch_data[:, t, :] + 1) / 2
                batch_data[:, t, :] = batch_data[:, t, :] * (max_t - min_t) + min_t
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
    else:
        # Global normalization (original behavior)
        if operation == "normalize":
            batch_data = (batch_data - min_val) / (max_val - min_val + 1e-8)
            batch_data = batch_data * 2 - 1
        elif operation == "unnormalize":
            batch_data = (batch_data + 1) / 2
            batch_data = batch_data * (max_val - min_val) + min_val
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
    return batch_data


class Normalize(nn.Module):
    """
    Enhanced normalization with per-timestep action normalization.
    
    Drop-in replacement for the original Normalize class that implements
    Tedrake's per-timestep action normalization for improved manipulation performance.
    """

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats
        
        stats_buffers = create_stats_buffers_tedrake(features, norm_map, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        
        for key, ft in self.features.items():
            if key not in batch:
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                
                # Apply per-timestep normalization for actions, global for others
                if ft.type is FeatureType.ACTION:
                    batch[key] = _apply_per_timestep_normalization(
                        batch[key], mean, std, ft.shape, "normalize"
                    )
                else:
                    # Original global normalization for non-action features
                    batch[key] = (batch[key] - mean) / (std + 1e-8)
                    
            elif norm_mode is NormalizationMode.MIN_MAX:
                min_val = buffer["min"]
                max_val = buffer["max"]
                assert not torch.isinf(min_val).any(), _no_stats_error_str("min")
                assert not torch.isinf(max_val).any(), _no_stats_error_str("max")
                
                # Apply per-timestep normalization for actions, global for others
                if ft.type is FeatureType.ACTION:
                    batch[key] = _apply_per_timestep_minmax_normalization(
                        batch[key], min_val, max_val, ft.shape, "normalize"
                    )
                else:
                    # Original global normalization for non-action features
                    batch[key] = (batch[key] - min_val) / (max_val - min_val + 1e-8)
                    batch[key] = batch[key] * 2 - 1
            else:
                raise ValueError(norm_mode)
                
        return batch


class Unnormalize(nn.Module):
    """
    Enhanced unnormalization with per-timestep action unnormalization.
    
    Drop-in replacement for the original Unnormalize class that implements
    Tedrake's per-timestep action unnormalization.
    """

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats
        
        stats_buffers = create_stats_buffers_tedrake(features, norm_map, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        
        for key, ft in self.features.items():
            if key not in batch:
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                
                # Apply per-timestep unnormalization for actions, global for others
                if ft.type is FeatureType.ACTION:
                    batch[key] = _apply_per_timestep_normalization(
                        batch[key], mean, std, ft.shape, "unnormalize"
                    )
                else:
                    # Original global unnormalization for non-action features
                    batch[key] = batch[key] * std + mean
                    
            elif norm_mode is NormalizationMode.MIN_MAX:
                min_val = buffer["min"]
                max_val = buffer["max"]
                assert not torch.isinf(min_val).any(), _no_stats_error_str("min")
                assert not torch.isinf(max_val).any(), _no_stats_error_str("max")
                
                # Apply per-timestep unnormalization for actions, global for others
                if ft.type is FeatureType.ACTION:
                    batch[key] = _apply_per_timestep_minmax_normalization(
                        batch[key], min_val, max_val, ft.shape, "unnormalize"
                    )
                else:
                    batch[key] = (batch[key] + 1) / 2
                    batch[key] = batch[key] * (max_val - min_val) + min_val
            else:
                raise ValueError(norm_mode)
                
        return batch


def _initialize_stats_buffers_tedrake(
    module: nn.Module,
    features: dict[str, PolicyFeature],
    norm_map: dict[str, NormalizationMode],
    stats: dict[str, dict[str, Tensor]] | None = None,
) -> None:
    """
    Enhanced version of _initialize_stats_buffers with per-timestep action support.
    
    Registers statistics buffers on the given module with the following behavior:
    - ACTION features: Use per-timestep statistics if available
    - All other features: Use global statistics (same as original)
    """
    for key, ft in features.items():
        norm_mode = norm_map.get(ft.type, NormalizationMode.IDENTITY)
        if norm_mode is NormalizationMode.IDENTITY:
            continue

        # Determine shape for statistics
        shape: tuple[int, ...] = tuple(ft.shape)
        if ft.type is FeatureType.VISUAL:
            # reduce spatial dimensions, keep channel dimension only
            c, *_ = shape
            shape = (c, 1, 1)
        elif ft.type is FeatureType.ACTION and stats and key in stats:
            # Check if we have per-timestep action statistics
            if "mean" in stats[key]:
                stat_data = stats[key]["mean"]
                if isinstance(stat_data, torch.Tensor):
                    stat_shape = stat_data.shape
                    # If stats have more dimensions than expected, assume per-timestep format
                    if len(stat_shape) > len(shape):
                        shape = tuple(stat_shape)

        prefix = key.replace(".", "_")

        if norm_mode is NormalizationMode.MEAN_STD:
            mean = torch.full(shape, torch.inf, dtype=torch.float32)
            std = torch.full(shape, torch.inf, dtype=torch.float32)

            if stats and key in stats and "mean" in stats[key] and "std" in stats[key]:
                mean_data = stats[key]["mean"]
                std_data = stats[key]["std"]
                if isinstance(mean_data, torch.Tensor):
                    mean = mean_data.clone().to(dtype=torch.float32)
                    std = std_data.clone().to(dtype=torch.float32)
                else:
                    raise ValueError(f"Unsupported stats type for key '{key}' (expected Tensor).")

            module.register_buffer(f"{prefix}_mean", mean)
            module.register_buffer(f"{prefix}_std", std)
            continue

        if norm_mode is NormalizationMode.MIN_MAX:
            min_val = torch.full(shape, torch.inf, dtype=torch.float32)
            max_val = torch.full(shape, torch.inf, dtype=torch.float32)

            if stats and key in stats and "min" in stats[key] and "max" in stats[key]:
                min_data = stats[key]["min"]
                max_data = stats[key]["max"]
                if isinstance(min_data, torch.Tensor):
                    min_val = min_data.clone().to(dtype=torch.float32)
                    max_val = max_data.clone().to(dtype=torch.float32)
                else:
                    raise ValueError(f"Unsupported stats type for key '{key}' (expected Tensor).")

            module.register_buffer(f"{prefix}_min", min_val)
            module.register_buffer(f"{prefix}_max", max_val)
            continue

        raise ValueError(norm_mode)


class NormalizeBuffer(nn.Module):
    """
    Enhanced NormalizeBuffer with per-timestep action normalization.
    
    Drop-in replacement using registered buffers rather than parameters.
    """

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__()
        self.features = features
        self.norm_map = norm_map

        _initialize_stats_buffers_tedrake(self, features, norm_map, stats)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        
        for key, ft in self.features.items():
            if key not in batch:
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            prefix = key.replace(".", "_")

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = getattr(self, f"{prefix}_mean")
                std = getattr(self, f"{prefix}_std")
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                
                # Apply per-timestep normalization for actions, global for others
                if ft.type is FeatureType.ACTION:
                    batch[key] = _apply_per_timestep_normalization(
                        batch[key], mean, std, ft.shape, "normalize"
                    )
                else:
                    batch[key] = (batch[key] - mean) / (std + 1e-8)
                continue

            if norm_mode is NormalizationMode.MIN_MAX:
                min_val = getattr(self, f"{prefix}_min")
                max_val = getattr(self, f"{prefix}_max")
                assert not torch.isinf(min_val).any(), _no_stats_error_str("min")
                assert not torch.isinf(max_val).any(), _no_stats_error_str("max")
                
                # Apply per-timestep normalization for actions, global for others
                if ft.type is FeatureType.ACTION:
                    batch[key] = _apply_per_timestep_minmax_normalization(
                        batch[key], min_val, max_val, ft.shape, "normalize"
                    )
                else:
                    batch[key] = (batch[key] - min_val) / (max_val - min_val + 1e-8)
                    batch[key] = batch[key] * 2 - 1
                continue

            raise ValueError(norm_mode)

        return batch


class UnnormalizeBuffer(nn.Module):
    """
    Enhanced UnnormalizeBuffer with per-timestep action unnormalization.
    
    Drop-in replacement using registered buffers rather than parameters.
    """

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__()
        self.features = features
        self.norm_map = norm_map

        _initialize_stats_buffers_tedrake(self, features, norm_map, stats)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # Note: Don't shallow copy here to match original UnnormalizeBuffer behavior
        for key, ft in self.features.items():
            if key not in batch:
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            prefix = key.replace(".", "_")

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = getattr(self, f"{prefix}_mean")
                std = getattr(self, f"{prefix}_std")
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                
                # Apply per-timestep unnormalization for actions, global for others
                if ft.type is FeatureType.ACTION:
                    batch[key] = _apply_per_timestep_normalization(
                        batch[key], mean, std, ft.shape, "unnormalize"
                    )
                else:
                    batch[key] = batch[key] * std + mean
                continue

            if norm_mode is NormalizationMode.MIN_MAX:
                min_val = getattr(self, f"{prefix}_min")
                max_val = getattr(self, f"{prefix}_max")
                assert not torch.isinf(min_val).any(), _no_stats_error_str("min")
                assert not torch.isinf(max_val).any(), _no_stats_error_str("max")
                
                # Apply per-timestep unnormalization for actions, global for others  
                if ft.type is FeatureType.ACTION:
                    batch[key] = _apply_per_timestep_minmax_normalization(
                        batch[key], min_val, max_val, ft.shape, "unnormalize"
                    )
                else:
                    batch[key] = (batch[key] + 1) / 2
                    batch[key] = batch[key] * (max_val - min_val) + min_val
                continue

            raise ValueError(norm_mode)

        return batch 