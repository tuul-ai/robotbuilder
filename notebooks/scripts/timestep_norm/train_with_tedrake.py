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
Minimal training wrapper with Tedrake's enhanced per-timestep action normalization.

This script enables per-timestep action normalization by:
1. Monkey-patching the normalization imports
2. Loading enhanced stats if provided OR computing on-the-fly
3. Running the original train.py with all its features intact

Usage:
    # With pre-computed enhanced stats (fastest)
    python train_with_tedrake.py --enhanced-stats=enhanced_stats.npz [train.py args...]
    
    # Compute enhanced stats on-the-fly (slower startup, but convenient)
    python train_with_tedrake.py --compute-enhanced-stats [train.py args...]
    
    # Without enhanced stats (uses global normalization - no improvement)
    python train_with_tedrake.py [train.py args...]

Examples:
    # Pre-computed stats (recommended for production)
    python train_with_tedrake.py \
        --enhanced-stats=enhanced_stats.npz \
        --config-name=smolvla_pusht \
        --dataset.repo_id=shreyasgite/tedrake_test
    
    # On-the-fly computation (good for experimentation)
    python train_with_tedrake.py \
        --compute-enhanced-stats \
        --config-name=smolvla_pusht \
        --dataset.repo_id=shreyasgite/tedrake_test
"""

import argparse
import logging
import sys
from pathlib import Path

# Monkey patch: Replace normalization classes with enhanced versions
import lerobot.policies.normalize as original_normalize
from lerobot.policies.normalize_tedrake import (
    Normalize as TedrakeNormalize,
    Unnormalize as TedrakeUnnormalize,
    NormalizeBuffer as TedrakeNormalizeBuffer,
    UnnormalizeBuffer as TedrakeUnnormalizeBuffer,
)

# Apply the monkey patch
original_normalize.Normalize = TedrakeNormalize
original_normalize.Unnormalize = TedrakeUnnormalize  
original_normalize.NormalizeBuffer = TedrakeNormalizeBuffer
original_normalize.UnnormalizeBuffer = TedrakeUnnormalizeBuffer

# Now import the original training script
from lerobot.scripts.train import train
from lerobot.utils.utils import init_logging


def load_enhanced_stats(stats_path):
    """Load enhanced stats from .npz file."""
    import numpy as np
    import torch
    
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Enhanced stats not found: {stats_path}")
    
    logging.info(f"Loading enhanced stats from: {stats_path}")
    
    raw_stats = np.load(stats_path)
    enhanced_stats = {}
    
    # Group by feature key
    for key in raw_stats.files:
        parts = key.split('_')
        feature_key = '_'.join(parts[:-1])
        stat_type = parts[-1]
        
        if feature_key not in enhanced_stats:
            enhanced_stats[feature_key] = {}
        
        enhanced_stats[feature_key][stat_type] = torch.from_numpy(raw_stats[key]).float()
    
    logging.info(f"✅ Loaded enhanced stats for: {list(enhanced_stats.keys())}")
    
    # Log which features have per-timestep normalization
    for key, stats in enhanced_stats.items():
        if "mean" in stats:
            shape = stats["mean"].shape
            if key.startswith("action") and len(shape) > 1 and shape[0] > 1:
                logging.info(f"   🎯 {key}: Per-timestep normalization {shape}")
            else:
                logging.info(f"   📊 {key}: Global normalization {shape}")
    
    return enhanced_stats


def compute_enhanced_stats_on_the_fly(dataset):
    """Compute enhanced stats from dataset on-the-fly."""
    import numpy as np
    import torch
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback if tqdm is not available
        def tqdm(iterable, desc=""):
            print(f"{desc}...")
            return iterable
    
    logging.info("🔄 Computing enhanced stats on-the-fly...")
    
    max_samples = 1000  # Limit for faster computation
    # Tip: Increase to 5000+ for more robust stats, or set to len(dataset) for full dataset
    
    # Find action key
    action_key = None
    for key in dataset.meta.features.keys():
        if key.startswith("action"):
            action_key = key
            break
    
    if not action_key:
        logging.warning("No action feature found - falling back to global stats")
        return dataset.meta.stats
    
    # Collect action trajectories
    all_actions = []
    dataset_size = len(dataset)
    sample_size = min(dataset_size, max_samples)
    
    for i in tqdm(range(sample_size), desc="Loading action trajectories"):
        try:
            sample = dataset[i]
            if action_key in sample:
                action = sample[action_key]
                if isinstance(action, torch.Tensor):
                    action = action.numpy()
                all_actions.append(action)
        except Exception as e:
            logging.warning(f"Failed to load sample {i}: {e}")
            continue
    
    if not all_actions:
        logging.warning("No valid action data found - falling back to global stats")
        return dataset.meta.stats
    
    # Stack and analyze actions
    actions = np.stack(all_actions, axis=0)
    logging.info(f"Action trajectories shape: {actions.shape}")
    
    if actions.ndim == 2:
        # Single timestep actions - add timestep dimension
        actions = actions[:, np.newaxis, :]
        logging.info("Single-timestep actions detected - no per-timestep benefit")
    
    num_samples, timesteps, action_dim = actions.shape
    
    if timesteps == 1:
        logging.info("Only 1 timestep - using global normalization")
        # Use existing global stats
        enhanced_stats = dict(dataset.meta.stats)
    else:
        logging.info(f"Computing per-timestep stats for {timesteps} timesteps")
        # Compute per-timestep statistics
        per_timestep_mean = np.zeros((timesteps, action_dim))
        per_timestep_std = np.zeros((timesteps, action_dim))
        
        for t in range(timesteps):
            per_timestep_mean[t] = np.mean(actions[:, t, :], axis=0)
            per_timestep_std[t] = np.std(actions[:, t, :], axis=0)
        
        # Create enhanced stats dict
        enhanced_stats = dict(dataset.meta.stats)
        enhanced_stats[action_key] = {
            "mean": torch.from_numpy(per_timestep_mean).float(),
            "std": torch.from_numpy(per_timestep_std).float()
        }
        
        logging.info(f"✅ Created per-timestep stats: {per_timestep_mean.shape}")
    
    # Handle other features with global stats
    for key, feature in dataset.meta.features.items():
        if not key.startswith("action") and key in dataset.meta.stats:
            enhanced_stats[key] = dataset.meta.stats[key]
    
    return enhanced_stats


def monkey_patch_dataset_stats(enhanced_stats, compute_on_the_fly=False):
    """Monkey patch make_dataset to inject enhanced stats."""
    from lerobot.datasets import factory
    
    original_make_dataset = factory.make_dataset
    
    def enhanced_make_dataset(cfg):
        dataset = original_make_dataset(cfg)
        
        # Inject enhanced stats into dataset metadata
        if enhanced_stats:
            dataset.meta.stats = enhanced_stats
            logging.info("✅ Injected pre-computed enhanced stats into dataset")
        elif compute_on_the_fly:
            dataset.meta.stats = compute_enhanced_stats_on_the_fly(dataset)
            logging.info("✅ Computed and injected enhanced stats on-the-fly")
        
        return dataset
    
    factory.make_dataset = enhanced_make_dataset


def main():
    """Main function that wraps the original train.py with Tedrake enhancements."""
    
    # Parse enhanced stats argument separately
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--enhanced-stats", type=str, help="Path to enhanced stats .npz file")
    parser.add_argument("--compute-enhanced-stats", action="store_true", help="Compute enhanced stats on-the-fly")
    args, remaining_args = parser.parse_known_args()
    
    # Initialize logging
    init_logging()
    
    if args.enhanced_stats:
        logging.info("🚀 Training with Tedrake's enhanced per-timestep action normalization!")
        
        # Load enhanced stats
        enhanced_stats = load_enhanced_stats(args.enhanced_stats)
        
        # Monkey patch dataset creation to inject enhanced stats
        monkey_patch_dataset_stats(enhanced_stats, compute_on_the_fly=False)
    elif args.compute_enhanced_stats:
        logging.info("🔄 Training with Tedrake's enhanced normalization - computing stats on-the-fly!")
        
        # Monkey patch dataset creation to compute enhanced stats on-the-fly
        monkey_patch_dataset_stats(None, compute_on_the_fly=True)
    else:
        logging.info("🔄 Training with enhanced normalization classes (but no per-timestep stats)")
        monkey_patch_dataset_stats(None, compute_on_the_fly=False)
    
    # Update sys.argv to pass remaining arguments to train.py
    sys.argv = [sys.argv[0]] + remaining_args
    
    # Run the original training function
    try:
        train()
        logging.info("🎉 Training completed successfully with Tedrake enhancements!")
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 