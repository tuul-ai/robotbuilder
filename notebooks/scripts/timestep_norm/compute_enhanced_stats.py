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
Simple script to compute enhanced dataset statistics with per-timestep action normalization.

Usage:
    python compute_enhanced_stats.py \
        --dataset=shreyasgite/tedrake_test \
        --output=enhanced_stats.npz \
        --episodes="0,2,3,5,6"  # For Mac compatibility
"""

import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging


def parse_episodes(episodes_str):
    """Parse episodes string into list of integers."""
    if not episodes_str:
        return None
    
    try:
        episodes = [int(x.strip()) for x in episodes_str.split(',')]
        return episodes
    except ValueError as e:
        raise ValueError(f"Invalid episodes format '{episodes_str}'. Use comma-separated integers like '0,2,3,5,6'") from e


def compute_enhanced_stats(dataset, max_samples=1000):
    """
    Compute enhanced statistics with per-timestep action normalization.
    
    Simple and bulletproof implementation.
    """
    logging.info("Computing enhanced statistics with per-timestep action normalization...")
    
    stats = {}
    
    # Find action key in dataset
    action_key = None
    for key, feature in dataset.meta.features.items():
        if key.startswith("action"):
            action_key = key
            break
    
    if not action_key:
        raise ValueError("No action feature found in dataset!")
    
    logging.info(f"Found action feature: {action_key}")
    
    # Collect action trajectories
    all_actions = []
    dataset_size = len(dataset)
    sample_size = min(dataset_size, max_samples)
    
    logging.info(f"Sampling {sample_size} trajectories from {dataset_size} total...")
    
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
        raise ValueError(f"No valid action data found!")
    
    # Stack actions and compute per-timestep stats
    actions = np.stack(all_actions, axis=0)
    logging.info(f"Action trajectories shape: {actions.shape}")
    
    if actions.ndim == 2:
        # Single timestep actions [num_samples, action_dim] -> add timestep dimension
        actions = actions[:, np.newaxis, :]
        logging.info("Added timestep dimension for single-step actions")
    
    num_samples, timesteps, action_dim = actions.shape
    
    # Compute per-timestep statistics
    per_timestep_mean = np.zeros((timesteps, action_dim))
    per_timestep_std = np.zeros((timesteps, action_dim))
    
    for t in range(timesteps):
        timestep_actions = actions[:, t, :]
        per_timestep_mean[t] = np.mean(timestep_actions, axis=0)
        per_timestep_std[t] = np.std(timestep_actions, axis=0)
    
    # Store enhanced action stats
    stats[action_key] = {
        "mean": per_timestep_mean,
        "std": per_timestep_std
    }
    
    # Log Tedrake's key insight!
    if timesteps > 1:
        mean_diff = np.mean(np.abs(per_timestep_mean[0] - per_timestep_mean[-1]))
        std_diff = np.mean(np.abs(per_timestep_std[0] - per_timestep_std[-1]))
        logging.info(f"🎯 Tedrake insight - Mean difference (t=0 vs t={timesteps-1}): {mean_diff:.6f}")
        logging.info(f"🎯 Tedrake insight - Std difference (t=0 vs t={timesteps-1}): {std_diff:.6f}")
        
        if mean_diff > 0.01 or std_diff > 0.01:
            logging.info("✅ Significant per-timestep differences detected - Tedrake normalization will help!")
        else:
            logging.info("ℹ️  Small per-timestep differences - may not see large gains from Tedrake normalization")
    
    # Compute standard stats for other features
    logging.info("Computing standard statistics for other features...")
    for key, feature in dataset.meta.features.items():
        if key.startswith("action"):
            continue  # Already handled
        
        # Sample other features
        sample_values = []
        sample_size_features = min(dataset_size, 100)
        
        for i in tqdm(range(sample_size_features), desc=f"Sampling {key}", leave=False):
            try:
                sample = dataset[i]
                if key in sample:
                    value = sample[key]
                    if isinstance(value, torch.Tensor):
                        value = value.numpy()
                    sample_values.append(value)
            except:
                continue
        
        if sample_values:
            all_values = np.stack(sample_values, axis=0)
            
            # Handle different feature types
            if "image" in key.lower() and all_values.ndim == 4:
                # Visual features: [batch, c, h, w] -> [c, 1, 1]
                mean = np.mean(all_values, axis=(0, 2, 3), keepdims=True)[0]
                std = np.std(all_values, axis=(0, 2, 3), keepdims=True)[0]
            else:
                # Other features: global stats
                mean = np.mean(all_values, axis=0)
                std = np.std(all_values, axis=0)
            
            stats[key] = {
                "mean": mean,
                "std": std
            }
            logging.info(f"✅ Computed global stats for {key}: {mean.shape}")
    
    logging.info(f"✅ Enhanced statistics computed for {len(stats)} features")
    return stats


def save_enhanced_stats(stats, output_path):
    """Save enhanced stats to .npz file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Flatten for npz format
    npz_data = {}
    for key, stat_dict in stats.items():
        for stat_type, stat_value in stat_dict.items():
            npz_key = f"{key}_{stat_type}"
            npz_data[npz_key] = stat_value
    
    np.savez(output_path, **npz_data)
    logging.info(f"✅ Enhanced stats saved to {output_path}")
    
    # Print summary
    for key, stat_dict in stats.items():
        if "mean" in stat_dict:
            shape = stat_dict["mean"].shape
            # Only actions with timesteps > 1 are per-timestep
            if key.startswith("action") and len(shape) > 1 and shape[0] > 1:
                logging.info(f"   {key}: Per-timestep normalization {shape}")
            else:
                logging.info(f"   {key}: Global normalization {shape}")


def main():
    parser = argparse.ArgumentParser(description="Compute enhanced dataset statistics with per-timestep action normalization")
    parser.add_argument("--dataset", required=True, help="Dataset repo_id (e.g., shreyasgite/tedrake_test)")
    parser.add_argument("--output", required=True, help="Output path for enhanced stats .npz file")
    parser.add_argument("--episodes", type=str, help="Comma-separated list of episodes for Mac compatibility (e.g., '0,2,3,5,6')")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum samples to use")
    
    args = parser.parse_args()
    
    # Initialize logging
    init_logging()
    
    logging.info(f"🚀 Computing enhanced stats for dataset: {args.dataset}")
    
    try:
        # Parse episodes if provided
        episodes = parse_episodes(args.episodes)
        if episodes:
            logging.info(f"Using episodes: {episodes} (Mac compatibility mode)")
        
        # Load dataset
        dataset = LeRobotDataset(repo_id=args.dataset, episodes=episodes)
        logging.info(f"Dataset loaded: {len(dataset)} samples, {dataset.num_episodes} episodes")
        
        # Compute enhanced stats
        enhanced_stats = compute_enhanced_stats(dataset, max_samples=args.max_samples)
        
        # Save enhanced stats
        save_enhanced_stats(enhanced_stats, args.output)
        
        logging.info("🎉 Enhanced statistics computation completed successfully!")
        
    except Exception as e:
        logging.error(f"❌ Failed to compute enhanced stats: {e}")
        raise


if __name__ == "__main__":
    main() 