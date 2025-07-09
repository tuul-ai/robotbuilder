# The 40x More Important Thing: Why Data Normalization accelerates Robot Learning

*"Data normalization had 40 times more impact than architectural changes" - Russ Tedrake, MIT/TRI*

Imagine you're teaching a robot to make breakfast. In the first step, the robot gently picks up an egg - a tiny, precise 2cm movement. Fifty steps later, the robot sweeps its arm across the entire counter to grab the coffee pot - a massive 150cm movement. 

Here's the mind-bending question: **Should these two completely different types of movements use the same statistical "ruler" for learning?**

For years, I assumed "yes" - however after watching Russ talk about data normalization for action sequences, I decided to try it out. His research showed that fixing this one issue provides **20+ percentage point improvements** in robot success rates. Not 2%. Not 10%. **Twenty plus percentage points**. 
And this improvement comes not from fancy new neural architectures or clever algorithms, but from something as mundane as **how we normalize our data**. It's like discovering that the secret to making Formula 1 cars faster isn't a new engine design - it's properly inflating the tires.

## What Is Data Normalization Anyway?

So let's understand normalization with a simple analogy.

Imagine you're a teacher grading students, but your class has wildly different types of assignments:
- Essay scores: 0-100 points  
- Quiz scores: 0-10 points
- Participation: 0-5 points

If you just average these raw numbers, the essays (with their big numbers) will completely dominate the final grade. A student could ace every quiz and participation but still fail because of one bad essay.

**Normalization** fixes this by putting everything on the same scale:

```python
# Raw scores - essays dominate everything
essay_score = 85      # 0-100 scale
quiz_score = 8        # 0-10 scale  
participation = 4     # 0-5 scale
raw_average = (85 + 8 + 4) / 3 = 32.3  # Meaningless!

# Normalized scores - fair comparison
essay_norm = (85 - 50) / 25 = 1.4      # Standardized
quiz_norm = (8 - 5) / 2.5 = 1.2        # Standardized  
participation_norm = (4 - 2.5) / 1.25 = 1.2  # Standardized
fair_average = (1.4 + 1.2 + 1.2) / 3 = 1.27   # Actually meaningful!
```

The normalization formula is simple: `(value - mean) / standard_deviation`. This puts everything on a scale where 0 means "average," +1 means "one standard deviation above average," etc.

## The Robot Problem: Not All Movements Are Created Equal

Now back to our robot making breakfast. Here's what a typical 50-step robot action sequence might look like:

```python
# A robot breakfast-making sequence (7-DOF arm, joint angle deltas in radians)
timestep_1  = [0.002, 0.001, 0.003, 0.001, 0.002, 0.001, 0.0]  # Gentle egg pickup
timestep_5  = [0.008, 0.005, 0.012, 0.003, 0.007, 0.002, 0.1]  # Crack egg  
timestep_25 = [0.035, 0.025, 0.048, 0.015, 0.031, 0.012, 0.3]  # Pour batter
timestep_50 = [0.085, 0.067, 0.102, 0.045, 0.078, 0.034, 0.8]  # Reach for coffee
```

Look at those numbers! Early movements are tiny and precise. Later movements are large and sweeping. They're not just different in magnitude - they're **fundamentally different types of actions** requiring completely different precision and control.

## The Standard Approach: One Size Fits None

Here's how traditional robot learning (including standard LeRobot) handles this:

```python
# Traditional normalization - SUBOPTIMAL
def old_school_normalization(all_robot_actions):
    # Flatten ALL timesteps together
    flattened = all_robot_actions.reshape(-1, action_dim)  # [50k samples, 7 actions]
    
    # Compute global statistics
    global_mean = np.mean(flattened, axis=0)  # One mean for everything
    global_std = np.std(flattened, axis=0)    # One std for everything
    
    # Apply same normalizer to delicate egg movements AND coffee-reaching
    normalized = (all_robot_actions - global_mean) / global_std
    
    return normalized  # Same ruler for precision and power movements!

# Result: Small movements get over-scaled, big movements get under-scaled
```

This is like using the same grading curve for kindergarten finger painting and PhD dissertation defenses. It doesn't make sense!

## The Tedrake Lesson: Per-Timestep Normalization


```python
# Tedrake's approach - OPTIMAL  
def tedrake_normalization(all_robot_actions):
    num_episodes, timesteps, action_dim = all_robot_actions.shape
    
    # Compute separate statistics for EACH timestep
    per_timestep_means = np.zeros((timesteps, action_dim))
    per_timestep_stds = np.zeros((timesteps, action_dim))
    
    for t in range(timesteps):
        # Only use data from this specific timestep
        timestep_data = all_robot_actions[:, t, :]  # [episodes, actions]
        per_timestep_means[t] = np.mean(timestep_data, axis=0)
        per_timestep_stds[t] = np.std(timestep_data, axis=0)
    
    # Apply timestep-specific normalization
    normalized = np.zeros_like(all_robot_actions)
    for t in range(timesteps):
        normalized[:, t, :] = (
            (all_robot_actions[:, t, :] - per_timestep_means[t]) / 
            (per_timestep_stds[t] + 1e-8)
        )
    
    return normalized  # Each timestep gets its own "ruler"!
```

Now delicate egg movements are normalized using delicate-movement statistics, and sweeping coffee-reach movements are normalized using sweeping-movement statistics. Each gets the statistical treatment it deserves!

## Why This Works: The Statistics Tell the Story

Let's look at what happens when we analyze real robot data:

```python
# Example from robot manipulation task (joint angle deltas across episodes)
timestep_1_movements = [0.002, 0.003, 0.001, 0.002, 0.004]  # Precise, small changes
timestep_50_movements = [0.078, 0.089, 0.065, 0.082, 0.071]  # Larger, sweeping changes

# Timestep 1 statistics
t1_mean = 0.0024
t1_std = 0.0012  # Very tight distribution!

# Timestep 50 statistics  
t50_mean = 0.077
t50_std = 0.009   # Wider distribution!

# Global statistics (the old way)
global_mean = 0.040  # Average of small and large movements
global_std = 0.038   # Standard deviation across everything
```

**The Problem with Global Stats:**
- A small early movement (0.002 rad) becomes: `(0.002 - 0.040) / 0.038 = -1.0` (massively negative)
- A large later movement (0.077 rad) becomes: `(0.077 - 0.040) / 0.038 = 0.97` (barely positive)

The delicate early movement gets treated as an enormous negative outlier, while the sweeping later movement barely registers as above average!

**The Magic of Per-Timestep Stats:**
- Same early movement: `(0.002 - 0.0024) / 0.0012 = -0.33` (slightly below normal for its timestep)
- Same later movement: `(0.077 - 0.077) / 0.009 = 0.0` (perfectly normal for its timestep)

Now each movement is judged by its peers, not by completely unrelated actions!

## How to Use Tedrake Normalization in LeRobot

I've implemented Tedrake's per-timestep normalization as a drop-in replacement for LeRobot's standard normalization. Here's how to use it:

### Option 1: Quick Start (On-the-fly computation)

The simplest way to try Tedrake normalization is to use the enhanced training script that computes statistics automatically:

```bash
# Replace your standard train.py command with train_with_tedrake.py
python lerobot/src/lerobot/scripts/train_with_tedrake.py \
    --compute-enhanced-stats \
    --config-name=smolvla_pusht \
    --dataset.repo_id=your_dataset \
    --policy.path=lerobot/smolvla_base \
    --output_dir=./outputs/enhanced_model
```

This will:
1. Load your dataset
2. Automatically compute per-timestep action statistics from ~1000 samples
3. Apply enhanced normalization during training
4. Use standard global normalization for visual features

### Option 2: Pre-computed Statistics (Recommended for production)

For faster training startup, pre-compute the enhanced statistics:

```bash
# Step 1: Compute enhanced statistics
python lerobot/src/lerobot/scripts/compute_enhanced_stats.py \
    --dataset=your_dataset \
    --output=enhanced_stats.npz

# Step 2: Train with pre-computed stats
python lerobot/src/lerobot/scripts/train_with_tedrake.py \
    --enhanced-stats=enhanced_stats.npz \
    --config-name=smolvla_pusht \
    --dataset.repo_id=your_dataset \
    --output_dir=./outputs/enhanced_model
```

### Option 3: Manual Integration (For custom workflows)

If you want to integrate into your own training pipeline:

```python
# In your training script
from lerobot.policies.normalize_tedrake import Normalize, Unnormalize

# Replace the standard imports:
# from lerobot.policies.normalize import Normalize, Unnormalize

# The enhanced classes automatically detect per-timestep vs global stats
# and apply the appropriate normalization
```

### What Happens Under the Hood

1. **Action Detection**: The system identifies action features in your dataset
2. **Statistics Computation**: For multi-timestep actions, it computes separate mean/std for each timestep position
3. **Automatic Application**: During training, actions get normalized using their timestep-specific statistics
4. **Backward Compatibility**: Visual features and single-timestep actions still use global normalization

### When Will This Help?

Tedrake normalization provides the most benefit when:
- Your actions have **multiple timesteps** (chunk_size > 1)
- **Early and late timesteps have different magnitudes** (common in long-horizon tasks)
- You're using policies like **ACT** (chunk_size=100) or **SmolVLA** (chunk_size=50)

For single-timestep policies or very short sequences, the improvement will be minimal since there's no temporal variation to exploit.


---

## Getting Started: Adding Files to Your LeRobot Repo

To use Tedrake normalization in your LeRobot setup, you need to add three files from this repository to your working LeRobot installation:

### Step 1: Locate the Files

In the `robotbuilder` repository, find the `timestep_norm` directory which contains three essential files:
- `normalize_tedrake.py` - Enhanced normalization classes
- `train_with_tedrake.py` - Training script with Tedrake normalization
- `compute_enhanced_stats.py` - Statistics computation utility

### Step 2: Add Files to Your LeRobot Repo

Copy these files to the appropriate locations in your LeRobot repository:

```bash
# From the robotbuilder/timestep_norm directory, copy:
cp normalize_tedrake.py /path/to/your/lerobot/src/lerobot/policies/
cp train_with_tedrake.py /path/to/your/lerobot/src/lerobot/scripts/
cp compute_enhanced_stats.py /path/to/your/lerobot/src/lerobot/scripts/
```

### Step 3: Start Training

Once the files are in place, you can start training with Tedrake normalization using this command:

```bash
python lerobot/src/lerobot/scripts/train_with_tedrake.py \
    --compute-enhanced-stats \
    --config-name=smolvla_pusht \
    --dataset.repo_id=your_dataset \
    --policy.path=lerobot/smolvla_base \
    --output_dir=./outputs/enhanced_model
```

Replace `your_dataset` with your actual dataset repository ID, and adjust other parameters as needed for your specific use case.

---

## Implementation Notes

- **Backward Compatible**: Works as drop-in replacement for existing LeRobot normalization
- **Automatic Detection**: Intelligently chooses per-timestep vs global normalization based on data structure
- **Production Ready**: Tested with ACT and SmolVLA policies on multi-timestep action sequences
- **Easy Integration**: Minimal code changes required to existing training pipelines

Try it on your robot learning tasks and see the difference proper normalization makes! 