# Week 1 Completion Guide

This guide helps you complete the remaining Week 1 checklist items.

## ✅ Already Completed

- [x] Project Setup & Structure
- [x] Data & Modeling (data.py, model.py, train.py implemented)
- [x] Dependency & Code Quality (pyproject.toml, PEP8, documentation)
- [x] CLI interfaces (tasks.py)
- [x] Hydra configuration files
- [x] Logging integration
- [x] W&B integration
- [x] Data version control (Hugging Face datasets + documentation)
- [x] Profiling support (profile.py module)
- [x] Hyperparameter sweep config (configs/sweep.yaml)

## 🔲 Remaining Tasks

### 1. Build and Verify Docker Images (M10)

```bash
# Build Docker images
invoke docker-build

# Or manually:
docker build -t train:latest . -f dockerfiles/train.dockerfile
docker build -t api:latest . -f dockerfiles/api.dockerfile

# Test training image (quick test with subset)
docker run --rm train:latest python -m mlops_project.train data.use_subset=true data.subset_ratio=0.01 training.num_epochs=1
```

### 2. Test Training Pipeline (M6)

Before running full training, enable subset mode for quick testing:

```bash
# Option 1: Edit configs/data.yaml and set:
#   use_subset: true
#   subset_ratio: 0.1

# Then run:
invoke train

# Option 2: Override via command line
uv run python -m mlops_project.train data.use_subset=true data.subset_ratio=0.1 training.num_epochs=2
```

### 3. Optional: Run Hyperparameter Sweep (M14)

```bash
# Initialize W&B sweep
wandb sweep configs/sweep.yaml

# Run agents (in separate terminals or as background jobs)
wandb agent <sweep-id>
```

### 4. Optional: Use Profiling (M12)

```python
# In your training script or notebook:
from mlops_project.profile import profile_training_step

# Profile a few training batches
stats = profile_training_step(model, train_loader, device, num_batches=10)
stats.print_stats(20)  # Print top 20 functions by cumulative time
```

## Quick Verification Checklist

Run these commands to verify everything works:

```bash
# 1. Smoke tests
invoke smoke-data
invoke smoke-model

# 2. Unit tests
invoke test

# 3. Quick training test (with subset)
# Edit configs/data.yaml: use_subset=true, subset_ratio=0.1
invoke train

# 4. Docker build test
invoke docker-build
```

## Notes

- **W&B**: If you don't have a W&B account, set `wandb.enabled: false` in `configs/wandb.yaml` or set `WANDB_MODE=disabled`
- **Data**: First run will download ~5GB from Hugging Face (cached for future runs)
- **Training**: Use subset mode (`use_subset: true`) for quick testing to avoid long training times

