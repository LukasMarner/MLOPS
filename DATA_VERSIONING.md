# Data Version Control

This project uses **Hugging Face Datasets** for data versioning and management.

## How Data Versioning Works

1. **Hugging Face Datasets**: The Food-101 dataset is loaded from `ethz/food101` which provides:
   - Automatic versioning through Hugging Face's dataset registry
   - Caching in `data/` directory (specified by `cache_dir` parameter)
   - Reproducible dataset loading via versioned splits

2. **Local Caching**: Datasets are cached locally in `data/ethz___food101/` with version information in the directory structure.

3. **Reproducibility**: The dataset version is locked by the Hugging Face library, ensuring consistent data across runs.

## Optional: DVC Setup

If you want additional local data versioning with DVC:

```bash
# Install DVC (optional)
pip install dvc

# Initialize DVC
dvc init

# Track data directory
dvc add data/

# Commit DVC files
git add data.dvc .dvcignore
git commit -m "Track data with DVC"
```

For this project, Hugging Face's built-in versioning is sufficient for Week 1 requirements.

