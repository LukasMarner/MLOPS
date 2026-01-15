# mlops_project

Project for DTU mlops January course.

# Food-101 Image Classification with Vision Transformer

## Project description

The goal of this project is to build a complete MLOps pipeline for an image classification task. We will train and deploy a model that can recognize different types of food from images. The focus of the project is not on reaching state-of-the-art accuracy, but on creating a reproducible, well-structured, and deployable machine learning system using the tools taught in the DTU MLOps course.

We use the **Food-101** dataset from ETH Zurich, which contains 101,000 images divided into 101 food categories. Each class has 750 training images and 250 validation images. The dataset is large enough to be realistic, but still manageable for fast experimentation. The images have different sizes and some noise in the labels, which makes it a good example of a real-world dataset. The data is loaded using the Hugging Face `datasets` library, which provides an easy and reproducible way to access and version the data.

For the model, we use a Vision Transformer (ViT), specifically the pretrained model `google/vit-base-patch16-224-in21k`. This model was trained on ImageNet-21k with more than 14 million images and can extract strong visual features. We fine-tune this model on Food-101 by adding a classification head with 101 output classes. By using a pretrained model, we avoid training from scratch and can focus on building a stable training and deployment pipeline.

The project includes data loading, preprocessing, training, evaluation, and deployment. We use configuration files and Hydra to manage hyperparameters and experiment settings. Training and evaluation metrics are logged using Weights & Biases. The code is structured in a modular way, with separate files for data handling, model definition, and training logic.

We also include unit tests for the data pipeline and the model, and run these tests automatically using GitHub Actions. Docker is used to make the training and inference environments reproducible. Finally, the trained model is served through a FastAPI application, allowing users to upload an image and receive a predicted food class.

Overall, this project demonstrates how a modern MLOps workflow can be applied to a realistic computer vision problem.


# MLOps Project Tracker

> This document is used to track ownership, progress, and responsibilities across the MLOps project.
> Not all items are required for the exam — this is a **menu**, not a contract.

Legend:
- **Status**: `Todo`, `In Progress`, `Blocked`, `Done`
- **Priority**: `Low`, `Medium`, `High`
- **Week**: Expected target week (1–3)

---

## Week 1 — Foundations

| Task | Module | Owner | Status | Priority | Notes |
|------|--------|-------|--------|----------|-------|
| Create GitHub repository | M5 | x | Todo | High |  |
| Grant team write access | M5 |  | Todo | High |  |
| Create project environment (venv/conda/uv) | M2 |  | Todo | High |  |
| Generate project template with Cookiecutter | M6 |  | Todo | Medium |  |
| Implement `data.py` (download + preprocessing) | M6 |  | Todo | High |  |
| Implement `model.py` | M6 |  | Todo | High |  |
| Implement `train.py` | M6 |  | Todo | High |  |
| Dependency management (requirements / pyproject) | M2, M6 |  | Todo | High |  |
| PEP8 compliance | M7 |  | Todo | Medium |  |
| Code documentation | M7 |  | Todo | Medium |  |
| Data versioning (DVC or similar) | M8 |  | Todo | Medium |  |
| Add CLI commands | M9 |  | Todo | Low |  |
| Create Dockerfile(s) | M10 |  | Todo | High |  |
| Build Docker images locally | M10 |  | Todo | High |  |
| Add experiment config files | M11 |  | Todo | Medium |  |
| Integrate Hydra | M11 |  | Todo | Medium |  |
| Add profiling | M12 |  | Todo | Low |  |
| Add logging | M14 |  | Todo | Medium |  |
| Add Weights & Biases | M14 |  | Todo | Medium |  |
| Hyperparameter sweep | M14 |  | Todo | Low |  |
| Use PyTorch Lightning | M15 |  | Todo | Low |  |

---

## Week 2 — Testing, CI & Deployment

| Task | Module | Owner | Status | Priority | Notes |
|------|--------|-------|--------|----------|-------|
| Data unit tests | M16 |  | Todo | High |  |
| Model / training unit tests | M16 |  | Todo | High |  |
| Code coverage | M16 |  | Todo | Medium |  |
| Setup GitHub Actions CI | M17 |  | Todo | High |  |
| Multi-OS / PyTorch / caching in CI | M17 |  | Todo | Medium |  |
| CI linting | M17 |  | Todo | Medium |  |
| Pre-commit hooks | M18 |  | Todo | Medium |  |
| Data change trigger workflow | M19 |  | Todo | Low |  |
| Model registry trigger workflow | M19 |  | Todo | Low |  |
| Create GCP bucket for data | M21 |  | Todo | High |  |
| Docker build workflow | M21 |  | Todo | High |  |
| Run training in GCP | M21 |  | Todo | Medium |  |
| FastAPI inference API | M22 |  | Todo | High |  |
| Deploy API (Cloud Run / Functions) | M23 |  | Todo | High |  |
| API tests + CI | M24 |  | Todo | Medium |  |
| Load testing | M24 |  | Todo | Low |  |
| ONNX or BentoML deployment | M25 |  | Todo | Low |  |
| Frontend for API | M26 |  | Todo | Low |  |

---

## Week 3 — Monitoring & Optimization

| Task | Module | Owner | Status | Priority | Notes |
|------|--------|-------|--------|----------|-------|
| Data drift evaluation | M27 |  | Todo | High |  |
| Collect inference data | M27 |  | Todo | High |  |
| Deploy drift detection API | M27 |  | Todo | Medium |  |
| Add system metrics | M28 |  | Todo | Medium |  |
| Cloud monitoring | M28 |  | Todo | Medium |  |
| Alerts in GCP | M28 |  | Todo | Medium |  |
| Distributed data loading | M29 |  | Todo | Low |  |
| Distributed training | M30 |  | Todo | Low |  |
| Model optimization (quantization, pruning, etc.) | M31 |  | Todo | Low |  |

---

## Extra / Delivery

| Task | Module | Owner | Status | Priority | Notes |
|------|--------|-------|--------|----------|-------|
| Write documentation | M32 |  | Todo | High |  |
| Publish to GitHub Pages | M32 |  | Todo | Medium |  |
| Revisit project goals | — |  | Todo | Medium |  |
| Create MLOps architecture diagram | — |  | Todo | High |  |
| Team knowledge sync | — |  | Todo | High |  |
| Push all code to GitHub | — |  | Todo | High |  |

---



## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
