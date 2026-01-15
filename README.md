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


# Project Checklist

> **Note**  
> All lists are *exhaustive*, meaning you are **not expected to have completed every single point** for the exam.  
> The module in parentheses indicates which course module the item is related to.

---

## Week 1

### Project Setup & Structure
- [x] Create a git repository (M5)  
- [x] Make sure that all team members have write access to the GitHub repository (M5)  
- [x] Create a dedicated environment for your project to keep track of your packages (M2)  
- [x] Create the initial file structure using Cookiecutter with an appropriate template (M6)  

### Data & Modeling
- [x] Fill out `data.py` so it downloads and preprocesses the required data (M6)  
- [x] Add a model to `model.py` and a training procedure to `train.py` and get it running (M6)  

### Dependency & Code Quality
- [ ] Keep `requirements.txt` / `requirements_dev.txt` or `pyproject.toml` / `uv.lock` up to date (M2, M6)  
- [ ] Follow good coding practices (PEP8) (M7)  
- [ ] Document essential parts of the code (M7)  

### MLOps Foundations
- [ ] Set up version control for your data or parts of your data (M8)  
- [ ] Add command-line interfaces and project commands where appropriate (M9)  
- [ ] Create one or more Dockerfiles (M10)  
- [ ] Build the Docker images locally and verify they work (M10)  
- [ ] Write one or more configuration files for experiments (M11)  
- [ ] Use Hydra to load configurations and manage hyperparameters (M11)  
- [ ] Use profiling to optimize your code (M12)  
- [ ] Use logging to log important events (M14)  
- [ ] Use Weights & Biases to log training progress and artifacts (M14)  
- [ ] Consider running a hyperparameter sweep (M14)  
- [ ] Use PyTorch Lightning (if applicable) to reduce boilerplate (M15)  

---

## Week 2

### Testing & CI
- [ ] Write unit tests for the data pipeline (M16)  
- [ ] Write unit tests for model construction and/or training (M16)  
- [ ] Calculate code coverage (M16)  
- [ ] Set up Continuous Integration (CI) on GitHub (M17)  
- [ ] Add caching and multi-OS / Python / PyTorch testing to CI (M17)  
- [ ] Add linting to CI (M17)  
- [ ] Add pre-commit hooks (M18)  

### Automation & Cloud
- [ ] Add a workflow that triggers when data changes (M19)  
- [ ] Add a workflow that triggers when the model registry changes (M19)  
- [ ] Create a GCP Bucket for data and connect it to data version control (M21)  
- [ ] Create a workflow that automatically builds Docker images (M21)  
- [ ] Run model training in GCP (Engine or Vertex AI) (M21)  

### Deployment
- [ ] Create a FastAPI inference API (M22)  
- [ ] Deploy the model using GCP Functions or Cloud Run (M23)  
- [ ] Write API tests and integrate them into CI (M24)  
- [ ] Load test the application (M24)  
- [ ] Create a specialized deployment API using ONNX and/or BentoML (M25)  
- [ ] Create a frontend for your API (M26)  

---

## Week 3

### Monitoring & Optimization
- [ ] Evaluate robustness to data drift (M27)  
- [ ] Collect input-output data from the deployed application (M27)  
- [ ] Deploy a drift detection API (M27)  
- [ ] Instrument the API with system metrics (M28)  
- [ ] Set up cloud monitoring (M28)  
- [ ] Create alerting in GCP for system failures or performance drops (M28)  
- [ ] Optimize data loading using distributed data loading (if applicable) (M29)  
- [ ] Optimize training using distributed training (if applicable) (M30)  
- [ ] Apply quantization, compilation, or pruning to improve inference speed (M31)  

---

## Extra

- [ ] Write documentation for your application (M32)  
- [ ] Publish documentation on GitHub Pages (M32)  
- [ ] Revisit the original project description — did it turn out as expected?  
- [ ] Create an architectural diagram of your MLOps pipeline  
- [ ] Ensure all group members understand all parts of the project  
- [ ] Upload all code to GitHub  

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
