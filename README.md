# mlops_project

Project for DTU mlops January course.

# Food-101 Image Classification with Vision Transformer

## Project description

The goal of this project is to build a complete MLOps pipeline for an image classification task. We will train and deploy a model that can recognize different types of food from images. The focus of the project is not on reaching state-of-the-art accuracy, but on creating a reproducible, well-structured, and deployable machine learning system using the tools taught in the DTU MLOps course.

We use the **Food-101** dataset from ETH Zurich, which contains 101,000 images divided into 101 food categories. Each class has 750 training images and 250 validation images. The dataset is large enough to be realistic, but still manageable for fast experimentation. The images have different sizes and some noise in the labels, which makes it a good example of a real-world dataset. The data is loaded using the Hugging Face `datasets` library, which provides an easy and reproducible way to access and version the data.

For the model, we use a **Vision Transformer (ViT)**, specifically the pretrained model `google/vit-base-patch16-224-in21k`. This model was trained on ImageNet-21k with more than 14 million images and can extract strong visual features. We fine-tune this model on Food-101 by adding a classification head with 101 output classes. By using a pretrained model, we avoid training from scratch and can focus on building a stable training and deployment pipeline.

The project includes data loading, preprocessing, training, evaluation, and deployment. We use configuration files and Hydra to manage hyperparameters and experiment settings. Training and evaluation metrics are logged using Weights & Biases. The code is structured in a modular way, with separate files for data handling, model definition, and training logic.

We also include unit tests for the data pipeline and the model, and run these tests automatically using GitHub Actions. Docker is used to make the training and inference environments reproducible. Finally, the trained model is served through a FastAPI application, allowing users to upload an image and receive a predicted food class.

Overall, this project demonstrates how a modern MLOps workflow can be applied to a realistic computer vision problem.



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
