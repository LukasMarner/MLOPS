import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops_project"
PYTHON_VERSION = "3.11"

# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run python -m {PROJECT_NAME}.data", echo=True, pty=not WINDOWS)

@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run python -m {PROJECT_NAME}.train", echo=True, pty=not WINDOWS)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)

@task
def smoke_data(ctx: Context, batch_size: int = 8) -> None:
    """Smoke test data loading (imports data.py and grabs 1 batch)."""
    ctx.run(
        "uv run python -c \""
        "from src.mlops_project.data import get_dataloaders;"
        f"train_loader, _ = get_dataloaders(batch_size={batch_size});"
        "x, y = next(iter(train_loader));"
        "print('x', x.shape, x.dtype);"
        "print('y', y.shape, y.dtype);"
        "print('x range', float(x.min()), float(x.max()));"
        "\"",
        echo=True,
        pty=not WINDOWS,
    )

@task
def smoke_model(ctx: Context) -> None:
    """Smoke test model forward pass."""
    ctx.run(
        "uv run python -c \""
        "import torch;"
        "from src.mlops_project.model import build_model;"
        "m = build_model(); m.eval();"
        "x = torch.randn(2,3,224,224);"
        "torch.set_grad_enabled(False);"
        "y = m(x);"
        "print('logits', y.shape);"
        "\"",
        echo=True,
        pty=not WINDOWS,
    )

