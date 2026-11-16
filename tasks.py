import os

from invoke import Context, task

WINDOWS = os.name == "nt"


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run pytest tests/", echo=True, pty=not WINDOWS)


@task
def fmt(ctx: Context) -> None:
    """Auto-fix lint issues and format the code."""
    ctx.run("uv run ruff check . --fix", echo=True, pty=not WINDOWS)
    ctx.run("uv run ruff format .", echo=True, pty=not WINDOWS)
