# Reddit network project

## Setup

Run these commands from this repository with `uv` installed:

```powershell
uv sync --locked
uv run python -c "import mdacn; print('Package import OK')"
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
