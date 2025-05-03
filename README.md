# PGKD

An unofficial python implementation of Performance Guided Knowledge Distillation (Di Palo et al., 2024)

## Setup

1. Install [uv](https://github.com/astral-sh/uv):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Development

- Run `pre-commit run --all-files` to check all files
- Run `black .` to format code
- Run `ruff check .` to lint code 