
```bash
uv sync
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=project
```

To run after installing just do:
```bash
uv run --with jupyter jupyter lab
```

Then run cells
