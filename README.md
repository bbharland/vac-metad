# Testing

`pytest` will look inside `pyproject.toml` to find test files and run them.  Use the .venv python environment:

```bash
uv run pytest -q
```

# Reloading modules

```python
import importlib
import src.module
importlib.reload(src.module)

from src.module import func
```


