## Obsidian
	ctrl-e    Toggle edit/read mode
## git 
```bash
# commit against current working tree (trucating hash ok)
git difftool fcf30a9 -- src/kernel.py 
# against HEAD or some other commit
git difftool fcf30a9 HEAD -- src/kernel.py
git difftool fcf30a96 a1b2c3d4 -- src/kernel.py
```
## Reloading modules
```python
import importlib
import src.module
importlib.reload(src.module)

from src.module import func
```
## Testing
```bash
uv run pytest -q
```
`pytest` will look inside `pyproject.toml` to find test files and run them.  Use the .venv python environment.

