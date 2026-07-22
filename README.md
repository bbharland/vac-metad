## Obsidian
```   ctrl-e    Toggle edit/read mode```

New vault:
* Close Obsidian first! (avoids settings being overwritten on quit)
```bash
VAULTNAME=Recipes # Edit here!
mkdir ~/obsidian/"$VAULTNAME"
cp -r ~/obsidian/vac-metad/.obsidian ~/obsidian/"$VAULTNAME"/
rm ~/obsidian/"$VAULTNAME"/.obsidian/workspace.json
```
* Then:
1. Open Obsidian → vault switcher → "Open folder as vault" → pick the new folder
   (NOT "Create new vault" — that gives a blank .obsidian)
2. If a Settings/plugins screen pops up: toggle on any community plugins
   (they clone in disabled by default), then close Settings.
3. Ctrl-= (or Ctrl+scroll) to fix zoom to taste.
   NOTE: zoom won't survive a full quit/reopen (known Linux bug, reported).
   Fix once: `body { zoom: 1.2; }` as a CSS snippet in vac-metad, enable it
   under Settings → Appearance → CSS snippets — it'll clone forward and
   survive restarts, unlike Ctrl-= zoom.
4. Window size: already handled globally via the KDE window rule for
   "obsidian" — nothing to do per-vault.

## git 
```bash
# commit against current working tree (trucating hash ok)
git difftool fcf30a9 -- src/kernel.py 
# against HEAD or some other commit
git difftool fcf30a9 HEAD -- src/kernel.py
git difftool fcf30a96 a1b2c3d4 -- src/kernel.py
# for notebooks
nbdiff-web file.ipynb
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

