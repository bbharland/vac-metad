# New Project Bootstrap: uv + PyTorch (CUDA) + OpenMM + Jupyter + GitHub

## 1. Create and enter the project directory

```bash
mkdir ~/work/vac-metad
cd ~/work/vac-metad
```

---

## 2. Determine your CUDA version and choose the right PyTorch wheels

### 2a. Check what CUDA version your driver supports

```bash
nvidia-smi
```

Look at the top-right corner of the output for **CUDA Version**. For example:

```
| NVIDIA-SMI 580.159.03   Driver Version: 580.159.03   CUDA Version: 13.0 |
```

This tells you the *maximum* CUDA version your driver supports. You can use any
PyTorch wheel built for this version or earlier — CUDA drivers are
backwards-compatible. So a driver reporting 13.0 can run `cu128`, `cu126`, etc.,
but `cu130` or `cu132` would also work if wheels exist.

### 2b. See which CUDA wheel indexes PyTorch publishes

```bash
curl -s https://download.pytorch.org/whl/ | grep -o 'cu[0-9]*' | sort -u
```

This returns something like: `cu121 cu124 cu126 cu128 cu129 cu130 cu132`.
Pick the **highest `cu` version that is ≤ your driver's CUDA version** and for
which PyTorch actually publishes wheels (not all of them do). In the example
above with CUDA 13.0, `cu130` is the right choice.

### 2c. Verify that wheels exist for your Python version

PyTorch wheels are compiled per Python version. Your Python version appears in
the wheel filename as `cp312` (Python 3.12), `cp313` (Python 3.13), etc. Check
that your chosen index actually has wheels for your target Python:

```bash
# Replace cu130 and cp313 with your chosen values
curl -s https://download.pytorch.org/whl/cu130/torch/ | grep 'cp313.*linux_x86_64'
```

If you get no output, that Python version has no wheel in that CUDA index — you
will need to choose a different CUDA index or a different Python version.

> **Key lesson from practice:** `cu128` had wheels up to torch 2.11.0 for cp312,
> but torch 2.12.0 cp312 wheels were not published there. `cu130` has torch 2.12.0
> but only for cp313 and cp314 — not cp312. The intersection of CUDA index, torch
> version, and Python version must all exist. Always check before pinning.

### 2d. Summary: choose your stack

Once you have confirmed the above, write down your three choices:

| What            | Example value    |
| --------------- | ---------------- |
| Python version  | 3.13             |
| PyTorch index   | `cu130`          |
| OpenMM CUDA pkg | `openmm-cuda-13` |

Everything in the rest of this guide follows from these three values.

---

## 3. Initialize the uv project

```bash
uv init --python 3.13
```

> NOTE: Python 3.13 is required if you are using the `cu130` PyTorch index, as
> the cu130 wheels for torch 2.12.0+ are only published for cp313 and cp314.
> If your scientific packages don't support 3.13, fall back to the highest
> `cu12x` index that has the torch version and cp312 wheels you need.

This creates: `pyproject.toml`, `README.md`, `.python-version`, `.venv/`, `hello.py`

```bash
rm main.py    # remove the placeholder script uv init creates
```

---

## 4. Configure PyTorch CUDA index in pyproject.toml

`uv add torch --index-url ...` does NOT persist correctly into the lockfile.
The correct approach is to declare the index in pyproject.toml.
Add the following block to `pyproject.toml` (after `[project]`), substituting
your chosen CUDA index from Step 2:

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
torchvision = { index = "pytorch-cu130" }
```

---

## 5. Add all dependencies

```bash
# Core scientific stack
uv add \
    torch torchvision \
    openmm openmm-cuda-13 \
    "openmmtorch[cuda13]" \
    mdtraj \
    pymbar \
    kdepy \
    scikit-learn \
    matplotlib \
    tqdm \
    ipykernel \
    ipywidgets \
    jupyterlab \
    jupyter

# Code formatting
uv add --dev black
```

> NOTE on OpenMM + CUDA: `uv add "openmm[cuda13]"` is broken — uv cannot resolve
> that extra. Instead, install `openmm` + `openmm-cuda-13` as separate packages.
> The `openmm-cuda-13` wheel on PyPI provides the CUDA 13 platform plugin.
> Substitute `openmm-cuda-12` / `openmmtorch[cuda12]` if you are on a cu12x stack.

---

## 6. Verify the full GPU stack

### 6a. Verify PyTorch sees the GPU

```bash
uv run python -c "
import torch
print('torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Compute capability:', torch.cuda.get_device_capability(0))
"
```

Expected output (example):

```
torch version: 2.12.0+cu130
CUDA available: True
GPU: NVIDIA GeForce GTX 1650
Compute capability: (7, 5)
```

> **GTX 1650 note:** The 1650 is Turing architecture (compute capability 7.5).
> If a future PyTorch release drops support for older architectures, `is_available()`
> will return False. The compute capability check lets you spot this early.

### 6b. Verify OpenMM sees your GPU

```bash
uv run python -m openmm.testInstallation
```

You should see the CUDA platform listed with passing tests.

---

## 7. Register the Jupyter kernel under your project name

```bash
uv run python -m ipykernel install --user --name vac-metad --display-name "vac-metad"
```

Your kernel will now appear as **vac-metad** in JupyterLab / VS Code.
To confirm it's registered:

```bash
uv run jupyter kernelspec list
```

---

## 8. Set up git and push to GitHub

```bash
# Initialize git (uv init may have already done this)
git init
git add .
git commit -m "initial project setup"

# Create the GitHub repo and push
gh repo create vac-metad --public --source=. --remote=origin --push
```

---

## 9. Write your CLAUDE.md for Claude Code

```bash
cat > CLAUDE.md << 'EOF'
# Project: vac-metad

## Environment
- Fedora Linux
- Package manager: uv (never use pip directly)
- Python: 3.13 (as set in .python-version)
- All source modules live in src/

## Code Style
- Formatter: black (line length 88)
- Type hints required on all public function signatures
- Docstrings: NumPy style
- Imports: stdlib → third-party → local, each group separated by a blank line

## Key Packages
- OpenMM for MD simulations (CUDA 13 via openmm-cuda-13)
- PyTorch (cu130 wheels from download.pytorch.org)
- openmmtorch[cuda13] for PyTorch/OpenMM integration
- mdtraj for trajectory analysis
- pymbar for free energy calculations
EOF
```

---

## 10. Recommended .gitignore additions

`uv init` creates a basic `.gitignore`. Add these:

```bash
cat >> .gitignore << 'EOF'

# Data and trajectories
data/
*.xtc
*.dcd
*.nc
*.trr

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# OS
.DS_Store
EOF
```

---

## 11. Sanity checks

```bash
# Python and uv env
uv run python --version
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run python -m openmm.testInstallation

# Jupyter
uv run jupyter lab --version
uv run jupyter kernelspec list

# Git
git log --oneline
gh repo view
```

---

## 12. Launch Claude Code

From the project root, start a session:

```bash
cd ~/work/vac-metad
claude
```

### Key file locations

- **Project instructions — `CLAUDE.md`**
  `~/work/vac-metad/CLAUDE.md`
  Checked into the repo (see Section 9). Loaded automatically into context
  at the start of every session in this directory.

- **Persistent memory — global context directory**
  Claude Code also keeps a per-project memory store *outside* the repo,
  under your home directory:
  
  ```
  ~/.claude/projects/-home-bharland-work-vac-metad/memory/
  ```
  
  (The directory name is the project's absolute path with `/` replaced by
  `-`, prefixed with `-`.) Inside it:
  
  - `MEMORY.md` — a short index of saved notes
  - topic files (e.g. `feedback_*.md`, `project_*.md`, `user_*.md`) — durable
    notes Claude writes over time about your preferences, project context,
    and corrections you've given it
  
  This is loaded automatically each session too — nothing to configure.
  It currently starts empty and fills in as you work with Claude.

---

## Quick-reference: day-to-day commands

| Task               | Command                                                                                 |
| ------------------ | --------------------------------------------------------------------------------------- |
| Add a package      | `uv add <pkg>`                                                                          |
| Run a script       | `uv run python script.py`                                                               |
| Start JupyterLab   | `uv run jupyter lab`                                                                    |
| Sync after pulling | `uv sync`                                                                               |
| Update all deps    | `uv lock --upgrade && uv sync`                                                          |
| Re-register kernel | `uv run python -m ipykernel install --user --name vac-metad --display-name "vac-metad"` |

---

## Appendix: Troubleshooting wheel resolution errors

If `uv sync` fails with `No solution found when resolving dependencies`, the most
common causes are a mismatch between CUDA index, torch version, and Python version.
Work through this checklist:

**1. Check what torch versions the index actually has for your Python:**

```bash
curl -s https://download.pytorch.org/whl/cu130/torch/ | grep 'cp313.*linux_x86_64' | grep -o 'torch-[^-]*'
```

**2. If no output, try the next CUDA index up or down:**

```bash
# Check cu132
curl -s https://download.pytorch.org/whl/cu132/torch/ | grep 'cp313.*linux_x86_64' | grep -o 'torch-[^-]*'
```

**3. If you need to stay on an older Python (e.g. cp312), check which indices have cp312 wheels:**

```bash
for idx in cu126 cu128 cu129 cu130 cu132; do
  echo -n "$idx: "
  curl -s https://download.pytorch.org/whl/$idx/torch/ | grep -c 'cp312.*linux_x86_64' || echo 0
done
```

**4. Once you find a working combination**, update both `requires-python` in
`pyproject.toml` and the `[[tool.uv.index]]` URL to match, then `rm -rf .venv && uv sync`.
