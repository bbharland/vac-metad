# Project: vac-metad

## Environment

- Fedora linux
- Package manager: uv
- Python: 3.13
- All source modules live in ~/work/vac-metad/src/
- Work with VS Cold on jupyter notebooks in ~/work/vac-metad/

## File Permissions

- You may create and edit files in src/ freely
- Do NOT modify .ipynb notebooks directly

## Workflow: Typical Loop

1. User gives an instruction; you edit the relevant `.py` file in `src/`.
2. The user may then make their own edits to that file.
3. The user may ask you to check that their edits didn't break anything —
 re-read the current file rather than assuming your last edit is still
 in place.

## Code Style

- Formatter: black (line length 88)
- Type hints required on all public function signatures
- Docstrings: NumPy style
```python
def f(
    u: np.ndarray,
    max_time: float,
) -> np.ndarray:
    """
    Compute C(τ) = ⟨u(0)u(τ)⟩ / ⟨u²⟩.

    Parameters
    ----------
    u : array, shape (num_frames,)
        Description
    tau : float
        Description

    Returns
    -------
    c : array, shape (num_lags,)
        Description
    """
```
- Imports: stdlib → third-party → local, each group separated by a blank line

## Key Packages

- OpenMM for MD simulations (CUDA 13 via openmm-cuda-13 & "openmmtorch[cuda13]")
- PyTorch (cu130 wheels from download.pytorch.org)

