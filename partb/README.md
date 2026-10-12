# Part (b)

All experiments are run inside `partb.ipynb`. Helper code is inside the `src/` folder.

## Environment

Python 3.12, please use `uv` and `uv sync` to install dependencies

```bash
cd ..
uv sync
```

## Seeds

`Config.seed = 42` in `src/config.py` is used. May be non-deterministic due to GPU usage in torch.

