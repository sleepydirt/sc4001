# Part (a)

All experiments are run in `parta.ipynb`. Helper methods and classes are found in the `src/` folder. To run, please use `uv` and run `uv sync`.

## Environment

Python 3.12. Dependencies are pinned in `uv.lock`:

```bash
cd parta
uv sync
```

Note that an NVIDIA GPU was used to speed up computation, where possible, the code will try to use a cuda device.

## Seeds

`Config.seed = 42` in `src/config.py` is used for the A1 grid search and the A1 final model.

The A2 and A3 comparisons use five different seeds, `(67, 420, 69, 21, 13)`
