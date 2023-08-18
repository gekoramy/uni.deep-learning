# uni.deep-learning

init python venv

```shell
python -m venv venv
venv/bin/pip install jupytext
```

enter python venv

```shell
source venv/bin/activate
```

from `[...].py` to `[...].ipynb`

```shell
jupytext --output notebook.ipynb notebook.py [--update]
```

from `[...].ipynb` to `[...].py`

```shell
jupytext --output notebook.py notebook.ipynb \
  --to py:percent \
  --opt notebook_metadata_filter="-all" \
  --opt cell_metadata_filter="-all"
```
