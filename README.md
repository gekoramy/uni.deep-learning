# uni.deep-learning

init python venv

```shell
python -m venv venv
venv/bin/pip install -r requirements.txt
```

populate `.git/config`

```shell
git config --local filter.strip-notebook-output.clean "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
```

populate `.git/hooks`

```shell
pre-commit install
```
