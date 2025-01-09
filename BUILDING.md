# Building

## Publish new version

```console
rm -rf dist
uv build
UV_PUBLISH_TOKEN="pypi token here" uv publish
```

## Format and lint

```console
uv run ruff format
uv run ruff check
```

## Log

Enable log with

```console
LOG_LEVEL=DEBUG python main.py
```
