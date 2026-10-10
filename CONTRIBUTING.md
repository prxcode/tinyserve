# Contributing

Thanks for taking an interest. Issues and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/prxcode/tinyserve.git
cd tinyserve
uv sync --extra cpu      # or --extra cu126 on a machine with an NVIDIA GPU
```

## Before opening a pull request

Run the same checks as CI. All of them must pass:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

- Keep each pull request focused on one change.
- New behavior needs tests, including failure paths.
- Performance claims need a benchmark run, stating the hardware and the command used.
- If a change affects the design, update `docs/architecture.md`, or add an entry to
  `docs/decisions.md` explaining the trade-off.

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(scheduler): preempt newest sequence when cache is full
fix(server): free blocks when client disconnects
docs: add benchmark methodology
```

Common types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`, `chore`.

## Dependencies

Dependencies are locked in `uv.lock`. To add or upgrade one, edit `pyproject.toml`, run
`uv lock`, and commit both files together. Do not commit model weights, datasets or
benchmark traces. They are ignored by `.gitignore` on purpose.
