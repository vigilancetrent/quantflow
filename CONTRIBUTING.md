# Contributing to QuantFlow

Thanks for considering a contribution. QuantFlow is small and opinionated — the goal is to keep it that way while growing the indicator catalogue, broker adapters, and reference strategies.

## Ground rules

1. **Read [`docs/design-decisions.md`](docs/design-decisions.md) first.** It explains *why* the library is shaped the way it is. Most "why don't you do X?" questions are answered there.
2. **No look-ahead bias, ever.** Any indicator, model, or backtest contribution must respect the next-bar-open execution rule. PRs that compute features on bar `t` and trade at bar `t`'s close will be rejected.
3. **Walk-forward validation is the default.** Random k-fold is unsafe for finance and we don't ship it.
4. **Citations matter.** Modern indicators (`quantflow/features/microstructure.py`, `fractal.py`, `regime.py`) carry the original paper's equation and reference. Match the existing format.

## How to contribute

### Reporting a bug
Open an issue with:
- The QuantFlow version (or commit hash)
- A minimal reproducer (synthetic data is fine — `SyntheticLoader` is in the library)
- The expected vs. actual behaviour

### Proposing a new indicator
1. Open an issue describing the indicator, its citation, and what gap it fills
2. Wait for a 👍 (we keep the surface small on purpose)
3. Submit a PR with: implementation, docstring with equation + citation, at least one invariant test

### Proposing an architectural change
1. Open an issue tagged `discussion` and outline the change
2. We may ask for a `docs/design-decisions.md` entry as part of the PR

## Development

```bash
git clone https://github.com/vigilancetrent/quantflow.git
cd quantflow
pip install -e .[dev]
pytest                          # run all tests
ruff check quantflow tests      # lint
```

## Code style

- Line length 100, ruff-formatted
- Type annotations on public APIs; internal helpers can skip them
- Docstrings: short summary, then equation (in Unicode math), citation, "Use" line
- No print/log debugging in committed code

## Commit messages

Conventional Commits style:

- `feat(features): add hurst exponent estimator`
- `fix(backtest): correct slippage sign for short positions`
- `docs(readme): clarify walk-forward CV rationale`
- `test(metrics): cover empty-input edge case`

## Code of conduct

Participation in this project is governed by the [Contributor Covenant](CODE_OF_CONDUCT.md).
