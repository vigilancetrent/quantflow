# Design Decisions

A running log of non-obvious decisions and the reasoning behind them. Future
contributors should read this before proposing structural changes.

## DD-01 — Why a registry instead of subclass auto-discovery

**Decision:** Plugins (indicators, models, agents, brokers) register themselves
via `@register("kind", "name")` rather than being discovered by importing all
subclasses of an ABC.

**Why:** Auto-discovery requires importing every module at startup, which couples
import time to the breadth of the library. A registry is opt-in: users only
import the modules they need, and registration is a side-effect of import. Cost:
slightly more boilerplate. Benefit: O(installed-modules) startup, not
O(everything-the-package-could-do).

## DD-02 — Why next-bar-open execution in the backtest engine

**Decision:** Trades signaled at the close of bar `t` execute at bar `t+1`'s
price, not at bar `t`'s close.

**Why:** Same-bar execution leaks the future. If an indicator computed on bar
`t`'s close says "go long," and you fill at `t`'s close, you've already seen
that close. Real-world execution can only happen *after* the data point is
known. Many published backtests inflate Sharpe by 0.3–0.7 from this single
bug; QuantFlow refuses to do it.

## DD-03 — Why walk-forward CV is the default, not random k-fold

**Decision:** `data.preprocessing.walk_forward_split` is recommended for every
finance-ML workflow. Random k-fold is *not provided* — users who want it can
import `sklearn.model_selection.KFold` and accept the consequences.

**Why:** Random k-fold places future bars in the training set and past bars in
the test set. Models trained this way memorise tomorrow's noise to predict
yesterday. Walk-forward respects time order. Cost: smaller effective training
set early. Benefit: out-of-sample numbers that survive contact with live
trading.

## DD-04 — Why a pluggable reward function on `TradingEnv`

**Decision:** Rewards are not hardcoded in `step()`. They're injected via a
callable taking a `RewardContext`.

**Why:** Reward shaping is the dominant lever in RL-for-trading research. PnL,
Sharpe-shaped, sparse-terminal, and risk-penalised rewards each induce
qualitatively different policies. Tying the env to one shape forces a fork to
try another. Cost: tiny indirection per step. Benefit: reward-engineering is a
single-import change, not a refactor.

## DD-05 — Why no built-in market-data vendor

**Decision:** QuantFlow ships a `SyntheticLoader` and a `CSVLoader`. We do not
include adapters for any specific data vendor (Polygon, IEX, Yahoo, etc.).

**Why:** Data terms-of-use vary, vendor APIs churn, and bundling them creates
an implicit endorsement. Loaders are 50 lines each — users add their own. We'd
rather support five vendors well in a separate `quantflow-data-*` package than
five vendors poorly in core.

## DD-06 — Why a vectorised backtester *and* an event-driven broker

**Decision:** Two execution paths exist: `evaluation.backtesting.BacktestEngine`
(vectorised, fast, single-instrument) and `execution.broker.PaperBroker`
(event-driven, multi-instrument).

**Why:** Vectorised backtests are an order of magnitude faster — required for
parameter sweeps and walk-forward CV across thousands of configurations. But
they don't capture multi-asset interactions, partial fills, or exotic order
types. The broker abstraction handles those, at the cost of speed. Both share
the same metric library and reporting layer, so reports are comparable.

## DD-07 — Why `BaseModel.fit` accepts NumPy, not pandas

**Decision:** All model interfaces consume and return `np.ndarray`, not
`pd.DataFrame`.

**Why:** NumPy arrays are the lowest common denominator across PyTorch,
TensorFlow, JAX, scikit-learn, XGBoost, ONNX. Pandas is a *user-facing*
convenience; coercing it inside the model layer creates ambiguity (which
column is the target? does the index matter?). Users are free to pass
`df.to_numpy()` and `df.target.to_numpy()`; we don't impose pandas internally.

## DD-08 — Why we store `is_active` and `delist_date` on OHLCV

**Decision:** The `OHLCV` data class explicitly includes `is_active`. Even
when only one symbol is being studied, the flag is required.

**Why:** Survivorship bias is the single most common silent error in equity
backtests. Forcing the field into the data structure makes it impossible to
ignore. A model that can't see delistings is a model that thinks every
company that ever existed still exists.

## DD-09 — Why an in-memory plugin registry, not a config file

**Decision:** The registry lives in process memory, populated at import time.

**Why:** Config-file plugins require a discovery and validation layer that
isn't worth its weight at this scale. Import-time registration is dead simple,
debuggable with `pdb`, and breaks loudly at startup if a plugin fails to load
— rather than silently mis-routing at request time.

## DD-10 — Why no native Jupyter integration

**Decision:** No `quantflow.notebook`, no IPython magics, no auto-`%matplotlib`.

**Why:** Jupyter integrations age badly. The library should work *equally well*
in a script, a notebook, a Sphinx doctest, and a serverless function.
`matplotlib` and `plotly` are listed as optional `[viz]` extras; users compose
their own visualisation. We refuse to make the library dependent on a UI layer.
