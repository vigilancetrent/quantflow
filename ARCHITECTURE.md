# QuantFlow — Architecture

## 1. Layered architecture

QuantFlow follows a **strict five-layer pipeline**. Each layer has a narrow contract and can be swapped independently.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                            │
│  examples/  •  user strategies  •  research notebooks  •  services  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────┐
│                          ORCHESTRATION                              │
│         core.pipeline   •   core.registry   •   core.config         │
│   (composes Data → Features → Model → Eval → Execution as a DAG)    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
   ┌──────────────┬──────────────┼──────────────┬──────────────┐
   ▼              ▼              ▼              ▼              ▼
┌────────┐  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  DATA  │─▶│ FEATURES │──▶│  MODELS  │──▶│  EVAL    │──▶│  EXEC    │
├────────┤  ├──────────┤   ├──────────┤   ├──────────┤   ├──────────┤
│Loaders │  │Indicators│   │ base.py  │   │ Metrics  │   │ Broker   │
│Streams │  │TimeSeries│   │Classical │   │Backtest  │   │ Orders   │
│Preproc │  │ Pipeline │   │   ML     │   │  Risk    │   │  Live    │
│        │  │          │   │  Deep    │   │WalkFwdCV │   │SmartExec │
│        │  │          │   │   RL     │   │          │   │          │
│        │  │          │   │ Ensemble │   │          │   │          │
└────────┘  └──────────┘   └──────────┘   └──────────┘   └──────────┘
                                  │
                                  ▼
                          ┌────────────────┐
                          │  RL SUBSYSTEM  │
                          ├────────────────┤
                          │  envs/         │  ← Gym-compatible
                          │  agents/       │  ← DQN, PPO, SAC
                          │  rewards.py    │  ← profit, sharpe, sparse
                          └────────────────┘
                                  │
                                  ▼
                          ┌────────────────┐
                          │  CROSS-CUTTING │
                          ├────────────────┤
                          │  utils.logging │
                          │  utils.cache   │
                          │  audit log     │
                          └────────────────┘
```

## 2. Data flow

### Offline (research / backtest)

```
   raw market data
        │
        ▼
   data.loaders ──► data.preprocessing ──► features.pipeline
                                                  │
                                                  ▼
                                          models.* (fit)
                                                  │
                                                  ▼
                                          evaluation.backtesting
                                                  │
                                                  ▼
                                          metrics + report
```

### Online (live / paper trading)

```
   broker stream ──► data.streams ──► features.pipeline (incremental)
                                              │
                                              ▼
                                       models.* (predict)
                                              │
                                              ▼
                                  execution.broker.place_order
                                              │
                                              ▼
                                       audit log + metrics
```

## 3. Why this shape?

### Pipeline as a DAG, not a linear chain
Real strategies branch: one feature pipeline feeds multiple models; one model output drives both an order *and* a risk gauge. `core.pipeline` is a DAG over named nodes, each implementing a `fit` / `transform` / `predict` contract.

### Plugin registry as the extension point
`core.registry` is a process-wide map: `kind → name → factory`. New indicators, models, agents, and brokers are added by **registering**, not by editing the core. This keeps the library forkable and lets domain extensions (robotics, supply chain) live in separate packages.

```python
from quantflow.core.registry import register

@register("indicator", "my_custom")
def my_custom(prices, period=14):
    ...
```

### Separation of model kinds behind one ABC
`models.base.BaseModel` defines `fit / predict / save / load`. Classical (statsmodels), ML (sklearn/XGBoost), deep (PyTorch), and RL (SB3) each subclass it. Backtest and pipeline code never branches on model type — they call the ABC.

### RL is a peer, not a special case
`rl.envs.TradingEnv` exposes the standard Gymnasium interface. Any RL library (SB3, RLlib, CleanRL, custom) works. Reward shaping is pluggable via `rl.rewards`, and the same `evaluation.backtesting` engine validates both supervised and RL-derived signals.

## 4. Performance design

| Concern | Approach |
|---|---|
| Indicator throughput | Vectorised NumPy first; Numba `@njit` planned for v0.2 hot paths |
| Inference latency | ONNX export from PyTorch models; pure-NumPy inference for classical |
| Training scale | Distributed via Ray (planned v0.3); single-node multi-GPU via PyTorch |
| Memory | Streaming loaders; `Polars` backend opt-in for >RAM datasets |
| Online updates | `BaseModel.partial_fit` contract for online-learning subclasses |

## 5. Cross-domain abstraction

The interface that makes the library reusable is **the `Env` + `Agent` + `Metric` triplet**, not the indicators. To extend:

| Domain | What you replace | What you keep |
|---|---|---|
| Robotics | `TradingEnv` → your sim env; reward = task completion | Agents, training loop, metrics, registry |
| Supply chain | State = inventory + demand; action = reorder qty | Same |
| RecSys | State = user context; action = item; reward = engagement | Same |

Indicators and brokers are the only finance-coupled modules. Everything else is pure sequential decision-making infrastructure.

## 6. Failure modes & mitigations

| Failure | Mitigation in QuantFlow |
|---|---|
| Look-ahead leak in features | `features.pipeline` enforces `shift(1)` on label; `WalkForwardSplit` is default |
| Survivorship bias | Loaders surface `is_active` and `delist_date` columns explicitly |
| Backtest over-fit | Built-in deflated Sharpe ratio + multiple-testing correction (planned) |
| Live/backtest divergence | Same `Pipeline` object runs in both modes; only the data source differs |
| Model staleness in prod | Drift monitor on prediction distribution (planned) |

## 7. Testing strategy

- **Unit** — every indicator vs. a hand-computed reference series
- **Property** — Hypothesis-based invariants (RSI ∈ [0, 100], drawdown ≤ 0, Sharpe scale-invariant under × constant)
- **Integration** — full pipeline end-to-end on synthetic data with known optimum
- **Regression** — pinned snapshots of backtest results on a fixed dataset

## 8. Non-goals

- A high-frequency C++ trading engine — use a dedicated HFT stack for sub-millisecond
- A data vendor — we wrap loaders, we don't ship data
- A research notebook UI — Jupyter integrates, but we don't replace it
- A regulatory compliance tool — we provide audit-log hooks; compliance is yours
