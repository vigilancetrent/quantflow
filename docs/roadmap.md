# QuantFlow Roadmap

The roadmap is a sequence of releases. Each release is a coherent step that
*ships* end-to-end value, not a grab-bag of features.

## v0.1 — Foundation (current)

**Status:** scaffolded.

- [x] Pipeline DAG (`core.pipeline`) and plugin registry (`core.registry`)
- [x] Vectorised classical indicators (RSI, MACD, Bollinger, ATR, EMA, SMA)
- [x] Time-series transforms (returns, log returns, lag, rolling z-score, rolling vol)
- [x] Walk-forward CV (`data.preprocessing.walk_forward_split`)
- [x] Event-driven `BacktestEngine` with commission + slippage
- [x] Financial / ML / RL metric library (Sharpe, Sortino, Calmar, drawdown, VaR, CVaR)
- [x] Synthetic + CSV data loaders
- [x] `BaseModel` ABC + `NaiveBaseline` reference
- [x] Gym-compatible `TradingEnv` with pluggable reward functions
- [x] PPO agent wrapper
- [x] PaperBroker reference implementation
- [x] Test suite (indicators, metrics, backtest, pipeline)

## v0.2 — Modern indicators & speed

- [ ] Microstructure indicators: Kyle's λ, VPIN, Amihud, Roll spread, Corwin–Schultz
- [ ] Fractal / info-theoretic: Hurst, fractional differentiation, sample entropy, permutation entropy, DFA
- [ ] Regime: realised variance / bipower variation / jump variation, CUSUM, GMM regime probability
- [ ] Numba `@njit` hot paths for the rolling-window indicators
- [ ] ONNX export utility for PyTorch models (`models.deep.export_onnx`)
- [ ] Numerically-stable streaming RSI / EMA (incremental updates)
- [ ] Polars-backed loaders for >RAM datasets

## v0.3 — Live infrastructure

- [ ] Broker adapters: Alpaca, IBKR, Binance (paper + live)
- [ ] WebSocket data streams (`data.streams`) with automatic reconnect
- [ ] Order management with idempotency keys + audit log
- [ ] Distributed training via Ray (`models.deep.train_distributed`)
- [ ] L2 order-book features (depth imbalance, micro-price)
- [ ] Live latency budgets enforced and logged

## v0.4 — Advanced research

- [ ] Hybrid Transformer + RL reference (encoder over price/orderbook → policy)
- [ ] Meta-learning: regime-conditioned policies that switch on detected regime
- [ ] Multi-asset portfolio RL (continuous Box action over weights)
- [ ] Deflated Sharpe ratio + multiple-testing correction in evaluation reports
- [ ] Explicit drift monitor — KL divergence between train and live feature dists

## v0.5 — Operability & explainability

- [ ] SHAP-based attribution for trading decisions
- [ ] Audit log with append-only signed entries (regulatory hooks)
- [ ] Built-in paper-trading sandbox with broker adapter
- [ ] Drift / staleness alerts surfaced through `utils.logging`
- [ ] Strategy report renderer (HTML + PDF) — equity, risk, trade list, attribution

## v1.0 — API stability + benchmarks

- [ ] Frozen public API (semver from here)
- [ ] Comprehensive docs site (mkdocs-material)
- [ ] Benchmark suite: QuantFlow vs. backtrader, vectorbt, zipline-reloaded on a fixed dataset
- [ ] Reference cookbook: 10 reproducible end-to-end strategies
- [ ] Cross-domain templates: robotics-control, supply-chain, recsys quickstart

## Beyond v1.0 — speculative

- Distributed live execution across multiple brokers with smart routing
- Differentiable backtester (PyTorch-native) for end-to-end strategy gradients
- Causal-inference layer for treating strategy changes as interventions
