"""Feature engineering: financial indicators and time-series transforms.

Three indicator families are provided:

- ``indicators``     — classical technical analysis (RSI, MACD, Bollinger, ATR, EMA, SMA)
- ``microstructure`` — modern market-microstructure / liquidity (Kyle's λ, VPIN, Amihud,
                       Roll, Corwin-Schultz, OFI, realised quarticity)
- ``fractal``        — fractal & information-theoretic (Hurst R/S, fractional differentiation,
                       sample/permutation entropy, Lyapunov, DFA, multiscale entropy)
- ``regime``         — volatility decomposition & regime detection (RV, BPV, semivariance,
                       jump variation, CUSUM change-point, GMM regime probability)
- ``timeseries``     — generic transforms (returns, log returns, lag, rolling z-score, vol)

Twenty modern indicators total, each with the closed-form equation and the
original citation in its docstring.
"""

from quantflow.features.indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
)
from quantflow.features.timeseries import (
    lag,
    log_returns,
    returns,
    rolling_volatility,
    rolling_zscore,
)

# Modern indicator families
from quantflow.features.microstructure import (
    amihud_illiquidity,
    corwin_schultz_spread,
    kyle_lambda,
    order_flow_imbalance,
    realised_quarticity,
    roll_effective_spread,
    vpin,
)
from quantflow.features.fractal import (
    dfa,
    frac_diff_ffd,
    hurst_rs,
    lyapunov_rosenstein,
    multiscale_entropy,
    permutation_entropy,
    sample_entropy,
)
from quantflow.features.regime import (
    bipower_variation,
    cusum_change_point,
    hmm_regime_probability,
    jump_variation,
    realised_semivariance,
    realised_volatility,
)

__all__ = [
    # classical
    "sma", "ema", "rsi", "macd", "bollinger_bands", "atr",
    # generic time series
    "returns", "log_returns", "lag", "rolling_zscore", "rolling_volatility",
    # microstructure
    "kyle_lambda", "vpin", "amihud_illiquidity", "order_flow_imbalance",
    "roll_effective_spread", "corwin_schultz_spread", "realised_quarticity",
    # fractal / info-theoretic
    "hurst_rs", "frac_diff_ffd", "sample_entropy", "permutation_entropy",
    "lyapunov_rosenstein", "dfa", "multiscale_entropy",
    # regime / vol-decomp
    "realised_volatility", "bipower_variation", "realised_semivariance",
    "jump_variation", "cusum_change_point", "hmm_regime_probability",
]
