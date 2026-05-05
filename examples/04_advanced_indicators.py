"""Advanced indicator showcase — 20 modern features in one pipeline.

Demonstrates the next-generation indicator suite:
- Microstructure / liquidity (Kyle's λ, VPIN, Amihud, OFI, Roll, Corwin-Schultz)
- Fractal / info-theoretic (Hurst, fractional differentiation, sample entropy,
  permutation entropy, DFA, multiscale entropy)
- Regime / volatility decomposition (RV, BPV, semivariance, jump variation,
  CUSUM, GMM regime probability)

Run:
    python examples/04_advanced_indicators.py
"""

from __future__ import annotations

import numpy as np

from quantflow.data import SyntheticLoader
from quantflow.features import (
    amihud_illiquidity,
    bipower_variation,
    corwin_schultz_spread,
    cusum_change_point,
    dfa,
    frac_diff_ffd,
    hmm_regime_probability,
    hurst_rs,
    jump_variation,
    kyle_lambda,
    log_returns,
    multiscale_entropy,
    order_flow_imbalance,
    permutation_entropy,
    realised_quarticity,
    realised_semivariance,
    realised_volatility,
    roll_effective_spread,
    sample_entropy,
)


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * (60 - len(title)))


def main() -> None:
    ohlcv = SyntheticLoader(n=2_000, sigma=0.3, seed=11).load()
    p = ohlcv.close
    v = ohlcv.volume
    h = ohlcv.high
    lo = ohlcv.low
    r = log_returns(p)[1:]

    print("=" * 64)
    print("  QuantFlow advanced indicators - 20 features on 2,000 GBM bars")
    print("=" * 64)

    section("Microstructure & liquidity")
    print(f"  Kyle's lambda    (last):   {kyle_lambda(p, v, window=60)[-1]:+.3e}")
    print(f"  Amihud ILLIQ     (last):   {amihud_illiquidity(p, v, window=21)[-1]:+.3e}")
    print(f"  Order-flow imbal.(last):   {order_flow_imbalance(p, v, window=50)[-1]:+.3f}")
    rs = roll_effective_spread(p, window=60)
    print(f"  Roll spread      (mean):   {np.nanmean(rs):.4f}")
    cs = corwin_schultz_spread(h, lo)
    print(f"  Corwin-Schultz   (mean):   {np.nanmean(cs):.4f}")
    print(f"  Realised quart.  (last):   {realised_quarticity(p, window=78)[-1]:.3e}")

    section("Fractal & information-theoretic")
    print(f"  Hurst R/S        (full):   {hurst_rs(p, min_window=16, max_window=512):.3f}")
    print(f"  DFA exponent     (full):   {dfa(p, min_window=16, max_window=512):.3f}")
    print(f"  Sample entropy   (full):   {sample_entropy(r, m=2):.3f}")
    print(f"  Permutation ent. (full):   {permutation_entropy(r, m=3):.3f}")
    fd = frac_diff_ffd(p, d=0.4, threshold=1e-3)
    print(f"  Frac-diff (d=0.4) (last):   {fd[-1]:+.3e}   (last warm: {np.where(np.isnan(fd))[0][-1] + 1})")
    mse = multiscale_entropy(r, scales=5, m=2)
    print(f"  Multiscale ent.  (mean):   {np.nanmean(mse):.3f}  across {len(mse)} scales")

    section("Regime & volatility decomposition")
    rv = realised_volatility(r, window=78)
    bpv = bipower_variation(r, window=78)
    jv = jump_variation(r, window=78)
    rs_asym = realised_semivariance(r, window=78, side="both")
    print(f"  Realised vol     (last):   {rv[-1]:.4f}")
    print(f"  Bipower variation(last):   {bpv[-1]:.4f}")
    print(f"  Jump variation   (last):   {jv[-1]:.4e}   (RV^2 - BPV, truncated)")
    print(f"  RS asymmetry     (last):   {rs_asym[-1]:+.4e}")
    cp = cusum_change_point(r, threshold=4.0, drift=0.5)
    print(f"  CUSUM triggers   (count):  {int(cp.sum())}")
    try:
        hmm = hmm_regime_probability(r, window=252, refit_every=42)
        print(f"  P(high-vol regime)(last):  {hmm[-1]:.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  HMM regime prob.: skipped ({exc})")

    print("\nSee `quantflow/features/{microstructure,fractal,regime}.py` for the equations.")


if __name__ == "__main__":
    main()
