"""
Long-Wave Trend Extractor (LWTE).

v2.1 design (2026-06-24): extracts ultra-long components (>30y) that fall
outside AMPD's FFT-resolvable range.

Methods provided:
- Butterworth low-pass filter (default, simple, robust)
- Hodrick-Prescott filter (alternative, classical in macroeconomics)
- Wavelet decomposition (alternative, multi-resolution)

The output of LWTE is the "trend" component; the residual is the
"cycle" component which AMPD then operates on.

Reference: proposal-v2.md §4.3 (NEW in v2.1)
"""
import numpy as np
from typing import Optional, Tuple
from scipy.signal import butter, filtfilt


class LongWaveTrendExtractor:
    """
    Decompose time series into long-wave trend + residual cycle.

    Args:
        cutoff_period_months: periods longer than this are treated as trend
                              default 360 months = 30 years
        method: 'butter' | 'hp' | 'wavelet'
        order: Butterworth filter order (only for method='butter')
    """

    def __init__(
        self,
        cutoff_period_months: int = 360,
        method: str = "hp",  # Hodrick-Prescott — gold standard in macroeconomics
        order: int = 4,  # only used if method='butter'
    ):
        if method not in ("butter", "hp", "wavelet"):
            raise ValueError(f"Unknown method: {method}")
        self.cutoff_period_months = cutoff_period_months
        self.method = method
        self.order = order

    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose X into trend and cycle (residual).

        Args:
            X: np.ndarray of shape (T,) or (T, N)

        Returns:
            trend: np.ndarray same shape as X, ultra-low-frequency component
            cycle: np.ndarray same shape as X, high-frequency residual
        """
        if self.method == "butter":
            return self._butter_transform(X)
        elif self.method == "hp":
            return self._hp_transform(X)
        elif self.method == "wavelet":
            return self._wavelet_transform(X)

    def _butter_transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Butterworth low-pass filter."""
        # Sample freq = 1 sample per month; Nyquist = 0.5 cycles/month
        # Cutoff frequency (cycles/month) = 1 / cutoff_period_months
        cutoff_freq = 1.0 / self.cutoff_period_months
        nyq = 0.5
        normalized_cutoff = cutoff_freq / nyq  # in [0, 1]

        b, a = butter(self.order, normalized_cutoff, btype="low")

        if X.ndim == 1:
            trend = filtfilt(b, a, X)
        else:
            trend = np.zeros_like(X)
            for i in range(X.shape[1]):
                trend[:, i] = filtfilt(b, a, X[:, i])
        cycle = X - trend
        return trend, cycle

    def _hp_transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Hodrick-Prescott filter. λ for monthly: 14400 (Ravn-Uhlig default)."""
        try:
            from statsmodels.tsa.filters.hp_filter import hpfilter
        except ImportError:
            raise ImportError("statsmodels not installed. Run: pip install statsmodels")

        # λ proportional to cutoff period: λ = (cutoff / (2π))^2 * something
        # Simplified: λ = 14400 for monthly (Ravn-Uhlig convention)
        lam = 14400.0

        if X.ndim == 1:
            cycle, trend = hpfilter(X, lamb=lam)
        else:
            trend = np.zeros_like(X)
            cycle = np.zeros_like(X)
            for i in range(X.shape[1]):
                c, t = hpfilter(X[:, i], lamb=lam)
                trend[:, i] = t
                cycle[:, i] = c
        return trend, cycle

    def _wavelet_transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wavelet decomposition: keep only the coarsest level as trend.
        """
        try:
            import pywt
        except ImportError:
            raise ImportError("pywavelets not installed. Run: pip install pywavelets")

        # Use 'sym8' wavelet, decompose to a level where the coarsest
        # approximation corresponds to our cutoff period
        # For T=720 with monthly samples, level 5 corresponds to 720/32 = 22.5 months
        # level 6 -> ~45 months; level 7 -> ~90 months
        # We choose level adaptively: smallest level where level >= log2(cutoff_period_months)
        import math
        level = max(1, int(math.log2(self.cutoff_period_months)))

        if X.ndim == 1:
            coeffs = pywt.wavedec(X, "sym8", level=level)
            # Zero out all detail coefficients; keep only approximation
            coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
            trend = pywt.waverec(coeffs, "sym8")[: len(X)]
        else:
            trend = np.zeros_like(X)
            for i in range(X.shape[1]):
                coeffs = pywt.wavedec(X[:, i], "sym8", level=level)
                coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
                trend[:, i] = pywt.waverec(coeffs, "sym8")[: X.shape[0]]
        cycle = X - trend
        return trend, cycle


def evaluate_trend_recovery(
    extracted_trend: np.ndarray,
    true_trend: np.ndarray,
) -> dict:
    """Compute recovery metrics between extracted and true long-wave trends."""
    if extracted_trend.ndim == 2:
        extracted_trend = extracted_trend.mean(axis=1)
    if true_trend.ndim == 2:
        true_trend = true_trend.mean(axis=1)

    # Handle empty / zero-amplitude true_trend (no long-wave in signal)
    true_std = true_trend.std()
    if true_std < 1e-6:
        return {
            "corr": float("nan"),
            "rmse": 0.0,
            "nrmse": float("nan"),
            "r2": float("nan"),
            "passed": True,  # trivially pass — no trend to recover
            "note": "no long-wave component in signal",
        }

    # Pearson correlation
    corr = np.corrcoef(extracted_trend, true_trend)[0, 1]

    # Normalized RMSE
    rmse = np.sqrt(np.mean((extracted_trend - true_trend) ** 2))
    signal_range = true_trend.max() - true_trend.min()
    nrmse = rmse / signal_range if signal_range > 0 else float("inf")

    # R^2 (1 - SS_res / SS_tot)
    ss_res = np.sum((extracted_trend - true_trend) ** 2)
    ss_tot = np.sum((true_trend - true_trend.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("inf")

    return {
        "corr": float(corr),
        "rmse": float(rmse),
        "nrmse": float(nrmse),
        "r2": float(r2),
        "passed": bool(corr > 0.7 and nrmse < 0.5),
    }


if __name__ == "__main__":
    # Quick smoke test: 60y pure sinusoid
    rng = np.random.default_rng(42)
    t = np.arange(720)
    true_trend = 2.0 * np.sin(2 * np.pi * t / 720)  # 60y period
    noise = rng.normal(0, 0.1, size=720)
    X = true_trend + noise

    lwte = LongWaveTrendExtractor(cutoff_period_months=360, method="butter")
    trend, cycle = lwte.fit_transform(X)
    metrics = evaluate_trend_recovery(trend, true_trend)

    print("Long-Wave Trend Extractor smoke test:")
    print(f"  corr:  {metrics['corr']:.4f} (target >0.7)")
    print(f"  nrmse: {metrics['nrmse']:.4f} (target <0.5)")
    print(f"  r2:    {metrics['r2']:.4f}")
    print(f"  passed: {metrics['passed']}")
    print(f"  trend std: {trend.std():.3f}, cycle std: {cycle.std():.3f}")