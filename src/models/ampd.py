"""
Adaptive Multi-Period Discovery (AMPD) module.

The core mechanism of CycleTimesNet v2:
- Take 1D time series
- FFT → amplitude spectrum
- Identify top-K frequencies (periods) within the mathematically resolvable range
- Optional: soft regularization toward domain theory periods

**v2.1 change (2026-06-24)**: added `max_period` parameter (default 360 months = 30 years)
because synthetic validation revealed that periods longer than T/2 cannot be reliably
recovered due to FFT Nyquist limit and bin discretization. See notes/06-key-finding-fft-limits.md.

Reference: proposal-v2.md §4.2
"""
import numpy as np
from typing import List, Optional


class AMPD:
    """
    Adaptive Multi-Period Discovery via FFT.

    Inputs are assumed monthly frequency. Output periods are in months.

    Constraints:
    - max_period: default 360 months (30 years) — above this, FFT bin quantization
                  and Nyquist limit make period recovery unreliable. For longer
                  cycles, use the Long-Wave Trend Extractor instead.
    """

    def __init__(
        self,
        top_k: int = 4,
        sample_freq: str = "M",  # monthly
        use_amplitude_weighted: bool = True,
        min_period: Optional[int] = 6,    # months; avoid trivial high-freq noise
        max_period: Optional[int] = 360,  # months = 30 years; FFT Nyquist limit
    ):
        self.top_k = top_k
        self.sample_freq = sample_freq
        self.use_amplitude_weighted = use_amplitude_weighted
        self.min_period = min_period
        self.max_period = max_period

    def fit(self, X: np.ndarray, n_fft: Optional[int] = None):
        """
        Compute and store the amplitude spectrum.

        Args:
            X: np.ndarray of shape (T,) or (T, N) — last axis averaged
            n_fft: FFT length (default 4*T for finer frequency resolution via zero-padding)
        """
        if X.ndim == 2:
            X_agg = X.mean(axis=1)  # average across assets
        else:
            X_agg = X
        self.X_ = X_agg
        self.T_ = len(X_agg)
        # Default: zero-pad to 4x for finer frequency resolution
        if n_fft is None:
            n_fft = self.T_ * 4
        # Detrend by removing mean (standard for FFT)
        X_detrend = X_agg - X_agg.mean()
        # Compute FFT amplitude spectrum (one-sided) with zero-padding
        fft_vals = np.fft.rfft(X_detrend, n=n_fft)
        self.amplitude_ = np.abs(fft_vals)
        self.freqs_ = np.fft.rfftfreq(n_fft, d=1.0)  # cycles per sample (month)
        self.n_fft_ = n_fft
        return self

    def _parabolic_interpolation(self, amp: np.ndarray, idx: int) -> float:
        """
        Parabolic interpolation around an FFT bin peak for sub-bin accuracy.

        Given amplitudes at idx-1, idx, idx+1, fit a parabola and return the
        sub-bin offset (delta in bin units).
        """
        if idx <= 0 or idx >= len(amp) - 1:
            return 0.0
        alpha = np.log(amp[idx - 1] + 1e-12)
        beta = np.log(amp[idx] + 1e-12)
        gamma = np.log(amp[idx + 1] + 1e-12)
        # Parabolic peak offset: 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
        denom = alpha - 2 * beta + gamma
        if abs(denom) < 1e-12:
            return 0.0
        delta = 0.5 * (alpha - gamma) / denom
        # Clamp to [-0.5, 0.5]
        return float(np.clip(delta, -0.5, 0.5))

    def discover(self) -> np.ndarray:
        """
        Discover top-k periods from the amplitude spectrum within [min_period, max_period].

        Uses parabolic interpolation around top-k peaks for sub-bin accuracy
        (mitigates the FFT bin discretization problem).

        Returns:
            periods: np.ndarray of shape (top_k,) — period lengths in months,
                     sorted by amplitude (largest first).
        """
        if not hasattr(self, "amplitude_"):
            raise RuntimeError("Call fit() before discover().")

        amp = self.amplitude_.copy()
        # Zero out DC component (freq=0)
        amp[0] = 0
        # Apply period bounds
        valid = np.ones(len(amp), dtype=bool)
        for i, f in enumerate(self.freqs_):
            if f == 0:
                valid[i] = False
                continue
            p = 1.0 / f  # period in samples
            if self.min_period is not None and p < self.min_period:
                valid[i] = False
            if self.max_period is not None and p > self.max_period:
                valid[i] = False
        amp = amp * valid

        # Pick top-k indices
        k = min(self.top_k, len(amp))
        top_idx = np.argpartition(amp, -k)[-k:]
        # Sort by amplitude descending
        top_idx = top_idx[np.argsort(amp[top_idx])[::-1]]

        # Apply parabolic interpolation for sub-bin accuracy
        refined_freqs = []
        refined_amps = []
        for idx in top_idx:
            delta = self._parabolic_interpolation(amp, idx)
            refined_freq_idx = idx + delta
            if 0 <= refined_freq_idx < len(self.freqs_):
                # Linear interpolation on frequency axis
                f_low = self.freqs_[int(np.floor(refined_freq_idx))]
                f_high = self.freqs_[int(np.ceil(refined_freq_idx))]
                frac = refined_freq_idx - np.floor(refined_freq_idx)
                f_refined = f_low * (1 - frac) + f_high * frac
            else:
                f_refined = self.freqs_[idx]
            # Convert freq to period in months
            if f_refined > 0:
                period = 1.0 / f_refined
                refined_freqs.append(period)
                refined_amps.append(amp[idx])

        periods = np.array(refined_freqs)
        self.last_periods_ = periods
        self.last_amplitudes_ = np.array(refined_amps)
        return periods

    def fit_discover(self, X: np.ndarray) -> np.ndarray:
        """Convenience: fit + discover."""
        self.fit(X)
        return self.discover()

    def fit_discover_multi(self, X: np.ndarray, merge_tol: float = 0.05) -> np.ndarray:
        """
        Multi-asset discovery: run on each asset independently, then merge similar periods.

        Args:
            X: np.ndarray of shape (T, N) — multi-asset time series
            merge_tol: relative tolerance for merging periods from different assets

        Returns:
            periods: np.ndarray of unique periods, sorted by aggregate amplitude descending
        """
        assert X.ndim == 2, "fit_discover_multi expects (T, N) shape"
        n_assets = X.shape[1]

        all_periods = []
        all_amplitudes = []
        for i in range(n_assets):
            self.fit(X[:, i])
            periods = self.discover()
            all_periods.extend(periods.tolist())
            all_amplitudes.extend(self.last_amplitudes_.tolist())

        all_periods = np.array(all_periods)
        all_amplitudes = np.array(all_amplitudes)

        if len(all_periods) == 0:
            return np.array([])

        # Cluster: merge periods within merge_tol relative distance
        # Greedy: sort by amplitude desc, then for each new period, merge with existing cluster if close
        sorted_idx = np.argsort(-all_amplitudes)
        sorted_periods = all_periods[sorted_idx]
        sorted_amps = all_amplitudes[sorted_idx]

        cluster_centers = []
        cluster_amps = []
        for p, a in zip(sorted_periods, sorted_amps):
            merged = False
            for i, c in enumerate(cluster_centers):
                if abs(p - c) / c < merge_tol:
                    # Amplitude-weighted average
                    total_amp = cluster_amps[i] + a
                    cluster_centers[i] = (cluster_centers[i] * cluster_amps[i] + p * a) / total_amp
                    cluster_amps[i] = total_amp
                    merged = True
                    break
            if not merged:
                cluster_centers.append(p)
                cluster_amps.append(a)

        # Sort by aggregate amplitude descending
        cluster_centers = np.array(cluster_centers)
        cluster_amps = np.array(cluster_amps)
        order = np.argsort(-cluster_amps)
        return cluster_centers[order][:self.top_k]

    def regularization_loss(
        self,
        learned_periods: np.ndarray,
        prior_periods: Optional[List[float]] = None,
        strength: float = 0.1,
    ) -> float:
        """
        Soft regularization toward domain prior periods.

        L_reg = strength * sum_i min_j |log(learned_i / prior_j)|^2

        Args:
            learned_periods: periods discovered by AMPD
            prior_periods: list of prior periods (in same unit as learned_periods)
            strength: regularization weight
        Returns:
            scalar loss
        """
        if prior_periods is None or len(prior_periods) == 0:
            return 0.0
        prior_arr = np.array(prior_periods)
        loss = 0.0
        for lp in learned_periods:
            if lp <= 0:
                continue
            log_ratios = np.log(lp / prior_arr)
            loss += np.min(log_ratios ** 2)
        return strength * loss


if __name__ == "__main__":
    # Quick smoke test
    rng = np.random.default_rng(42)
    t = np.arange(720)
    # v2.1: only resolvable periods ≤30y
    true_periods = [300, 216, 84]  # 25y, 18y, 7y
    X = sum(
        1.0 * np.sin(2 * np.pi * t / p + rng.uniform(0, 2 * np.pi))
        for p in true_periods
    )
    X += rng.normal(0, 0.1, size=720)

    ampd = AMPD(top_k=3, max_period=360)  # max 30y
    discovered = ampd.fit_discover(X)
    print(f"True periods (months):      {true_periods}")
    print(f"True periods (years):       {[p/12 for p in true_periods]}")
    print(f"Discovered periods (months): {discovered.astype(int).tolist()}")
    print(f"Discovered periods (years):  {(discovered/12).round(1).tolist()}")

    from scipy.optimize import linear_sum_assignment
    cost = np.abs(discovered[:, None] / np.array(true_periods)[None, :] - 1.0)
    row_idx, col_idx = linear_sum_assignment(cost)
    matched = [(discovered[r], true_periods[c], cost[r, c]) for r, c in zip(row_idx, col_idx)]
    print(f"\nOptimal matching (discovered -> true):")
    for d, t_true, c in matched:
        print(f"  {d:.1f}mo ({d/12:.2f}y) -> {t_true}mo ({t_true/12:.2f}y), rel_err={c*100:.1f}%")