"""
Synthetic data generator for Multi-Period Fourier Representation Learning.

Generates multi-asset time series as superposition of known sinusoidal periods
plus noise. Used as ground-truth sanity check for period recovery.

Reference: proposal-v2.md §6.5
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional


def generate_synthetic(
    true_periods_months: List[int] = [720, 300, 216, 84],  # 60y, 25y, 18y, 7y in months
    n_assets: int = 4,
    T: int = 720,
    noise_std: float = 0.1,
    seed: int = 42,
    n_periods_per_asset: Optional[int] = None,
    amplitude_range: tuple = (0.5, 1.5),
    phase_range: tuple = (0, 2 * np.pi),
) -> tuple:
    """
    Generate multi-asset synthetic time series with known multi-period structure.

    X_i(t) = sum_k a_{i,k} * sin(2*pi*t / T_k + phi_{i,k}) + epsilon_i(t)

    Args:
        true_periods_months: list of period lengths in months (e.g., 720=60y)
        n_assets: number of assets (channels)
        T: number of time steps (months). T=720 = 60 years
        noise_std: standard deviation of Gaussian noise
        seed: random seed
        n_periods_per_asset: if set, each asset uses only this many top periods
                              (simulates real data where some assets don't show all cycles)
        amplitude_range: uniform range for amplitude coefficients
        phase_range: uniform range for phase offsets

    Returns:
        X: np.ndarray of shape (T, n_assets)
        metadata: dict with true_periods, true_amplitudes, true_phases
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)  # monthly time index

    # Generate per-asset amplitudes and phases
    K = len(true_periods_months)
    amplitudes = rng.uniform(amplitude_range[0], amplitude_range[1], size=(n_assets, K))
    phases = rng.uniform(phase_range[0], phase_range[1], size=(n_assets, K))

    # Optionally mask some periods per asset (zero amplitude)
    if n_periods_per_asset is not None and n_periods_per_asset < K:
        # Each asset keeps n_periods_per_asset random periods, zeros others
        mask = np.zeros((n_assets, K), dtype=bool)
        for i in range(n_assets):
            keep = rng.choice(K, size=n_periods_per_asset, replace=False)
            mask[i, keep] = True
        amplitudes = amplitudes * mask

    # Construct the signal
    X = np.zeros((T, n_assets))
    for i in range(n_assets):
        for k in range(K):
            if amplitudes[i, k] > 0:
                X[:, i] += amplitudes[i, k] * np.sin(
                    2 * np.pi * t / true_periods_months[k] + phases[i, k]
                )

    # Add noise
    noise = rng.normal(0, noise_std, size=(T, n_assets))
    X += noise

    metadata = {
        "true_periods_months": true_periods_months,
        "true_periods_years": [p / 12 for p in true_periods_months],
        "amplitudes": amplitudes,  # (n_assets, K)
        "phases": phases,  # (n_assets, K)
        "noise_std": noise_std,
        "T": T,
        "n_assets": n_assets,
        "seed": seed,
    }
    return X, metadata


def save_synthetic_csv(X: np.ndarray, metadata: dict, out_dir: Path, name: str = "synthetic"):
    """Save synthetic data + metadata to CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save data: columns are asset_0, asset_1, ...
    cols = [f"asset_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df.index.name = "t"
    data_path = out_dir / f"{name}.csv"
    df.to_csv(data_path)

    # Save metadata
    meta_path = out_dir / f"{name}_meta.json"
    import json
    meta_save = {
        "true_periods_months": metadata["true_periods_months"],
        "true_periods_years": metadata["true_periods_years"],
        "noise_std": metadata["noise_std"],
        "T": metadata["T"],
        "n_assets": metadata["n_assets"],
        "seed": metadata["seed"],
        "amplitudes_per_asset": metadata["amplitudes"].tolist(),
        "phases_per_asset": metadata["phases"].tolist(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta_save, f, indent=2)

    print(f"Saved {data_path} and {meta_path}")
    return data_path, meta_path


def generate_negative_control(
    T: int = 720, n_assets: int = 4, noise_std: float = 0.1, seed: int = 123
):
    """
    Negative control: deliberately wrong periods (close to plausible but off).
    Used to test if theory prior anchor actually helps.
    """
    wrong_periods = [13 * 12, 17 * 12, 22 * 12, 33 * 12]  # 13y, 17y, 22y, 33y
    return generate_synthetic(
        true_periods_months=wrong_periods,
        n_assets=n_assets,
        T=T,
        noise_std=noise_std,
        seed=seed,
    )


if __name__ == "__main__":
    # Quick sanity print
    X, meta = generate_synthetic()
    print(f"X shape: {X.shape} (T={meta['T']}, n_assets={meta['n_assets']})")
    print(f"True periods (months): {meta['true_periods_months']}")
    print(f"True periods (years):  {meta['true_periods_years']}")
    print(f"Amplitudes per asset:\n{meta['amplitudes']}")
    print(f"X stats: mean={X.mean():.3f}, std={X.std():.3f}, min={X.min():.3f}, max={X.max():.3f}")