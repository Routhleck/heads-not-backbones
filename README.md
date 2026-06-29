# Heads, Not Backbones — Reproducibility Artifact

> *Heads, Not Backbones: Output Heads Dominate Architectures on Fat-Tailed Returns* (arXiv:2606.XXXXX)

This is the reproducibility artifact for the paper. Every number in the
paper can be regenerated from the committed data and scripts in this
repository. Wall-clock on a single NVIDIA RTX 4060 laptop GPU is
~22 minutes for the main 720-run protocol (12 variants × 5 folds × 4
horizons × 3 seeds); on Apple-silicon CPU the same protocol takes
~7-9 hours.

## What is in this repo

```
src/                 # model architecture code (TimesNet, DLinear, N-BEATS,
                     # iTransformer, GMM head, walk-forward protocol)
experiments/         # 57 numbered experiment scripts, one per (panel, claim)
results/             # 27 paper-canonical JSON outputs (proof of
                     # reproducibility — re-run and diff with these files)
data/raw/            # Shiller monthly log-returns + daily panel CSVs (small)
paper/               # main_acm.tex + main_acm.pdf (the paper itself)
figures/             # all 5 figures referenced by \includegraphics (PDF + PNG)
run_reproduce.sh     # one-shot driver that runs the headline protocol
```

## Quick start

```bash
# (1) Get the data (Shiller monthly + daily panel CSVs are committed;
#     only FRED pulls need network — see data/README.md for sources).

# (2) Reproduce the headline 12-variant × 5-fold × 4-horizon × 3-seed grid
python experiments/22_sota_comparison.py     # ~22 min on RTX 4060
python experiments/34_gaussian_head.py      # ~22 min on RTX 4060
python experiments/35_merge_12variants.py   # combines into results/35_combined_12variants.json

# (3) Regenerate the headline figure (and 4 supplementary figures)
python experiments/35c_make_figures.py       # figs 1, 2, 5
python experiments/35f_more_figures.py       # figs 3, 4

# (4) Verify against committed results
diff <(jq -S . results/22_sota_comparison.json) \
     <(jq -S . <your-rerun.json>)
# Numbers should match within ±0.05pp (seed-level noise on per-fold means).
```

## Mapping: paper table / figure → experiment script

| Paper asset                  | Reproduce with                       | Result file                          | Figure file                          |
|------------------------------|--------------------------------------|--------------------------------------|--------------------------------------|
| **Figure 1** (heatmap)        | 22b or 35c                            | `results/35_combined_12variants.json` | `figures/fig_12variant_heatmap.pdf`   |
| **Figure 2** (coverage 90%)   | 35c                                   | `results/35_combined_12variants.json` | `figures/fig_coverage_12variants.pdf` |
| **Figure 3** (pinball quant.) | 35f                                   | `results/35e_pinball_12variants.json` | `figures/fig_pinball_12variants.pdf`  |
| **Figure 4** (classical 3h)   | 35f                                   | `results/35d_classical_3heads.json`   | `figures/fig_classical_3heads.pdf`    |
| **Figure 5** (GMM vs Gauss)   | 35c                                   | `results/35b_regime_with_gauss.json`  | `figures/fig_gmm_vs_gauss_regime.pdf` |
| **Table 2** (12-variant CSS)  | 22 + 34 → 35                           | `results/35_combined_12variants.json` | n/a                                  |
| **Table 4 + Table 8**        | 35d                                   | `results/35d_classical_3heads.json`   | n/a                                  |
| **Table 6** (per-regime)      | 35b                                   | `results/35b_regime_with_gauss.json`  | n/a                                  |
| **Table 7** (bootstrap CI)    | 26                                    | `results/26_bootstrap_crps_ss.json`   | n/a                                  |
| **Table 7** (DM per fold)     | 35g                                   | `results/35g_dm_per_fold.json`        | n/a                                  |
| **Table 10** (DA + R²)        | 41                                    | `results/41_da_r2_summary.json`       | n/a                                  |
| §5.1 cross-asset table       | 36e (× 4) → 36f                        | `results/36f_cross_asset_summary.txt` | n/a                                  |
| §5.1 cross-asset raw         | 36e × 4 panels                        | `results/36e_daily_{SP500_Return,VIX,DGS10,EURUSD_Return}.json` | n/a |
| §5.3 tail risk               | 38                                    | `results/38_tail_risk_summary.json`   | n/a                                  |
| §5.4 economic value          | 38b                                   | `results/38b_economic_value.json`     | n/a                                  |
| §5.5 FRTB capital            | 38c                                   | `results/38c_frtb_capital.json`       | n/a                                  |
| §5.6 VaR backtest + ES       | 42                                    | `results/42_var_backtest.json`        | n/a                                  |

## Reproducibility caveats — read first

1. **Seed determinism.** `torch`, `numpy`, `random` are all seeded at the
   top of every experiment script. But PyTorch on CUDA is not bit-deterministic
   across GPU architectures even with the same seed; what is deterministic is
   that the per-fold mean CRPS-Skill-Score is within ±0.05pp of the committed
   value. We report per-cell numbers in the paper as the mean over
   `5 folds × 3 seeds = 15` observations.

2. **Walk-forward protocol.** Training windows are anchored at the earliest
   50% of the series, then 5 folds × 128-month test windows stepped by
   183 months. Re-running `experiments/22_sota_comparison.py` regenerates
   exactly the same per-fold CRPS-Skill-Scores within ±0.05pp on the
   committed seeds.

3. **Head choice matters most.** The paper's headline number (`+3.7pp`
   point→GMM gradient on backbone-mean axis) is robust to seed; the per-cell
   breakdown in Table 2 is what you should diff against, not the headline
   mean.

4. **Classical baselines (Table 4) are under GMM-on-residuals protocol.**
   `35d_classical_3heads.py` uses ARIMA/GARCH point forecasts with a
   residual GMM. A separate protocol (`37a_garch_monthly.py`) gives
   different numbers (-2.76% at h=1 for GARCH_gmm); this difference is
   discussed in §5.1 and is the reason GARCH_gmm h=6 = -50.5% in Table 4
   but Table 2 in earlier phases of this project showed a different value.

## License

MIT. See `LICENSE` for the full text.

## Citation

```
@misc{he2026heads,
  title  = {Heads, Not Backbones: Output Heads Dominate Architectures on Fat-Tailed Returns},
  author = {He, Sichao},
  year   = {2026},
  note   = {arXiv preprint}
}
```
