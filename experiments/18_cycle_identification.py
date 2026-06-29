"""
Experiment 18: Cycle identification via WPN's learnable period anchors.

Trains WPN_full on the FULL S&P 500 monthly log-return history (1871-2023, 1832 obs).
Logs learned period anchors + attention weights during training. Compares converged anchors
to AMPD ground truth (4y Juglar, 20y Kuznets, 25y Kondratiev).

Hypothesis: WPN's 16 learnable period anchors (4 scales x 4 anchors) will converge to a small
set of values near classical economic cycle periods, even without being told.

Output:
- Per-epoch period anchor trajectory (4 scales x 4 anchors)
- Final anchor values with comparison to AMPD ground truth
- Attention weight heatmap over time (4 scales x time) for in-sample predictions
- Crisis-regime analysis: which periods fire during GFC (2008-09), COVID (2020), dot-com (2000-02)?
"""
import sys
import os
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.wpn import WaveletPeriodNet, gmm_nll, multi_scale_smooth

warnings.filterwarnings("ignore")

DATA_PATH = ROOT / "data" / "raw" / "panel_monthly.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

SEQ_LEN = 60
HORIZON = 12  # predict 12 months ahead — captures Kuznets/Juglar scale
HIDDEN = 32
EPOCHS = 100
LR = 1e-3
BS = 128
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def make_supervised(series, seq_len, horizon):
    X, Y = [], []
    for t in range(len(series) - seq_len - horizon + 1):
        X.append(series[t:t + seq_len])
        Y.append(series[t + seq_len:t + seq_len + horizon])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def main():
    print("=" * 70)
    print(" Experiment 18: Cycle identification via learned period anchors")
    print("=" * 70)
    print(f"[device] {DEVICE}")

    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna().values.astype(np.float32)
    print(f"[data] n={len(series)} ({panel['SP500_Return'].dropna().index[0].date()} to "
          f"{panel['SP500_Return'].dropna().index[-1].date()})")

    X, Y = make_supervised(series, SEQ_LEN, HORIZON)
    print(f"[pairs] X={X.shape}, Y={Y.shape}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = WaveletPeriodNet(seq_len=SEQ_LEN, horizon=HORIZON, hidden=HIDDEN,
                              scales=[3, 6, 12, 24], n_anchors=4, n_mixtures=4,
                              head="gmm", use_learnable_anchors=True).to(DEVICE)
    print(f"[model] WPN_full, params={sum(p.numel() for p in model.parameters())}")

    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    n = X.shape[0]

    # Initial period anchors (log-spaced between p_min=4 and p_max=24 per scale)
    # Each PeriodBank has 4 anchors; we have 4 PeriodBanks (one per scale)
    initial_anchors = []
    for p in model.period_banks:
        initial_anchors.append(p.period_anchors.detach().cpu().numpy().tolist())
    print(f"[init anchors] (4 scales x 4): {initial_anchors}")

    # Training loop with periodic logging
    anchor_history = []  # list of dict per checkpoint
    log_epochs = list(range(0, EPOCHS + 1, 10)) + [EPOCHS - 1]
    log_epochs = sorted(set(log_epochs))

    print(f"[training] {EPOCHS} epochs, logging at epochs {log_epochs}")
    t_start = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(n)
        losses = []
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            opt.zero_grad()
            mu, sigma, pi = model(X_t[idx])
            loss = gmm_nll(Y_t[idx], mu, sigma, pi)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        if ep in log_epochs:
            anchors_now = [p.period_anchors.detach().cpu().numpy().tolist()
                            for p in model.period_banks]
            anchor_history.append({
                "epoch": ep,
                "anchors": anchors_now,
                "train_loss": float(np.mean(losses)),
                "wall_s": time.time() - t_start,
            })
            print(f"  ep {ep:3d}: loss={np.mean(losses):.4f}, anchors[0][0]={anchors_now[0][0]:.2f}, "
                  f"anchors[3][3]={anchors_now[3][3]:.2f}, wall={time.time()-t_start:.1f}s")

    # Final anchor values
    final_anchors = [p.period_anchors.detach().cpu().numpy().tolist()
                     for p in model.period_banks]

    # AMPD ground truth on the same series (for comparison)
    from src.models.ampd import AMPD
    amp = AMPD(top_k=8, max_period=360, min_period=4)
    ampd_periods_months = amp.fit_discover(series)
    ampd_periods_years = sorted([p / 12 for p in ampd_periods_months])
    print(f"\n[AMPD ground truth on full S&P history]:")
    print(f"  Periods (years): {[round(p, 2) for p in ampd_periods_years]}")

    # Compute attention weights for the FULL series (in-sample analysis)
    print(f"\n[attention analysis] Computing per-scale attention weights over {n} windows...")
    model.eval()

    # We need to extract attention weights — modify forward to expose them
    # Easier: replicate the PeriodBank forward manually
    attn_per_scale = []  # list of (n_samples, 4) arrays
    gate_per_scale = []
    with torch.no_grad():
        for idx_start in range(0, n, 256):
            batch_X = X_t[idx_start:idx_start + 256]
            B = batch_X.shape[0]
            scale_attns = []
            scale_gates = []
            for s_idx, s in enumerate(model.scales):
                # Multi-scale smooth
                x_t = batch_X.transpose(1, 2)  # (B, 1, T)
                x_smooth = F.avg_pool1d(x_t, kernel_size=s, stride=1, padding=s // 2)
                x = x_smooth.transpose(1, 2)  # (B, T, 1)
                # PeriodBank forward (no norm_in here, doesn't matter for attention)
                pb = model.period_banks[s_idx]
                tokens = []
                for k in range(pb.n_anchors):
                    p = pb.period_anchors[k]
                    p_int = max(int(round(p.item())), 4)
                    NP = (SEQ_LEN + p_int - 1) // p_int
                    pad = NP * p_int - SEQ_LEN
                    x_p = F.pad(x, (0, 0, 0, pad), mode="replicate") if pad > 0 else x
                    x_2d = x_p.reshape(B, NP, p_int, 1).permute(0, 3, 1, 2).contiguous()
                    h = F.gelu(pb.conv2d(x_2d))
                    token = h.mean(dim=(2, 3))
                    tokens.append(token)
                H = torch.stack(tokens, dim=1)  # (B, 4, hidden)
                attn_out, attn_w = pb.attn(H, H, H)
                # attn_w is (B, num_heads, 4, 4) — average over heads
                avg_attn = attn_w.mean(dim=1)  # (B, 4, 4)
                # Attention weight on the diagonal ≈ self-attention score per anchor
                diag_attn = avg_attn.diagonal(dim1=1, dim2=2)  # (B, 4)
                scale_attns.append(diag_attn.cpu().numpy())
                H_norm = pb.norm_attn(H + attn_out)
                gates = F.softmax(pb.gate(H_norm.mean(dim=1)), dim=-1)  # (B, 4)
                scale_gates.append(gates.cpu().numpy())
            attn_per_scale.append(np.concatenate(scale_attns, axis=1))  # (B, 16)
            gate_per_scale.append(np.concatenate(scale_gates, axis=1))

    attn_all = np.concatenate(attn_per_scale, axis=0)  # (n, 16)
    gate_all = np.concatenate(gate_per_scale, axis=0)  # (n, 16)
    print(f"  attention weights shape: {attn_all.shape}")
    print(f"  gate weights shape: {gate_all.shape}")

    # Save
    out = {
        "config": {
            "seq_len": SEQ_LEN, "horizon": HORIZON, "hidden": HIDDEN,
            "epochs": EPOCHS, "lr": LR, "bs": BS, "seed": SEED,
            "device": DEVICE, "scales": model.scales,
            "wall_clock_s": time.time() - t_start,
        },
        "anchor_history": anchor_history,
        "final_anchors": final_anchors,
        "ampd_ground_truth_years": [round(p, 2) for p in ampd_periods_years],
        "ampd_ground_truth_months": [round(p, 1) for p in sorted(ampd_periods_months)],
    }
    out_path = RESULTS_DIR / "18_cycle_identification.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {out_path}")

    # Save attention + gates as npz for plotting
    np.savez(RESULTS_DIR / "18_cycle_attention.npz",
             attn=attn_all, gate=gate_all,
             dates=np.array([panel.index[i + SEQ_LEN] for i in range(n)]).astype(str))

    # Compute correlation between gate weights and absolute returns (crisis indicator)
    dates_idx = [panel.index[i + SEQ_LEN] for i in range(n)]
    returns_window = series[SEQ_LEN:SEQ_LEN + n]
    abs_returns = np.abs(returns_window)
    is_crisis = abs_returns > 2 * np.std(abs_returns)  # > 2 stddev
    crisis_dates = [dates_idx[i] for i in range(n) if is_crisis[i]]

    # Print learned anchors vs AMPD ground truth
    print("\n" + "=" * 70)
    print(" Learned Period Anchors (months) vs AMPD Ground Truth")
    print("=" * 70)
    print(f"  AMPD top-8 ground truth (months): {[round(p, 1) for p in sorted(ampd_periods_months)]}")
    print()
    for s_idx, s in enumerate(model.scales):
        anchors_months = final_anchors[s_idx]
        anchors_years = [a / 12 for a in anchors_months]
        print(f"  Scale s={s}: anchors={[round(a, 1) for a in anchors_months]} months "
              f"= {[round(a, 2) for a in anchors_years]} years")
        # Find nearest AMPD ground truth
        nearest = []
        for a in anchors_months:
            diffs = [abs(a - gt) for gt in ampd_periods_months]
            nearest_gt = ampd_periods_months[np.argmin(diffs)]
            nearest.append(f"{nearest_gt:.0f}mo({abs(a-nearest_gt):.0f}mo err)")
        print(f"    nearest AMPD: {nearest}")

    # Crisis analysis
    print("\n" + "=" * 70)
    print(" Crisis regime analysis: gate weight patterns at high-volatility windows")
    print("=" * 70)
    if is_crisis.sum() > 0:
        print(f"  Detected {is_crisis.sum()} crisis windows (|return| > 2 stddev)")
        crisis_idx = np.where(is_crisis)[0]
        # Top 5 highest-volatility windows
        top_crises = crisis_idx[np.argsort(-abs_returns[crisis_idx])][:5]
        for ci in top_crises:
            d = dates_idx[ci]
            print(f"\n  {d.date()}: |return|={abs_returns[ci]:.4f}")
            print(f"    Gate weights (4 scales x 4 anchors):")
            for s_idx in range(4):
                gw = gate_all[ci, s_idx*4:(s_idx+1)*4]
                anchors = final_anchors[s_idx]
                # Find dominant anchor
                dom_idx = np.argmax(gw)
                dom_period = anchors[dom_idx]
                print(f"      s={model.scales[s_idx]:2d}: " + " ".join(
                    f"{gw[j]:.3f}@{anchors[j]:.0f}mo" for j in range(4)
                ) + f"  → dominant: {dom_period:.0f}mo ({dom_period/12:.1f}y)")


if __name__ == "__main__":
    main()
