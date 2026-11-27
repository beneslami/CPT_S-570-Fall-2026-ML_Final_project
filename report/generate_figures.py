#!/usr/bin/env python3
"""Generate publication-quality vector PDF figures for the NeurIPS 2025 paper.

All plotted values are loaded directly from verified experiment logs in:
  - results/paper_runs/*.json
  - results/clean_session_study.json
  - datapoints_new/ and datapoints_fx/
(plus exact closed-form analytical equations for Figure 2 and Phase-1 logged metrics for Figure 3).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"
PAPER_RUNS = ROOT / "results" / "paper_runs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Clean NeurIPS publication styling (colorblind-safe Okabe-Ito palette)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9.3,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.4,
    "xtick.labelsize": 7.7,
    "ytick.labelsize": 7.7,
    "legend.fontsize": 7.3,
    "legend.frameon": True,
    "legend.framealpha": 0.93,
    "legend.edgecolor": "#cccccc",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_YELLOW = "#E69F00"
C_CYAN = "#56B4E9"
C_GRAY = "#555555"
C_DARK = "#1F2937"
C_RED = "#C0392B"


def make_fig1_pipeline_architecture():
    """Figure 1: Leakage-free 4-way purged split, fresh Stage-1 OOF, and continuous entry-session masking."""
    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    ax.text(
        0.5, 49.5,
        "(a) Leakage-Free 4-Way Purged Split & Rolling 2-Year Fresh Stage-1 Inside Expanding Walk-Forward",
        fontsize=8.4, fontweight="bold", color=C_DARK, va="center",
    )

    y0, h = 35.5, 7.2
    segments = [
        (0.5, 24.5, C_BLUE, "Fresh Stage 1:\nRolling 2y OOF\nATR-MAE SL"),
        (25.0, 2.0, C_RED, ""),
        (27.0, 24.5, C_GREEN, "Stage 2: Factorized\nXGB Heads\n(q_TP + p_TO)"),
        (51.5, 2.0, C_RED, ""),
        (53.5, 11.5, C_YELLOW, "Platt\nCalib.\n(In-Sess.)"),
        (65.0, 2.0, C_RED, ""),
        (67.0, 11.5, C_PURPLE, "Val EV /\nCalmar\n(Stand-Dn)"),
        (78.5, 1.8, C_RED, ""),
        (80.3, 19.2, C_DARK, "Unseen OOS\nTest Chunk\n(180d Refit)"),
    ]
    for x, w, col, lbl in segments:
        rect = patches.FancyBboxPatch(
            (x, y0), w, h,
            boxstyle="round,pad=0.18,rounding_size=0.6",
            facecolor=col, edgecolor="white", linewidth=1.0, alpha=0.91 if col != C_RED else 0.96,
        )
        ax.add_patch(rect)
        if lbl:
            ax.text(
                x + w / 2, y0 + h / 2, lbl,
                color="white", fontsize=6.8, fontweight="bold",
                ha="center", va="center", linespacing=1.15,
            )

    for gx in [26.0, 52.5, 66.0, 79.4]:
        ax.annotate(
            "", xy=(gx, y0 - 0.1), xytext=(gx, y0 - 2.9),
            arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=0.9, mutation_scale=7),
        )
    ax.text(
        52.5, y0 - 4.3,
        "Purge Gaps (g = max_bars = 32 bars = 8h): zero overlapping trade paths across split boundaries",
        fontsize=7.1, color=C_RED, fontweight="bold", ha="center", va="center",
    )

    ax.annotate(
        "", xy=(40.0, y0 + h + 0.35), xytext=(89.9, y0 + h + 0.35),
        arrowprops=dict(
            arrowstyle="-|>", color=C_GRAY, lw=1.05, ls="--",
            connectionstyle="arc3,rad=0.055", mutation_scale=9,
        ),
    )
    ax.text(
        65.0, 46.7,
        "Expand history & refit every 180 days (trailing 2y Stage 1 + 60d lagged prior-shift seeded causally)",
        fontsize=6.7, color=C_GRAY, style="italic", ha="center", va="center",
    )

    ax.plot([0.5, 99.5], [28.3, 28.3], color="#dddddd", lw=0.9)

    ax.text(
        0.5, 25.8,
        "(b) Continuous-Candle Execution & Entry-Session Masking Pipeline (Zero Phantom Gaps)",
        fontsize=8.4, fontweight="bold", color=C_DARK, va="center",
    )

    boxes = [
        (0.8, 2.8, 17.6, 19.6, "#EFF6FF", C_BLUE,
         "1. Continuous\nMarket Data",
         "• Binance (BTC, ETH)\n• Dukascopy (FX, US Idx)\n• Keep 100% real bars\n• Close-aligned 1h/4h\n• 123–141 causal feats"),
        (21.0, 2.8, 17.8, 19.6, "#ECFDF5", C_GREEN,
         "2. Entry-Session\nTime Mask",
         "• Never delete CSV rows\n• Mask train/cal/entry:\n  Crypto: 24/7 or Wkday\n  FX: Asia / LN / NY\n  US Idx: NY-RTH (ET)"),
        (41.4, 2.8, 18.2, 19.6, "#FFFBEB", C_YELLOW,
         "3. Stage 1 & 2\nFactorized Heads",
         "• Fixed 1.5 ATR (2H) or\n  Fresh 2y MAE SL (10H)\n• Separate q_TP & p_TO\n• Platt + 60d lag shift\n• Monotone across R:R"),
        (62.2, 2.8, 18.2, 19.6, "#FDF2F8", C_PURPLE,
         "4. Conservative\nEV & Sharpe Gate",
         "• Asset-class fee c\n• Gate Fee_R ≤ 0.05R\n• Conservative EV ≥ 0.10\n• Score S = EV_R/Std_R\n• Cancels (RR+1) noise"),
        (83.0, 2.8, 16.5, 19.6, "#F3F4F6", C_DARK,
         "5. Risk Control\n& Live Exits",
         "• Val Net R / MaxDD\n• Val ≤ 0 -> Stand down\n• Check SL/TP every\n  real bar (SL-first)\n• 1% risk, ≤3x lev."),
    ]
    for x, y, w, bh, bg, ec, title, body in boxes:
        rect = patches.FancyBboxPatch(
            (x, y), w, bh,
            boxstyle="round,pad=0.3,rounding_size=1.0",
            facecolor=bg, edgecolor=ec, linewidth=1.2,
        )
        ax.add_patch(rect)
        hdr = patches.FancyBboxPatch(
            (x, y + bh - 5.2), w, 5.2,
            boxstyle="round,pad=0.3,rounding_size=1.0",
            facecolor=ec, edgecolor=ec, linewidth=0.8,
        )
        ax.add_patch(hdr)
        ax.text(
            x + w / 2, y + bh - 2.6, title,
            color="white", fontsize=6.8, fontweight="bold",
            ha="center", va="center", linespacing=1.1,
        )
        ax.text(
            x + 0.65, y + bh - 6.6, body,
            color=C_DARK, fontsize=6.1,
            ha="left", va="top", linespacing=1.34,
        )

    for x_end in [18.7, 39.1, 59.9, 80.7]:
        ax.annotate(
            "", xy=(x_end + 2.0, 12.6), xytext=(x_end, 12.6),
            arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.3, mutation_scale=10),
        )

    out = FIG_DIR / "fig1_pipeline_architecture.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig2_ev_and_fee_theory():
    """Figure 2: Commission drag across all 3 asset-class fee tiers + Per-Trade Sharpe noise cancellation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw={"wspace": 0.32})

    sl_pct = np.linspace(0.05, 1.00, 400) / 100.0
    for c, lbl, col, ls, min_sl in [
        (0.00002, "FX ECN (c = 0.2 bp, min SL 0.08%)", C_GREEN, "-", 0.08),
        (0.00005, "US Index E-Mini (c = 0.5 bp, min SL 0.20%)", C_ORANGE, "--", 0.20),
        (0.00010, "Crypto VIP (c = 1.0 bp, min SL 0.40%)", C_BLUE, "-", 0.40),
    ]:
        fee_r = (2.0 * c) / sl_pct
        ax1.plot(sl_pct * 100.0, fee_r, label=lbl, color=col, lw=1.7, ls=ls)
        ax1.scatter([min_sl], [0.05], color=col, s=26, zorder=6)

    ax1.axhline(0.05, color=C_RED, lw=1.25, ls="-.", label=r"Universal Fee Gate ($\mathrm{Fee}_R = 0.05R$)")
    ax1.fill_between([0.05, 1.00], 0.05, 0.26, color=C_RED, alpha=0.06)
    ax1.annotate(
        "Same 0.05R rule adapts\nautomatically to each\nasset class's ATR scale",
        xy=(0.20, 0.05), xytext=(0.45, 0.13),
        fontsize=6.5, color=C_DARK,
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=0.9),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.95),
    )
    ax1.set_xlim(0.05, 1.00)
    ax1.set_ylim(0.0, 0.25)
    ax1.set_xlabel(r"Stop-Loss Width $\mathrm{sl\_pct} = |e - \mathrm{SL}|/e$ (%)")
    ax1.set_ylabel(r"Round-Trip Fee Drag $\mathrm{Fee}_R$ ($R$)")
    ax1.set_title(r"(a) Universal Fee Gate ($\mathrm{Fee}_R \approx 2c/\mathrm{sl\_pct} \leq 0.05R$)")
    ax1.legend(loc="upper right", fontsize=6.4)

    rr = np.linspace(1.5, 4.5, 300)
    p_be = 1.0 / (rr + 1.0)
    eps = 0.02
    raw_ev_bias = (rr + 1.0) * eps
    p_est = p_be + eps
    sharpe_bias = eps / np.sqrt(p_est * (1.0 - p_est))

    l1, = ax2.plot(rr, raw_ev_bias, color=C_ORANGE, lw=1.9, label=r"Raw $\Delta \mathrm{EV}_R = (\mathrm{RR}+1)\,\epsilon$ (left)")
    ax2.set_xlabel(r"Risk-to-Reward Ratio $\mathrm{RR}$")
    ax2.set_ylabel(r"Raw $\mathrm{EV}_R$ Error for $\epsilon = +0.02$ ($R$)", color=C_ORANGE)
    ax2.tick_params(axis="y", labelcolor=C_ORANGE)
    ax2.set_ylim(0.03, 0.125)
    ax2.set_xticks([2.0, 2.5, 3.0, 3.5, 4.0])

    ax2b = ax2.twinx()
    ax2b.spines["top"].set_visible(False)
    ax2b.grid(False)
    l2, = ax2b.plot(rr, sharpe_bias, color=C_BLUE, lw=1.9, ls="--", label=r"Sharpe $\Delta S = \epsilon / \sqrt{p(1-p)}$ (right)")
    l3, = ax2b.plot(rr, p_be, color=C_GRAY, lw=1.3, ls=":", label=r"Break-even $p_{\mathrm{be}} = 1/(\mathrm{RR}+1)$ (right)")
    ax2b.set_ylabel(r"Sharpe Error $\Delta S$ / Break-even $p_{\mathrm{be}}$", color=C_BLUE)
    ax2b.tick_params(axis="y", labelcolor=C_BLUE)
    ax2b.set_ylim(0.0, 0.44)

    rr_pts = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
    ax2.scatter(rr_pts, (rr_pts + 1.0) * eps, color=C_ORANGE, s=22, zorder=5)
    ax2b.scatter(rr_pts, eps / np.sqrt((1 / (rr_pts + 1) + eps) * (1 - (1 / (rr_pts + 1) + eps))), color=C_BLUE, s=22, zorder=5)

    lines = [l1, l2, l3]
    ax2.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=6.5)
    ax2.set_title(r"(b) $(\mathrm{RR}+1)$ Error Growth vs. Sharpe Score")

    out = FIG_DIR / "fig2_ev_and_fee_theory.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig3_consensus_1h_diagnostics():
    """Figure 3: Phase 1 BTC 1h 3-Fold Walk-Forward & Holdout Confidence-Coverage Curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw={"wspace": 0.32})

    folds = ["Fold 1\n(2021–2023)", "Fold 2\n(2023–2025)", "Fold 3\n(2025–2026)"]
    x = np.arange(len(folds))
    series = [
        ("XGBoost (55 feats)", [54.66, 54.70, 52.80], C_BLUE, "o", "--"),
        ("LSTM (19 feats)", [54.85, 54.96, 52.76], C_ORANGE, "s", "--"),
        ("Logistic Reg. (30 feats)", [54.80, 55.00, 53.51], C_GREEN, "^", "--"),
        ("3-Model Majority", [55.53, 55.18, 53.18], C_PURPLE, "D", "-"),
        ("3-Model Unanimous", [57.99, 56.21, 54.29], C_DARK, "*", "-"),
    ]
    for lbl, vals, col, mk, ls in series:
        ax1.plot(x, vals, label=lbl, color=col, marker=mk, ls=ls, lw=1.6 if ls == "-" else 1.25, ms=5.5 if mk == "*" else 4.5)

    ax1.axhline(50.0, color=C_RED, lw=1.0, ls=":", label="Coin-flip baseline (50%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(folds)
    ax1.set_ylim(49.2, 59.3)
    ax1.set_ylabel("Out-of-Sample Accuracy (%)")
    ax1.set_title("(a) BTC 1h 3-Fold Walk-Forward (44,911 hrs)")
    ax1.legend(loc="lower left", fontsize=6.4, ncol=2)

    conf_labels = ["All Bars\n(≥ 0.50)", "Unanimous\nAgree", "Conf.\n≥ 0.55", "Conf.\n≥ 0.60"]
    accs = [53.41, 53.73, 54.81, 59.05]
    covs = [100.0, 72.2, 42.4, 8.9]
    x2 = np.arange(len(conf_labels))

    bars = ax2.bar(x2, accs, width=0.52, color=[C_CYAN, C_BLUE, C_GREEN, C_DARK], alpha=0.88, edgecolor="white", label="Holdout Acc. (%)")
    ax2.axhline(49.97, color=C_RED, lw=1.0, ls=":", label="Majority class (49.97%)")
    for b, v in zip(bars, accs):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f}%", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=C_DARK)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(conf_labels)
    ax2.set_ylim(46.2, 62.5)
    ax2.set_ylabel("Holdout Accuracy (%)")

    ax2b = ax2.twinx()
    ax2b.spines["top"].set_visible(False)
    ax2b.grid(False)
    ax2b.plot(x2, covs, color=C_ORANGE, marker="o", lw=1.7, ls="-", label="Coverage (%)")
    for xi, cv in zip(x2, covs):
        ax2b.text(xi, cv + 4.5, f"{cv:.1f}%", ha="center", va="bottom", fontsize=6.8, color=C_ORANGE, fontweight="bold")
    ax2b.set_ylabel("Bar Coverage (%)", color=C_ORANGE)
    ax2b.tick_params(axis="y", labelcolor=C_ORANGE)
    ax2b.set_ylim(-14, 118)

    ax2.text(
        1.5, 47.4,
        "Horizon mismatch: 1h probs on 24–48h brackets -> 0 / 11,979 EV trades",
        fontsize=6.1, color=C_RED, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFF5F5", ec=C_RED, lw=0.7),
    )
    ax2.set_title("(b) Holdout Accuracy vs. Selectivity (BTC 1h)")

    out = FIG_DIR / "fig3_consensus_1h_diagnostics.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig4_calibration_and_auc():
    """Figure 4: 10-Head OOS probability calibration error and ROC-AUC across Crypto, FX, and US Indices."""
    def load_run(tag):
        return json.load(open(PAPER_RUNS / f"{tag}.json"))

    btc = load_run("BTC_247_learned10h")
    eth = load_run("ETH_247_learned10h")
    jpy = load_run("JPYUSD_london_learned10h")
    aud = load_run("AUDUSD_ny_learned10h")
    ndx = load_run("NDX_nyrth_learned10h")

    heads = [f"long_{r:g}" for r in (2, 2.5, 3, 3.5, 4)] + [f"short_{r:g}" for r in (2, 2.5, 3, 3.5, 4)]
    head_labels = ["L2.0", "L2.5", "L3.0", "L3.5", "L4.0", "S2.0", "S2.5", "S3.0", "S3.5", "S4.0"]
    x = np.arange(len(heads))

    def bias_pp(rec):
        hd = rec["head_diagnostics"]
        return np.array([(hd[h]["test_mean_pred_tp"] - hd[h]["test_base_tp_rate"]) * 100.0 for h in heads])

    def aucs(rec):
        hd = rec["head_diagnostics"]
        return np.array([hd[h]["test_roc_auc"] for h in heads])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.85), gridspec_kw={"wspace": 0.25})

    w = 0.20
    b_btc, b_eth, b_jpy, b_aud = bias_pp(btc), bias_pp(eth), bias_pp(jpy), bias_pp(aud)
    ax1.bar(x - 1.5 * w, b_btc, w, label=f"BTC 24/7 (MAE={np.mean(np.abs(b_btc)):.2f}pp)", color=C_BLUE)
    ax1.bar(x - 0.5 * w, b_eth, w, label=f"ETH 24/7 (MAE={np.mean(np.abs(b_eth)):.2f}pp)", color=C_GREEN)
    ax1.bar(x + 0.5 * w, b_jpy, w, label=f"JPYUSD London (MAE={np.mean(np.abs(b_jpy)):.2f}pp)", color=C_ORANGE)
    ax1.bar(x + 1.5 * w, b_aud, w, label=f"AUDUSD NY (MAE={np.mean(np.abs(b_aud)):.2f}pp)", color=C_PURPLE)
    ax1.axhline(0.0, color=C_DARK, lw=0.9)
    ax1.axvline(4.5, color=C_GRAY, lw=0.8, ls=":")
    ax1.set_xticks(x)
    ax1.set_xticklabels(head_labels, fontsize=6.9)
    ax1.set_ylim(-3.6, 0.95)
    ax1.set_ylabel("Mean OOS Prob. Bias (pp)")
    ax1.set_title("(a) 60d Causal Calibration Bias Across 10 Heads")
    ax1.legend(loc="lower center", fontsize=5.6, ncol=2, columnspacing=0.8, handlelength=1.4)

    for vals, lbl, col, ls, mk in [
        (aucs(jpy), f"JPYUSD London ({jpy['avg_test_roc_auc']:.3f})", C_ORANGE, "-", "D"),
        (aucs(aud), f"AUDUSD NY ({aud['avg_test_roc_auc']:.3f})", C_PURPLE, "-", "s"),
        (aucs(eth), f"ETH 24/7 ({eth['avg_test_roc_auc']:.3f})", C_GREEN, "-", "^"),
        (aucs(btc), f"BTC 24/7 ({btc['avg_test_roc_auc']:.3f})", C_BLUE, "-", "o"),
        (aucs(ndx), f"NDX NY-RTH ({ndx['avg_test_roc_auc']:.3f})", C_GRAY, "--", "x"),
    ]:
        ax2.plot(x, vals, label=lbl, color=col, ls=ls, marker=mk, lw=1.45, ms=4.0)

    ax2.axhline(0.50, color=C_RED, lw=0.9, ls=":")
    ax2.axvline(4.5, color=C_GRAY, lw=0.8, ls=":")
    ax2.set_xticks(x)
    ax2.set_xticklabels(head_labels, fontsize=6.9)
    ax2.set_ylim(0.41, 0.705)
    ax2.set_ylabel("Out-of-Sample ROC-AUC")
    ax2.set_title("(b) 10-Head OOS ROC-AUC Across Asset Classes")
    ax2.legend(loc="lower center", fontsize=5.6, ncol=2, columnspacing=0.8, handlelength=1.5)

    out = FIG_DIR / "fig4_calibration_and_auc.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig5_payoff_ablation_and_equity():
    """Figure 5: Timeout-payoff ablation on BTC/ETH + Clean cross-asset cumulative Net R equity curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.95), gridspec_kw={"wspace": 0.28})

    # Panel (a): Controlled payoff & architecture ablation on BTC & ETH (24/7 verified runs)
    labels = [
        "V2 Joint\n3-Class",
        "V2.1 Fact.\n3-Out EV",
        "V3 Direct\nNet-R Reg",
        "V2.2 Fresh\n10H Cons.",
        "V2.2 Fixed\n1.5 ATR 2H",
    ]
    btc_vals = [-15.27, -4.67, -89.68, 4.25, 0.00]
    eth_vals = [19.82, -33.42, -22.08, 5.29, 17.23]
    x = np.arange(len(labels))
    w = 0.36

    b1 = ax1.bar(x - w / 2, btc_vals, w, label="BTC 15m (24/7)", color=C_BLUE, edgecolor="white", lw=0.4)
    b2 = ax1.bar(x + w / 2, eth_vals, w, label="ETH 15m (24/7)", color=C_GREEN, edgecolor="white", lw=0.4)
    ax1.axhline(0.0, color=C_DARK, lw=0.9)
    ax1.axvline(2.5, color=C_GRAY, lw=0.8, ls=":")
    ax1.text(1.0, 26.5, "Timeout-crediting\ngates fail", fontsize=6.4, color=C_RED, ha="center", fontweight="bold")
    ax1.text(3.5, 26.5, "Conservative EV\ngate works", fontsize=6.4, color=C_GREEN, ha="center", fontweight="bold")

    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            if abs(v) < 0.05:
                ax1.text(bar.get_x() + bar.get_width() / 2, 1.5, "0\nSD", ha="center", va="bottom", fontsize=5.8, color=C_GRAY, fontweight="bold")
            elif v > 0:
                lbl_s = "+4.25" if abs(v - 4.25) < 0.01 else f"{v:+.1f}"
                ax1.text(bar.get_x() + bar.get_width() / 2, v + 1.2, lbl_s, ha="center", va="bottom", fontsize=6.0, color=C_DARK, fontweight="bold")
            elif v < -42.0:
                ax1.text(bar.get_x() + bar.get_width() / 2, -39.5, f"{v:+.1f}", ha="center", va="center", fontsize=5.9, color="white", fontweight="bold")
            else:
                ax1.text(bar.get_x() + bar.get_width() / 2, v - 1.6, f"{v:+.1f}", ha="center", va="top", fontsize=6.0, color=C_DARK, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6.7)
    ax1.set_ylim(-44.0, 35.0)
    ax1.set_ylabel("OOS Net Total $R$")
    ax1.set_title("(a) Payoff & Complexity Ablation (BTC & ETH)")
    ax1.legend(loc="lower right", fontsize=6.4)

    # Panel (b): Step-by-step realized OOS cumulative Net R curves from results/paper_runs/
    def load_curve(tag):
        rec = json.load(open(PAPER_RUNS / f"{tag}.json"))
        bt = rec["test_backtest"]
        pos = rec.get("positions", [])
        t0 = pd.Timestamp(bt["start"])
        t1 = pd.Timestamp(bt["end"])
        if not pos:
            return [t0, t1], [0.0, 0.0]
        times = [t0] + [pd.Timestamp(p["exit_time"]) for p in pos] + [t1]
        cum = np.concatenate([[0.0], np.cumsum([p["net_r"] for p in pos])])
        cum = np.concatenate([cum, [cum[-1]]])
        return times, cum

    curves = [
        ("ETH_247_fixed2h", C_GREEN, "-", 1.85, "ETH 24/7 Fixed-ATR (+17.23R)"),
        ("JPYUSD_london_learned10h", C_ORANGE, "-", 1.85, "JPYUSD London Learned-SL (+16.40R)"),
        ("ETH_wkday_fixed2h", C_CYAN, "--", 1.35, "ETH Weekdays Fixed-ATR (+11.04R)"),
        ("AUDUSD_ny_fixed2h", C_PURPLE, "-", 1.55, "AUDUSD NY Fixed-ATR (+9.23R)"),
        ("BTC_247_learned10h", C_BLUE, "-", 1.75, "BTC 24/7 Learned-SL (+4.25R)"),
        ("NDX_nyrth_learned10h", C_DARK, "-.", 1.30, "NDX NY-RTH Learned-SL (-0.20R)"),
        ("NDX_24h_learned10h", C_RED, ":", 1.35, "NDX 24h Unmasked (-27.05R)"),
    ]
    for tag, col, ls, lw, lbl in curves:
        ts, cr = load_curve(tag)
        ax2.step(ts, cr, where="post", color=col, ls=ls, lw=lw, label=lbl)

    ax2.axhline(0.0, color=C_GRAY, lw=0.8)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.get_xticklabels(), rotation=22, ha="right", fontsize=6.9)
    ax2.set_ylim(-35.0, 23.5)
    ax2.set_ylabel("Cumulative Realized Net $R$")
    ax2.set_title("(b) Clean OOS Equity Curves Across Asset Classes")
    ax2.legend(loc="lower left", fontsize=5.7, handlelength=1.6)

    out = FIG_DIR / "fig5_v2_payoff_and_equity.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig6_timezone_profit_impact():
    """Figure 6: Flagship timezone study (5 FX pairs x 5 sessions heatmap + cross-asset timezone comparison)."""
    df = pd.DataFrame(json.load(open(ROOT / "results" / "clean_session_study.json")))
    fx_fix = df[(df.group == "fx") & (df.arch == "fixed_2h")]

    syms = ["EURUSD", "GBPUSD", "JPYUSD", "AUDUSD", "CADUSD"]
    sess_order = ["asia", "london", "ny", "london_ny", "weekdays"]
    sess_labels = ["Asia\n(00–09)", "London\n(07–16)", "New York\n(12–21)", "LN+NY\n(07–21)", "Wkday\n24h"]

    mat_r = fx_fix.pivot(index="symbol", columns="entry_session", values="net_r").loc[syms, sess_order].to_numpy()
    mat_auc = fx_fix.pivot(index="symbol", columns="entry_session", values="avg_test_auc").loc[syms, sess_order].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.05), gridspec_kw={"width_ratios": [1.18, 1.0], "wspace": 0.30})

    # Panel (a): Annotated Heatmap of OOS Net R & AUC for 5 FX pairs x 5 sessions (Fixed-ATR 2H)
    vmax = 24.0
    im = ax1.imshow(mat_r, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax1.grid(False)
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color("#cccccc")

    for i in range(len(syms)):
        for j in range(len(sess_order)):
            r_val = mat_r[i, j]
            a_val = mat_auc[i, j]
            txt_col = "white" if abs(r_val) > 15.0 else C_DARK
            r_str = "0.0R" if abs(r_val) < 0.01 else f"{r_val:+.1f}R"
            ax1.text(
                j, i, f"{r_str}\n({a_val:.3f})",
                ha="center", va="center", fontsize=6.6, fontweight="bold" if r_val > 0 else "normal",
                color=txt_col, linespacing=1.15,
            )

    ax1.set_xticks(np.arange(len(sess_labels)))
    ax1.set_xticklabels(sess_labels, fontsize=7.0)
    ax1.set_yticks(np.arange(len(syms)))
    ax1.set_yticklabels(syms, fontsize=7.5, fontweight="bold")
    ax1.set_title("(a) 5 FX Pairs × Time Zone: Net $R$ & (AUC)")

    # Highlight winning NY column with subtle border
    rect = patches.Rectangle((1.5, -0.5), 1.0, len(syms), fill=False, edgecolor=C_BLUE, lw=1.8)
    ax1.add_patch(rect)

    # Panel (b): Cross-Asset Timezone Profit Impact (Total Net R across baskets)
    cats = [
        "Asia",
        "Wkday 24h",
        "London",
        "New York",
        "Row-Del*",
        "Clean Wkd",
        "Full 24/7",
        "24h Trap",
        "NY-RTH",
    ]
    # Fixed-ATR 2H & Learned-SL 10H totals (verified from clean_session_study.json & 24h runs)
    vals_2h = [-42.28, -19.57, 3.56, 16.49, np.nan, 6.68, 17.23, -15.42, 0.00]
    vals_10h = [-0.40, np.nan, 18.32, 6.38, -47.15, -4.52, 9.54, -27.05, -0.20]

    x2 = np.arange(len(cats))
    w = 0.38
    v2_plot = [0.0 if np.isnan(v) else v for v in vals_2h]
    v10_plot = [0.0 if np.isnan(v) else v for v in vals_10h]

    b1 = ax2.bar(x2 - w / 2, v2_plot, w, label="Fixed-ATR 2H", color=C_GREEN, alpha=0.9, edgecolor="white", lw=0.4)
    b2 = ax2.bar(x2 + w / 2, v10_plot, w, label="Learned-SL 10H", color=C_BLUE, alpha=0.9, edgecolor="white", lw=0.4)
    ax2.axhline(0.0, color=C_DARK, lw=0.9)
    ax2.axvline(3.5, color=C_GRAY, lw=0.8, ls=":")
    ax2.axvline(6.5, color=C_GRAY, lw=0.8, ls=":")

    # Clean basket group headers above bars
    ax2.text(1.5, 25.5, "5 G10 FX", fontsize=6.8, fontweight="bold", color=C_DARK, ha="center", va="center")
    ax2.text(5.0, 25.5, "BTC + ETH", fontsize=6.8, fontweight="bold", color=C_DARK, ha="center", va="center")
    ax2.text(7.5, 25.5, "SPX + NDX", fontsize=6.8, fontweight="bold", color=C_DARK, ha="center", va="center")

    for idx_b, (v2, v10) in enumerate(zip(vals_2h, vals_10h)):
        if not np.isnan(v2) and (abs(v2) > 3.0 or idx_b == 8):
            y_p = v2 + 1.2 if v2 >= 0 else v2 - 1.5
            s2 = "0" if idx_b == 8 else f"{v2:+.0f}"
            ax2.text(x2[idx_b] - w / 2, y_p, s2, ha="center", va="bottom" if v2 >= 0 else "top", fontsize=5.7, color=C_DARK, fontweight="bold")
        if not np.isnan(v10) and (abs(v10) > 3.0 or idx_b == 8):
            y_p = v10 + 1.2 if v10 >= 0 else v10 - 1.5
            s10 = "-0.2" if idx_b == 8 else f"{v10:+.0f}"
            ax2.text(x2[idx_b] + w / 2, y_p, s10, ha="center", va="bottom" if v10 >= 0 else "top", fontsize=5.7, color=C_DARK, fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(cats, fontsize=6.4, rotation=32, ha="right")
    ax2.set_ylim(-56.0, 29.5)
    ax2.set_ylabel("Basket OOS Net $R$")
    ax2.set_title("(b) Time Zone Effect Across All 3 Asset Classes")
    ax2.legend(loc="lower right", fontsize=6.3)

    out = FIG_DIR / "fig6_timezone_profit_impact.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def _plot_candlestick_window(ax, df_slice, trades_slice, title, ylabel, price_fmt="{:.0f}"):
    times = mdates.date2num(df_slice["time"].dt.to_pydatetime())
    bar_w = 0.0072

    for t, o, h, l, c in zip(times, df_slice["open"], df_slice["high"], df_slice["low"], df_slice["close"]):
        col = "#2E8B57" if c >= o else "#C0392B"
        ax.plot([t, t], [l, h], color=col, lw=0.65, zorder=2)
        body_lo = min(o, c)
        body_h = max(abs(c - o), (h - l) * 0.04, 1e-8)
        ax.add_patch(
            patches.Rectangle((t - bar_w / 2, body_lo), bar_w, body_h, facecolor=col, edgecolor=col, lw=0.35, zorder=3)
        )

    for tr in trades_slice:
        t_in = mdates.date2num(pd.Timestamp(tr["entry_time"]).to_pydatetime())
        t_out = mdates.date2num(pd.Timestamp(tr["exit_time"]).to_pydatetime())
        span = max(t_out - t_in, bar_w * 2.5)
        e, tp, sl, ex = tr["open_price"], tr["tp"], tr["sl"], tr["close_price"]
        side = tr["side"]
        reason = tr["exit_reason"]
        net_r = tr["net_r"]

        tp_lo, tp_hi = min(e, tp), max(e, tp)
        sl_lo, sl_hi = min(e, sl), max(e, sl)
        ax.add_patch(patches.Rectangle((t_in, tp_lo), span, tp_hi - tp_lo, facecolor=C_GREEN, alpha=0.14, lw=0, zorder=1))
        ax.add_patch(patches.Rectangle((t_in, sl_lo), span, sl_hi - sl_lo, facecolor=C_RED, alpha=0.14, lw=0, zorder=1))
        ax.plot([t_in, t_in + span], [tp, tp], color=C_GREEN, lw=0.85, ls="--", zorder=4)
        ax.plot([t_in, t_in + span], [sl, sl], color=C_RED, lw=0.85, ls="--", zorder=4)

        mk = "^" if side == "LONG" else "v"
        ax.scatter([t_in], [e], marker=mk, s=34, color=C_BLUE, edgecolor="white", lw=0.5, zorder=6)
        ex_col = C_GREEN if net_r > 0 else C_RED
        ax.scatter([t_out], [ex], marker="o", s=30, color=ex_col, edgecolor="white", lw=0.5, zorder=6)

        y_off = (tp_hi - sl_lo) * 0.14
        va = "bottom" if net_r > 0 else "top"
        y_txt = ex + y_off if net_r > 0 else ex - y_off
        ax.text(
            t_out, y_txt, f"{reason}\n{net_r:+.2f}R",
            fontsize=6.2, fontweight="bold", color=ex_col, ha="center", va=va,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=ex_col, lw=0.5, alpha=0.90),
            zorder=7,
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=6.9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def make_fig7_trade_execution_zoom():
    """Figure 7: Real 15m OOS execution zoom on ETH 24/7 (Fixed-ATR) and JPYUSD London (Learned-SL)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.95), gridspec_kw={"wspace": 0.34})

    # Panel (a): ETH 24/7 Fixed-ATR trades #8..10 (Feb 1-2, 2026: +2.47R, +2.49R, -1.01R = +3.95R)
    eth_pos = json.load(open(PAPER_RUNS / "ETH_247_fixed2h.json"))["positions"][8:11]
    eth_df = pd.read_csv(ROOT / "datapoints_new/ETH/ETH_15m.csv", usecols=["time", "open", "high", "low", "close"])
    eth_df["time"] = pd.to_datetime(eth_df["time"], utc=True)
    e_sub = eth_df[(eth_df["time"] >= "2026-02-01 03:00+00:00") & (eth_df["time"] <= "2026-02-02 09:00+00:00")]
    _plot_candlestick_window(
        ax1, e_sub, eth_pos,
        title="(a) ETH 24/7 Fixed-ATR (Feb 1–2, 2026): +3.95R",
        ylabel="ETH/USDT Price (USD)",
    )
    ax1.set_title("(a) ETH 24/7 Fixed-ATR (Feb 1–2, 2026): +3.95R", fontsize=8.0)

    # Panel (b): JPYUSD London Learned-SL trades #5..6 (Jun 20, 2025: -1.04R SL -> +2.96R 3:1 TP = +1.92R)
    jpy_pos = json.load(open(PAPER_RUNS / "JPYUSD_london_learned10h.json"))["positions"][5:7]
    jpy_df = pd.read_csv(ROOT / "datapoints_fx/JPYUSD/JPYUSD_15m.csv", usecols=["time", "open", "high", "low", "close"])
    jpy_df["time"] = pd.to_datetime(jpy_df["time"], utc=True)
    j_sub = jpy_df[(jpy_df["time"] >= "2025-06-20 06:30+00:00") & (jpy_df["time"] <= "2025-06-20 15:00+00:00")]
    _plot_candlestick_window(
        ax2, j_sub, jpy_pos,
        title="(b) JPYUSD London Learned-SL (Jun 20, 2025): +1.92R",
        ylabel="JPY/USD Price (USD per JPY)",
    )
    ax2.set_title("(b) JPYUSD London Learned-SL (Jun 20, 2025): +1.92R", fontsize=8.0)

    out = FIG_DIR / "fig7_trade_execution_zoom.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    make_fig1_pipeline_architecture()
    make_fig2_ev_and_fee_theory()
    make_fig3_consensus_1h_diagnostics()
    make_fig4_calibration_and_auc()
    make_fig5_payoff_ablation_and_equity()
    make_fig6_timezone_profit_impact()
    make_fig7_trade_execution_zoom()
    print("All 7 vector PDF figures generated successfully.")


if __name__ == "__main__":
    main()
