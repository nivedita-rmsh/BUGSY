
"""
BUGSY figure generator (updated).

Put this file in the project root (next to Assets and results) together with:

A) Measurement CSVs written by the CoverageTracker logger:
   baseline_random.csv
   baseline_wallfollower.csv
   baseline_v8.csv
   baseline_nomap300.csv
   baseline_v8_unseen.csv

B) CSVs downloaded from TensorBoard (optional):
   tb_cells_v8.csv
   tb_cells_nomap.csv
   tb_cells_task.csv
   tb_cells_v4.csv ... tb_cells_v7.csv

   tb_entropy_v8.csv
   tb_entropy_nomap.csv
   tb_entropy_task.csv
   tb_entropy_v1.csv ... tb_entropy_v7.csv

Run:
   pip install pandas matplotlib numpy
   python make_figures.py

Figures are written to the folder "figures".
Missing files are skipped.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

REACHABLE = 492
OUT = "figures"

os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(0)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def boot_ci(x, n=5000):
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(
        p * (1 - p) / n +
        z * z / (4 * n * n)
    ) / d

    return max(0.0, c - h), min(1.0, c + h)


def ema(v, w=0.6):
    out = []
    last = v[0]

    for x in v:
        last = last * w + (1 - w) * x
        out.append(last)

    return np.array(out)


def load_cells(path):
    if not os.path.exists(path):
        print("  (skipped, not found):", path)
        return None

    return pd.read_csv(path)["cells_this_episode"].to_numpy()


def load_csv_full(path):
    """Return full DataFrame or None."""

    if not os.path.exists(path):
        print("  (skipped, not found):", path)
        return None

    return pd.read_csv(path)


def load_tb(metric, run):
    p = f"tb_{metric}_{run}.csv"

    if not os.path.exists(p):
        return None

    df = pd.read_csv(p)

    return (
        df["Step"].to_numpy(),
        df["Value"].to_numpy()
    )


def steps_to_threshold(steps, sm, thr, K=3):
    """
    N* = first step where smoothed value is >= thr
    for K consecutive points.
    """

    for i in range(len(sm) - K + 1):
        if np.all(sm[i:i + K] >= thr):
            return int(steps[i])

    return None


def save(name):
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUT, name),
        dpi=160
    )

    plt.close()

    print("  wrote", name)


# ------------------------------------------------------------------
# Part A: Measured agents
# ------------------------------------------------------------------

AGENTS = [
    ("Random", "baseline-results/baseline_random.csv", "#888888"),
    ("Wall follower", "baseline-results/baseline_wallfollower.csv", "#d98c1f"),
    ("v8 (final)", "baseline-results/baseline_v8.csv", "#1f6fd9"),
    ("No visited map", "baseline-results/baseline_nomap300.csv", "#2aa36b"),
]

UNSEEN = (
    "v8 unseen maze",
    "baseline-results/baseline_v8_unseen.csv",
    "#d93b3b"
)


print("\nPart A: measured agents")

data = {}
full = {}

for label, path, col in AGENTS + [UNSEEN]:

    df = load_csv_full(path)

    if df is not None:

        data[label] = (
            df["cells_this_episode"].to_numpy(),
            col
        )

        full[label] = df


main = [
    a for a in AGENTS
    if a[0] in data
]


if main:

    # --------------------------------------------------------------
    # Summary table
    # --------------------------------------------------------------

    rows = []

    for label, _, _ in main + (
        [UNSEEN] if UNSEEN[0] in data else []
    ):

        x = data[label][0]

        lo, hi = boot_ci(x)

        s49 = int((x >= 49).sum())
        s25 = int((x >= 25).sum())

        cum = (
            int(full[label]["cumulative_unique_cells"].iloc[-1])
            if "cumulative_unique_cells" in full[label].columns
            else None
        )

        rows.append({

            "agent": label,
            "episodes": len(x),

            "mean_cells": round(x.mean(), 1),

            "pct_of_maze": round(
                100 * x.mean() / REACHABLE, 1
            ),

            "ci95_low": round(lo, 1),
            "ci95_high": round(hi, 1),

            "median": float(np.median(x)),

            "std": round(x.std(ddof=1), 1),

            "best": int(x.max()),

            "success_ge49_pct": round(
                100 * s49 / len(x), 1
            ),

            "success_ge25_pct": round(
                100 * s25 / len(x), 1
            ),

            "avg_regret": round(
                REACHABLE - x.mean(), 1
            ),

            "cumulative_regret": int(
                (REACHABLE - x).sum()
            ),

            "final_cumulative_unique": cum,

            "final_pct_of_maze": (
                round(100 * cum / REACHABLE, 1)
                if cum is not None else None
            ),

        })

    table = pd.DataFrame(rows)

    table.to_csv(
        os.path.join(OUT, "summary_table.csv"),
        index=False
    )

    print("\nSummary table:")
    print(table.to_string(index=False))

    # Figure 1: Cumulative unique cells
    plt.figure(figsize=(7.5, 4.5))
    for label, _, col in main:
        df = full[label]
        if "cumulative_unique_cells" in df.columns:
            y = df["cumulative_unique_cells"].to_numpy()
            plt.plot(np.arange(1, len(y) + 1), y, label=label, color=col, lw=2)

    if UNSEEN[0] in full and "cumulative_unique_cells" in full[UNSEEN[0]].columns:
        y = full[UNSEEN[0]]["cumulative_unique_cells"].to_numpy()
        plt.plot(np.arange(1, len(y) + 1), y, label=UNSEEN[0], color=UNSEEN[2], lw=2, ls="--")

    plt.axhline(REACHABLE, color="k", ls="--", lw=1, alpha=0.6, label=f"All reachable ({REACHABLE})")
    plt.xlabel("Evaluation episode")
    plt.ylabel("Cumulative unique cells discovered")
    plt.title("Total maze coverage over 100 evaluation episodes")
    plt.legend(fontsize=8)
    plt.grid(alpha=.3)
    save("fig1_cumulative_unique.png")

    # Figure 2: Coverage comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    for i, (label, _, col) in enumerate(main):
        x = data[label][0]
        lo, hi = boot_ci(x)
        ax1.bar(i, x.mean(), color=col, yerr=[[x.mean() - lo], [hi - x.mean()]], capsize=5)
        ax1.text(i, hi + 1.5, f"{x.mean():.1f}", ha="center", fontsize=9)
    ax1.set_xticks(range(len(main)))
    ax1.set_xticklabels([a[0] for a in main], fontsize=9, rotation=15)
    ax1.set_ylabel("Cells visited per episode (mean, 95% CI)")
    ax1.set_title("Coverage per episode")
    ax1.set_ylim(0, 55)
    ax1.grid(axis="y", alpha=.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for i, (label, _, col) in enumerate(main):
        df = full[label]
        if "cumulative_unique_cells" in df.columns:
            cum = int(df["cumulative_unique_cells"].iloc[-1])
            pct = 100.0 * cum / REACHABLE
            ax2.bar(i, pct, color=col)
            ax2.text(i, pct + 1.8, f"{pct:.1f}%\n({cum})", ha="center", fontsize=9)

    ax2.axhline(100, color="gray", ls="--", lw=1, alpha=0.6)
    ax2.set_xticks(range(len(main)))
    ax2.set_xticklabels([a[0] for a in main], fontsize=9, rotation=15)
    ax2.set_ylabel(f"Cumulative unique cells (% of {REACHABLE})")
    ax2.set_title("Total maze coverage after 100 episodes")
    ax2.set_ylim(0, 115)
    ax2.grid(axis="y", alpha=.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.suptitle("BUGSY exploration performance (100 evaluation episodes)", y=1.02)
    save("fig2_coverage_improved.png")

    # Figure 3: Success rate
    plt.figure(figsize=(7.5, 4.5))
    w = 0.38
    for j, (thr, lab) in enumerate([(49, ">=49 cells (10% of maze)"), (25, ">=25 cells (5% of maze)")]):
        for i, (label, _, col) in enumerate(main):
            x = data[label][0]
            s = int((x >= thr).sum())
            lo, hi = wilson(s, len(x))
            pct = 100 * s / len(x)
            plt.bar(i + (j - 0.5) * w, pct, width=w, color=col, alpha=1 if j == 0 else .5,
                    yerr=[[pct - 100 * lo], [100 * hi - pct]], capsize=3,
                    label=lab if i == 0 else None)
    plt.xticks(range(len(main)), [a[0] for a in main], fontsize=9)
    plt.ylabel("Episodes meeting the threshold (%), 95% Wilson CI")
    plt.title("Coverage success rate (dark = >=49 cells, light = >=25 cells)")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=.3)
    save("fig3_success_rate.png")

    # Figure 5: Distribution boxplot
    plt.figure(figsize=(7.5, 4.5))
    plt.boxplot([data[a[0]][0] for a in main], showmeans=True)
    plt.xticks(range(1, len(main) + 1), [a[0] for a in main], fontsize=9)
    plt.ylabel("Cells visited per episode")
    plt.title("Distribution of coverage per episode (box = quartiles, triangle = mean)")
    plt.grid(axis="y", alpha=.3)
    save("fig5_coverage_distribution.png")

    # Figure 11: Cumulative regret vs Random
    if "Random" in data:
        random_cells = data["Random"][0]
        plt.figure(figsize=(8, 4.8))
        random_cum_regret = np.cumsum(REACHABLE - random_cells)
        for label, _, col in main:
            x = data[label][0]
            if len(x) != len(random_cells):
                print(f"  (skipped in fig11, episode count differs): {label}")
                continue
            cumulative_regret = np.cumsum(REACHABLE - x)
            regret_gap = random_cum_regret - cumulative_regret
            plt.plot(np.arange(1, len(x) + 1), regret_gap, label=label, color=col, lw=2)
        plt.axhline(0, color="black", ls="--", lw=1)
        plt.xlabel("Evaluation episode")
        plt.ylabel("Cumulative regret avoided vs Random (cells)")
        plt.title("Cumulative regret improvement")
        plt.legend(fontsize=9)
        plt.grid(alpha=.3)
        save("fig11_cumulative_regret_difference.png")


# Figure 4: Generalization gap
if "v8 (final)" in data and UNSEEN[0] in data:
    a = data["v8 (final)"][0]
    b = data[UNSEEN[0]][0]
    plt.figure(figsize=(4.8, 4.5))
    for i, (x, c) in enumerate([(a, "#1f6fd9"), (b, "#d93b3b")]):
        lo, hi = boot_ci(x)
        plt.bar(i, x.mean(), color=c, yerr=[[x.mean() - lo], [hi - x.mean()]], capsize=5)
        plt.text(i, hi + 1, f"{x.mean():.1f}", ha="center")
    plt.xticks([0, 1], ["Seen maze\n(seed 21)", "Unseen maze\n(seed 7)"])
    plt.ylabel("Cells per episode (mean, 95% CI)")
    plt.title(f"v8 generalization gap = {a.mean() - b.mean():.1f} cells")
    plt.grid(axis="y", alpha=.3)
    save("fig4_generalization_gap.png")


# ------------------------------------------------------------------
# Part B: TensorBoard curves
# ------------------------------------------------------------------

print("\nPart B: TensorBoard curves")

MAIN_RUNS = [
    ("v8", "v8 (novelty + map)", "#1f6fd9"),
    ("nomap", "No visited map", "#2aa36b"),
    ("task", "Finish reward only (task PPO)", "#d93b3b")
]

tab = plt.get_cmap("tab10")


def overlay(metric, runs, ylabel, title, fname, hline=None):
    plt.figure(figsize=(7.5, 4.5))
    any_ = False
    for key, label, col in runs:
        d = load_tb(metric, key)
        if d is None:
            continue
        any_ = True
        s, v = d
        plt.plot(s, v, color=col, alpha=.25, lw=1)
        plt.plot(s, ema(v), color=col, lw=2, label=label)
    if not any_:
        plt.close()
        print("  (skipped, no files):", fname)
        return
    if hline is not None:
        plt.axhline(hline, color="k", ls="--", lw=1)
    plt.xlabel("Environment steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(alpha=.3)
    save(fname)


overlay("cells", MAIN_RUNS, "Cells visited per episode (training)",
        "Training coverage: PPO variants (smoothed)", "fig6_training_cells_main.png")

HIST_CELLS = [
    ("v4", "v4 ICM"),
    ("v5", "v5 ICM + cell bonus"),
    ("v6", "v6 bonus + map (start-pad spawns)"),
    ("v7", "v7 random spawns"),
    ("v8", "v8 final (stuck rule)")
]

overlay("cells", [(k, l, tab(i)) for i, (k, l) in enumerate(HIST_CELLS)],
        "Cells visited per episode (training)",
        "Development history (v4-v6: start-pad-only spawns, not directly comparable)",
        "fig7_history_cells.png")

overlay("entropy", MAIN_RUNS, "Policy entropy (action diversity)",
        "Policy entropy during training (smoothed)", "fig8_entropy_main.png")

HIST_ENT = [(f"v{i}", f"v{i}") for i in range(1, 9)]
overlay("entropy", [(k, l, tab(i)) for i, (k, l) in enumerate(HIST_ENT)],
        "Policy entropy",
        "Entropy across versions (max 4.09 for v1-v7 layout, 3.18 for v8)",
        "fig9_entropy_history.png")


# Figure 10: Convergence speed
THR = 40
conv = []
plt.figure(figsize=(7.5, 4.5))
any_ = False
for key, label, col in MAIN_RUNS:
    d = load_tb("cells", key)
    if d is None:
        continue
    any_ = True
    s, v = d
    sm = ema(v)
    plt.plot(s, sm, color=col, lw=2, label=label)
    n30 = steps_to_threshold(s, sm, 30)
    n40 = steps_to_threshold(s, sm, THR)
    conv.append({
        "run": label,
        "steps_to_30_cells": n30 if n30 is not None else "not reached",
        "steps_to_40_cells": n40 if n40 is not None else "not reached",
        "final_smoothed_cells": round(float(sm[-1]), 1)
    })
    if n40 is not None:
        plt.axvline(n40, color=col, ls=":", lw=1)

if any_:
    plt.axhline(THR, color="k", ls="--", lw=1, label=f"threshold = {THR} cells")
    plt.xlabel("Environment steps")
    plt.ylabel("Smoothed cells per episode (training)")
    plt.title("Convergence speed: steps to reach the threshold (3 consecutive points)")
    plt.legend(fontsize=8)
    plt.grid(alpha=.3)
    save("fig10_convergence.png")
    cdf = pd.DataFrame(conv)
    cdf.to_csv(os.path.join(OUT, "convergence_table.csv"), index=False)
    print("\nConvergence table:")
    print(cdf.to_string(index=False))
else:
    plt.close()

print("\nDone. Figures are in the folder:", OUT)

