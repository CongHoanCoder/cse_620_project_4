#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate every figure used in the CSE 620 GA mini-project report from
``results/experiments.json`` (produced by ``niching_ga_experiments.py``).

Figures written to ``figures/``:

  fig_convergence_{M1,M4}.png   Fmax / Favg / Fmin vs generation, one
                                panel per method, averaged over 10 runs,
                                at that method's best (Pc, Pm) config
  fig_landscape_{M1,M4}.png     initial (red) and final (blue) population
                                on the fitness landscape, best run of the
                                best config, one panel per method
  fig_peaks_time.png           average number of peaks represented in
                               the population vs generation (both
                               functions side by side)
  fig_peaks_bars_{M1,M4}.png    average peaks found for every
                                (method, Pc, Pm) configuration
  fig_conv_grid_{M1,M4}.png     appendix grid: convergence for every
                                method x configuration
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from niching_ga_experiments import COST_FUNCS, PEAKS, PM_VALUES, PC_VALUES

RESULTS = "results/experiments.json"
FIGDIR = "figures"

METHOD_LABELS = {
    "standard": "Standard GA",
    "crowding": "Deterministic crowding",
    "sharing": "Fitness sharing",
}
METHOD_ORDER = ["standard", "crowding", "sharing"]
METHOD_COLORS = {
    "standard": "#d62728",
    "crowding": "#1f77b4",
    "sharing": "#2ca02c",
}

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load():
    with open(RESULTS) as fh:
        return json.load(fh)


def config_str(Pc, Pm):
    return f"$P_c$={Pc:.2f}, $P_m$={Pm:.2f}"


def runs_at(res, func, method, Pc, Pm):
    return [r for r in res["runs"]
            if r["function"] == func and r["method"] == method
            and r["Pc"] == Pc and r["Pm"] == Pm]


def best_cfg(res, func, method):
    """Best (Pc, Pm) for a (function, method) from the stored summary."""
    key = f"{func}|{method}"
    row = res["best_configs"][key]
    return row["Pc"], row["Pm"]


def best_run(runs):
    """Best single run: most peaks, ties broken by best fitness."""
    return max(runs, key=lambda r: (r["peaks_found"], r["best_fitness"]))


def mean_history(runs, field):
    """Mean over runs of a fitness-history list (min/max/mean)."""
    arr = np.array([r["fitness_history"][field] for r in runs])
    return arr.mean(axis=0)


def plot_convergence(res, func):
    """Fmax/Favg/Fmin vs generation, one panel per method (best config)."""
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3), sharey=True)
    cost = COST_FUNCS[func]
    for ax, method in zip(axes, METHOD_ORDER):
        Pc, Pm = best_cfg(res, func, method)
        runs = runs_at(res, func, method, Pc, Pm)
        gens = np.arange(len(runs[0]["fitness_history"]["max"]))
        for field, label, color in [("max", "$F_{max}$", "#1f77b4"),
                                    ("mean", "$F_{avg}$", "#ff7f0e"),
                                    ("min", "$F_{min}$", "#d62728")]:
            ax.plot(gens, mean_history(runs, field), label=label,
                    color=color, lw=1.2)
        ax.set_title(f"{METHOD_LABELS[method]}\n{config_str(Pc, Pm)} "
                     f"(avg. 10 runs)", fontsize=7.5)
        ax.set_xlabel("Generation")
        if ax is axes[0]:
            ax.set_ylabel("Fitness")
        ax.set_xlim(0, len(gens) - 1)
        ax.set_ylim(-0.03, 1.05)
        ax.legend(loc="lower right", framealpha=0.9)
    fig.suptitle(f"Fitness statistics on {func}", fontsize=9, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGDIR, f"fig_convergence_{func}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_landscape(res, func):
    """Initial (red) vs final (blue) population on the landscape."""
    cost = COST_FUNCS[func]
    x = np.arange(0, 1, 0.001)
    y = [cost(v) for v in x]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharey=True)
    for ax, method in zip(axes, METHOD_ORDER):
        Pc, Pm = best_cfg(res, func, method)
        runs = runs_at(res, func, method, Pc, Pm)
        r = best_run(runs)
        ax.plot(x, y, color="grey", lw=0.8, zorder=1, label="Landscape")
        for p in PEAKS[func]:
            ax.axvline(p, color="green", ls=":", lw=0.7, zorder=2)
        ax.scatter(r["initial_pop"], [cost(v) for v in r["initial_pop"]],
                   color="red", s=8, zorder=3, label="Initial")
        ax.scatter(r["final_pop"], [cost(v) for v in r["final_pop"]],
                   color="blue", s=8, zorder=4, label="Final")
        ax.set_title(f"{METHOD_LABELS[method]}\n{config_str(Pc, Pm)}, "
                     f"seed {r['seed']} ({r['peaks_found']}/5 peaks)",
                     fontsize=7.5)
        ax.set_xlabel("Phenotype value $x$")
        if ax is axes[0]:
            ax.set_ylabel("Fitness")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1.12)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=6.5)
    fig.suptitle(f"Initial and final population on the {func} landscape "
                 f"(best run per method)", fontsize=9, y=1.03)
    fig.tight_layout()
    out = os.path.join(FIGDIR, f"fig_landscape_{func}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_peaks_time(res):
    """Average number of peaks represented vs generation (both functions)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4), sharey=True)
    for ax, f in zip(axes, ["M1", "M4"]):
        for method in METHOD_ORDER:
            Pc, Pm = best_cfg(res, f, method)
            runs = runs_at(res, f, method, Pc, Pm)
            hist = np.array([r["peaks_history"] for r in runs])
            gens = np.arange(hist.shape[1])
            ax.plot(gens, hist.mean(axis=0), label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method], lw=1.2)
        ax.set_title(f"{f}  (best config per method)", fontsize=7.5)
        ax.set_xlabel("Generation")
        ax.set_xlim(0, hist.shape[1] - 1)
        ax.set_ylim(0, 5.4)
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        if ax is axes[0]:
            ax.set_ylabel("Peaks represented (of 5)")
        ax.legend(loc="center right", framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_peaks_time.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_peaks_bars(res, func):
    """Bar chart: average peaks for every (method, Pc, Pm) configuration."""
    labels, series = [], {m: [] for m in METHOD_ORDER}
    stds = {m: [] for m in METHOD_ORDER}
    for Pc in PC_VALUES:
        for Pm in PM_VALUES:
            labels.append(f"{Pc:.2f}\n{Pm:.2f}")
            for method in METHOD_ORDER:
                row = next(s for s in res["summary"]
                           if s["function"] == func
                           and s["method"] == method
                           and s["Pc"] == Pc and s["Pm"] == Pm)
                series[method].append(row["peaks_mean"])
                stds[method].append(row["peaks_std"])

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    for i, method in enumerate(METHOD_ORDER):
        ax.bar(x + (i - 1) * width, series[method], width,
               yerr=stds[method], capsize=2, label=METHOD_LABELS[method],
               color=METHOD_COLORS[method], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("$P_c$ over $P_m$")
    ax.set_ylabel("Peaks found (of 5)")
    ax.set_ylim(0, 5.6)
    ax.legend(fontsize=6.5, ncol=3, loc="upper center")
    ax.set_title(f"{func}: average peaks over 10 runs", fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIGDIR, f"fig_peaks_bars_{func}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_conv_grid(res, func):
    """Appendix grid: convergence for every method x (Pc, Pm)."""
    fig, axes = plt.subplots(3, 4, figsize=(7.16, 6.2), sharex=True,
                             sharey=True)
    for i, method in enumerate(METHOD_ORDER):
        for j, (Pc, Pm) in enumerate([(pc, pm) for pc in PC_VALUES
                                      for pm in PM_VALUES]):
            ax = axes[i][j]
            runs = runs_at(res, func, method, Pc, Pm)
            gens = np.arange(len(runs[0]["fitness_history"]["max"]))
            for field, color in [("max", "#1f77b4"), ("mean", "#ff7f0e"),
                                 ("min", "#d62728")]:
                ax.plot(gens, mean_history(runs, field), color=color,
                        lw=0.9)
            if i == 0:
                ax.set_title(config_str(Pc, Pm), fontsize=7)
            if j == 0:
                ax.set_ylabel(METHOD_LABELS[method] + "\nFitness",
                              fontsize=7)
            if i == 2:
                ax.set_xlabel("Generation", fontsize=7)
            ax.set_xlim(0, len(gens) - 1)
            ax.set_ylim(-0.03, 1.05)
    fig.suptitle(f"{func}: convergence for all configurations "
                 "(blue=$F_{max}$, orange=$F_{avg}$, red=$F_{min}$; "
                 "avg. 10 runs)", fontsize=8, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = os.path.join(FIGDIR, f"fig_conv_grid_{func}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    res = load()
    for func in ("M1", "M4"):
        plot_convergence(res, func)
        plot_landscape(res, func)
        plot_peaks_bars(res, func)
        plot_conv_grid(res, func)
    plot_peaks_time(res)
    print("done")


if __name__ == "__main__":
    main()
