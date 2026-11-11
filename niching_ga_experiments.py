#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSE 620 GA Mini-Project -- Niching Genetic Algorithms on benchmark
functions M1 and M4.

This script re-implements and extends the course-provided notebook
``niching_ga_examplever.py`` (K. Spurlock & M. Fukuyama, CSE 620 Fall 2022):

  * benchmark problems M1 and M4 (as used by Mahfoud, ICGA 1995),
  * three algorithms: standard GA (no niching), deterministic crowding,
    and fitness sharing,
  * a full experimental protocol: 2 values of Pc x 2 values of Pm x 10
    seeds for every (function, method) combination,
  * robust peak counting and JSON serialisation of every run.

Differences from the original notebook (bug fixes / extensions):
  1. Both children of a mating pair are now inserted into the population
     (the original loop only inserted the first child of each pair).
  2. ``compute_fitness`` no longer calls the undefined ``bin_to_float``.
  3. Peak counting no longer mutates the list it iterates over and now
     requires both proximity to the peak and near-peak fitness.
  4. Added the fitness-sharing niching method (Goldberg & Richardson, 1987).
  5. Roulette selection now mixes in a small (5%) uniform component;
     otherwise zero-fitness individuals become permanently unselectable
     and their slots fill up with stuck clones, shrinking the effective
     population of the standard GA and the sharing GA.
  6. The two fitness-based methods (standard, sharing) now use classic
     generational replacement with single-slot elitism.  The notebook's
     steady-state "children overwrite their parents' slots" scheme lets
     cross-niche children destroy occupied peaks permanently, which
     prevents fitness sharing from doing its job.  Deterministic crowding
     keeps its steady-state replacement, for which that scheme is the
     correct (and textbook) design.

Usage
-----
    python3 niching_ga_experiments.py             # full protocol
    python3 niching_ga_experiments.py --quick     # smoke test
    python3 niching_ga_experiments.py --num-runs 10 --num-gen 300
"""

import argparse
import json
import math
import os
import time

import numpy as np

# ----------------------------------------
# Benchmark functions (M1 and M4) and their global peak locations
# ----------------------------------------

# Global maxima of M1 and M4 on [0, 1] (5 peaks each).
PEAKS = {
    "M1": [0.1, 0.3, 0.5, 0.7, 0.9],
    "M4": [0.080, 0.245, 0.45, 0.68, 0.93],
}


def M1(x):
    """M1(x) = sin^6(5 * pi * x)  -- five equal-height peaks."""
    return np.sin(5 * np.pi * x) ** 6


def M4(x):
    """M4(x) = exp(-2 ln2 ((x-0.08)/0.854)^2) * sin^6(5 pi (x^0.75 - 0.05)).

    Five peaks of unequal height: the Gaussian envelope decays away from
    x = 0.08, so later peaks are progressively lower.
    """
    return (math.exp(-2 * math.log(2) * ((x - 0.08) / 0.854) ** 2)
            * (math.sin(5 * math.pi * (x ** 0.75 - 0.05)) ** 6))


COST_FUNCS = {"M1": M1, "M4": M4}

# ----------------------------------------
# Binary encoding / decoding
# ----------------------------------------

GENO_LEN = 10  # 10-bit strings represent values in [0, 1 - 2^-10]

# Mix this fraction of uniform probability into roulette selection.  Pure
# fitness-proportionate selection makes zero-fitness individuals
# effectively unselectable, so their slots are never replaced and the
# population permanently accumulates stuck clones (observed with the
# original notebook code).  A 5% uniform component fixes that while
# barely changing selection pressure among fitter individuals.
SELECTION_UNIFORM_MIX = 0.05


def constrained_bin_float(binary_list):
    """Decode a bit string to a value in [0, 1).

    value = sum_i b_i * 2^-(i+1), i.e. a 10-bit fractional representation.
    """
    value = 0.0
    for i, v in enumerate(binary_list):
        if v == 1:
            value += 2 ** -(i + 1)
    return value


def ham_dist(x, y):
    """Hamming distance between two bit strings."""
    return len([1 for i, j in zip(x, y) if i != j])


# ----------------------------------------
# GA building blocks (adapted from the course notebook)
# ----------------------------------------

def initialize_pop(pop_size, rand_gen=None):
    """Initialize the population as a (pop_size, GENO_LEN) integer matrix."""
    population = []
    for _ in range(pop_size):
        population.append(rand_gen.integers(0, 2, size=(GENO_LEN,)))
    return np.array(population).astype("int")


def compute_fitness(batch, cost_func):
    """Compute raw fitness for one solution or a batch of solutions."""
    phenotypes = np.array([constrained_bin_float(g) for g in batch])
    fitness = np.array([cost_func(p) for p in phenotypes])
    if fitness.ndim == 0:  # single solution passed as one-element batch
        return float(fitness)
    return fitness


def roulette_selection(fitness, num_mating=0, rand_gen=None,
                       replace=False):
    """Fitness-proportionate selection.

    ``replace=False`` (the default) samples without replacement, as in the
    original notebook; the generational methods use ``replace=True`` so
    that fitter individuals can be drawn multiple times.
    """
    if num_mating == 0:
        num_mating = len(fitness) // 2
        num_mating = num_mating - (num_mating % 2)

    total = fitness.sum()
    if total <= 0:
        fit_dist = np.full(len(fitness), 1.0 / len(fitness))
    else:
        fit_dist = fitness / total
    fit_dist = ((1 - SELECTION_UNIFORM_MIX) * fit_dist
                + SELECTION_UNIFORM_MIX / len(fitness))

    selected = rand_gen.choice(
        len(fitness), size=(num_mating,), replace=replace, p=fit_dist
    )
    return selected


def random_selection(fitness, num_mating=0, rand_gen=None):
    """Uniform random selection without replacement (used by crowding)."""
    N = len(fitness) if num_mating == 0 else num_mating
    selected = rand_gen.choice(
        np.arange(len(fitness)), size=(N,), replace=False
    )
    return selected


def one_point_cross(parent1, parent2, rand_gen=None):
    """One-point crossover."""
    length = len(parent1)
    cut = rand_gen.integers(1, length - 2)
    child1 = np.concatenate((parent1[0:cut], parent2[cut:length]))
    child2 = np.concatenate((parent2[0:cut], parent1[cut:length]))
    return (child1, child2)


def two_point_cross(parent1, parent2, rand_gen=None):
    """Two-point crossover."""
    length = len(parent1)
    cut1 = rand_gen.integers(0, length - 2)
    cut2 = cut1 + rand_gen.integers(1, length - cut1 - 1)
    child1 = np.concatenate(
        (parent1[0:cut1], parent2[cut1:cut2], parent1[cut2:length])
    )
    child2 = np.concatenate(
        (parent2[0:cut1], parent1[cut1:cut2], parent2[cut2:length])
    )
    return (child1, child2)


def mutation(solution, Pm=0.003, rand_gen=None):
    """Bit-flip mutation with per-bit probability Pm."""
    sol_copy = solution.copy()
    for i, bit in enumerate(sol_copy):
        if Pm > rand_gen.random():
            sol_copy[i] = 0 if bit == 1 else 1
    return sol_copy


def reproduce(population, selected, cross_method, cost_func, Pc, Pm,
              rand_gen=None):
    """Apply crossover + mutation to the selected parents, in pairs.

    Returns ``replacements`` mapping parent-index pairs to child pairs and
    ``child_fitness`` mapping the same keys to the children's raw fitness.
    Only pairs that actually crossed over are present in the dictionaries.
    """
    first_parents = selected[:len(selected) // 2]
    second_parents = selected[len(selected) // 2:]

    replacements = {}
    child_fitness = {}

    for p1, p2 in zip(first_parents, second_parents):
        if Pc > rand_gen.random():
            parent1, parent2 = population[p1], population[p2]
            children = cross_method(parent1, parent2, rand_gen)
            children = [mutation(c, Pm, rand_gen) for c in children]
            child_fit = compute_fitness(children, cost_func)
            replacements[(p1, p2)] = (children[0], children[1])
            child_fitness[(p1, p2)] = (child_fit[0], child_fit[1])

    return replacements, child_fitness


# ----------------------------------------
# Niching method 1: deterministic crowding
# ----------------------------------------

def crowding(population, fitness, replacements, child_fitness):
    """Deterministic crowding.

    Each child is matched to the closer of the two parents (Hamming
    distance on genotypes) and replaces that parent only if it is fitter.
    """
    new_replacements = {}
    new_child_fitness = {}

    for p1, p2 in replacements.keys():
        p1_ge, p2_ge = population[p1], population[p2]
        c1_ge, c2_ge = replacements[(p1, p2)][0], replacements[(p1, p2)][1]

        p1_fit, p2_fit = fitness[p1], fitness[p2]
        c1_fit, c2_fit = child_fitness[(p1, p2)][0], child_fitness[(p1, p2)][1]

        # Default: keep the parents.
        new_replacements[(p1, p2)] = [p1_ge, p2_ge]
        new_child_fitness[(p1, p2)] = [p1_fit, p2_fit]

        if (ham_dist(p1_ge, c1_ge) + ham_dist(p2_ge, c2_ge)) <= \
           (ham_dist(p1_ge, c2_ge) + ham_dist(p2_ge, c1_ge)):
            # p1 closer to c1, p2 closer to c2
            if c1_fit > p1_fit:
                new_replacements[(p1, p2)][0] = c1_ge
                new_child_fitness[(p1, p2)][0] = c1_fit
            if c2_fit > p2_fit:
                new_replacements[(p1, p2)][1] = c2_ge
                new_child_fitness[(p1, p2)][1] = c2_fit
        else:
            # p2 closer to c1, p1 closer to c2
            if c1_fit > p2_fit:
                new_replacements[(p1, p2)][1] = c1_ge
                new_child_fitness[(p1, p2)][1] = c1_fit
            if c2_fit > p1_fit:
                new_replacements[(p1, p2)][0] = c2_ge
                new_child_fitness[(p1, p2)][0] = c2_fit

    return new_replacements, new_child_fitness


# ----------------------------------------
# Niching method 2: fitness sharing
# ----------------------------------------

def sharing_function(d, sigma, alpha=1.0):
    """Sharing function sh(d) = 1 - (d/sigma)^alpha for d < sigma, else 0."""
    if d < sigma:
        return 1.0 - (d / sigma) ** alpha
    return 0.0


def shared_fitness(population, fitness, sigma=0.1, alpha=1.0):
    """Fitness sharing: f'_i = f_i / sum_j sh(d(x_i, x_j)).

    Distances are phenotypic (|x_i - x_j| on [0, 1]) so that the niche
    radius sigma has a direct meaning in decision space.  The self term
    d = 0 always contributes 1, so the denominator is >= 1.
    """
    phenotypes = np.array([constrained_bin_float(g) for g in population])
    n = len(population)
    shared = np.empty(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            d = abs(phenotypes[i] - phenotypes[j])
            s += sharing_function(d, sigma, alpha)
        shared[i] = fitness[i] / s
    return shared


# ----------------------------------------
# Main GA loop
# ----------------------------------------

def genetic_algorithm(method, cost_func, num_gen, pop_size,
                      Pc, Pm, rand_gen, sigma_share=0.1,
                      cross_method=two_point_cross, peaks_fn=None):
    """Run one GA experiment.

    method : "standard" | "crowding" | "sharing"
      * standard : roulette selection on raw fitness + generational
                   replacement (with elitism)
      * sharing  : identical to standard, but the roulette wheel uses
                   fitness-shared values
      * crowding : random pairing of the whole population + deterministic
                   crowding replacement (steady state, elitist by
                   construction)

    The genetic operators are identical for all three methods: a pair
    exchanges material with probability Pc and both children are then
    mutated bit-wise with probability Pm; pairs that do not cross over
    are passed through unchanged.  The methods differ ONLY in how parents
    are chosen and how children are inserted.

    peaks_fn : optional callable(phenotypes, fitness) -> int recorded after
               every generation (used to track how many peaks survive).

    Returns a dict with the raw-fitness history (min/max/mean per
    generation) and the initial/final populations as phenotypes.
    """
    if method not in ("standard", "crowding", "sharing"):
        raise ValueError(f"unknown method {method!r}")

    population = initialize_pop(pop_size, rand_gen=rand_gen)
    fitness = compute_fitness(population, cost_func)

    fitness_stats = {
        "min": [float(fitness.min())],
        "max": [float(fitness.max())],
        "mean": [float(fitness.mean())],
    }
    peaks_history = []
    if peaks_fn is not None:
        phen = [constrained_bin_float(g) for g in population]
        peaks_history.append(int(peaks_fn(phen, fitness)))
    initial_pop = np.array([constrained_bin_float(g) for g in population])

    for _ in range(num_gen):
        if method in ("standard", "sharing"):
            # ----------------------------------------
            # Generational replacement (standard GA / fitness sharing)
            # ----------------------------------------
            if method == "sharing":
                sel_fitness = shared_fitness(population, fitness,
                                             sigma=sigma_share)
            else:
                sel_fitness = fitness
            # Draw pop_size parents with replacement, pair them up and
            # breed a full new population.
            parents = roulette_selection(sel_fitness, num_mating=pop_size,
                                         rand_gen=rand_gen, replace=True)

            new_population = np.empty_like(population)
            for i in range(0, pop_size, 2):
                p1, p2 = parents[i], parents[i + 1]
                if Pc > rand_gen.random():
                    c1, c2 = cross_method(population[p1], population[p2],
                                          rand_gen)
                    c1 = mutation(c1, Pm, rand_gen)
                    c2 = mutation(c2, Pm, rand_gen)
                else:
                    # Pair did not cross over: pass the parents through.
                    c1 = population[p1].copy()
                    c2 = population[p2].copy()
                new_population[i] = c1
                new_population[i + 1] = c2

            new_fitness = compute_fitness(new_population, cost_func)

            # Elitism: the single best individual survives if it would
            # otherwise be lost (and is not already present).
            old_best = int(np.argmax(fitness))
            new_worst = int(np.argmin(new_fitness))
            if (fitness[old_best] > new_fitness[new_worst]
                    and not np.any(np.all(
                        new_population == population[old_best], axis=1))):
                new_population[new_worst] = population[old_best]
                new_fitness[new_worst] = fitness[old_best]

            population, fitness = new_population, new_fitness

        else:
            # ----------------------------------------
            # Deterministic crowding (steady-state replacement)
            # ----------------------------------------
            selected = random_selection(fitness, num_mating=pop_size,
                                        rand_gen=rand_gen)
            replacements, child_fitness = reproduce(
                population, selected, cross_method, cost_func, Pc, Pm,
                rand_gen
            )
            replacements, child_fitness = crowding(
                population, fitness, replacements, child_fitness
            )
            # Apply replacements: both children of a mating pair are
            # inserted (the original notebook only inserted the first
            # child of each pair).
            for k in replacements.keys():
                population[k[0]] = replacements[k][0]
                fitness[k[0]] = child_fitness[k][0]
                population[k[1]] = replacements[k][1]
                fitness[k[1]] = child_fitness[k][1]

        fitness_stats["min"].append(float(fitness.min()))
        fitness_stats["max"].append(float(fitness.max()))
        fitness_stats["mean"].append(float(fitness.mean()))
        if peaks_fn is not None:
            phen = [constrained_bin_float(g) for g in population]
            peaks_history.append(int(peaks_fn(phen, fitness)))

    final_pop = np.array([constrained_bin_float(g) for g in population])

    return {
        "fitness_history": fitness_stats,
        "peaks_history": peaks_history,
        "initial_pop": initial_pop.tolist(),
        "final_pop": final_pop.tolist(),
        "final_fitness": fitness.tolist(),
    }


# ----------------------------------------
# Evaluation: peak counting
# ----------------------------------------

def count_peaks(final_pop, final_fitness, func_name, tol=0.01,
                fit_ratio=0.9):
    """Count how many of the known peaks are represented in the final
    population.

    A peak counts as found if at least one individual lies within ``tol``
    of the peak location AND its fitness is at least ``fit_ratio`` of the
    value of that peak (guards against sitting next to, but not on, a peak).
    """
    cost = COST_FUNCS[func_name]
    found = 0
    for p in PEAKS[func_name]:
        peak_val = cost(p)
        for x, f in zip(final_pop, final_fitness):
            if abs(x - p) < tol and f >= fit_ratio * peak_val:
                found += 1
                break
    return found


# ----------------------------------------
# Experiment protocol
# ----------------------------------------

METHODS = ["standard", "crowding", "sharing"]
PC_VALUES = [0.60, 0.95]
PM_VALUES = [0.01, 0.10]
SIGMA_SHARE = 0.10          # sharing niche radius (phenotype distance)
PEAK_TOL = 0.01             # peak acceptance tolerance
PEAK_FIT_RATIO = 0.90       # required fitness relative to the peak value


def run_protocol(num_runs=10, num_gen=300, pop_size=50,
                 seeds=None, functions=("M1", "M4"), methods=None,
                 verbose=True):
    """Run the full experimental protocol and return a results dict."""
    methods = methods or METHODS
    seeds = seeds if seeds is not None else list(range(num_runs))
    cross_method = two_point_cross

    results = {
        "config": {
            "num_runs": len(seeds),
            "seeds": seeds,
            "num_gen": num_gen,
            "pop_size": pop_size,
            "genotype_len": GENO_LEN,
            "crossover": "two_point",
            "pc_values": PC_VALUES,
            "pm_values": PM_VALUES,
            "methods": methods,
            "functions": list(functions),
            "sigma_share": SIGMA_SHARE,
            "selection_uniform_mix": SELECTION_UNIFORM_MIX,
            "peak_tol": PEAK_TOL,
            "peak_fit_ratio": PEAK_FIT_RATIO,
            "peaks": {k: v for k, v in PEAKS.items() if k in functions},
        },
        "runs": [],
    }

    total = len(functions) * len(methods) * len(PC_VALUES) * len(PM_VALUES) \
        * len(seeds)
    done = 0
    t0 = time.time()

    for func_name in functions:
        cost = COST_FUNCS[func_name]

        def peaks_fn(phen, fit, _fn=func_name):
            return count_peaks(phen, fit, _fn, tol=PEAK_TOL,
                               fit_ratio=PEAK_FIT_RATIO)

        for method in methods:
            for Pc in PC_VALUES:
                for Pm in PM_VALUES:
                    for seed in seeds:
                        rand_gen = np.random.default_rng(seed)
                        out = genetic_algorithm(
                            method=method,
                            cost_func=cost,
                            num_gen=num_gen,
                            pop_size=pop_size,
                            Pc=Pc, Pm=Pm,
                            rand_gen=rand_gen,
                            sigma_share=SIGMA_SHARE,
                            cross_method=cross_method,
                            peaks_fn=peaks_fn,
                        )
                        peaks = count_peaks(
                            out["final_pop"], out["final_fitness"],
                            func_name, tol=PEAK_TOL,
                            fit_ratio=PEAK_FIT_RATIO,
                        )
                        run = {
                            "function": func_name,
                            "method": method,
                            "Pc": Pc,
                            "Pm": Pm,
                            "seed": seed,
                            "peaks_found": peaks,
                            "peaks_history": out["peaks_history"],
                            "best_fitness": float(max(out["final_fitness"])),
                            "mean_final_fitness":
                                float(np.mean(out["final_fitness"])),
                            "fitness_history": out["fitness_history"],
                            "initial_pop": out["initial_pop"],
                            "final_pop": out["final_pop"],
                            "final_fitness": out["final_fitness"],
                        }
                        results["runs"].append(run)
                        done += 1
                        if verbose:
                            print(f"[{done:4d}/{total}] {func_name} "
                                  f"{method:9s} Pc={Pc:.2f} Pm={Pm:.2f} "
                                  f"seed={seed} peaks={peaks} "
                                  f"best={run['best_fitness']:.4f} "
                                  f"({time.time() - t0:.1f}s)", flush=True)
    return results


def summarise(results):
    """Aggregate the raw runs into per-configuration averages."""
    groups = {}
    for run in results["runs"]:
        key = (run["function"], run["method"], run["Pc"], run["Pm"])
        groups.setdefault(key, []).append(run)

    summary = []
    for (func, method, Pc, Pm), runs in sorted(groups.items()):
        peaks = [r["peaks_found"] for r in runs]
        best = [r["best_fitness"] for r in runs]
        meanf = [r["mean_final_fitness"] for r in runs]
        summary.append({
            "function": func,
            "method": method,
            "Pc": Pc,
            "Pm": Pm,
            "n": len(runs),
            "peaks_mean": float(np.mean(peaks)),
            "peaks_std": float(np.std(peaks, ddof=1)) if len(peaks) > 1 else 0.0,
            "best_fitness_mean": float(np.mean(best)),
            "best_fitness_std": float(np.std(best, ddof=1)) if len(best) > 1 else 0.0,
            "mean_final_fitness_mean": float(np.mean(meanf)),
            "mean_final_fitness_std": float(np.std(meanf, ddof=1)) if len(meanf) > 1 else 0.0,
        })
    return summary


def best_configs(summary):
    """Pick the best (Pc, Pm) configuration for each (function, method).

    Primary criterion: average number of peaks found; ties broken by the
    average best fitness of the final population.
    """
    best = {}
    for row in summary:
        key = (row["function"], row["method"])
        cur = best.get(key)
        score = (row["peaks_mean"], row["best_fitness_mean"])
        if cur is None or score > (cur["peaks_mean"],
                                   cur["best_fitness_mean"]):
            best[key] = row
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 1 seed, 30 generations")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--num-gen", type=int, default=300)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--out", default="results/experiments.json")
    args = parser.parse_args()

    num_runs = args.num_runs
    num_gen = args.num_gen
    if args.quick:
        num_runs, num_gen = 1, 30

    results = run_protocol(num_runs=num_runs, num_gen=num_gen,
                           pop_size=args.pop_size)

    summary = summarise(results)
    best = best_configs(summary)
    results["summary"] = summary
    results["best_configs"] = {
        f"{k[0]}|{k[1]}": v for k, v in best.items()
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh)
    print(f"\nSaved {len(results['runs'])} runs to {args.out}")

    # Console summary
    print("\n=== Average peaks found (of 5) ===")
    for row in summary:
        print(f"{row['function']} {row['method']:9s} "
              f"Pc={row['Pc']:.2f} Pm={row['Pm']:.2f} : "
              f"{row['peaks_mean']:.1f} +/- {row['peaks_std']:.2f} "
              f"(best fitness {row['best_fitness_mean']:.4f})")

    print("\n=== Best configuration per (function, method) ===")
    for key, row in sorted(best.items()):
        print(f"{key[0]} {key[1]:9s}: Pc={row['Pc']:.2f} Pm={row['Pm']:.2f} "
              f"-> {row['peaks_mean']:.1f} peaks, "
              f"best fitness {row['best_fitness_mean']:.4f}")


if __name__ == "__main__":
    main()
