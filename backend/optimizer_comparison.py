"""
Optimizer benchmarking layer.

Advisory gap addressed: "The Meerkat Optimization Algorithm is assumed to
be effective." -> "Treat MOA as a candidate method and benchmark it
against agronomic rules, linear or constrained optimization, and simpler
ML methods."

This module treats MOA as ONE candidate among several and runs the same
chemical-dose-reduction problem through four different optimizers so they
can be compared side by side on the same fitness function:

  1. MOA          -- backend.meerkat_optimizer.meerkat_chemical_reduction
                      (population-based metaheuristic, already implemented)
  2. PSO           -- Particle Swarm Optimization (new, this file)
  3. GA            -- Genetic Algorithm (new, this file)
  4. Rule-based    -- fixed-percentage agronomic rule of thumb, no search
                      at all (the simplest possible baseline)

All four optimize the SAME objective: minimize chemical dose while the
non-linear dose-response curve in meerkat_optimizer stays at/above a
target yield fraction. Reduction %, resulting yield-response %, and wall
clock runtime are reported for each, so a reviewer/agronomist can judge
whether MOA is actually earning its complexity over simpler alternatives.
"""

import time
import random
import numpy as np

from backend.meerkat_optimizer import (
    run_meerkat_search,
    _yield_response_ratio,
    _min_ratio_for_target,
)


def _fitness(dose_ratio, initial_dose, target_yield, chemical_category):
    """
    Shared objective for PSO/GA: minimize dose ratio subject to the yield
    response staying >= target_yield. Encoded as a penalized scalar so
    both optimizers can just minimize one number.
    """
    dose_ratio = max(0.0, min(1.0, dose_ratio))
    y = _yield_response_ratio(dose_ratio, initial_dose, chemical_category)
    penalty = max(0.0, target_yield - y) * 10.0  # heavily penalize missing the yield target
    return dose_ratio + penalty


def run_pso(initial_dose, target_yield_percentage=0.95, chemical_category="fertilizer",
            n_particles=20, n_iterations=40, seed=42):
    """
    Particle Swarm Optimization over dose_ratio in [0, 1].
    Simple textbook PSO: each particle has a position + velocity, pulled
    toward its own best and the swarm's best each iteration.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    target = max(0.5, min(0.99, target_yield_percentage))

    w, c1, c2 = 0.6, 1.5, 1.5
    positions = rng.uniform(0.3, 1.0, n_particles)
    velocities = rng.uniform(-0.1, 0.1, n_particles)
    personal_best = positions.copy()
    personal_best_val = np.array([
        _fitness(p, initial_dose, target, chemical_category) for p in positions
    ])
    global_best_idx = int(np.argmin(personal_best_val))
    global_best = personal_best[global_best_idx]
    global_best_val = personal_best_val[global_best_idx]

    for _ in range(n_iterations):
        r1, r2 = rng.random(n_particles), rng.random(n_particles)
        velocities = (
            w * velocities
            + c1 * r1 * (personal_best - positions)
            + c2 * r2 * (global_best - positions)
        )
        positions = np.clip(positions + velocities, 0.0, 1.0)
        values = np.array([
            _fitness(p, initial_dose, target, chemical_category) for p in positions
        ])
        improved = values < personal_best_val
        personal_best[improved] = positions[improved]
        personal_best_val[improved] = values[improved]
        best_idx = int(np.argmin(personal_best_val))
        if personal_best_val[best_idx] < global_best_val:
            global_best = personal_best[best_idx]
            global_best_val = personal_best_val[best_idx]

    runtime_ms = (time.perf_counter() - start) * 1000
    optimal_dose = round(float(global_best) * initial_dose, 3)
    achieved_yield = _yield_response_ratio(global_best, initial_dose, chemical_category)
    return {
        "method": "PSO (Particle Swarm Optimization)",
        "optimal_chemical_dose": optimal_dose,
        "reduction_percentage": round((1 - global_best) * 100, 1),
        "achieved_yield_pct": round(achieved_yield * 100, 1),
        "runtime_ms": round(runtime_ms, 2),
    }


def run_ga(initial_dose, target_yield_percentage=0.95, chemical_category="fertilizer",
           population_size=24, generations=40, mutation_rate=0.15, seed=42):
    """
    Genetic Algorithm over dose_ratio in [0, 1]: tournament selection,
    arithmetic crossover, gaussian mutation, elitism of 2.
    """
    start = time.perf_counter()
    rng = random.Random(seed)
    target = max(0.5, min(0.99, target_yield_percentage))

    population = [rng.uniform(0.3, 1.0) for _ in range(population_size)]

    def fit(ind):
        return _fitness(ind, initial_dose, target, chemical_category)

    for _ in range(generations):
        scored = sorted(population, key=fit)
        elite = scored[:2]
        next_gen = list(elite)
        while len(next_gen) < population_size:
            a = min(rng.sample(scored, 3), key=fit)
            b = min(rng.sample(scored, 3), key=fit)
            alpha = rng.random()
            child = alpha * a + (1 - alpha) * b
            if rng.random() < mutation_rate:
                child += rng.gauss(0, 0.08)
            next_gen.append(max(0.0, min(1.0, child)))
        population = next_gen

    best = min(population, key=fit)
    runtime_ms = (time.perf_counter() - start) * 1000
    optimal_dose = round(best * initial_dose, 3)
    achieved_yield = _yield_response_ratio(best, initial_dose, chemical_category)
    return {
        "method": "GA (Genetic Algorithm)",
        "optimal_chemical_dose": optimal_dose,
        "reduction_percentage": round((1 - best) * 100, 1),
        "achieved_yield_pct": round(achieved_yield * 100, 1),
        "runtime_ms": round(runtime_ms, 2),
    }


def run_rule_based(initial_dose, target_yield_percentage=0.95, chemical_category="fertilizer"):
    """
    Simplest possible baseline -- NOT a search method at all. Applies a
    fixed agronomic rule of thumb (flat 10% cut for fertilizers, flat 5%
    cut for pest-control products, since under-dosing pest control is
    riskier) and reports the resulting yield response for comparison. No
    optimization loop, so runtime is effectively zero -- that is the
    point: it shows what you get WITHOUT any optimizer.
    """
    start = time.perf_counter()
    flat_cut = 0.05 if chemical_category == "pest_control" else 0.10
    dose_ratio = 1.0 - flat_cut
    achieved_yield = _yield_response_ratio(dose_ratio, initial_dose, chemical_category)
    runtime_ms = (time.perf_counter() - start) * 1000
    return {
        "method": "Rule-based (flat % cut)",
        "optimal_chemical_dose": round(dose_ratio * initial_dose, 3),
        "reduction_percentage": round(flat_cut * 100, 1),
        "achieved_yield_pct": round(achieved_yield * 100, 1),
        "runtime_ms": round(runtime_ms, 3),
    }


def run_moa(initial_dose, target_yield_percentage=0.95, chemical_category="fertilizer", seed=42):
    """
    Runs MOA against the SAME objective PSO/GA/rule-based are scored
    against (_fitness, defined above) -- not the richer, more
    conservative fitness meerkat_chemical_reduction() uses for the
    actual per-farmer recommendation elsewhere in the app. Using the
    identical objective here is what makes the four-way comparison fair:
    whichever method wins does so because of search quality, not because
    it was solving an easier problem. Search dynamics come from
    backend.meerkat_optimizer.run_meerkat_search() -- Levy-flight
    exploration, sentry-pull exploitation, stagnation-triggered partial
    reinitialization (see that function's docstring for the full math).
    """
    start = time.perf_counter()
    target = max(0.5, min(0.99, target_yield_percentage))

    def fitness_fn(dose_ratio):
        # run_meerkat_search maximizes; _fitness is a cost (lower is
        # better), so negate it.
        return -_fitness(dose_ratio, initial_dose, target, chemical_category)

    result = run_meerkat_search(
        fitness_fn, lower_bound=0.0, upper_bound=1.0,
        population_size=12, max_iterations=60, seed=seed,
    )
    best_ratio = result["best_solution"]
    runtime_ms = (time.perf_counter() - start) * 1000
    optimal_dose = round(best_ratio * initial_dose, 3)
    achieved_yield = _yield_response_ratio(best_ratio, initial_dose, chemical_category)
    return {
        "method": "MOA (Meerkat Optimization Algorithm)",
        "optimal_chemical_dose": optimal_dose,
        "reduction_percentage": round((1 - best_ratio) * 100, 1),
        "achieved_yield_pct": round(achieved_yield * 100, 1),
        "runtime_ms": round(runtime_ms, 2),
    }


def compare_optimizers(initial_dose, target_yield_percentage=0.95, chemical_category="fertilizer"):
    """
    Runs all four candidate methods on the identical problem and returns
    a ranked comparison. This is the function the UI should call instead
    of calling meerkat_chemical_reduction() directly -- it makes MOA one
    row in a table rather than the assumed answer, per the advisory.

    Returns a dict: {"rows": [...], "recommended": <method name>, "note": str}
    """
    if initial_dose <= 0:
        return {"rows": [], "recommended": None, "note": "Initial dose must be greater than zero."}

    rows = [
        run_moa(initial_dose, target_yield_percentage, chemical_category),
        run_pso(initial_dose, target_yield_percentage, chemical_category),
        run_ga(initial_dose, target_yield_percentage, chemical_category),
        run_rule_based(initial_dose, target_yield_percentage, chemical_category),
    ]

    # Only consider methods that actually met the yield target when
    # picking a "recommended" row -- a method that cuts more but misses
    # the yield target is not actually better.
    target_pct = max(0.5, min(0.99, target_yield_percentage)) * 100
    feasible = [r for r in rows if r["achieved_yield_pct"] >= target_pct - 0.5]
    pool = feasible if feasible else rows
    best = max(pool, key=lambda r: r["reduction_percentage"])

    return {
        "rows": rows,
        "recommended": best["method"],
        "note": (
            "Recommendation is the candidate with the highest chemical reduction among "
            "methods that still met the target yield response. MOA is shown alongside "
            "PSO, GA and a rule-based baseline rather than assumed to be best -- "
            "per the advisory recommendation to benchmark MOA against simpler methods."
        ),
    }
