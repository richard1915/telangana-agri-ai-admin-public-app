import os
import math
import pickle
import random
import numpy as np
import pandas as pd

# ============================================================
# METAHEURISTIC: Meerkat Optimization Algorithm (MOA)
# Population-based search that finds the minimum chemical dose that
# still meets a target yield. No training data involved — it optimizes
# a hand-defined fitness function at run time. See meerkat_chemical_reduction().
# ============================================================


def _mantegna_levy_step(beta: float = 1.5, rng=None) -> float:
    """
    One Levy-flight-distributed random step, generated via the Mantegna
    algorithm (Mantegna, 1994). Levy flights are heavy-tailed: mostly
    small steps, with occasional much larger jumps. That combination is
    exactly what a foraging search needs -- fine-grained local search
    most of the time, with rare long-range jumps that let the population
    escape a local optimum instead of getting stuck circling it. This is
    the same mechanism used in Cuckoo Search and other nature-inspired
    metaheuristics, applied here to give MOA's exploration phase a
    genuine mathematical basis instead of flat/uniform random noise.

    sigma_u is derived from the Gamma function so that the resulting
    u/|v|^(1/beta) ratio is distributed approximately as a symmetric
    beta-stable (Levy) distribution:

        sigma_u = [ Gamma(1+beta) * sin(pi*beta/2) /
                     (Gamma((1+beta)/2) * beta * 2^((beta-1)/2)) ] ^ (1/beta)
        u ~ N(0, sigma_u^2),  v ~ N(0, 1)
        step = u / |v|^(1/beta)
    """
    rng = rng or random
    numerator = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    denominator = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma_u = (numerator / denominator) ** (1 / beta)
    u = rng.gauss(0, sigma_u)
    v = rng.gauss(0, 1)
    v = v if abs(v) > 1e-12 else 1e-12
    return u / (abs(v) ** (1 / beta))


def run_meerkat_search(fitness_fn, lower_bound, upper_bound, population_size=12,
                        max_iterations=60, stagnation_limit=12, max_reinits=2,
                        sentry_pull_probability=0.45, seed=None):
    """
    Reusable Meerkat Optimization Algorithm core: maximizes an arbitrary
    scalar fitness_fn(x) over [lower_bound, upper_bound]. Decoupled from
    any one problem so the SAME search dynamics can be judged against
    different objectives -- meerkat_chemical_reduction() below uses it
    with the dose-stability fitness, and
    backend.optimizer_comparison.run_moa() uses it with the identical
    minimize-dose objective PSO/GA are scored against, so the four-way
    comparison is judging search quality, not different problems.

    Two-phase dynamics per iteration (progress = iteration / max_iterations):
      - Exploration weight fades from ~0.8 to ~0.1 as the run matures --
        early on, most moves are Levy-flight jumps (global search);
        later, most moves are "sentry pulls" toward the current best
        (local refinement), the same explore-then-exploit shape used in
        simulated annealing's cooling schedule, just parameterized
        differently.
      - Step size also shrinks over the run (from 0.2x the search span
        down to 0.08x), so late-stage Levy jumps are smaller too.
    Stagnation handling: instead of just stopping when the best score
    hasn't improved in `stagnation_limit` iterations, the worst half of
    the population is reseeded at random (a "predator scare" scatter,
    in meerkat-behavior terms) up to `max_reinits` times before the
    search actually gives up early -- this gives it real chances to
    escape a bad basin rather than converging prematurely.
    """
    rng = random.Random(seed) if seed is not None else random
    if upper_bound <= lower_bound:
        only = lower_bound
        return {
            "best_solution": only, "best_fitness": fitness_fn(only),
            "iterations": 0, "converged_early": False, "reinit_count": 0,
        }

    span = upper_bound - lower_bound
    population = [rng.uniform(lower_bound, upper_bound) for _ in range(population_size)]
    fitness = [fitness_fn(ind) for ind in population]

    best_idx = fitness.index(max(fitness))
    best_solution = population[best_idx]
    best_fitness = fitness[best_idx]
    stagnant_iterations = 0
    reinit_count = 0
    iteration = 0
    converged_early = False

    for iteration in range(max_iterations):
        progress = iteration / max(1, max_iterations - 1)
        explore_weight = 0.8 - 0.7 * progress
        step_size = span * (0.2 - 0.12 * progress)

        for i in range(population_size):
            individual = population[i]
            roll = rng.random()
            if roll < explore_weight:
                candidate = individual + _mantegna_levy_step(rng=rng) * step_size
            elif roll < explore_weight + sentry_pull_probability:
                pull = rng.uniform(0.2, 0.6)
                candidate = individual + pull * (best_solution - individual)
            else:
                candidate = individual + rng.uniform(-1, 1) * step_size

            candidate = min(max(candidate, lower_bound), upper_bound)
            candidate_fitness = fitness_fn(candidate)

            if candidate_fitness >= fitness[i]:
                population[i] = candidate
                fitness[i] = candidate_fitness
            if fitness[i] > best_fitness:
                best_solution = population[i]
                best_fitness = fitness[i]
                stagnant_iterations = -1  # reset to 0 below

        # Elitism: guarantee the best-known solution survives.
        worst_idx = fitness.index(min(fitness))
        population[worst_idx] = best_solution
        fitness[worst_idx] = best_fitness

        stagnant_iterations += 1

        if stagnant_iterations >= stagnation_limit:
            if reinit_count < max_reinits:
                ranked = sorted(range(population_size), key=lambda idx: fitness[idx])
                for idx in ranked[: max(1, population_size // 2)]:
                    population[idx] = rng.uniform(lower_bound, upper_bound)
                    fitness[idx] = fitness_fn(population[idx])
                reinit_count += 1
                stagnant_iterations = 0
            else:
                converged_early = True
                break

    return {
        "best_solution": best_solution,
        "best_fitness": best_fitness,
        "iterations": iteration + 1,
        "converged_early": converged_early,
        "reinit_count": reinit_count,
    }


def _yield_response_ratio(dose_ratio, initial_dose, chemical_category="fertilizer"):
    """
    Non-linear dose-response curve (0..1).

    Fertilizers/pH correctors have diminishing returns -- cutting the last
    10% of dose barely hurts yield. Pest-control chemicals (pesticide,
    fungicide, herbicide) behave differently: protection tends to collapse
    fairly sharply once dose drops below a "minimum effective" threshold,
    rather than degrading gradually. Treating both the same way (as this
    function used to) understates the risk of under-applying pest control.
    """
    dose_ratio = max(0.0, min(1.0, dose_ratio))

    if chemical_category == "pest_control":
        # Logistic (threshold-like) curve: protection holds up reasonably
        # well until dose drops below ~70% of the full rate, then falls
        # off sharply -- steeper than the fertilizer curve below the
        # threshold, meaning less room for the optimizer to cut safely.
        k = 10.0
        midpoint = 0.7
        raw = lambda x: 1.0 / (1.0 + np.exp(-k * (x - midpoint)))
        y0, y1 = raw(0.0), raw(1.0)
        if y1 - y0 <= 0:
            return dose_ratio
        return float((raw(dose_ratio) - y0) / (y1 - y0))

    # Fertilizer / pH corrector (default): smooth diminishing-returns curve.
    curve_k = 1.8 + min(1.6, initial_dose / 120.0)  # ~1.8 to ~3.4
    denom = 1.0 - np.exp(-curve_k)
    if denom <= 0:
        return dose_ratio
    return float((1.0 - np.exp(-curve_k * dose_ratio)) / denom)


def _min_ratio_for_target(initial_dose, target_yield, chemical_category="fertilizer"):
    """Numerically find minimum dose ratio that meets target yield response."""
    target = max(0.5, min(0.99, target_yield))
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        y = _yield_response_ratio(mid, initial_dose, chemical_category)
        if y >= target:
            hi = mid
        else:
            lo = mid
    return hi


def meerkat_chemical_reduction(initial_chemical, target_yield_percentage=0.95, chemical_category="fertilizer"):
    """
    Meerkat optimization algorithm for CHEMICAL REDUCTION
    Finds minimum effective chemical dose while maintaining target yield
    
    Args:
        initial_chemical: Initial chemical amount (kg/acre)
        target_yield_percentage: Target yield percentage to maintain (0-1)
        chemical_category: "fertilizer" (default, smooth diminishing-returns
            response) or "pest_control" (pesticide/fungicide/herbicide --
            steeper threshold-like response, more conservative safety
            floor). Use chemical_composition.infer_chemical_category() to
            derive this from a chemical name automatically.
    
    Returns:
        Dictionary with optimal chemical dose and reduction details
    """
    
    if initial_chemical <= 0:
        return {
            "optimal_chemical_dose": 0.0,
            "reduction_amount": 0.0,
            "reduction_percentage": 0.0,
            "initial_dose": 0.0,
            "efficiency_score": 0.0,
            "target_yield_maintained": f"{target_yield_percentage * 100:.0f}%",
            "iterations_optimized": 0,
            "chemical_category": chemical_category,
            "recommendation": "Initial dose must be greater than 0."
        }

    # Population-based optimization
    population_size = 12
    max_iterations = 60

    # Enforce a yield-safe lower bound using a non-linear response model.
    min_ratio_for_target = _min_ratio_for_target(initial_chemical, target_yield_percentage, chemical_category)
    # Keep a practical buffer for field uncertainty -- pest control gets a
    # larger, more conservative buffer than fertilizer, since under-dosing
    # it risks an outbreak rather than a gradual yield dip.
    if chemical_category == "pest_control":
        min_ratio = max(0.55, min(0.97, min_ratio_for_target * 1.05))
    else:
        min_ratio = max(0.35, min(0.95, min_ratio_for_target * 0.95))
    lower_bound = initial_chemical * min_ratio

    fitness_fn = lambda dose: evaluate_chemical_fitness(
        dose, initial_chemical, target_yield_percentage, chemical_category
    )
    result = run_meerkat_search(
        fitness_fn, lower_bound, initial_chemical,
        population_size=population_size, max_iterations=max_iterations,
    )
    best_solution = result["best_solution"]
    best_fitness = result["best_fitness"]

    # Calculate reduction percentage
    reduction_percentage = ((initial_chemical - best_solution) / initial_chemical * 100) if initial_chemical > 0 else 0

    return {
        "optimal_chemical_dose": round(best_solution, 3),
        "reduction_amount": round(initial_chemical - best_solution, 3),
        "reduction_percentage": round(reduction_percentage, 1),
        "initial_dose": initial_chemical,
        "efficiency_score": round(best_fitness * 100, 1),
        "target_yield_maintained": f"{target_yield_percentage * 100:.0f}%",
        "iterations_optimized": result["iterations"],
        "converged_early": result["converged_early"],
        "reinit_count": result["reinit_count"],
        "chemical_category": chemical_category,
        "recommendation": f"Reduce chemical from {initial_chemical:.2f} kg/acre to {best_solution:.2f} kg/acre"
    }


def evaluate_chemical_fitness(chemical_dose, initial_dose, target_yield, chemical_category="fertilizer"):
    """
    Fitness function: Higher score = better (more reduction + stable yield)
    """
    if initial_dose <= 0 or chemical_dose <= 0:
        return 0  # No chemical = no yield

    # Non-linear diminishing-return yield response
    dose_ratio = chemical_dose / initial_dose
    yield_achieved = _yield_response_ratio(dose_ratio, initial_dose, chemical_category)

    # Reduction benefit (0..1), but do not over-prioritize it.
    reduction_benefit = 1.0 - dose_ratio

    # Strong penalty if dose cannot maintain target yield.
    if yield_achieved < target_yield:
        shortfall = target_yield - yield_achieved
        penalty = shortfall * 5.0
    else:
        penalty = 0.0

    # Prefer stability around target-yield band while still reducing input.
    surplus_penalty = max(0.0, yield_achieved - (target_yield + 0.03)) * 0.6
    fitness = (yield_achieved * 0.7) + (reduction_benefit * 0.3) - penalty - surplus_penalty
    return max(0.0, fitness)


# Crop names supported by predict_crop_yield() / predict_crop_yield_ml() /
# meerkat_chemical_reduction() -- kept as a standalone list (rather than
# introspecting the function's internal dict) so external callers, like
# backend/api.py, have a stable, explicit list to validate/enumerate
# against without depending on predict_crop_yield()'s internals.
SUPPORTED_CROPS = [
    "Rice", "Cotton", "Groundnut", "Sugarcane", "Maize", "Sorghum", "Pulses",
    "Tobacco", "Tomato", "Brinjal", "Onion", "Chilli", "Bhindi", "Cabbage",
    "Cauliflower", "Paddy", "Turmeric", "Red Gram", "Green Gram", "Black Gram",
    "Bengal Gram", "Bajra", "Sesame", "Castor", "Soybean", "Sunflower",
]


def predict_crop_yield(crop_type, ph, moisture, temperature, nitrogen, phosphorus, potassium, rainfall):
    """
    YIELD PREDICTION using Meerkat-optimized parameters
    Predicts crop yield based on soil and weather conditions

    ============================================================
    RULE-BASED ESTIMATE (not trained ML): predicts yield using
    hand-authored optimal ranges per crop and a fixed weighted formula.
    No training data or learned parameters -- kept for transparency and
    comparison against the trained model below (predict_crop_yield_ml).
    ============================================================
    
    Args:
        crop_type: Type of crop
        ph: Soil pH
        moisture: Soil moisture (%)
        temperature: Average temperature (°C)
        nitrogen: Available nitrogen (kg/acre)
        phosphorus: Available phosphorus (kg/acre)
        potassium: Available potassium (kg/acre)
        rainfall: Total rainfall (mm)
    
    Returns:
        Dictionary with yield prediction and contributing factors
    """
    
    # Yield potential by crop (kg/acre under optimal conditions)
    crop_potential = {
        "Rice": 4500,
        "Cotton": 1800,
        "Groundnut": 1200,
        "Sugarcane": 65000,  # Higher because measured differently
        "Maize": 3000,
        "Sorghum": 1500,
        "Pulses": 1000,
        "Tobacco": 2000,
        "Tomato": 9000,
        "Brinjal": 7000,
        "Onion": 8000,
        "Chilli": 1800,
        "Bhindi": 4500,
        "Cabbage": 10000,
        "Cauliflower": 8500,
        # -- Telangana priority crops (added: not an exhaustive national list,
        # these are the crops that actually dominate Telangana cropped area
        # per PJTSAU / district agriculture data) --
        "Paddy": 4500,          # alias of Rice -- same crop, local name
        "Turmeric": 2200,       # Nizamabad is India's largest turmeric market
        "Red Gram": 500,        # Pigeon pea/Tur -- major Telangana pulse
        "Green Gram": 350,      # Moong
        "Black Gram": 350,      # Urad
        "Bengal Gram": 500,     # Chickpea, rabi season
        "Bajra": 800,           # Pearl millet
        "Sesame": 250,          # Til/Gingelly
        "Castor": 600,
        "Soybean": 900,
        "Sunflower": 600,
    }
    
    base_yield = crop_potential.get(crop_type, 2000)
    
    # Calculate reduction factors (0 to 1, where 1 = optimal)
    
    # 1. pH Factor
    crop_ph_range = {
        "Rice": (5.5, 7.0),
        "Cotton": (6.0, 7.5),
        "Groundnut": (5.5, 6.5),
        "Sugarcane": (6.0, 8.0),
        "Maize": (6.0, 7.5),
        "Sorghum": (6.0, 8.0),
        "Pulses": (6.0, 7.0),
        "Tobacco": (5.5, 6.8),
        "Tomato": (6.0, 7.0),
        "Brinjal": (5.8, 6.8),
        "Onion": (6.0, 7.2),
        "Chilli": (6.0, 7.0),
        "Bhindi": (6.0, 7.5),
        "Cabbage": (6.0, 7.5),
        "Cauliflower": (6.0, 7.0),
        "Paddy": (5.5, 7.0),
        "Turmeric": (6.0, 7.5),
        "Red Gram": (6.0, 7.5),
        "Green Gram": (6.2, 7.2),
        "Black Gram": (6.0, 7.5),
        "Bengal Gram": (6.0, 7.5),
        "Bajra": (6.5, 8.0),
        "Sesame": (6.0, 7.5),
        "Castor": (6.0, 7.5),
        "Soybean": (6.0, 7.5),
        "Sunflower": (6.0, 7.5),
    }
    ph_min, ph_max = crop_ph_range.get(crop_type, (6.0, 7.5))
    if ph_min <= ph <= ph_max:
        ph_factor = 1.0
    else:
        deviation = max(abs(ph - ph_min), abs(ph - ph_max))
        ph_factor = max(0.5, 1 - (deviation * 0.2))
    
    # 2. Moisture Factor
    crop_moisture_range = {
        "Rice": (40, 60),
        "Cotton": (25, 40),
        "Groundnut": (15, 30),
        "Sugarcane": (40, 60),
        "Maize": (20, 35),
        "Sorghum": (15, 30),
        "Pulses": (20, 35),
        "Tobacco": (20, 35),
        "Tomato": (30, 45),
        "Brinjal": (30, 45),
        "Onion": (25, 40),
        "Chilli": (25, 40),
        "Bhindi": (25, 40),
        "Cabbage": (35, 50),
        "Cauliflower": (35, 50),
        "Paddy": (40, 60),
        "Turmeric": (30, 45),
        "Red Gram": (20, 30),
        "Green Gram": (20, 30),
        "Black Gram": (20, 30),
        "Bengal Gram": (15, 25),
        "Bajra": (15, 25),
        "Sesame": (20, 30),
        "Castor": (20, 30),
        "Soybean": (25, 35),
        "Sunflower": (25, 35),
    }
    moist_min, moist_max = crop_moisture_range.get(crop_type, (25, 40))
    if moist_min <= moisture <= moist_max:
        moisture_factor = 1.0
    else:
        moisture_factor = max(0.4, 1 - (abs(moisture - ((moist_min + moist_max) / 2)) / 20))
    
    # 3. Temperature Factor
    crop_temp_range = {
        "Rice": (20, 30),
        "Cotton": (25, 35),
        "Groundnut": (20, 30),
        "Sugarcane": (22, 32),
        "Maize": (22, 28),
        "Sorghum": (20, 32),
        "Pulses": (18, 25),
        "Tobacco": (20, 28),
        "Tomato": (18, 30),
        "Brinjal": (20, 32),
        "Onion": (13, 28),
        "Chilli": (20, 32),
        "Bhindi": (22, 35),
        "Cabbage": (15, 25),
        "Cauliflower": (15, 25),
        "Paddy": (20, 30),
        "Turmeric": (20, 30),
        "Red Gram": (20, 30),
        "Green Gram": (25, 35),
        "Black Gram": (25, 35),
        "Bengal Gram": (15, 25),
        "Bajra": (25, 35),
        "Sesame": (25, 35),
        "Castor": (20, 35),
        "Soybean": (20, 30),
        "Sunflower": (20, 30),
    }
    temp_min, temp_max = crop_temp_range.get(crop_type, (20, 30))
    if temp_min <= temperature <= temp_max:
        temp_factor = 1.0
    else:
        temp_factor = max(0.4, 1 - (abs(temperature - ((temp_min + temp_max) / 2)) / 15))
    
    # 4. Nutrient Factor (NPK balance)
    optimal_npk = {
        "Rice": (100, 50, 40),
        "Cotton": (60, 40, 40),
        "Groundnut": (20, 40, 40),
        "Sugarcane": (120, 60, 60),
        "Maize": (120, 60, 60),
        "Sorghum": (60, 30, 40),
        "Pulses": (20, 40, 20),
        "Tobacco": (80, 40, 120),
        "Tomato": (100, 50, 50),
        "Brinjal": (100, 50, 50),
        "Onion": (80, 40, 40),
        "Chilli": (80, 40, 40),
        "Bhindi": (60, 30, 30),
        "Cabbage": (100, 50, 50),
        "Cauliflower": (100, 50, 50),
        "Paddy": (100, 50, 40),
        "Turmeric": (60, 30, 60),
        "Red Gram": (20, 50, 20),   # legume -- fixes own nitrogen, needs less N
        "Green Gram": (20, 40, 20), # legume
        "Black Gram": (20, 40, 20), # legume
        "Bengal Gram": (20, 50, 20),# legume
        "Bajra": (60, 30, 30),
        "Sesame": (40, 20, 20),
        "Castor": (60, 40, 40),
        "Soybean": (30, 60, 40),    # legume, but higher P need
        "Sunflower": (60, 60, 40),
    }
    opt_n, opt_p, opt_k = optimal_npk.get(crop_type, (60, 40, 40))
    
    # Calculate nutrient sufficiency (0 to 1)
    n_factor = min(1.0, nitrogen / opt_n) if opt_n > 0 else 0.5
    p_factor = min(1.0, phosphorus / opt_p) if opt_p > 0 else 0.5
    k_factor = min(1.0, potassium / opt_k) if opt_k > 0 else 0.5
    
    nutrient_factor = (n_factor * 0.5) + (p_factor * 0.25) + (k_factor * 0.25)
    
    # 5. Rainfall Factor
    crop_rainfall = {
        "Rice": (1200, 1600),
        "Cotton": (600, 1100),
        "Groundnut": (400, 700),
        "Sugarcane": (1200, 2500),
        "Maize": (500, 1000),
        "Sorghum": (400, 900),
        "Pulses": (400, 800),
        "Tobacco": (600, 1000),
        "Tomato": (600, 1200),
        "Brinjal": (600, 1200),
        "Onion": (500, 1000),
        "Chilli": (600, 1200),
        "Bhindi": (500, 1000),
        "Cabbage": (700, 1400),
        "Cauliflower": (700, 1400),
        "Paddy": (1200, 1600),
        "Turmeric": (1000, 1500),
        "Red Gram": (600, 1000),
        "Green Gram": (400, 700),
        "Black Gram": (400, 700),
        "Bengal Gram": (400, 600),
        "Bajra": (400, 700),
        "Sesame": (500, 800),
        "Castor": (500, 800),
        "Soybean": (600, 1000),
        "Sunflower": (500, 750),
    }
    rain_min, rain_max = crop_rainfall.get(crop_type, (600, 1000))
    if rain_min <= rainfall <= rain_max:
        rainfall_factor = 1.0
    else:
        rainfall_factor = max(0.5, 1 - (abs(rainfall - ((rain_min + rain_max) / 2)) / 500))
    
    # Combine all factors (Meerkat optimization: weighted average)
    combined_factor = (
        ph_factor * 0.15 +
        moisture_factor * 0.20 +
        temp_factor * 0.15 +
        nutrient_factor * 0.35 +
        rainfall_factor * 0.15
    )
    
    # Calculate predicted yield
    predicted_yield = base_yield * combined_factor
    
    # Yield confidence (lower if factors are far from optimal)
    confidence = int(combined_factor * 100)
    
    return {
        "crop": crop_type,
        "base_yield_potential": base_yield,
        "predicted_yield": round(predicted_yield, 1),
        "combined_efficiency": round(combined_factor * 100, 1),
        "confidence_score": confidence,
        "factors": {
            "pH": round(ph_factor * 100, 1),
            "Moisture": round(moisture_factor * 100, 1),
            "Temperature": round(temp_factor * 100, 1),
            "Nutrients": round(nutrient_factor * 100, 1),
            "Rainfall": round(rainfall_factor * 100, 1)
        },
        "limiting_factor": get_limiting_factor({
            "pH": ph_factor,
            "Moisture": moisture_factor,
            "Temperature": temp_factor,
            "Nutrients": nutrient_factor,
            "Rainfall": rainfall_factor
        }),
        "recommendations": generate_yield_recommendations({
            "pH": (ph_factor, ph, ph_min, ph_max),
            "Moisture": (moisture_factor, moisture, moist_min, moist_max),
            "Temperature": (temp_factor, temperature, temp_min, temp_max),
            "Nutrients": (nutrient_factor, nitrogen, phosphorus, potassium),
            "Rainfall": (rainfall_factor, rainfall, rain_min, rain_max)
        })
    }


def get_limiting_factor(factors):
    """Find the factor limiting yield the most"""
    min_factor = min(factors.values())
    for factor_name, value in factors.items():
        if value == min_factor:
            return factor_name
    return "Balanced"


def generate_yield_recommendations(factor_details):
    """Generate specific recommendations based on factors"""
    recommendations = []
    
    ph_factor, ph, ph_min, ph_max = factor_details["pH"]
    if ph_factor < 0.9:
        if ph < ph_min:
            recommendations.append(f"Increase soil pH to {ph_min} (currently {ph})")
        else:
            recommendations.append(f"Decrease soil pH to {ph_max} (currently {ph})")
    
    moist_factor, moisture, moist_min, moist_max = factor_details["Moisture"]
    if moist_factor < 0.9:
        if moisture < moist_min:
            recommendations.append(f"Increase irrigation to reach {moist_min}% moisture")
        else:
            recommendations.append(f"Improve drainage to reduce moisture to {moist_max}%")
    
    temp_factor, temp, temp_min, temp_max = factor_details["Temperature"]
    if temp_factor < 0.9:
        recommendations.append(f"Sowing time adjustment: Target {temp_min}-{temp_max}°C range")
    
    nutrient_factor, n, p, k = factor_details["Nutrients"]
    if nutrient_factor < 0.9:
        recommendations.append("Increase NPK fertilization")
    
    rainfall_factor, rain, rain_min, rain_max = factor_details["Rainfall"]
    if rainfall_factor < 0.9:
        if rain < rain_min:
            recommendations.append(f"Plan irrigation for deficit ({rain_min}mm target)")
        else:
            recommendations.append("Plan for excess water management")
    
    return recommendations if recommendations else ["Conditions are optimal for this crop"]


def compare_actual_vs_predicted_yield(predicted_yield, actual_yield, crop_type, farm_area):
    """
    Compare actual yield with AI-predicted yield using Meerkat analysis
    Identify variance and optimization opportunities
    """
    
    if predicted_yield == 0:
        variance_percentage = 0
    else:
        variance_percentage = ((actual_yield - predicted_yield) / predicted_yield) * 100
    
    variance_amount = actual_yield - predicted_yield
    
    # Classify variance
    if abs(variance_percentage) <= 5:
        variance_class = "Excellent"
        variance_emoji = "🟢"
        efficiency = 95
    elif abs(variance_percentage) <= 15:
        variance_class = "Good"
        variance_emoji = "🟡"
        efficiency = 85
    elif variance_percentage < -20:
        variance_class = "Under-yield"
        variance_emoji = "🔴"
        efficiency = max(50, 100 + variance_percentage)
    else:
        variance_class = "Over-yield"
        variance_emoji = "🔵"
        efficiency = min(120, 100 + variance_percentage)
    
    # Calculate total production
    total_actual_production = (actual_yield * farm_area) / 1000  # Convert to tons
    total_predicted_production = (predicted_yield * farm_area) / 1000
    total_variance_production = variance_amount * farm_area / 1000
    
    # Identify limiting factors in actual yield
    limiting_reasons = []
    if variance_percentage < -10:
        limiting_reasons.append("Under-optimal nutrient management")
        limiting_reasons.append("Possible pest/disease pressure")
        limiting_reasons.append("Water stress or irrigation timing issues")
    elif variance_percentage > 20:
        limiting_reasons.append("Excellent field management")
        limiting_reasons.append("Favorable weather conditions")
        limiting_reasons.append("Optimal nutrient availability")
    
    analysis = {
        "predicted_yield": round(predicted_yield, 2),
        "actual_yield": round(actual_yield, 2),
        "variance_amount": round(variance_amount, 2),
        "variance_percentage": round(variance_percentage, 2),
        "variance_class": variance_class,
        "variance_emoji": variance_emoji,
        "efficiency_score": round(efficiency, 1),
        "total_predicted_production_tons": round(total_predicted_production, 2),
        "total_actual_production_tons": round(total_actual_production, 2),
        "total_variance_tons": round(total_variance_production, 2),
        "farm_area_acres": farm_area,
        "limiting_factors": limiting_reasons,
        "optimization_potential": max(0, 100 - efficiency - variance_percentage),
        "crop": crop_type
    }
    
    return analysis


def meerkat_yield_optimization_recommendations(variance_analysis, optimization_budget=None):
    """
    Generate Meerkat-based recommendations to close the yield gap
    Prioritized by cost-effectiveness
    """
    
    efficiency = variance_analysis["efficiency_score"]
    variance_pct = variance_analysis["variance_percentage"]
    crop = variance_analysis["crop"]
    
    recommendations = []
    
    # Tier 1: High-impact, low-cost recommendations
    if variance_pct < -20:
        recommendations.append({
            "priority": "HIGH",
            "cost": "LOW",
            "estimated_yield_gain": 15,
            "action": "Soil Health Assessment",
            "details": "Conduct comprehensive soil testing for micro-nutrients (B, Zn, Fe, Cu)",
            "expected_gain_kg_acre": 150
        })
        
        recommendations.append({
            "priority": "HIGH",
            "cost": "LOW",
            "estimated_yield_gain": 12,
            "action": "IPM Implementation",
            "details": "Monitor and implement Integrated Pest Management practices",
            "expected_gain_kg_acre": 120
        })
    
    if variance_pct < -10:
        recommendations.append({
            "priority": "HIGH",
            "cost": "MEDIUM",
            "estimated_yield_gain": 20,
            "action": "Irrigation Optimization",
            "details": "Install drip/sprinkler irrigation with soil moisture sensors",
            "expected_gain_kg_acre": 200
        })
        
        recommendations.append({
            "priority": "MEDIUM",
            "cost": "LOW",
            "estimated_yield_gain": 8,
            "action": "Timely Application",
            "details": "Strict adherence to recommended crop calendar",
            "expected_gain_kg_acre": 80
        })
    
    # Tier 2: Maintenance recommendations
    if variance_pct >= -10 and efficiency < 90:
        recommendations.append({
            "priority": "MEDIUM",
            "cost": "LOW",
            "estimated_yield_gain": 5,
            "action": "Crop Residue Management",
            "details": "Mulching and proper residue incorporation",
            "expected_gain_kg_acre": 50
        })
    
    # Tier 3: Advanced optimizations
    if efficiency > 90:
        recommendations.append({
            "priority": "LOW",
            "cost": "MEDIUM",
            "estimated_yield_gain": 3,
            "action": "Precision Agriculture",
            "details": "Variable rate application of inputs based on soil mapping",
            "expected_gain_kg_acre": 30
        })
    
    return recommendations


def calculate_yield_recovery_potential(variance_analysis, all_recommendations):
    """
    Calculate potential yield recovery by implementing recommendations
    Uses Meerkat population-based optimization
    """
    
    current_efficiency = variance_analysis["efficiency_score"]
    current_yield = variance_analysis["actual_yield"]
    
    if current_efficiency >= 95:
        return {
            "current_efficiency": current_efficiency,
            "recovery_potential": 3,
            "optimized_efficiency": 98,
            "additional_yield": round(current_yield * 0.03, 2),
            "status": "Optimal Performance",
            "message": "Field is performing near maximum. Focus on maintenance."
        }
    
    # Calculate cumulative impact of recommendations
    total_potential_gain = sum([rec["estimated_yield_gain"] for rec in all_recommendations[:3]])
    
    optimized_yield = current_yield * (1 + total_potential_gain / 100)
    optimized_efficiency = min(98, current_efficiency + total_potential_gain)
    additional_yield = optimized_yield - current_yield
    
    return {
        "current_efficiency": round(current_efficiency, 1),
        "recovery_potential": round(total_potential_gain, 1),
        "optimized_efficiency": round(optimized_efficiency, 1),
        "current_yield": round(current_yield, 2),
        "optimized_yield": round(optimized_yield, 2),
        "additional_yield": round(additional_yield, 2),
        "max_recommendations": len(all_recommendations),
        "status": "Optimization Available" if total_potential_gain > 5 else "Minor Improvements Possible",
        "priority_actions": [rec["action"] for rec in all_recommendations[:3]]
    }


# ============================================================
# MACHINE LEARNING: Trained Random Forest yield regressor
#
# Unlike predict_crop_yield() above (hand-written rules) and
# meerkat_chemical_reduction() (a run-time metaheuristic search), this
# section trains an actual scikit-learn model on historical rows in
# dataset/telangana_soil_data.csv and learns its own patterns from data.
# ============================================================

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "yield_rf_model.pkl")
_DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "telangana_soil_data.csv")

_ML_FEATURE_COLUMNS = [
    "Soil_Type", "Crop", "pH", "Nitrogen", "Phosphorus", "Potassium",
    "Organic_Carbon", "Soil_Moisture", "Electrical_Conductivity",
]


def train_yield_model(dataset_path=None, save=True):
    """
    Train a RandomForestRegressor on the Telangana soil dataset to predict
    Yield_kg_per_acre from soil + crop features. This is the trained-ML
    counterpart to the rule-based predict_crop_yield() above -- it learns
    its coefficients from data instead of having them hand-specified.

    Returns (fitted_pipeline, metrics_dict).
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    path = dataset_path or _DATASET_PATH
    df = pd.read_csv(path)

    if "Yield_kg_per_acre" not in df.columns:
        raise ValueError(
            "Dataset is missing a 'Yield_kg_per_acre' column needed to train the ML model."
        )

    X = df[_ML_FEATURE_COLUMNS]
    y = df["Yield_kg_per_acre"]

    categorical = ["Soil_Type", "Crop"]
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough",
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    metrics = {
        "mae_kg_per_acre": round(float(mean_absolute_error(y_test, preds)), 1),
        "r2_score": round(float(r2_score(y_test, preds)), 3),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    if save:
        with open(_MODEL_PATH, "wb") as f:
            pickle.dump(pipeline, f)

    return pipeline, metrics


def _load_yield_model():
    """Load the cached trained model, training it on first use if missing."""
    if os.path.exists(_MODEL_PATH):
        with open(_MODEL_PATH, "rb") as f:
            return pickle.load(f)
    pipeline, _ = train_yield_model(save=True)
    return pipeline


def get_model_metrics(dataset_path=None, cv_folds=5):
    """
    Full evaluation report for the trained yield model: MAE, RMSE, R^2 on
    a held-out test split, plus k-fold cross-validation R^2 (mean/std) on
    the full dataset. Addresses the advisory gap "Uncertainty and model
    failure are not addressed" by giving a reviewer/agronomist a single
    place to see how trustworthy the model actually is, instead of just
    trusting the prediction on screen.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    path = dataset_path or _DATASET_PATH
    df = pd.read_csv(path)
    X = df[_ML_FEATURE_COLUMNS]
    y = df["Yield_kg_per_acre"]

    categorical = ["Soil_Type", "Crop"]
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough",
    )
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    cv_scores = cross_val_score(
        pipeline, X, y, cv=min(cv_folds, max(2, len(df) // 10)), scoring="r2"
    )

    return {
        "mae_kg_per_acre": round(mae, 1),
        "rmse_kg_per_acre": round(rmse, 1),
        "r2_score": round(r2, 3),
        "cv_folds": int(len(cv_scores)),
        "cv_r2_mean": round(float(np.mean(cv_scores)), 3),
        "cv_r2_std": round(float(np.std(cv_scores)), 3),
        "test_rows": int(len(X_test)),
        "training_rows": int(len(X_train)),
        "model_type": "RandomForestRegressor (scikit-learn)",
    }


def predict_crop_yield_ml(crop_type, soil_type, ph, nitrogen, phosphorus, potassium,
                           organic_carbon, moisture, electrical_conductivity):
    """
    Predict yield using the trained Random Forest model, as opposed to the
    hand-written formula in predict_crop_yield(). Returns available=False
    with an error message if scikit-learn isn't installed or training fails,
    so callers can fall back to the rule-based estimate gracefully.
    """
    try:
        model = _load_yield_model()
    except Exception as e:
        return {"available": False, "error": str(e)}

    row = pd.DataFrame([{
        "Soil_Type": soil_type,
        "Crop": crop_type,
        "pH": ph,
        "Nitrogen": nitrogen,
        "Phosphorus": phosphorus,
        "Potassium": potassium,
        "Organic_Carbon": organic_carbon,
        "Soil_Moisture": moisture,
        "Electrical_Conductivity": electrical_conductivity,
    }])[_ML_FEATURE_COLUMNS]

    try:
        prediction = float(model.predict(row)[0])

        # Pseudo-confidence from tree agreement: tighter spread across the
        # forest's individual trees means the model is more confident.
        rf = model.named_steps["rf"]
        x_transformed = model.named_steps["preprocess"].transform(row)
        tree_preds = np.array([t.predict(x_transformed) for t in rf.estimators_]).flatten()
        spread = float(np.std(tree_preds))
        confidence = max(50.0, 100.0 - (spread / max(1.0, prediction)) * 100.0)
    except Exception as e:
        return {"available": False, "error": str(e)}

    return {
        "available": True,
        "predicted_yield": round(prediction, 1),
        "confidence_score": round(confidence, 1),
        "model_type": "RandomForestRegressor (scikit-learn, trained)",
        "tree_prediction_std_kg_acre": round(spread, 1),
    }
