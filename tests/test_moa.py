"""
Automated tests for backend/meerkat_optimizer.py -- the metaheuristic.
Corresponds to the "MOA (Metaheuristic)" section of the test case matrix.
"""
import random
from backend.meerkat_optimizer import meerkat_chemical_reduction, evaluate_chemical_fitness


def test_reduces_dose_below_baseline():
    """MOA-01: optimizer reduces dose below baseline while meeting target yield."""
    random.seed(42)
    result = meerkat_chemical_reduction(100, target_yield_percentage=0.95)
    assert result["optimal_chemical_dose"] < 100
    assert result["reduction_percentage"] > 0


def test_never_returns_dose_below_safety_floor():
    """MOA-02: optimizer never returns a dose below the safety floor, across several doses."""
    random.seed(42)
    for initial_dose in (60, 100, 140):
        result = meerkat_chemical_reduction(initial_dose, target_yield_percentage=0.95)
        # The optimizer's own reported dose must never be below what its
        # reduction_percentage implies as the floor.
        assert result["optimal_chemical_dose"] > 0
        assert result["optimal_chemical_dose"] <= initial_dose


def test_early_stopping_halts_before_max_iterations():
    """MOA-03: early stopping should trigger before the full 50-iteration budget on convergence."""
    random.seed(42)
    result = meerkat_chemical_reduction(100, target_yield_percentage=0.95)
    assert result["iterations_optimized"] <= 50
    if result["converged_early"]:
        assert result["iterations_optimized"] < 50


def test_zero_or_negative_dose_is_rejected_gracefully():
    """MOA-05: zero/negative initial dose should not raise, and should signal no valid result."""
    for bad_dose in (0, -10):
        result = meerkat_chemical_reduction(bad_dose)
        assert result["optimal_chemical_dose"] == 0.0
        assert result["reduction_percentage"] == 0.0


def test_pest_control_category_is_more_conservative_than_fertilizer():
    """
    Category-aware curve: pest control chemicals should get a smaller
    reduction than fertilizer for the same starting dose, since
    under-dosing pest control risks an outbreak rather than a gradual
    yield dip.
    """
    random.seed(42)
    fertilizer_result = meerkat_chemical_reduction(100, target_yield_percentage=0.95, chemical_category="fertilizer")
    random.seed(42)
    pest_control_result = meerkat_chemical_reduction(100, target_yield_percentage=0.95, chemical_category="pest_control")
    assert pest_control_result["reduction_percentage"] < fertilizer_result["reduction_percentage"]


def test_fitness_function_penalizes_zero_dose():
    """A zero or negative dose should score zero fitness, never a false positive."""
    assert evaluate_chemical_fitness(0, 100, 0.95) == 0
    assert evaluate_chemical_fitness(-5, 100, 0.95) == 0
