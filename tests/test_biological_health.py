from backend.biological_health import estimate_biological_activity


def test_no_carbon_reading_returns_unavailable():
    result = estimate_biological_activity(organic_carbon_pct=None)
    assert result["available"] is False


def test_negative_carbon_returns_unavailable():
    result = estimate_biological_activity(organic_carbon_pct=-0.1)
    assert result["available"] is False


def test_high_organic_carbon_gives_high_level():
    result = estimate_biological_activity(organic_carbon_pct=0.9)
    assert result["available"] is True
    assert result["biological_activity_level"] == "High"


def test_moderate_organic_carbon_gives_moderate_level():
    result = estimate_biological_activity(organic_carbon_pct=0.6)
    assert result["biological_activity_level"] == "Moderate"


def test_low_organic_carbon_gives_low_level():
    result = estimate_biological_activity(organic_carbon_pct=0.3)
    assert result["biological_activity_level"] == "Low"


def test_high_carbon_downgraded_by_low_moisture():
    # High carbon alone would be "High" -- low moisture should pull it down,
    # proving this isn't just a re-label of the carbon-only indicator.
    result = estimate_biological_activity(organic_carbon_pct=0.9, moisture=10)
    assert result["biological_activity_level"] == "Moderate"
    assert any("moisture" in n.lower() for n in result["constraint_notes"])


def test_high_carbon_downgraded_by_waterlogging():
    result = estimate_biological_activity(organic_carbon_pct=0.9, moisture=92)
    assert result["biological_activity_level"] == "Moderate"
    assert any("waterlogging" in n.lower() for n in result["constraint_notes"])


def test_high_carbon_downgraded_by_high_ec():
    result = estimate_biological_activity(organic_carbon_pct=0.9, electrical_conductivity=5.0)
    assert result["biological_activity_level"] == "Moderate"
    assert any("ec" in n.lower() for n in result["constraint_notes"])


def test_low_moisture_and_high_ec_together_floor_at_low_not_negative():
    result = estimate_biological_activity(organic_carbon_pct=0.6, moisture=10, electrical_conductivity=5.0)
    assert result["biological_activity_level"] == "Low"
    assert len(result["constraint_notes"]) == 2


def test_normal_moisture_and_ec_do_not_add_constraint_notes():
    result = estimate_biological_activity(organic_carbon_pct=0.6, moisture=40, electrical_conductivity=1.0)
    assert result["constraint_notes"] == []


def test_trend_insufficient_data_without_history():
    result = estimate_biological_activity(organic_carbon_pct=0.6)
    assert result["organic_carbon_trend"] == "insufficient data"


def test_trend_improving_with_rising_history():
    result = estimate_biological_activity(organic_carbon_pct=0.7, carbon_history=[0.4, 0.5, 0.7])
    assert result["organic_carbon_trend"] == "improving"


def test_trend_declining_with_falling_history():
    result = estimate_biological_activity(organic_carbon_pct=0.4, carbon_history=[0.8, 0.6, 0.4])
    assert result["organic_carbon_trend"] == "declining"


def test_trend_stable_with_flat_history():
    result = estimate_biological_activity(organic_carbon_pct=0.6, carbon_history=[0.61, 0.6, 0.6])
    assert result["organic_carbon_trend"] == "stable"


def test_available_n_note_reports_band_not_ratio():
    result = estimate_biological_activity(organic_carbon_pct=0.6, nitrogen=200)
    assert result["available_n_note"] is not None
    assert "low" in result["available_n_note"].lower()
    assert "illustrative" not in result["available_n_note"].lower()  # no unverified numeric ratio is computed


def test_available_n_note_medium_band():
    result = estimate_biological_activity(organic_carbon_pct=0.6, nitrogen=400)
    assert "medium" in result["available_n_note"].lower()


def test_available_n_note_high_band():
    result = estimate_biological_activity(organic_carbon_pct=0.6, nitrogen=700)
    assert "high" in result["available_n_note"].lower()


def test_available_n_note_absent_without_nitrogen():
    result = estimate_biological_activity(organic_carbon_pct=0.6)
    assert result["available_n_note"] is None


def test_no_c_n_ratio_key_exists_anymore():
    result = estimate_biological_activity(organic_carbon_pct=0.6, nitrogen=200)
    assert "c_n_illustrative_note" not in result


def test_c_n_ratio_absent_by_default():
    result = estimate_biological_activity(organic_carbon_pct=0.6, nitrogen=200)
    assert "c_n_ratio" not in result
    assert "c_n_ratio_note" not in result


def test_c_n_ratio_computed_when_lab_total_n_given():
    # OC 0.6% / Total N 0.06% = 10.0
    result = estimate_biological_activity(organic_carbon_pct=0.6, total_nitrogen_pct=0.06)
    assert result["c_n_ratio"] == 10.0
    assert "lab-verified" in result["c_n_ratio_note"].lower()


def test_c_n_ratio_narrow_band_note():
    # OC 0.3 / Total N 0.06 = 5.0 -> narrow
    result = estimate_biological_activity(organic_carbon_pct=0.3, total_nitrogen_pct=0.06)
    assert result["c_n_ratio"] == 5.0
    assert "narrow" in result["c_n_ratio_note"].lower()


def test_c_n_ratio_typical_band_note():
    # OC 0.66 / Total N 0.06 = 11.0 -> typical fertile range
    result = estimate_biological_activity(organic_carbon_pct=0.66, total_nitrogen_pct=0.06)
    assert result["c_n_ratio"] == 11.0
    assert "typical" in result["c_n_ratio_note"].lower()


def test_c_n_ratio_wide_band_note():
    # OC 0.9 / Total N 0.05 = 18.0 -> wide
    result = estimate_biological_activity(organic_carbon_pct=0.9, total_nitrogen_pct=0.05)
    assert result["c_n_ratio"] == 18.0
    assert "wide" in result["c_n_ratio_note"].lower()


def test_c_n_ratio_invalid_zero_total_n():
    result = estimate_biological_activity(organic_carbon_pct=0.6, total_nitrogen_pct=0)
    assert result["c_n_ratio"] is None
    assert "invalid" in result["c_n_ratio_note"].lower()


def test_c_n_ratio_independent_of_available_n_note():
    # Both can be present at once -- lab C:N ratio doesn't suppress the
    # Available-N band, since a caller might want to see both.
    result = estimate_biological_activity(
        organic_carbon_pct=0.6, nitrogen=200, total_nitrogen_pct=0.06
    )
    assert result["c_n_ratio"] == 10.0
    assert result["available_n_note"] is not None


def test_disclaimer_always_present_when_available():
    result = estimate_biological_activity(organic_carbon_pct=0.6)
    assert "does not influence any recommendation" in result["disclaimer"]
