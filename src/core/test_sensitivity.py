import os
import sys
import json

# Ensure the local src directory is importable for tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.bsm_calculator import sensitivity_test


def test_sensitivity_length_and_fields():
    """Run sensitivity_test and verify returned structure and basic fields."""
    res = sensitivity_test("call", 100.0, 100.0, 1.0, 0.01, 0.2)
    data = json.loads(res)

    assert isinstance(data, list), "sensitivity_test should return a JSON list"
    # expected spot changes from -0.025 to +0.025 step 0.005 => 11 entries
    assert len(data) == 11, f"expected 11 sensitivity entries, got {len(data)}"

    for entry in data:
        assert 'spot_change' in entry
        # either greeks present or error field
        assert ('price' in entry) or ('error' in entry)
        if 'price' in entry:
            assert isinstance(entry['price'], (float, int))
            # basic presence checks for greek keys
            for k in ('delta', 'gamma', 'vega', 'rho', 'theta'):
                assert k in entry
