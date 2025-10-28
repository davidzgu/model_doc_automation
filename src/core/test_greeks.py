import json
from src.core.bsm_calculator import greeks_calculator


def test_greeks_call_basic():
    # Use parameters where S=K to make d1 = ... reasonable
    res = greeks_calculator("call", 100.0, 100.0, 1.0, 0.01, 0.2)
    data = json.loads(res)
    assert "price" in data
    assert "delta" in data
    assert data["price"] > 0
    assert 0 <= data["delta"] <= 1


def test_greeks_put_call_parity():
    # Check put-call parity: C - P = S - K*e^{-rT}
    call = json.loads(greeks_calculator("call", 100.0, 100.0, 0.5, 0.01, 0.25))
    put = json.loads(greeks_calculator("put", 100.0, 100.0, 0.5, 0.01, 0.25))
    lhs = call["price"] - put["price"]
    rhs = 100.0 - 100.0 * (2.718281828459045 ** (-0.01 * 0.5))
    assert abs(lhs - rhs) < 1e-6
