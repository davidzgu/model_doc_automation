# -*- coding: utf-8 -*-
"""
Test generation tools for Black-Scholes option pricing model validation.

Provides sensitivity analysis, stress testing, and P&L analysis for equity options.
Includes:
  - Sensitivity Tests: Spot price, volatility, and curve sensitivity
  - Stress Tests: Extreme market moves and volatility scenarios
  - P&L Tests: Historical P&L attribution and Greeks-based P&L explain
"""

import json
import os
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import norm

from langchain.tools import tool
from langgraph.prebuilt import InjectedState

from .tool_registry import register_tool
from .utils import load_json_as_df


# ============================================================================
# BLACK-SCHOLES PRICING ENGINE (Helper)
# ============================================================================

def _bsm_price(
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """
    Calculate Black-Scholes price for European option.
    
    Args:
        option_type: 'call' or 'put'
        S: Current spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility (annualized)
    
    Returns:
        Option price (float)
    """
    if T <= 0:
        if option_type.lower() == 'call':
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return price


def _bsm_delta(
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """Calculate option delta (∂Price/∂S)."""
    if T <= 0:
        return 1.0 if option_type.lower() == 'call' and S > K else 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option_type.lower() == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


def _bsm_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """Calculate option gamma (∂²Price/∂S²)."""
    if T <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def _bsm_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """Calculate option vega (∂Price/∂σ, per 1% change)."""
    if T <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100.0  # per 1% vol change


def _bsm_theta(
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """Calculate option theta (∂Price/∂T, per day)."""
    if T <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    theta_annual = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)))
    
    if option_type.lower() == 'call':
        theta_annual -= r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta_annual += r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    return theta_annual / 365.0  # per day


# ============================================================================
# SENSITIVITY TEST
# ============================================================================

def run_sensitivity_test(
    option_data: str,
    output_dir: str = "outputs/tests",
    spot_bumps: str = "[-0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05]",
    vol_bumps: str = "[-0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05]"
) -> str:
    """
    Run sensitivity analysis on option prices.
    
    Tests impact of:
      - Spot price changes (percentage bumps)
      - Volatility changes (absolute bumps)
      - Implied vol curve shifts
    
    Args:
        option_data: JSON string with option parameters
          Required fields: option_type, S, K, T, r, sigma
        output_dir: Directory to save results
        spot_bumps: JSON array of spot price bumps (as percentages)
        vol_bumps: JSON array of volatility bumps (as absolute changes)
    
    Returns:
        JSON with sensitivity results:
        {
            "spot_sensitivity": [...],
            "vol_sensitivity": [...],
            "curve_shift": [...],
            "output_file": "path/to/sensitivity_results.csv"
        }
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Parse inputs
        try:
            option_params = json.loads(option_data)
        except:
            option_params = option_data if isinstance(option_data, dict) else {}
        
        try:
            spot_bumps_list = json.loads(spot_bumps)
        except:
            spot_bumps_list = [-0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05]
        
        try:
            vol_bumps_list = json.loads(vol_bumps)
        except:
            vol_bumps_list = [-0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05]
        
        required_fields = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        if not all(field in option_params for field in required_fields):
            return json.dumps({
                "error": f"Missing required fields. Need: {required_fields}",
                "received": list(option_params.keys())
            })
        asset_class = option_params['asset_class']
        opt_type = option_params['option_type']
        S = float(option_params['S'])
        K = float(option_params['K'])
        T = float(option_params['T'])
        r = float(option_params['r'])
        sigma = float(option_params['sigma'])
        
        # Base price and Greeks
        base_price = _bsm_price(opt_type, S, K, T, r, sigma)
        delta = _bsm_delta(opt_type, S, K, T, r, sigma)
        gamma = _bsm_gamma(S, K, T, r, sigma)
        vega = _bsm_vega(S, K, T, r, sigma)
        theta = _bsm_theta(opt_type, S, K, T, r, sigma)
        
        # Spot sensitivity
        spot_sensitivity = []
        for bump in spot_bumps_list:
            S_bumped = S * (1 + bump)
            price_bumped = _bsm_price(opt_type, S_bumped, K, T, r, sigma)
            pnl = price_bumped - base_price
            pnl_linear = delta * S * bump  # Greeks-based estimate
            
            spot_sensitivity.append({
                "spot_bump_pct": bump,
                "spot_level": S_bumped,
                "price": price_bumped,
                "pnl": pnl,
                "pnl_linear_estimate": pnl_linear,
                "pnl_error": pnl - pnl_linear
            })
        
        # Volatility sensitivity
        vol_sensitivity = []
        for bump in vol_bumps_list:
            sigma_bumped = sigma + bump
            if sigma_bumped <= 0:
                continue
            
            price_bumped = _bsm_price(opt_type, S, K, T, r, sigma_bumped)
            pnl = price_bumped - base_price
            pnl_linear = vega * bump  # Greeks-based estimate (per 1%)
            
            vol_sensitivity.append({
                "vol_bump_pct": bump,
                "vol_level": sigma_bumped,
                "price": price_bumped,
                "pnl": pnl,
                "pnl_linear_estimate": pnl_linear,
                "pnl_error": pnl - pnl_linear
            })
        
        # Curve shift (parallel shift of entire yield curve)
        curve_shift = []
        rate_bumps = [-0.01, -0.005, 0, 0.005, 0.01]
        for bump in rate_bumps:
            r_bumped = r + bump
            price_bumped = _bsm_price(opt_type, S, K, T, r_bumped, sigma)
            pnl = price_bumped - base_price
            
            curve_shift.append({
                "rate_bump_pct": bump,
                "rate_level": r_bumped,
                "price": price_bumped,
                "pnl": pnl
            })
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"sensitivity_test_{asset_class}_{opt_type}.csv")
        
        # Create comprehensive results dataframe
        results_data = {
            "test_type": ["spot_sensitivity"] * len(spot_sensitivity) + 
                         ["vol_sensitivity"] * len(vol_sensitivity) + 
                         ["curve_shift"] * len(curve_shift),
            "parameter": (
                [f"spot_bump_{s['spot_bump_pct']}" for s in spot_sensitivity] +
                [f"vol_bump_{v['vol_bump_pct']}" for v in vol_sensitivity] +
                [f"rate_bump_{c['rate_bump_pct']}" for c in curve_shift]
            ),
            "pnl": (
                [s['pnl'] for s in spot_sensitivity] +
                [v['pnl'] for v in vol_sensitivity] +
                [c['pnl'] for c in curve_shift]
            )
        }
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_file, index=False)
        
        return json.dumps({
            "success": True,
            "base_price": base_price,
            "base_greeks": {
                "delta": delta,
                "gamma": gamma,
                "vega": vega,
                "theta": theta
            },
            "spot_sensitivity": spot_sensitivity,
            "vol_sensitivity": vol_sensitivity,
            "curve_shift": curve_shift,
            "output_file": output_file
        })
    
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# STRESS TEST
# ============================================================================

def run_stress_test(
    option_data: str,
    output_dir: str = "outputs/tests",
    scenarios: Optional[str] = None
) -> str:
    """
    Run stress test with extreme market scenarios.
    
    Predefined scenarios:
      - Black Monday (Oct 1987): ~-20% stock, +50% vol
      - Dot-com (2000-2002): -70% stock, +200% vol
      - 2008 Crisis: -50% stock, +150% vol
      - VIX Spike: 0% stock, +100% vol
      - Rate Shock: 0% stock, +2% rates
      - Liquidation: -30% stock, +300% vol
    
    Args:
        option_data: JSON string with option parameters
        output_dir: Directory to save results
        scenarios: Optional JSON array of custom scenarios
          Format: [{"name": "...", "spot_change": X, "vol_change": X, "rate_change": X}, ...]
    
    Returns:
        JSON with stress test results
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Parse inputs
        try:
            option_params = json.loads(option_data)
        except:
            option_params = option_data if isinstance(option_data, dict) else {}
        
        required_fields = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        if not all(field in option_params for field in required_fields):
            return json.dumps({"error": f"Missing required fields. Need: {required_fields}"})
        
        opt_type = option_params['option_type']
        S = float(option_params['S'])
        K = float(option_params['K'])
        T = float(option_params['T'])
        r = float(option_params['r'])
        sigma = float(option_params['sigma'])
        
        base_price = _bsm_price(opt_type, S, K, T, r, sigma)
        
        # Default stress scenarios (historical + hypothetical)
        stress_scenarios = [
            {"name": "Black Monday (1987)", "spot_change": -0.20, "vol_change": 0.50, "rate_change": -0.005},
            {"name": "Dot-com Crash (2000)", "spot_change": -0.70, "vol_change": 2.00, "rate_change": -0.02},
            {"name": "2008 Financial Crisis", "spot_change": -0.50, "vol_change": 1.50, "rate_change": -0.01},
            {"name": "VIX Spike (No Stock Move)", "spot_change": 0.0, "vol_change": 1.00, "rate_change": 0.0},
            {"name": "Rate Shock (+200bps)", "spot_change": 0.0, "vol_change": 0.0, "rate_change": 0.02},
            {"name": "Liquidation Scenario", "spot_change": -0.30, "vol_change": 3.00, "rate_change": 0.01},
            {"name": "Volatility Collapse", "spot_change": 0.05, "vol_change": -0.50, "rate_change": 0.0},
        ]
        
        # Override with custom scenarios if provided
        if scenarios:
            try:
                custom_scenarios = json.loads(scenarios)
                stress_scenarios = custom_scenarios
            except:
                pass
        
        # Run stress tests
        results = []
        for scenario in stress_scenarios:
            name = scenario.get('name', 'Unknown')
            spot_change = float(scenario.get('spot_change', 0))
            vol_change = float(scenario.get('vol_change', 0))
            rate_change = float(scenario.get('rate_change', 0))
            
            S_stressed = S * (1 + spot_change)
            sigma_stressed = max(0.001, sigma + vol_change)  # Ensure positive vol
            r_stressed = r + rate_change
            
            stressed_price = _bsm_price(opt_type, S_stressed, K, T, r_stressed, sigma_stressed)
            pnl = stressed_price - base_price
            pnl_pct = (pnl / base_price * 100) if base_price != 0 else 0
            
            results.append({
                "scenario_name": name,
                "spot_change_pct": spot_change * 100,
                "vol_change_pct": vol_change * 100,
                "rate_change_pct": rate_change * 100,
                "base_price": base_price,
                "stressed_price": stressed_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct
            })
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"stress_test_{timestamp}.csv")
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        
        # Identify worst-case scenario
        worst_idx = np.argmin([r['pnl'] for r in results])
        worst_scenario = results[worst_idx]
        
        return json.dumps({
            "success": True,
            "base_price": base_price,
            "num_scenarios": len(results),
            "results": results,
            "worst_case": worst_scenario,
            "output_file": output_file
        })
    
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# P&L ANALYSIS TEST
# ============================================================================

def run_pnl_test(
    option_data: str,
    market_moves: str,
    output_dir: str = "outputs/tests"
) -> str:
    """
    Run P&L analysis and attribution test.
    
    Analyzes:
      - Greeks-based P&L estimate vs. actual P&L
      - P&L attribution (delta, gamma, vega, theta contributions)
      - Gamma P&L (realized variance impact)
      - Hedging effectiveness (delta-hedged returns)
    
    Args:
        option_data: JSON string with initial option parameters
        market_moves: JSON array of market scenarios
          Format: [{"spot": X, "vol": X, "days_passed": X, "rate": X}, ...]
        output_dir: Directory to save results
    
    Returns:
        JSON with P&L analysis results
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Parse inputs
        try:
            option_params = json.loads(option_data)
        except:
            option_params = option_data if isinstance(option_data, dict) else {}
        
        try:
            moves = json.loads(market_moves)
        except:
            moves = market_moves if isinstance(market_moves, list) else []
        
        required_fields = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        if not all(field in option_params for field in required_fields):
            return json.dumps({"error": f"Missing required fields. Need: {required_fields}"})
        
        opt_type = option_params['option_type']
        S0 = float(option_params['S'])
        K = float(option_params['K'])
        T0 = float(option_params['T'])
        r = float(option_params['r'])
        sigma0 = float(option_params['sigma'])
        
        # Base case
        V0 = _bsm_price(opt_type, S0, K, T0, r, sigma0)
        delta0 = _bsm_delta(opt_type, S0, K, T0, r, sigma0)
        gamma0 = _bsm_gamma(S0, K, T0, r, sigma0)
        vega0 = _bsm_vega(S0, K, T0, r, sigma0)
        theta0 = _bsm_theta(opt_type, S0, K, T0, r, sigma0)
        
        if not moves:
            # Default market moves for demonstration
            moves = [
                {"spot": S0 * 1.01, "vol": sigma0 + 0.01, "days_passed": 1, "rate": r},
                {"spot": S0 * 0.99, "vol": sigma0 - 0.01, "days_passed": 1, "rate": r},
                {"spot": S0 * 1.05, "vol": sigma0 + 0.05, "days_passed": 5, "rate": r},
                {"spot": S0 * 0.95, "vol": sigma0 + 0.03, "days_passed": 5, "rate": r},
            ]
        
        pnl_analysis = []
        
        for move in moves:
            S1 = float(move.get('spot', S0))
            sigma1 = float(move.get('vol', sigma0))
            days_passed = float(move.get('days_passed', 0))
            r1 = float(move.get('rate', r))
            
            # Time decay
            T1 = max(0, T0 - days_passed / 365.0)
            
            # New option price
            V1 = _bsm_price(opt_type, S1, K, T1, r1, sigma1)
            actual_pnl = V1 - V0
            
            # Greeks-based P&L estimate
            spot_move = S1 - S0
            vol_move = sigma1 - sigma0
            rate_move = r1 - r
            time_decay = days_passed / 365.0
            
            # P&L components
            delta_pnl = delta0 * spot_move
            gamma_pnl = 0.5 * gamma0 * (spot_move ** 2)
            vega_pnl = vega0 * vol_move
            theta_pnl = theta0 * time_decay
            
            # Total estimated P&L
            estimated_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
            pnl_error = actual_pnl - estimated_pnl
            
            # Realized variance (gamma P&L proxy)
            realized_var = (spot_move / S0) ** 2 if S0 != 0 else 0
            
            # Delta-hedged P&L (excluding delta component)
            hedged_pnl = gamma_pnl + vega_pnl + theta_pnl
            
            pnl_analysis.append({
                "spot_level": S1,
                "spot_move": spot_move,
                "vol_level": sigma1,
                "vol_move": vol_move,
                "days_passed": days_passed,
                "rate_level": r1,
                "rate_move": rate_move,
                "base_price": V0,
                "new_price": V1,
                "actual_pnl": actual_pnl,
                "delta_pnl": delta_pnl,
                "gamma_pnl": gamma_pnl,
                "vega_pnl": vega_pnl,
                "theta_pnl": theta_pnl,
                "estimated_pnl": estimated_pnl,
                "pnl_error": pnl_error,
                "realized_variance": realized_var,
                "delta_hedged_pnl": hedged_pnl
            })
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"pnl_analysis_{timestamp}.csv")
        
        analysis_df = pd.DataFrame(pnl_analysis)
        analysis_df.to_csv(output_file, index=False)
        
        # Summary statistics
        actual_pnls = [p['actual_pnl'] for p in pnl_analysis]
        estimated_pnls = [p['estimated_pnl'] for p in pnl_analysis]
        errors = [p['pnl_error'] for p in pnl_analysis]
        
        summary = {
            "num_scenarios": len(pnl_analysis),
            "avg_actual_pnl": float(np.mean(actual_pnls)),
            "max_actual_pnl": float(np.max(actual_pnls)),
            "min_actual_pnl": float(np.min(actual_pnls)),
            "avg_pnl_error": float(np.mean(errors)),
            "max_pnl_error": float(np.max(np.abs(errors))),
            "avg_delta_hedged_pnl": float(np.mean([p['delta_hedged_pnl'] for p in pnl_analysis]))
        }
        
        return json.dumps({
            "success": True,
            "base_greeks": {
                "price": V0,
                "delta": delta0,
                "gamma": gamma0,
                "vega": vega0,
                "theta": theta0
            },
            "pnl_analysis": pnl_analysis,
            "summary": summary,
            "output_file": output_file
        })
    
    except Exception as e:
        return json.dumps({"error": str(e)})

# Create StructuredTool wrappers and register them in the tool registry
from langchain.tools import tool as _lc_tool
from .tool_registry import REGISTRY as _REGISTRY

_sens_tool = _lc_tool("run_sensitivity_test")(run_sensitivity_test)
_REGISTRY.register(
    tool=_sens_tool,
    name="run_sensitivity_test",
    tags=["test", "sensitivity", "analysis"],
    roles=["validator"],
)

_stress_tool = _lc_tool("run_stress_test")(run_stress_test)
_REGISTRY.register(
    tool=_stress_tool,
    name="run_stress_test",
    tags=["test", "stress", "extreme"],
    roles=["validator"],
)

_pnl_tool = _lc_tool("run_pnl_test")(run_pnl_test)
_REGISTRY.register(
    tool=_pnl_tool,
    name="run_pnl_test",
    tags=["test", "pnl", "attribution"],
    roles=["validator"],
)
