# -*- coding: utf-8 -*-
"""
Chart generation tools for option analysis visualization.

Creates plots for option prices, Greeks, and sensitivity analysis.
"""
import json
import os
from typing import Union, Dict, Any
from pathlib import Path
from langchain_core.tools import tool

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@tool
def create_option_price_chart(calculation_data: Union[str, Dict[str, Any]], output_dir: str = "outputs/charts") -> str:
    """
    Create a chart showing option prices for different strike prices or spot prices.

    Args:
        calculation_data: JSON string or dict with BSM calculation results
        output_dir: Directory to save the chart

    Returns:
        JSON string with chart file path and description
    """
    if not MATPLOTLIB_AVAILABLE:
        return json.dumps({
            "status": "error",
            "message": "matplotlib not available. Install with: pip install matplotlib"
        })

    try:
        # Parse input
        if isinstance(calculation_data, str):
            data = json.loads(calculation_data)
        else:
            data = calculation_data

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        if isinstance(data, dict) and 'S' in data and isinstance(data['S'], dict):
            # Pandas-style dict
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return json.dumps({
                "status": "error",
                "message": "Invalid data format"
            })

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Option Prices by Spot Price
        if 'BSM_Price' in df.columns and 'S' in df.columns:
            calls = df[df['option_type'].str.lower() == 'call']
            puts = df[df['option_type'].str.lower() == 'put']

            if not calls.empty:
                ax1.plot(calls['S'], calls['BSM_Price'], 'o-', label='Call', color='green', linewidth=2)
            if not puts.empty:
                ax1.plot(puts['S'], puts['BSM_Price'], 's-', label='Put', color='red', linewidth=2)

            ax1.set_xlabel('Spot Price (S)', fontsize=12)
            ax1.set_ylabel('Option Price', fontsize=12)
            ax1.set_title('BSM Option Prices vs Spot Price', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Plot 2: Option Prices by Time to Maturity
        if 'BSM_Price' in df.columns and 'T' in df.columns:
            calls = df[df['option_type'].str.lower() == 'call']
            puts = df[df['option_type'].str.lower() == 'put']

            if not calls.empty:
                ax2.plot(calls['T'], calls['BSM_Price'], 'o-', label='Call', color='green', linewidth=2)
            if not puts.empty:
                ax2.plot(puts['T'], puts['BSM_Price'], 's-', label='Put', color='red', linewidth=2)

            ax2.set_xlabel('Time to Maturity (years)', fontsize=12)
            ax2.set_ylabel('Option Price', fontsize=12)
            ax2.set_title('BSM Option Prices vs Time to Maturity', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save chart
        chart_path = output_path / "option_prices.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        return json.dumps({
            "status": "success",
            "chart_path": str(chart_path),
            "description": "Option prices visualization showing relationship between prices and spot/time"
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error creating price chart: {str(e)}"
        })


@tool
def create_greeks_chart(greeks_data: Union[str, Dict[str, Any]], output_dir: str = "outputs/charts") -> str:
    """
    Create charts showing option Greeks (delta, gamma, vega, rho, theta).

    Args:
        greeks_data: JSON string or dict with Greeks calculation results
        output_dir: Directory to save the chart

    Returns:
        JSON string with chart file path and description
    """
    if not MATPLOTLIB_AVAILABLE:
        return json.dumps({
            "status": "error",
            "message": "matplotlib not available. Install with: pip install matplotlib"
        })

    try:
        # Parse input
        if isinstance(greeks_data, str):
            data = json.loads(greeks_data)
        else:
            data = greeks_data

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # For sensitivity test data (list of dicts)
        if isinstance(data, list):
            df = pd.DataFrame(data)

            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            greeks = ['price', 'delta', 'gamma', 'vega', 'rho', 'theta']
            titles = ['Price', 'Delta (Δ)', 'Gamma (Γ)', 'Vega (ν)', 'Rho (ρ)', 'Theta (Θ)']

            for idx, (greek, title) in enumerate(zip(greeks, titles)):
                if greek in df.columns and 'spot_change' in df.columns:
                    axes[idx].plot(df['spot_change'] * 100, df[greek], 'o-', linewidth=2, markersize=6)
                    axes[idx].set_xlabel('Spot Price Change (%)', fontsize=10)
                    axes[idx].set_ylabel(title, fontsize=10)
                    axes[idx].set_title(f'{title} Sensitivity', fontsize=12, fontweight='bold')
                    axes[idx].grid(True, alpha=0.3)
                    axes[idx].axhline(y=0, color='k', linestyle='--', alpha=0.3)

            plt.tight_layout()

            # Save chart
            chart_path = output_path / "greeks_sensitivity.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return json.dumps({
                "status": "success",
                "chart_path": str(chart_path),
                "description": "Greeks sensitivity analysis showing how each Greek changes with spot price"
            })
        else:
            return json.dumps({
                "status": "error",
                "message": "Greeks data must be a list of sensitivity results"
            })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error creating Greeks chart: {str(e)}"
        })


@tool
def create_summary_charts(calculation_data: str, greeks_data: str, output_dir: str = "outputs/charts") -> str:
    """
    Create a comprehensive set of charts including prices and Greeks.

    Args:
        calculation_data: JSON string with BSM calculation results
        greeks_data: JSON string with Greeks sensitivity data
        output_dir: Directory to save charts

    Returns:
        JSON string with list of chart paths and descriptions
    """
    try:
        charts = []

        # Create price chart
        price_result = create_option_price_chart.invoke({
            "calculation_data": calculation_data,
            "output_dir": output_dir
        })
        price_info = json.loads(price_result)
        if price_info.get("status") == "success":
            charts.append(price_info)

        # Create Greeks chart
        greeks_result = create_greeks_chart.invoke({
            "greeks_data": greeks_data,
            "output_dir": output_dir
        })
        greeks_info = json.loads(greeks_result)
        if greeks_info.get("status") == "success":
            charts.append(greeks_info)

        return json.dumps({
            "status": "success",
            "charts": charts,
            "total_charts": len(charts)
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error creating summary charts: {str(e)}"
        })
