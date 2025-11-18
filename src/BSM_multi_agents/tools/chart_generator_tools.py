# -*- coding: utf-8 -*-
"""
Chart generation tools for option analysis visualization.

Creates plots for option prices, Greeks, and sensitivity analysis.
"""
import json
import os
from typing import Union, Dict, Any, List
from pathlib import Path
from langchain_core.tools import tool
from .tool_registry import register_tool

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@register_tool(tags=["chart","generate","price"], roles=["chart_generator"])
@tool("create_option_price_chart")
def create_option_price_chart(bsm_results: str, output_dir: str = "outputs/charts") -> str:
    """
    Create a chart showing option prices for different strike prices or spot prices.

    Args:
        calculation_data: JSON string of list[dict] with BSM calculation results
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
        data = json.loads(bsm_results)
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        df = pd.DataFrame(data)

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


@register_tool(tags=["chart","generate","greeks"], roles=["chart_generator"])
@tool("create_greeks_chart")
def create_greeks_chart(greeks_results: str, output_dir: str = "outputs/charts") -> str:
    """
    Create portfolio Greeks distribution analysis charts (Bank OPA style).

    Generates a 2x3 grid showing:
    1. Portfolio Greeks Summary Table
    2. Delta Distribution by Option Type
    3. Gamma Distribution
    4. Vega Distribution
    5. Greeks by Moneyness (ITM/ATM/OTM)
    6. Theta & Rho Summary

    Args:
        greeks_data: JSON string of list[dict] with Greeks calculation results
                    Expected fields: option_type, S, K, delta, gamma, vega, theta, rho, T, price
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
        data = json.loads(greeks_results)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # For sensitivity test data (list of dicts)
        df = pd.DataFrame(data)

        # Ensure numeric columns
        numeric_cols = ['S', 'K', 'delta', 'gamma', 'vega', 'theta', 'rho', 'price', 'T']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate Moneyness
        if 'S' in df.columns and 'K' in df.columns and 'option_type' in df.columns:
            df['moneyness_ratio'] = df['S'] / df['K']

            def classify_moneyness(row):
                ratio = row['moneyness_ratio']
                opt_type = str(row['option_type']).lower()

                if 0.95 <= ratio <= 1.05:
                    return 'ATM'
                elif opt_type == 'call':
                    return 'ITM' if ratio > 1.05 else 'OTM'
                else:  # put
                    return 'ITM' if ratio < 0.95 else 'OTM'

            df['moneyness'] = df.apply(classify_moneyness, axis=1)

        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))

        # Define colors
        call_color = '#2E86AB'  # Blue for calls
        put_color = '#A23B72'   # Red/Magenta for puts

        # ===== Subplot 1: Portfolio Greeks Summary Table =====
        ax1 = plt.subplot(2, 3, 1)
        ax1.axis('tight')
        ax1.axis('off')

        # Calculate portfolio Greeks
        greeks_cols = ['delta', 'gamma', 'vega', 'theta', 'rho']
        summary_data = []

        for greek in greeks_cols:
            if greek in df.columns:
                total = df[greek].sum()
                call_val = df[df['option_type'].str.lower() == 'call'][greek].sum() if 'option_type' in df.columns else 0
                put_val = df[df['option_type'].str.lower() == 'put'][greek].sum() if 'option_type' in df.columns else 0
                summary_data.append([greek.capitalize(), f'{total:.4f}', f'{call_val:.4f}', f'{put_val:.4f}'])

        table = ax1.table(cellText=summary_data,
                         colLabels=['Greek', 'Portfolio', 'Calls', 'Puts'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.2, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax1.set_title('Portfolio Greeks Summary', fontsize=14, fontweight='bold', pad=20)

        # ===== Subplot 2: Delta Distribution =====
        ax2 = plt.subplot(2, 3, 2)
        if 'delta' in df.columns and 'option_type' in df.columns:
            calls = df[df['option_type'].str.lower() == 'call']['delta']
            puts = df[df['option_type'].str.lower() == 'put']['delta']

            positions = np.arange(len(df))
            width = 0.35

            call_positions = positions[df['option_type'].str.lower() == 'call']
            put_positions = positions[df['option_type'].str.lower() == 'put']

            if len(call_positions) > 0:
                ax2.bar(call_positions, calls, width, label='Call', color=call_color, alpha=0.8)
            if len(put_positions) > 0:
                ax2.bar(put_positions, puts, width, label='Put', color=put_color, alpha=0.8)

            ax2.set_xlabel('Option Index', fontsize=10)
            ax2.set_ylabel('Delta (Δ)', fontsize=10)
            ax2.set_title('Delta Distribution by Option Type', fontsize=12, fontweight='bold')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')

        # ===== Subplot 3: Gamma Distribution =====
        ax3 = plt.subplot(2, 3, 3)
        if 'gamma' in df.columns and 'option_type' in df.columns:
            calls_gamma = df[df['option_type'].str.lower() == 'call']['gamma']
            puts_gamma = df[df['option_type'].str.lower() == 'put']['gamma']

            call_positions = positions[df['option_type'].str.lower() == 'call']
            put_positions = positions[df['option_type'].str.lower() == 'put']

            if len(call_positions) > 0:
                ax3.bar(call_positions, calls_gamma, width, label='Call', color=call_color, alpha=0.8)
            if len(put_positions) > 0:
                ax3.bar(put_positions, puts_gamma, width, label='Put', color=put_color, alpha=0.8)

            ax3.set_xlabel('Option Index', fontsize=10)
            ax3.set_ylabel('Gamma (Γ)', fontsize=10)
            ax3.set_title('Gamma Distribution', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')

        # ===== Subplot 4: Vega Distribution =====
        ax4 = plt.subplot(2, 3, 4)
        if 'vega' in df.columns and 'option_type' in df.columns:
            calls_vega = df[df['option_type'].str.lower() == 'call']['vega']
            puts_vega = df[df['option_type'].str.lower() == 'put']['vega']

            call_positions = positions[df['option_type'].str.lower() == 'call']
            put_positions = positions[df['option_type'].str.lower() == 'put']

            if len(call_positions) > 0:
                ax4.bar(call_positions, calls_vega, width, label='Call', color=call_color, alpha=0.8)
            if len(put_positions) > 0:
                ax4.bar(put_positions, puts_vega, width, label='Put', color=put_color, alpha=0.8)

            ax4.set_xlabel('Option Index', fontsize=10)
            ax4.set_ylabel('Vega (ν)', fontsize=10)
            ax4.set_title('Vega Distribution', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')

        # ===== Subplot 5: Greeks by Moneyness =====
        ax5 = plt.subplot(2, 3, 5)
        if 'moneyness' in df.columns and all(col in df.columns for col in ['delta', 'gamma', 'vega']):
            moneyness_groups = df.groupby('moneyness')[['delta', 'gamma', 'vega']].mean()

            x_pos = np.arange(len(moneyness_groups))
            width = 0.25

            if 'delta' in moneyness_groups.columns:
                ax5.bar(x_pos - width, moneyness_groups['delta'], width, label='Avg Delta', color='#2E86AB')
            if 'gamma' in moneyness_groups.columns:
                ax5.bar(x_pos, moneyness_groups['gamma'] * 10, width, label='Avg Gamma (×10)', color='#F18F01')
            if 'vega' in moneyness_groups.columns:
                ax5.bar(x_pos + width, moneyness_groups['vega'] / 10, width, label='Avg Vega (÷10)', color='#C73E1D')

            ax5.set_xlabel('Moneyness', fontsize=10)
            ax5.set_ylabel('Greek Value', fontsize=10)
            ax5.set_title('Greeks by Moneyness', fontsize=12, fontweight='bold')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(moneyness_groups.index)
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')

        # ===== Subplot 6: Theta & Rho Summary =====
        ax6 = plt.subplot(2, 3, 6)
        if 'theta' in df.columns and 'rho' in df.columns:
            positions = np.arange(len(df))

            ax6_twin = ax6.twinx()

            ax6.bar(positions - width/2, df['theta'], width, label='Theta (Θ)', color='#6A4C93', alpha=0.7)
            ax6_twin.bar(positions + width/2, df['rho'], width, label='Rho (ρ)', color='#1982C4', alpha=0.7)

            ax6.set_xlabel('Option Index', fontsize=10)
            ax6.set_ylabel('Theta (Θ)', fontsize=10, color='#6A4C93')
            ax6_twin.set_ylabel('Rho (ρ)', fontsize=10, color='#1982C4')
            ax6.set_title('Theta & Rho Summary', fontsize=12, fontweight='bold')

            ax6.tick_params(axis='y', labelcolor='#6A4C93')
            ax6_twin.tick_params(axis='y', labelcolor='#1982C4')

            # Combined legend
            lines1, labels1 = ax6.get_legend_handles_labels()
            lines2, labels2 = ax6_twin.get_legend_handles_labels()
            ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            ax6.grid(True, alpha=0.3, axis='y')
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

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error creating Greeks chart: {str(e)}"
        })


@register_tool(tags=["chart","generate","summary"], roles=["chart_generator"])
@tool("create_summary_charts")
def create_summary_charts(
    bsm_results: str,
    greeks_results: str,
    output_dir: str = "outputs/charts"
) -> str:
    """
    Create a comprehensive set of charts including prices and Greeks.

    Args:
        bsm_results: JSON string of list[dict] with BSM calculation results
        greeks_results: JSON string of list[dict] with Greeks calculation results
        output_dir: Directory in str to save charts

    Returns:
        JSON string with list of chart paths and descriptions
    """
    try:
        # bsm_results and greeks_results are already JSON strings, use them directly
        charts = []

        # Create price chart
        price_result = create_option_price_chart.invoke({
            "bsm_results": bsm_results,
            "output_dir": output_dir
        })
        price_info = json.loads(price_result)
        if price_info.get("status") == "success":
            charts.append(price_info)

        # Create Greeks chart
        greeks_result = create_greeks_chart.invoke({
            "greeks_results": greeks_results,
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
