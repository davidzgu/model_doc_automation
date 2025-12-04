import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Annotated

import pandas as pd
from langchain.tools import tool
from langgraph.prebuilt import InjectedState

from .tool_registry import register_tool
from .utils import load_json_as_df


def _load_template(template_path: Optional[str]) -> str:
    """Load template from path or default location."""
    if template_path:
        p = Path(template_path)
        if p.exists():
            return p.read_text(encoding="utf-8")
    
    # Default path
    default_path = Path(__file__).resolve().parents[1] / "templates" / "summary_template.md"
    if default_path.exists():
        return default_path.read_text(encoding="utf-8")
    
    raise FileNotFoundError("Summary template not found.")


def _parse_input_data(state: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract and convert input data from state to DataFrames."""
    bsm_df = load_json_as_df(state.get("bsm_results"))
    greeks_df = load_json_as_df(state.get("greeks_results"))
    vr_df = load_json_as_df(state.get("validate_results"))

    if bsm_df is False: bsm_df = pd.DataFrame()
    if greeks_df is False: greeks_df = pd.DataFrame()
    if vr_df is False: vr_df = pd.DataFrame()

    return bsm_df, greeks_df, vr_df


def _format_range(series: pd.Series, fmt: str = "{:.2f}") -> str:
    """Helper to format min-max range."""
    if series.empty:
        return "N/A"
    try:
        vals = pd.to_numeric(series, errors='coerce').dropna()
        if vals.empty:
            return "N/A"
        return f"[{fmt.format(vals.min())}, {fmt.format(vals.max())}]"
    except:
        return "N/A"


def _format_avg(series: pd.Series, fmt: str = "{:.2f}") -> str:
    """Helper to format average."""
    if series.empty:
        return "N/A"
    try:
        vals = pd.to_numeric(series, errors='coerce').dropna()
        if vals.empty:
            return "N/A"
        return fmt.format(vals.mean())
    except:
        return "N/A"


def _compute_market_stats(vr_df: pd.DataFrame) -> Dict[str, str]:
    """Compute statistics for market data (S, K, T, sigma)."""
    stats = {}
    
    # Spot Price
    s_series = vr_df.get('S', pd.Series())
    stats['spot_range'] = _format_range(s_series)
    stats['spot_avg'] = _format_avg(s_series)

    # Strike Price
    k_series = vr_df.get('K', pd.Series())
    stats['strike_range'] = _format_range(k_series)
    stats['strike_avg'] = _format_avg(k_series)

    # Maturity
    t_series = vr_df.get('T', pd.Series())
    stats['maturity_range'] = _format_range(t_series)
    stats['maturity_avg'] = _format_avg(t_series)

    # Volatility
    sigma_series = vr_df.get('sigma', pd.Series())
    stats['volatility_range'] = _format_range(sigma_series, "{:.2%}")
    stats['volatility_avg'] = _format_avg(sigma_series, "{:.2%}")

    return stats


def _compute_greek_stats(vr_df: pd.DataFrame) -> Dict[str, str]:
    """Compute statistics for Greeks."""
    stats = {}
    for greek in ['delta', 'gamma', 'vega', 'theta', 'rho']:
        series = vr_df.get(greek, pd.Series())
        stats[f'{greek}_range'] = _format_range(series, "{:.4f}")
        stats[f'{greek}_avg'] = _format_avg(series, "{:.4f}")
    return stats


def _compute_bsm_stats(bsm_df: pd.DataFrame) -> str:
    """Generate summary string for BSM pricing."""
    if bsm_df.empty or "BSM_Price" not in bsm_df.columns:
        return "No BSM pricing data available."

    try:
        prices = pd.to_numeric(bsm_df['BSM_Price'], errors='coerce').dropna()
        if prices.empty:
            return "No valid BSM prices found."
            
        return (
            f"- **Total Priced:** {len(prices)}\n"
            f"- **Avg Price:** ${prices.mean():.4f}\n"
            f"- **Range:** [${prices.min():.4f}, ${prices.max():.4f}]\n"
            f"- **Total Value:** ${prices.sum():.2f}"
        )
    except Exception:
        return "Error calculating BSM stats."


def _generate_recommendations(fail_cnt: int, vr_df: pd.DataFrame) -> str:
    """Generate recommendations based on failures and data checks."""
    recs = []
    
    if fail_cnt == 0:
        recs.append("✅ **System Healthy:** All validations passed.")
    else:
        recs.append(f"⚠️ **Action Required:** Investigate {fail_cnt} failed options.")

    # Check specific conditions
    try:
        if 'sigma' in vr_df.columns:
            max_vol = pd.to_numeric(vr_df['sigma'], errors='coerce').max()
            if max_vol > 1.0:
                recs.append("- **High Volatility:** Detected >100% volatility. Verify input data.")
        
        if 'T' in vr_df.columns:
            min_t = pd.to_numeric(vr_df['T'], errors='coerce').min()
            if min_t < 0.1:
                recs.append("- **Near Expiry:** Short-dated options (T < 0.1y) detected. Monitor Theta.")
    except:
        pass

    return "\n".join(recs)


@register_tool(tags=["summary", "generate"], roles=["summary_generator"])
@tool("generate_summary")
def generate_summary(
    state: Annotated[dict, InjectedState],
    template_path: Optional[str] = None,
    save_md: bool = True,
) -> str:
    """
    Generate a concise Markdown summary report for BSM model performance.
    
    Args:
        state: Injected state containing bsm_results, greeks_results, validate_results.
        template_path: Optional path to custom template.
        save_md: Whether to save the report to disk.
        
    Returns:
        JSON string with state_update containing report_md and optional report_path.
    """
    try:
        # 1. Load Data
        bsm_df, greeks_df, vr_df = _parse_input_data(state)
        if bsm_df.empty:
            return json.dumps({"errors": ["No BSM results found to summarize."]})
        if greeks_df.empty:
            return json.dumps({"errors": ["No Greeks results found to summarize."]})
        if vr_df.empty:
            return json.dumps({"errors": ["No validation results found to summarize."]})

        # 2. Calculate General Stats
        total = len(vr_df)
        pass_cnt = int((vr_df.get("validations_result") == "passed").sum()) if "validations_result" in vr_df else 0
        fail_cnt = total - pass_cnt
        pass_rate = f"{(pass_cnt / total * 100):.1f}" if total > 0 else "0.0"
        
        # Option Types
        option_types_inline = "N/A"
        if "option_type" in vr_df.columns:
            counts = vr_df["option_type"].value_counts().to_dict()
            option_types_inline = ", ".join([f"{k}: {v}" for k, v in counts.items()])

        # Critical Issues
        issues = []
        if "validations_details" in vr_df.columns:
            raw_issues = vr_df[vr_df["validations_result"] == "failed"]["validations_details"].tolist()
            for item in raw_issues:
                if isinstance(item, list):
                    issues.extend([str(i) for i in item if i])
                elif item:
                    issues.append(str(item))
        
        critical_issues = "\n".join([f"- {i}" for i in issues[:10]]) if issues else "None"
        if len(issues) > 10:
            critical_issues += f"\n- ... and {len(issues) - 10} more."

        # 3. Prepare Template Variables
        market_stats = _compute_market_stats(vr_df)
        greek_stats = _compute_greek_stats(vr_df)
        
        context = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_options": total,
            "validation_status": "✅ PASSED" if fail_cnt == 0 else "⚠️ ISSUES FOUND",
            "pass_rate": pass_rate,
            "failed_count": fail_cnt,
            "option_types_inline": option_types_inline,
            "bsm_pricing_summary": _compute_bsm_stats(bsm_df),
            "critical_issues": critical_issues,
            "recommendations": _generate_recommendations(fail_cnt, vr_df),
            **market_stats,
            **greek_stats
        }

        # 4. Render Template
        template_txt = _load_template(template_path)
        report_md = template_txt.format(**context)

        # 5. Save Report
        report_path = None
        if save_md:
            out_dir = Path(__file__).resolve().parents[3] / "data" / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = f"validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            p = out_dir / filename
            p.write_text(report_md, encoding="utf-8")
            report_path = str(p)

        return json.dumps({
                "report_md": report_md,
                "report_path": report_path
            })

    except Exception as e:
        return json.dumps({"errors": [f"Summary generation failed: {str(e)}"]})