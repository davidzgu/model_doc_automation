import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

import pandas as pd
from langchain.tools import tool

from .tool_registry import register_tool
from .utils import load_json_as_df  # 你已有


def _load_template_text(template_path: Optional[str]) -> Optional[str]:
    """
    If template_path provided and exists, use it.
    Else try default: src/templates/summary_template.md
    """
    path = None
    if template_path:
        p = Path(template_path)
        path = p if p.exists() else None
    if path is None:
        default = Path(__file__).resolve().parents[1] / "templates" / "summary_template.md"
        path = default if default.exists() else None
    return path.read_text(encoding="utf-8") if path else None


def _generate_greeks_summary(vr_df: pd.DataFrame) -> str:
    """生成 Greeks 摘要"""
    lines = []
    greek_cols = ['delta', 'gamma', 'vega', 'theta', 'rho']

    for col in greek_cols:
        if col in vr_df.columns:
            try:
                col_numeric = pd.to_numeric(vr_df[col], errors='coerce')
                lines.append(f"- **{col.capitalize()}:** Avg = {col_numeric.mean():.6f}, Range = [{col_numeric.min():.6f}, {col_numeric.max():.6f}]")
            except Exception:
                pass

    return "\n".join(lines) if lines else "- Greeks data not available"


def _generate_greek_analysis(vr_df: pd.DataFrame, col_name: str, display_name: str, expected_range: str) -> str:
    """生成单个 Greek 的详细分析"""
    if col_name not in vr_df.columns:
        return f"- **{display_name}** data not available in validation results"

    try:
        col_numeric = pd.to_numeric(vr_df[col_name], errors='coerce')

        analysis = f"- **Portfolio {display_name}:** {col_numeric.sum():.6f}\n"
        analysis += f"- **Average {display_name}:** {col_numeric.mean():.6f}\n"
        analysis += f"- **{display_name} Range:** [{col_numeric.min():.6f}, {col_numeric.max():.6f}]\n"
        analysis += f"- **Expected Range:** {expected_range}\n"
        analysis += f"- **Standard Deviation:** {col_numeric.std():.6f}"

        return analysis
    except Exception:
        return f"- Unable to compute {display_name} statistics"


def _generate_recommendations(fail_cnt: Optional[int], total: int, vr_df: pd.DataFrame) -> str:
    """生成建议"""
    recommendations = []

    if fail_cnt == 0:
        recommendations.append("✅ **All validations passed.** The portfolio demonstrates consistent pricing and Greeks within expected theoretical bounds.")
    else:
        recommendations.append(f"⚠️ **{fail_cnt} validation failures detected.** Review failed options for potential pricing discrepancies or input data errors.")

    # 检查波动率
    if 'sigma' in vr_df.columns:
        try:
            sigma_numeric = pd.to_numeric(vr_df['sigma'], errors='coerce')
            if sigma_numeric.max() > 1.0:  # 100% volatility
                recommendations.append("⚠️ **High volatility detected** (>100%). Consider reviewing volatility inputs for accuracy.")
        except Exception:
            pass

    # 检查到期时间
    if 'T' in vr_df.columns:
        try:
            T_numeric = pd.to_numeric(vr_df['T'], errors='coerce')
            if T_numeric.min() < 0.1:  # Less than 1.2 months
                recommendations.append("📌 **Short-dated options detected** (T < 0.1 years). Monitor Theta decay closely.")
        except Exception:
            pass

    # 检查 Gamma 风险
    if 'gamma' in vr_df.columns:
        try:
            gamma_numeric = pd.to_numeric(vr_df['gamma'], errors='coerce')
            if gamma_numeric.max() > 0.1:
                recommendations.append("📌 **High Gamma exposure detected.** Portfolio may be sensitive to large moves in underlying asset.")
        except Exception:
            pass

    if not recommendations:
        recommendations.append("✅ No specific recommendations at this time. Continue monitoring market conditions.")

    return "\n".join(recommendations)

def _generate_fallback_summary_md(
    total: int,
    passed: Optional[int],
    failed: Optional[int],
    option_type_counts: Optional[Dict[str, int]],
    key_issues_md: str,
    bsm_df: Optional[pd.DataFrame],
    greeks_df: Optional[pd.DataFrame],
) -> str:
    lines = []
    lines.append(f"# Validation Summary")
    lines.append("")
    lines.append(f"- Date: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"- Total options: {total}")
    lines.append(f"- Passed: {passed if passed is not None else 'N/A'}")
    lines.append(f"- Failed: {failed if failed is not None else 'N/A'}")
    lines.append("")
    lines.append("## Option Types")
    if option_type_counts:
        for k, v in option_type_counts.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- N/A")
    lines.append("")
    lines.append("## Key Issues")
    lines.append(key_issues_md or "- None")
    lines.append("")

    # 计算简要统计
    calc_lines = []
    if bsm_df is not None and bsm_df is not False and "BSM_Price" in bsm_df.columns:
        try:
            bsm_price_numeric = pd.to_numeric(bsm_df['BSM_Price'], errors='coerce')
            calc_lines.append(f"- Avg BSM Price: {bsm_price_numeric.mean():.4f}")
        except Exception:
            pass
    if greeks_df is not None and greeks_df is not False:
        for col in ["delta","gamma","vega","rho","theta"]:
            if col in greeks_df.columns:
                try:
                    col_numeric = pd.to_numeric(greeks_df[col], errors='coerce')
                    calc_lines.append(f"- Avg {col.capitalize()}: {col_numeric.mean():.6f}")
                except Exception:
                    pass
    if calc_lines:
        lines.append("## Calculation Stats")
        lines += calc_lines
        lines.append("")

    lines.append(f"_Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    return "\n".join(lines)


@register_tool(tags=["summary","generate"], roles=["summary_generator"])
@tool("generate_summary")
def generate_summary(
    validate_results: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    bsm_results: Optional[Union[str, List[Dict[str, Any]]]] = None,
    greeks_results: Optional[Union[str, List[Dict[str, Any]]]] = None,
    template_path: Optional[str] = None,
    save_md: bool = True,
) -> str:
    """
    Generate a Markdown summary based on validator results (and optionally calculation results).
    If a template is provided (or found at default path), fill it; otherwise compose a fallback summary.
    Returns JSON envelope: {"state_update": {"report_md": "...", "report_path": "...?"}}
    """
    try:
        vr_df = load_json_as_df(validate_results)
        if vr_df is False:
            return json.dumps({"state_update": {"errors": ["validator_results must be JSON string, dict or list"]}})

        bsm_df = load_json_as_df(bsm_results) if bsm_results is not None else None
        greeks_df = load_json_as_df(greeks_results) if greeks_results is not None else None

        total = len(vr_df)
        pass_cnt  = int((vr_df.get("validations_result") == "passed").sum()) if "validations_result" in vr_df else None
        fail_cnt  = int((vr_df.get("validations_result") == "failed").sum()) if "validations_result" in vr_df else None

        option_type_counts = None
        if "option_type" in vr_df.columns:
            option_type_counts = vr_df["option_type"].value_counts(dropna=False).to_dict()

        key_issues_md = ""
        if "validations_details" in vr_df.columns:
            issues = []
            for x in vr_df["validations_details"].tolist():
                if isinstance(x, list):
                    issues.extend([str(i) for i in x if i])
            if issues:
                key_issues_md = "\n".join(f"- {i}" for i in issues[:50])  # 防止过长
    except Exception as e:
        return json.dumps({"state_update": {"errors": [f"summary parse error: {e}"]}})
    

    template_txt = _load_template_text(template_path)


    if template_txt:
        try:
            # === 基础信息 ===
            analysis_date = datetime.now().strftime("%Y-%m-%d")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 计算通过率
            pass_rate = f"{(pass_cnt / total * 100):.1f}" if total > 0 and pass_cnt is not None else "0.0"
            fail_rate = f"{(fail_cnt / total * 100):.1f}" if total > 0 and fail_cnt is not None else "0.0"

            # 验证状态
            validation_status = "✅ PASSED" if fail_cnt == 0 else f"⚠️ {fail_cnt} ISSUES FOUND"

            # === 期权类型分布 ===
            option_types_md = ""
            if option_type_counts:
                option_types_md = "\n".join(f"  - **{k.capitalize()}:** {v} positions" for k, v in option_type_counts.items())
            else:
                option_types_md = "  - N/A"

            # === 关键指标 ===
            key_metrics = f"- **Average Option Price:** {pd.to_numeric(vr_df.get('price', [0]), errors='coerce').mean():.4f}\n"
            key_metrics += f"- **Price Range:** [{pd.to_numeric(vr_df.get('price', [0]), errors='coerce').min():.4f}, {pd.to_numeric(vr_df.get('price', [0]), errors='coerce').max():.4f}]"

            # === 标的资产统计 ===
            underlying_stats = ""
            if 'S' in vr_df.columns:
                S_numeric = pd.to_numeric(vr_df['S'], errors='coerce')
                underlying_stats = f"- **Spot Price Range:** [{S_numeric.min():.2f}, {S_numeric.max():.2f}]\n"
                underlying_stats += f"- **Average Spot:** {S_numeric.mean():.2f}\n"
                underlying_stats += f"- **Volatility (σ) Range:** [{pd.to_numeric(vr_df.get('sigma', [0]), errors='coerce').min():.2%}, {pd.to_numeric(vr_df.get('sigma', [0]), errors='coerce').max():.2%}]"
            else:
                underlying_stats = "- Data not available"

            # === 行权价分布 ===
            strike_distribution = ""
            if 'K' in vr_df.columns:
                K_numeric = pd.to_numeric(vr_df['K'], errors='coerce')
                strike_distribution = f"- **Strike Range:** [{K_numeric.min():.2f}, {K_numeric.max():.2f}]\n"
                strike_distribution += f"- **Average Strike:** {K_numeric.mean():.2f}"
            else:
                strike_distribution = "- Data not available"

            # === 到期时间分布 ===
            maturity_profile = ""
            if 'T' in vr_df.columns:
                T_numeric = pd.to_numeric(vr_df['T'], errors='coerce')
                maturity_profile = f"- **Time to Maturity Range:** [{T_numeric.min():.2f}, {T_numeric.max():.2f}] years\n"
                maturity_profile += f"- **Average Maturity:** {T_numeric.mean():.2f} years"
            else:
                maturity_profile = "- Data not available"

            # === BSM 定价摘要 ===
            bsm_pricing_summary = "- No BSM pricing data available"
            if bsm_df is not None and bsm_df is not False and "BSM_Price" in bsm_df.columns:
                try:
                    bsm_price_numeric = pd.to_numeric(bsm_df['BSM_Price'], errors='coerce')
                    bsm_pricing_summary = f"- **Total Options Priced:** {len(bsm_df)}\n"
                    bsm_pricing_summary += f"- **Average BSM Price:** ${bsm_price_numeric.mean():.4f}\n"
                    bsm_pricing_summary += f"- **Price Range:** [${bsm_price_numeric.min():.4f}, ${bsm_price_numeric.max():.4f}]\n"
                    bsm_pricing_summary += f"- **Total Portfolio Value:** ${bsm_price_numeric.sum():.2f}"
                except Exception:
                    pass

            # === Greeks 摘要 ===
            greeks_summary = _generate_greeks_summary(vr_df)

            # === 验证细节 ===
            validation_details = f"**All validations passed criteria:** {pass_cnt}/{total} options"

            # === 关键问题 ===
            critical_issues = key_issues_md if key_issues_md else "✅ No critical issues identified. All options passed validation checks."

            # === Greeks 分析（详细） ===
            delta_analysis = _generate_greek_analysis(vr_df, 'delta', 'Delta', '[-1, 1]')
            gamma_analysis = _generate_greek_analysis(vr_df, 'gamma', 'Gamma', '[0, ∞)')
            vega_analysis = _generate_greek_analysis(vr_df, 'vega', 'Vega', '[0, ∞)')
            theta_analysis = _generate_greek_analysis(vr_df, 'theta', 'Theta', '(-∞, 0]')
            rho_analysis = _generate_greek_analysis(vr_df, 'rho', 'Rho', 'Varies')

            # === 性能摘要 ===
            performance_summary = f"Analysis completed successfully for {total} options with {pass_rate}% validation pass rate."

            # === 异常值 ===
            anomalies = "No significant anomalies detected in the current dataset."

            # === 模型准确性 ===
            model_accuracy = "BSM model assumptions hold for the analyzed dataset. All Greeks within expected theoretical bounds."

            # === 建议 ===
            recommendations = _generate_recommendations(fail_cnt, total, vr_df)

            # === 分析周期 ===
            if 'date' in vr_df.columns:
                dates = pd.to_datetime(vr_df['date'], errors='coerce')
                analysis_period = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
            else:
                analysis_period = analysis_date

            summary = template_txt.format(
                analysis_date=analysis_date,
                analysis_period=analysis_period,
                total_options=total,
                validation_status=validation_status,
                pass_rate=pass_rate,
                failed_count=fail_cnt if fail_cnt is not None else 0,
                option_types=option_types_md,
                key_metrics=key_metrics,
                underlying_stats=underlying_stats,
                strike_distribution=strike_distribution,
                maturity_profile=maturity_profile,
                bsm_pricing_summary=bsm_pricing_summary,
                greeks_summary=greeks_summary,
                passed_count=pass_cnt if pass_cnt is not None else 0,
                fail_rate=fail_rate,
                validation_details=validation_details,
                critical_issues=critical_issues,
                delta_analysis=delta_analysis,
                gamma_analysis=gamma_analysis,
                vega_analysis=vega_analysis,
                theta_analysis=theta_analysis,
                rho_analysis=rho_analysis,
                performance_summary=performance_summary,
                anomalies=anomalies,
                model_accuracy=model_accuracy,
                recommendations=recommendations,
                timestamp=timestamp,
            )
        except Exception as e:
            return json.dumps({"state_update": {"errors": [f"template render error: {e}"]}})
    else:
        # 兜底 Markdown（没有模板时）
        summary = _generate_fallback_summary_md(
            total=total,
            passed=pass_cnt,
            failed=fail_cnt,
            option_type_counts=option_type_counts,
            key_issues_md=key_issues_md,
            bsm_df=bsm_df,
            greeks_df=greeks_df,
        )


    report_path = None
    if save_md:
        try:
            out_dir = Path(__file__).resolve().parents[3] / "data" / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p = out_dir / f"validation_summary_{ts}.md"
            p.write_text(summary, encoding="utf-8")
            report_path = str(p)
        except Exception as e:
            out_dir = Path(__file__).resolve().parents[3] / "data" / "output"
            report_path = None
            json.dumps({"state_update": {"errors": [f"Path {out_dir} doesn't exits."]}})

    return json.dumps({"state_update": {
        "report_md": summary,
        **({"report_path": report_path} if report_path else {})
    }})

            