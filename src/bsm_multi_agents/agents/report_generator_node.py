from typing import Optional, Dict, List
import pandas as pd
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import datetime 
from io import BytesIO
import matplotlib.pyplot as plt
from docx.shared import Inches
from pathlib import Path
import seaborn as sns
import re

from langchain_core.messages import SystemMessage, HumanMessage
from bsm_multi_agents.config.llm_config import get_llm
from bsm_multi_agents.graph.state import WorkflowState


def report_generator_agent_node(state: WorkflowState) -> WorkflowState:
    """Refactored report generator agent node"""
    print("\n>>> [Report Generator Agent] Compiling final report...")

    ## Load parameters
    errors = state.get("errors", [])
    csv_file_path = state.get("csv_file_path")
    output_dir = state.get("output_dir")
    
    p = Path(csv_file_path)
    pricing_results_path = f"{output_dir}/analytics/analyzed_options.csv"
    sensitivity_test_results_path = f"{output_dir}/analytics/sensitivity_test_results.csv"
    stress_test_results_path = f"{output_dir}/analytics/stress_test_results.csv"
    put_call_parity_path = f"{output_dir}/analytics/put_call_parity.csv"
    final_report_path = state.get("final_report_path")
    
    # Load DataFrames
    df_pricing = pd.read_csv(pricing_results_path)
    sensitivity_test_results = pd.read_csv(sensitivity_test_results_path)
    stress_test_results = pd.read_csv(stress_test_results_path)
    put_call_parity = pd.read_csv(put_call_parity_path)

    ## Set parameters
    report_params = {
        "title": "Ongoing Monitoring Analysis Report",
        "model_name": "Option Pricing, BSM",
        "author_name": "John Doe",
        "group_name": "Front Desk Modeling and Analytics",
        "version": "v1.0"
    }


    llm = get_llm()
    doc = Document()

    # 1. Title Page
    print(">>>>>> [Report Generator Agent] Compiling title page...")
    _add_title_page(doc, report_params)

    # 2. Section 1
    print(">>>>>> [Report Generator Agent] Compiling section 1...")
    _add_section_1(doc, "1. Introduction", None)

    # 3. Section 2 (Loop per asset)
    print(">>>>>> [Report Generator Agent] Compiling section 2...")
    doc.add_heading("2. Summary of Analysis", level=1)
    
    section_ordering = 0
    # Assuming df_pricing has asset_class column. If not present in all rows, might crash.
    # We loop specific assets as per original code.
    for asset in df_pricing['asset_class'].unique():
        section_ordering += 1
        print(f">>>>>>>>> [Report Generator Agent] Compiling section 2.{section_ordering}: {asset}...")

        df_pricing_sub = df_pricing[df_pricing["asset_class"] == asset]
        sensitivity_test_results_sub = sensitivity_test_results[sensitivity_test_results["asset_class"] == asset]
        stress_test_results_sub = stress_test_results[stress_test_results["asset_class"] == asset]
        put_call_parity_sub = put_call_parity[put_call_parity["asset_class_put"] == asset]
        
        print(f">>>>>>>>>>>> [Report Generator Agent] Compiling section 2.{section_ordering}: {asset} pricing summary...")
        _generate_pricing_summary(doc, llm, asset, df_pricing_sub, section_ordering)
        
        print(f">>>>>>>>>>>> [Report Generator Agent] Compiling section 2.{section_ordering}: {asset} sensitivity test summary...")
        _generate_sensitivity_test_summary(doc, llm, asset, sensitivity_test_results_sub, section_ordering)
        
        print(f">>>>>>>>>>>> [Report Generator Agent] Compiling section 2.{section_ordering}: {asset} stress test summary...")
        _generate_stress_test_summary(doc, llm, asset, stress_test_results_sub, section_ordering)

        print(f">>>>>>>>>>>> [Report Generator Agent] Compiling section 2.{section_ordering}: {asset} put-call parity summary...")
        _generate_parity_summary(doc, llm, asset, put_call_parity_sub, section_ordering)

    # 4. Save
    doc.save(final_report_path)
    print(f">>> Report saved to {final_report_path}")

    state["errors"] = errors
    state["final_report_path"] = final_report_path
    
    return state

def _add_title_page(doc, params: Dict[str, str]):
    """Add title page and TOC"""
    # Title
    title_run = doc.add_paragraph().add_run(params["title"])
    title_run.bold = True
    title_run.font.size = doc.styles["Title"].font.size
    doc.paragraphs[-1].alignment = 1  # 1=center

    # Subtitle (Model Name)
    subtitle_para = doc.add_paragraph()
    subtitle_run = subtitle_para.add_run(params["model_name"])
    subtitle_para.alignment = 1
    subtitle_run.bold = True

    doc.add_paragraph("")  # spacer

    # Author
    author_para = doc.add_paragraph()
    author_para.add_run("Author: ").bold = True
    author_para.add_run(params["author_name"]).italic = False
    author_para.alignment = 1

    # Group
    group_para = doc.add_paragraph()
    group_para.add_run("Group: ").bold = True
    group_para.add_run(params["group_name"])
    group_para.alignment = 1

    # Report date
    date_str = datetime.date.today().strftime("%B %d, %Y")
    date_para = doc.add_paragraph()
    date_para.add_run("Report Date: ").bold = True
    date_para.add_run(date_str)
    date_para.alignment = 1

    # Version
    version_para = doc.add_paragraph()
    version_para.add_run("Document Version: ").bold = True
    version_para.add_run(params["version"])
    version_para.alignment = 1

    doc.add_page_break()

    # Table of Contents
    doc.add_heading("Table of Contents", level=1)
    toc_paragraph = doc.add_paragraph()
    toc_field = OxmlElement("w:fldSimple")
    toc_field.set(qn("w:instr"), 'TOC \\o "1-3" \\h \\z \\u')
    toc_paragraph._p.append(toc_field)
    doc.add_page_break()


def _add_section_1(doc, heading: str, paragraph: Optional[str]):
    """Add Introduction Section"""
    if not paragraph:
        paragraph = (
            "This section provides contextual background, objectives, and relevant "
            "considerations for the ongoing monitoring analysis. Subsequent sections "
            "expand on methodology, insights, and results."
        )

    doc.add_heading(heading, level=1)
    doc.add_paragraph(paragraph)


def _generate_pricing_summary(doc, llm, asset: str, df_pricing_sub: pd.DataFrame, section_ordering: int):
    """Generate pricing summary and validation section"""
    doc.add_heading(f"2.{section_ordering}.1 Summary of Pricing for {asset}", level=3)
    doc.add_paragraph(f"The pricing output of {asset} listed in the below table,")
    
    df = df_pricing_sub.sort_values("T").dropna(subset=['price', 'T'])
    
    

    # LLM Summary

    system_prompt = (
        "You are an expert Quantitative Risk Analyst. Your task is to perform a comprehensive audit and analysis of option portfolios based on provided CSV data."

        "Your analysis MUST follow this logical structure:"
        "1. **Data Quality & Integrity**: Check for logical consistency (e.g., non-negative prices, valid T/sigma values)."
        "2. **Pricing Dynamics**: Analyze the relationship between Spot (S) and Strike (K) to determine Moneyness (ITM/OTM) and how it affects price levels."
        "3. **Term Structure & Time Decay**: Evaluate how Time to Maturity (T) influences the pricing and the magnitude of Theta and Gamma."
        "4. **Risk Profile (Greeks Analysis)**: Contrast Call and Put behavior, focusing on Delta sensitivity and Vega exposure across different underlyings."
        "5. **Insights & Red Flags**: Identify any anomalies, such as extreme unexplained Greek values or potential arbitrage indicators in Put-Call pairs."

        "Maintain a professional, data-driven tone. Use precise financial terminology (e.g., time decay, convexity, rho sensitivity)."
    )
    user_prompt = (
        f"The following is an Option Analytics dataset of {asset} in CSV format:"
        f"Columns: {', '.join(df.columns)}\n"
        f"{df.to_csv(index=False)}\n\n"

        "Please provide a detailed Analysis Report covering the following points:"

        "### 1. Overall Data Quality"
        "- Evaluate if the dataset is consistent. "
        "- Do the Greeks (Delta, Gamma, etc.) align with standard Black-Scholes expectations for the given S, K, and T?"

        "### 2. Pricing Level & Moneyness"
        "- For each underlying, analyze the 'Price' relative to its 'Moneyness' (S/K ratio)."
        "- Identify which options are deep ITM or OTM and how that reflects in their Delta."

        "### 3. Term Structure Analysis"
        "- How does the longer duration (T > 1.0) impact the Vega and Rho compared to the shorter-dated options?"

        "### 4. Call vs Put Behavior"
        "- Explain the difference in their Delta and Rho values based on put-call parity logic."
        "- Comment on the 'Theta' behavior—which side is losing value faster?"

        "Final Summary: What is the primary risk exposure for this portfolio (e.g., directional risk, volatility risk, or interest rate risk)?"
    )

    ai_msg = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    ai_content = ai_msg.content
    add_markdown_to_docx(doc, ai_content)

    # Add Figure
    doc.add_heading(f"2.{section_ordering}.2 Visualization of Pricing for {asset}", level=3)
    for opt_type in ["call", "put"]:
        subset = df[df["option_type"] == opt_type]
        if subset.empty:
            continue
            
        # --- 1. 创建画布 ---
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # --- 2. 绘图逻辑 ---
        sns.lineplot(
            data=subset, 
            x="T", 
            y="price", 
            hue="underlying",  # 自动按 underlying 区分颜色
            marker="o", 
            ax=ax
        )
        
        ax.set_title(f"{asset} {opt_type.capitalize()} Price vs Time to Maturity")
        ax.set_xlabel("Time to Maturity (T)")
        ax.set_ylabel("Option Price")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Underlying", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # --- 3. 使用 BytesIO 保存到内存 ---
        img_stream = BytesIO()
        fig.savefig(img_stream, format="png", dpi=200, bbox_inches="tight")
        img_stream.seek(0)
        
        doc.add_picture(img_stream, width=Inches(5.5))
        
        # --- 5. 清理内存 ---
        plt.close(fig)      # 释放内存
        img_stream.close()  # 关闭流




def _generate_sensitivity_test_summary(doc, llm, asset: str, df_stress: pd.DataFrame, section_ordering: int):
    """Generate Sensitivity Test Summary"""
    doc.add_heading(f"2.{section_ordering}.2 Summary of Sensitivity Testing for {asset}", level=3)
    
    add_comparative_sensitivity_plots(doc, df_stress)

    system_prompt = (
        "You are a Senior Market Risk Controller. Your expertise is in evaluating option price sensitivities "
        "under various market shocks (Spot, Volatility, and Interest Rates).\n\n"
        "Your analysis must evaluate the following dimensions:\n"
        "1. **Delta/Gamma Profile**: Observe the price change across Spot Bumps. Is the relationship linear (Delta) or curved (Gamma)?\n"
        "2. **Vega Exposure**: Analyze how a +/- 5% move in Volatility affects the premium. Identify which underlyings are most 'Vega-sensitive'.\n"
        "3. **Rho Sensitivity**: Evaluate the impact of Rate shocks. Note if long-dated options (High T) show significantly higher sensitivity.\n"
        "4. **Risk Asymmetry**: Check if a -5% shock causes a larger loss than a +5% gain (Negative Skew/Gamma risk).\n\n"
        "Identify any 'Risk Outliers'—options that exhibit extreme price swings compared to their peers."
    )

    scen_cols = [c for c in df_stress.columns if 'sensitivity_scen' in c]
    base_cols = ['ID', 'underlying', 'S', 'K', 'T', 'option_type', 'price']
    
    display_df = df_stress[base_cols + scen_cols]
    
    user_prompt = (
        f"The following is a Sensitivity Analysis report for {asset} options portfolio (CSV format):\n\n"
        f"{display_df.to_csv(index=False)}\n\n"
        "Please provide a Risk Assessment Report:\n"
        "1. **Spot Shock Impact**: Which options are most at risk during a 5% market drop? Quantify the potential P&L loss.\n"
        "2. **Volatility Sensitivity**: How does a 5% increase in Volatility affect the portfolio value? Is the impact uniform across all assets?\n"
        "3. **Rate Sensitivity (Rho)**: Discuss the impact of a +500bps (+0.05) rate move, especially for longer-dated options.\n"
        "4. **Outlier Detection**: Are there any options where the price becomes zero or behaves erratically under stress?\n"
        "5. **Summary**: Provide a 'Top Risk' ranking for these options."
    )
    

    ai_msg = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    ai_content = ai_msg.content
    add_markdown_to_docx(doc, ai_content)


def _generate_stress_test_summary(doc, llm, asset: str, df_stress: pd.DataFrame, section_ordering: int):
    """Generate Stress Test Summary"""
    doc.add_heading(f"2.{section_ordering}.3 Summary of Stress Testing for {asset}", level=3)
    
    add_cross_asset_stress_plots(doc, df_stress)

    system_prompt = (
        "You are a Chief Risk Officer (CRO). You are reviewing the results of a high-impact Stress Testing suite.\n\n"
        "Your objective is to identify systemic vulnerabilities in the options portfolio. Focus your analysis on:\n"
        "1. **Tail Risk Severity**: Which historical scenarios (e.g., 1987 Black Monday, 2008 Crisis) cause the most catastrophic loss?\n"
        "2. **Factor Sensitivity**: Determine if the P&L impact is driven by directional moves (Spot), volatility spikes (VIX Spike), or rate shocks.\n"
        "3. **Non-linear Behavior**: Identify cases where the price change is disproportionate to the shock (Gamma/Vega acceleration).\n"
        "4. **Portfolio Robustness**: Compare different underlyings. Which asset class or specific ticker is the 'weak link' in the current portfolio?\n"
        "5. **Operational Integrity**: Flag any negative prices or zero-value scenarios that may indicate model breakdown or total wipeout of premium."
    )

    stress_cols = [c for c in df_stress.columns if c.startswith('stress_scen_')]
    base_info = ['ID', 'underlying', 'S', 'K', 'T', 'option_type', 'price', 'delta', 'vega']
    
    display_df = df_stress[base_info + stress_cols]
    
    user_prompt = (
        f"Portfolio Stress Test Data for {asset}:\n\n"
        f"{display_df.to_csv(index=False)}\n\n"
        "Please provide a Strategic Risk Memo:\n"
        "1. **Worst-Case Scenario**: Identify the single scenario that creates the maximum drawdown. Explain why based on the option's Delta/Vega.\n"
        "2. **Volatility vs. Spot**: Compare the 'VIX Spike' impact vs. the '2008 Financial Crisis'. Which is more damaging?\n"
        "3. **Greeks Correlation**: How did the initial Delta and Vega predict these outcomes? Were there any surprises?\n"
        "4. **Liquidation Risk**: In the 'Liquidation Scenario', evaluate the surge in option premiums. What does this mean for margin requirements?\n"
        "5. **Actionable Recommendation**: Suggest one hedging action to mitigate the top identified risk."
    )

    ai_msg = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    ai_content = ai_msg.content
    add_markdown_to_docx(doc, ai_content)


def _generate_parity_summary(doc, llm, asset: str, df: pd.DataFrame, section_ordering: int):
    """Generate Gamma Positivity Summary"""
    doc.add_heading(f"2.{section_ordering}.4 Summary of Put-Call Parity Testing for {asset}", level=3)
    
    add_parity_test_plots(doc, df)

    system_prompt = (
        "You are a Senior Quantitative Model Validator. Your task is to audit the Put-Call Parity consistency "
        "of an options pricing engine.\n\n"
        "Your analysis must follow these principles:\n"
        "1. **Theoretical Alignment**: Verify the formula C + K*exp(-rT) = S + P. LHS and RHS must match.\n"
        "2. **Numerical Precision**: Distinguish between 'Model Errors' and 'Floating-point Noise'. "
        "Absolute differences (abs_diff) in the magnitude of 1e-12 or smaller should be classified as acceptable "
        "computational rounding errors.\n"
        "3. **Arbitrage Identification**: If significant deviations exist, identify if the mispricing favors "
        "'Conversion' (Long Stock/Put, Short Call) or 'Reversal' (Short Stock/Put, Long Call) strategies.\n"
        "4. **Greeks Consistency**: Briefly comment on the Delta relationship (Delta_call - Delta_put should equal 1).\n\n"
        "Provide a professional, concise summary of whether the model passes the logical consistency check."
    )
    analysis_cols = [
        'underlying', 'S', 'K', 'T', 'r', 
        'price_call', 'price_put', 'lhs', 'rhs', 'abs_diff', 'is_parity_valid'
    ]
    
    csv_snippet = df[analysis_cols].to_csv(index=False)
    
    user_prompt = (
        f"The following is the Put-Call Parity test data for {asset}:\n\n"
        f"{csv_snippet}\n"
        "Please provide a Model Consistency Report:\n"
        "1. **Global Verdict**: Does the pricing engine satisfy the Put-Call Parity theorem across all assets?\n"
        "2. **Precision Audit**: Explain the 'abs_diff' values found for NVDA and MSFT (e.g., 2.84e-14). "
        "Are these indicative of a model flaw or numerical noise?\n"
        "3. **Asset Specifics**: Highlight any specific ticker that shows unusual behavior, or confirm if the "
        "performance is uniform.\n"
        "4. **Confidence Statement**: State whether the generated Call and Put prices are 'clean' and reliable "
        "for downstream risk management and P&L attribution."
    )

    ai_msg = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    ai_content = ai_msg.content
    add_markdown_to_docx(doc, ai_content)


def add_markdown_to_docx(doc, text):
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('###'):
            clean_text = line.lstrip('#').strip()
            level = line.count('#') - 1 
            doc.add_heading(clean_text, level=min(level, 3))
            
        elif line.startswith(('-', '*')):
            clean_text = line.lstrip('-*').strip()
            p = doc.add_paragraph(style='List Bullet')
            _add_formatted_text(p, clean_text)
            
        elif line == '---':
            doc.add_page_break() 
            
        else:
            p = doc.add_paragraph()
            _add_formatted_text(p, line)

def _add_formatted_text(paragraph, text):
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            run = paragraph.add_run(part.strip('*'))
            run.bold = True
        else:
            paragraph.add_run(part)

def add_comparative_sensitivity_plots(doc, df_sensitivity):
    factors = {
        'spot': 'Spot Price Bump (%)',
        'vol': 'Volatility Bump (abs)',
        'rate': 'Interest Rate Bump (abs)'
    }
    
    for opt_type in ["call", "put"]:
        subset_orig = df_sensitivity[df_sensitivity["option_type"] == opt_type]
        if subset_orig.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f"Cross-Asset {opt_type.upper()} Sensitivity Comparison", fontsize=16)

        for i, (factor, label) in enumerate(factors.items()):
            scen_cols = [c for c in df_sensitivity.columns if f"scen_{factor}_bump" in c]
            
            comparison_data = []
            for _, row in subset_orig.iterrows():
                for col in scen_cols:
                    try:
                        bump_val = float(col.split('_bump_')[-1])
                        comparison_data.append({
                            'Bump': bump_val,
                            'Price': row[col],
                            'Underlying': row['underlying'],
                            'ID': row['ID']
                        })
                    except: continue
            
            df_plot = pd.DataFrame(comparison_data).sort_values('Bump')

            sns.lineplot(
                data=df_plot, 
                x='Bump', 
                y='Price', 
                hue='Underlying', 
                marker='o', 
                ax=axes[i]
            )
            
            axes[i].set_title(f"Sensitivity to {factor.capitalize()}")
            axes[i].set_xlabel(label)
            axes[i].set_ylabel("Option Price")
            axes[i].grid(True, alpha=0.3)
            axes[i].axvline(x=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        img_stream = BytesIO()
        fig.savefig(img_stream, format="png", dpi=200, bbox_inches="tight")
        img_stream.seek(0)
        doc.add_picture(img_stream, width=Inches(6.2))
        
        plt.close(fig)
        img_stream.close()


def add_cross_asset_stress_plots(doc, df_stress):
    stress_cols = [c for c in df_stress.columns if c.startswith('stress_scen_')]
    
    for opt_type in ["call", "put"]:
        subset = df_stress[df_stress["option_type"] == opt_type]
        if subset.empty:
            continue
            
        plot_data = []
        for _, row in subset.iterrows():
            plot_data.append({
                'Underlying': row['underlying'],
                'Price': row['price'],
                'Scenario': 'Baseline'
            })
            for col in stress_cols:
                scen_name = col.replace('stress_scen_', '')
                plot_data.append({
                    'Underlying': row['underlying'],
                    'Price': row[col],
                    'Scenario': scen_name
                })
        
        df_plot = pd.DataFrame(plot_data)

        plt.figure(figsize=(14, 8))
        ax = sns.barplot(
            data=df_plot, 
            x='Underlying', 
            y='Price', 
            hue='Scenario',
            palette='tab10'
        )
        
        plt.title(f"Cross-Asset Impact Analysis: {opt_type.upper()} Options", fontsize=15)
        plt.ylabel("Option Price")
        plt.xlabel("Underlying Asset")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.legend(title="Stress Scenarios", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        img_stream = BytesIO()
        plt.savefig(img_stream, format="png", dpi=200, bbox_inches="tight")
        img_stream.seek(0)
        
        doc.add_heading(f"{opt_type.capitalize()} Stress Exposure by Underlying", level=2)
        doc.add_picture(img_stream, width=Inches(6.2))
        
        plt.close()
        img_stream.close()


def add_parity_test_plots(doc, df_parity):
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.scatterplot(data=df_parity, x='lhs', y='rhs', hue='underlying', s=100, ax=ax1)
    
    max_val = max(df_parity['lhs'].max(), df_parity['rhs'].max())
    min_val = min(df_parity['lhs'].min(), df_parity['rhs'].min())
    ax1.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', alpha=0.6, label='Ideal Parity')
    
    ax1.set_title("Put-Call Parity: LHS vs RHS", fontsize=14)
    ax1.set_xlabel("LHS (Call + PV of Strike)")
    ax1.set_ylabel("RHS (Spot + Put)")
    ax1.legend()

    sns.barplot(data=df_parity, x='ID_call', y='abs_diff', hue='underlying', ax=ax2)
    ax2.set_title("Arbitrage Deviation (Abs Diff)", fontsize=14)
    ax2.set_ylabel("Difference Value")
    ax2.set_xlabel("Pair ID")
    
    plt.tight_layout()

    img_stream = BytesIO()
    fig.savefig(img_stream, format="png", dpi=200, bbox_inches="tight")
    img_stream.seek(0)
    
    doc.add_heading("Put-Call Parity Validation Analysis", level=2)
    doc.add_picture(img_stream, width=Inches(6.0))
    
    plt.close(fig)
    img_stream.close()