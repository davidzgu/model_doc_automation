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
import io

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


def _generate_pricing_summary(doc, llm, asset: str, df: pd.DataFrame, section_ordering: int):
    """Generate pricing summary and validation section"""
    doc.add_heading(f"2.{section_ordering}.1 Summary of Pricing for {asset}", level=3)
    doc.add_paragraph(f"The pricing output of {asset} listed in the below table,")
    
    df = df.sort_values("T").dropna(subset=['price', 'T'])

    df_str = df.round(4).to_csv(index=False)

    system_prompt_text = (
        "You are an expert Quantitative Analyst writing a formal component for an Option Pricing Analysis (OPA) Report. "
        "Your output will be directly embedded into a professional document for senior management.\n\n"
        "### STRICT GUIDELINES:\n"
        "1. **Tone**: Use strictly professional, objective, and formal financial language. (e.g., Use 'The data indicates...' instead of 'I think...').\n"
        "2. **No Conversational Fillers**: Do NOT use phrases like 'Okay', 'Let's see', 'Let me check', 'Wait', or 'Here is the analysis'.\n"
        "3. **No Internal Monologue**: Do NOT output your thinking process. Only output the final analytical conclusions.\n"
        "4. **Direct Start**: Start directly with the bullet points or the section content.\n"
        "5. **Formatting**: Use standard Markdown (bolding for key metrics) suitable for a final report."
    )
    sys_msg = SystemMessage(content=system_prompt_text)

    prompt_quality = (
        f"Dataset for {asset}:\n{df_str}\n\n"
        "### Task: Write the 'Data Quality & Integrity' section.\n"
        "Draft a concise assessment covering:\n"
        "- Consistency of prices and volatility.\n"
        "- Alignment of Greeks with Black-Scholes expectations.\n"
        "- Specific trade anomalies (if any).\n"
        "Output Format: Bullet points."
    )

    # prompt_pricing = (
    #     f"Dataset for {asset}:\n{df_str}\n\n"
    #     "### Task: Pricing Dynamics & Moneyness Analysis\n"
    #     "1. Analyze Price relative to Moneyness (S/K). Verify ITM vs OTM pricing levels.\n"
    #     "2. Explain Call vs Put behavior: Compare Delta/Rho based on parity logic.\n"
    #     "3. Highlight how deep ITM/OTM status reflects in the Greeks.\n"
    #     "Output concise bullet points."
    # )

    prompt_term = (
        f"Dataset for {asset}:\n{df_str}\n\n"
        "### Task: Write the 'Term Structure & Time Decay' section.\n"
        "Draft a concise analysis covering:\n"
        "- Impact of Time to Maturity (T) on Price, Vega, and Rho.\n"
        "- Identification of options with the highest Theta decay.\n"
        "- Gamma risk concentration.\n"
        "Output Format: Bullet points."
    )

    batch_inputs = [
        [sys_msg, HumanMessage(content=prompt_quality)],
        # [sys_msg, HumanMessage(content=prompt_pricing)],
        [sys_msg, HumanMessage(content=prompt_term)]
    ]
    results = llm.batch(batch_inputs)

    res_quality = results[0].content
    res_term = results[1].content

    # summary_prompt = (
    #     f"Based on the following three analyses for {asset}, provide a 2-sentence 'Final Risk Summary' "
    #     "identifying the primary exposure (directional, vol, or rates).\n\n"
    #     f"1. Quality: {res_quality}\n"
    #     f"2. Pricing: {res_pricing}\n"
    #     f"3. Term: {res_term}"
    # )

    # final_risk_msg = llm.invoke([sys_msg, HumanMessage(content=summary_prompt)])
    # final_risk_summary = final_risk_msg.content


    full_report = (
        f"### 1. Data Quality & Integrity\n{res_quality}\n\n"
        f"### 2. Term Structure & Time Decay\n{res_term}\n\n"
    )

    add_markdown_to_docx(doc, full_report)

    # Add Figure
    doc.add_heading(f"2.{section_ordering}.2 Visualization of Pricing for {asset}", level=3)
    for opt_type in ["call", "put"]:
        subset = df[df["option_type"] == opt_type]
        if subset.empty:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.lineplot(
            data=subset, 
            x="T", 
            y="price", 
            hue="underlying",  
            marker="o", 
            ax=ax
        )
        
        ax.set_title(f"{asset} {opt_type.capitalize()} Price vs Time to Maturity")
        ax.set_xlabel("Time to Maturity (T)")
        ax.set_ylabel("Option Price")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Underlying", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        img_stream = BytesIO()
        fig.savefig(img_stream, format="png", dpi=200, bbox_inches="tight")
        img_stream.seek(0)
        
        doc.add_picture(img_stream, width=Inches(5.5))
        
        plt.close(fig)      
        img_stream.close()  




def _generate_sensitivity_test_summary(doc, llm, asset: str, df: pd.DataFrame, section_ordering: int):
    """Generate Sensitivity Test Summary"""
    doc.add_heading(f"2.{section_ordering}.2 Summary of Sensitivity Testing for {asset}", level=3)
    
    df_spot = aggregate_sensitivity_data(df, 'spot')
    df_vol = aggregate_sensitivity_data(df, 'vol')
    df_rate = aggregate_sensitivity_data(df, 'rate')

    cols_to_show = ['Bump', 'Net_PnL_Disp', 'Portfolio_Value_Disp']
    spot_str = df_spot[cols_to_show].to_markdown(index=False) if df_spot is not None else "No Data"
    vol_str = df_vol[cols_to_show].to_markdown(index=False) if df_vol is not None else "No Data"
    rate_str = df_rate[cols_to_show].to_markdown(index=False) if df_rate is not None else "No Data"

    sys_msg = SystemMessage(content=(
        "You are a Senior Risk Manager analyzing Portfolio Stress Tests. "
        "Analyze the aggregated PnL data provided. Be concise, professional, and focus on risk exposures."
    ))

    # Define the 3 prompts
    prompts = [
        # Spot / Gamma
        f"Portfolio Spot Sensitivity for {asset}:\n{spot_str}\n\n"
        "### Task: Spot & Gamma Analysis\n"
        "1. **Directional Bias**: Does the portfolio gain on Up or Down moves? (Delta view)\n"
        "2. **Convexity**: Is the PnL curve convex (Long Gamma) or concave (Short Gamma)? Check if losses accelerate in one direction.\n"
        "3. **Tail Risk**: Comment on the worst-case loss in the table.",
        
        # Vol / Vega
        f"Portfolio Volatility Sensitivity for {asset}:\n{vol_str}\n\n"
        "### Task: Volatility Analysis\n"
        "1. **Vega Exposure**: Is the portfolio Long Vega (gains when vol rises) or Short Vega?\n"
        "2. **Impact**: Is the PnL impact significant compared to the spot movements?",
        
        # Rate / Rho
        f"Portfolio Rate Sensitivity for {asset}:\n{rate_str}\n\n"
        "### Task: Interest Rate Analysis\n"
        "1. **Rho Exposure**: How does the portfolio react to rate hikes?\n"
        "2. **Implication**: Briefly explain what this implies about the positioning (e.g., Net Long/Short)."
    ]

    # Run LLM (Batch)
    inputs = [[sys_msg, HumanMessage(content=p)] for p in prompts]
    results = llm.batch(inputs)

    res_spot = results[0].content
    res_vol = results[1].content
    res_rate = results[2].content
    full_report = (
        f"### Portfolio Spot Stress (Gamma Profile)\n{res_spot}\n\n"
        f"### Portfolio Volatility Stress (Vega Profile)\n{res_vol}\n\n"
        f"### Portfolio Rate Stress (Rho Profile)\n{res_rate}"
    )
    add_markdown_to_docx(doc, full_report)

    # 1. Spot Chart (Most Important)
    img_spot = create_sensitivity_chart(df_spot, 'spot', asset)
    if img_spot:
        doc.add_picture(img_spot, width=Inches(5))
        doc.add_paragraph(f"Figure: Spot Price Sensitivity for {asset}", style="Caption")

    # 2. Vol Chart
    img_vol = create_sensitivity_chart(df_vol, 'vol', asset)
    if img_vol:
        doc.add_picture(img_vol, width=Inches(5))
        doc.add_paragraph(f"Figure: Volatility Sensitivity for {asset}", style="Caption")
        
    # 3. Rate Chart (Optional, usually less visual, but good for completeness)
    img_rate = create_sensitivity_chart(df_rate, 'rate', asset)
    if img_rate:
        doc.add_picture(img_rate, width=Inches(5))
        doc.add_paragraph(f"Figure: Interest Rate Sensitivity for {asset}", style="Caption")
    
    add_comparative_sensitivity_plots(doc, df)

    


def _generate_stress_test_summary(doc, llm, asset: str, df: pd.DataFrame, section_ordering: int):
    """Generate Stress Test Summary"""
    doc.add_heading(f"2.{section_ordering}.3 Summary of Stress Testing for {asset}", level=3)
    
    df_stress_raw, df_llm_clean = prepare_stress_test_data(df)
    table_str = df_llm_clean.to_markdown(index=False)

    system_prompt = (
        "You are a Chief Risk Officer (CRO). You are reviewing the Stress Test Report for an investment portfolio. "
        "Your job is to identify catastrophic risks and assess portfolio survivability. "
        "Be direct, identify the worst-case scenarios, and explain WHY the portfolio behaves this way based on the scenarios."
        "IMPORTANT FORMATTING RULES:\n"
        "1. Do NOT include a document title (e.g., 'Risk Assessment').\n"
        "2. Do NOT include metadata like 'Date', 'Prepared by', or 'To/From'.\n"
        "3. Do NOT include introductory phrases (e.g., 'Here is the report').\n"
        "4. Start directly with the first section header (e.g., ### 1. Tail Risk Severity).\n"
        "5. Use Markdown formatting."
    )
    
    user_prompt = (
        f"Stress Test Results for {asset} (Sorted by worst loss first):\n\n"
        f"{table_str}\n\n"
        "Please provide a Risk Assessment covering:\n"
        "1. **Tail Risk Severity**: Which historic or hypothetical scenario causes the most damage? Is it a 'Terminal Event' (wipeout)?\n"
        "2. **Factor Sensitivity**: Compare scenarios. Does the portfolio suffer more from Market Crashes (e.g., 2008) or Rate/Vol Shocks?\n"
        "3. **Hedge Effectiveness**: Look at the 'Worst Link'. Are we properly hedged, or is one specific asset dragging the whole portfolio down?\n"
        "4. **Operational Check**: Are there any scenarios showing positive PnL? If so, does that make sense (e.g., Short positions or Long Volatility)?\n"
        "5. **Final Verdict**: Classify the portfolio risk profile as 'Robust', 'Fragile', or 'Directionally Biased'."
    )
    ai_msg = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    add_markdown_to_docx(doc, ai_msg.content)

    img_stress = create_stress_chart(df_stress_raw, asset)
    if img_stress:
        doc.add_picture(img_stress, width=Inches(5.5))
        doc.add_paragraph(f"Figure: Stress Scenario Impact on {asset}", style="Caption")

    add_cross_asset_stress_plots(doc, df)

    


def _generate_parity_summary(doc, llm, asset: str, df: pd.DataFrame, section_ordering: int):
    """Generate Gamma Positivity Summary"""
    doc.add_heading(f"2.{section_ordering}.4 Summary of Put-Call Parity Testing for {asset}", level=3)
    

    _, summary_md = prepare_parity_summary(df)


    system_prompt = (
        "You are an Arbitrage Trader and Quantitative Researcher. "
        "Your task is to validate the 'Put-Call Parity' relationship for an options portfolio. "
        "Identify if deviations are noise or actionable arbitrage opportunities.\n\n"
        "FORMATTING RULES:\n"
        "1. Do NOT use document titles, dates, or signatures.\n"
        "2. Start directly with the analysis headers.\n"
        "3. Be concise and data-driven."
    )
    
    user_prompt = (
        f"Put-Call Parity Analysis for {asset}:\n\n"
        f"{summary_md}\n\n"
        "Please provide a structured analysis:\n"
        "1. **Model Consistency**: Is the pass rate acceptable? If low, what does it imply about the data quality or market efficiency?\n"
        "2. **Arbitrage Assessment**: Look at the 'Top 5 Violations'. Are the 'abs_diff' values large enough to cover transaction costs, or are they negligible?\n"
        "3. **Strike Bias**: (Hypothetically) Does the violation usually occur at deep ITM/OTM strikes due to liquidity issues?\n"
        "4. **Recommendation**: Should we flag these data points for cleaning, or execute arbitrage trades?"
    )

    ai_msg = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    add_markdown_to_docx(doc, ai_msg.content)

    img_reg, img_dev = create_parity_charts(df, asset)
    
    if img_reg:
        doc.add_paragraph("Visual verification of the parity relationship:")
        doc.add_picture(img_reg, width=Inches(5))
        doc.add_paragraph(f"Figure: Parity Regression (LHS vs RHS) for {asset}", style="Caption")
        
    if img_dev:
        doc.add_picture(img_dev, width=Inches(5))
        doc.add_paragraph(f"Figure: Deviation Magnitude across Strikes for {asset}", style="Caption")
    
    add_parity_test_plots(doc, df)



def _process_inline_formatting(paragraph, text):
    pattern = r'(\*\*.*?\*\*|\*.*?\*)'
    parts = re.split(pattern, text)

    for part in parts:
        if not part:
            continue
        
        run = paragraph.add_run()
        
        if part.startswith('**') and part.endswith('**') and len(part) >= 4:
            clean_text = part[2:-2]
            run.text = clean_text
            run.bold = True
            
        elif part.startswith('*') and part.endswith('*') and len(part) >= 2:
            clean_text = part[1:-1]
            run.text = clean_text
            run.italic = True
            
        else:
            run.text = part


def add_markdown_to_docx(doc, text):
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('###'):
            clean_text = line.replace('#', '').strip()
            clean_text = clean_text.replace('**', '').replace('*', '')
            doc.add_heading(clean_text, level=3)
            
        elif line.startswith('**') and line.endswith('**') and len(line) < 80:
            clean_text = line[2:-2].strip()
            doc.add_heading(clean_text, level=3)
        elif line.startswith(('-', '*')) and len(line) > 1 and line[1] == ' ':
            clean_text = line[2:].strip()
            p = doc.add_paragraph(style='List Bullet')
            _process_inline_formatting(p, clean_text)
            
        elif re.match(r'^\d+\.\s', line):
            clean_text = re.sub(r'^\d+\.\s', '', line).strip()
            p = doc.add_paragraph(style='List Number')
            _process_inline_formatting(p, clean_text)
            
        elif line == '---':
            p = doc.add_paragraph()
            p.add_run().add_break()
            
        else:
            p = doc.add_paragraph()
            _process_inline_formatting(p, line)

def aggregate_sensitivity_data(df, scenario_type='spot'):
    """
    Aggregates sensitivity columns across ALL trades to create a Portfolio View.
    Returns a DataFrame suitable for both LLM analysis and Plotting.
    
    Args:
        df: The raw dataframe containing trade-level sensitivity columns.
        scenario_type: 'spot', 'vol', or 'rate'.
    """
    # Identify columns based on naming convention
    prefix = f"sensitivity_scen_{scenario_type}_bump_"
    cols = [c for c in df.columns if prefix in c]
    
    if not cols:
        return None

    # Sum across all rows (trades) to get Portfolio Total Price per bump
    portfolio_totals = df[cols].sum()
    
    # Restructure into a clean DataFrame
    data_points = []
    for col_name, total_price in portfolio_totals.items():
        try:
            # Extract numerical bump value from column name (e.g., "-0.05")
            bump_val = float(col_name.replace(prefix, ""))
            data_points.append({"Bump": bump_val, "Portfolio_Value": total_price})
        except ValueError:
            continue
            
    df_agg = pd.DataFrame(data_points).sort_values("Bump")
    
    # Calculate PnL (Change from Baseline)
    # Try to find exactly 0 bump, otherwise use mean (fallback)
    if 0 in df_agg['Bump'].values:
        base_val = df_agg.loc[df_agg['Bump'] == 0, 'Portfolio_Value'].values[0]
    else:
        base_val = df_agg['Portfolio_Value'].mean()

    df_agg['Net_PnL'] = df_agg['Portfolio_Value'] - base_val
    
    # Rounding for token efficiency
    df_agg['Portfolio_Value_Disp'] = df_agg['Portfolio_Value'].round(2)
    df_agg['Net_PnL_Disp'] = df_agg['Net_PnL'].round(2)
    
    return df_agg

def create_sensitivity_chart(df_agg, scenario_type, asset_name):
    """
    Generates a PnL profile chart and returns a BytesIO buffer.
    """
    if df_agg is None or df_agg.empty:
        return None

    plt.figure(figsize=(6, 3.5)) # Compact size for Word doc
    
    # Plot Line
    plt.plot(df_agg['Bump'], df_agg['Net_PnL'], marker='o', linewidth=2, color='#1f77b4', label='Net PnL')
    
    # Reference Lines
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')

    # Color zones (Green for profit, Red for loss)
    plt.fill_between(df_agg['Bump'], df_agg['Net_PnL'], 0, where=(df_agg['Net_PnL']>=0), 
                     facecolor='green', alpha=0.1, interpolate=True)
    plt.fill_between(df_agg['Bump'], df_agg['Net_PnL'], 0, where=(df_agg['Net_PnL']<0), 
                     facecolor='red', alpha=0.1, interpolate=True)

    # Styling
    scenario_label = scenario_type.capitalize()
    if scenario_type == 'vol': scenario_label = "Volatility"
    
    plt.title(f"{asset_name}: {scenario_label} Sensitivity PnL", fontsize=10, weight='bold')
    plt.xlabel(f"{scenario_label} Bump", fontsize=9)
    plt.ylabel("Portfolio PnL ($)", fontsize=9)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    # Save to buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150)
    plt.close()
    img_buffer.seek(0)
    
    return img_buffer


def prepare_stress_test_data(df):
    """
    Aggregates stress test results to Portfolio Level.
    Also identifies the 'Weak Link' (worst performing asset) for each scenario.
    """
    # Filter columns
    prefix = "stress_scen_"
    cols = [c for c in df.columns if prefix in c]
    
    if not cols:
        return None, None

    # Calculate Current Portfolio Value (Baseline)
    current_val = df['price'].sum()
    
    # Structure to hold summary
    summary_rows = []
    
    for col in cols:
        scenario_name = col.replace(prefix, "")
        
        # 1. Total Portfolio Value in this scenario
        scenario_val = df[col].sum()
        
        # 2. Net PnL (Scenario - Current)
        pnl = scenario_val - current_val
        pnl_pct = (pnl / current_val) * 100 if current_val != 0 else 0
        
        # 3. Find the "Weak Link" (The underlying/asset contributing most to the loss)
        # We calculate row-wise PnL for this scenario
        df['temp_pnl'] = df[col] - df['price']
        # Group by Underlying or Asset Class to find the worst performer
        # Assuming 'underlying' column exists, otherwise use ID
        group_col = 'underlying' if 'underlying' in df.columns else 'ID'
        worst_performer = df.groupby(group_col)['temp_pnl'].sum().idxmin()
        worst_loss = df.groupby(group_col)['temp_pnl'].sum().min()
        
        summary_rows.append({
            "Scenario": scenario_name,
            "Portfolio_PnL": pnl,
            "PnL_Pct": pnl_pct,
            "Worst_Link": f"{worst_performer} (${worst_loss:,.0f})"
        })
        
    df_stress = pd.DataFrame(summary_rows)
    
    # Sort by PnL ascending (Worst losses first)
    df_stress = df_stress.sort_values("Portfolio_PnL", ascending=True)
    
    # Create a clean version for LLM (Rounding)
    df_llm = df_stress.copy()
    df_llm['Portfolio_PnL'] = df_llm['Portfolio_PnL'].apply(lambda x: f"${x:,.0f}")
    df_llm['PnL_Pct'] = df_llm['PnL_Pct'].round(2).astype(str) + "%"
    
    return df_stress, df_llm

def create_stress_chart(df_stress, asset_name):
    """
    Generates a Horizontal Bar Chart for Stress Scenarios.
    Red bars for losses, Green for gains.
    """
    if df_stress is None or df_stress.empty:
        return None
        
    plt.figure(figsize=(7, 4))
    
    # Color logic
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in df_stress['Portfolio_PnL']]
    
    # Horizontal Bar Plot
    bars = plt.barh(df_stress['Scenario'], df_stress['Portfolio_PnL'], color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width > 0 else width
        align = 'left' if width > 0 else 'right'
        
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                 f' ${width:,.0f}', 
                 va='center', ha=align, fontsize=8, fontweight='bold')

    plt.axvline(0, color='black', linewidth=0.8)
    plt.title(f"Stress Test Results: {asset_name}", fontsize=11, weight='bold')
    plt.xlabel("Estimated PnL ($)", fontsize=9)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save to buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150)
    plt.close()
    img_buffer.seek(0)
    
    return img_buffer

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

def prepare_parity_summary(df):
    """
    Calculates statistical summaries of Put-Call Parity checks.
    Returns a dictionary of stats and a Markdown string for the LLM.
    """
    total_pairs = len(df)
    if total_pairs == 0:
        return None, "No data available."
    
    # 1. Basic Stats
    num_valid = df['is_parity_valid'].sum()
    pass_rate = (num_valid / total_pairs) * 100
    
    mean_diff = df['abs_diff'].mean()
    max_diff = df['abs_diff'].max()
    median_diff = df['abs_diff'].median()
    
    # 2. Arbitrage Opportunities (Assume threshold > 0.1 or specific column logic)
    # If 'arbitrage_opportunity' column is boolean:
    arb_ops = df[df['arbitrage_opportunity'] == True]
    num_arbs = len(arb_ops)
    
    # 3. Identify Top Violations (Worst 5)
    # Sort by absolute difference descending
    worst_cases = df.sort_values('abs_diff', ascending=False).head(5)
    
    # Format worst cases for LLM
    # We select key columns to keep tokens low
    cols_to_show = ['K', 'T', 'price_call', 'price_put', 'abs_diff', 'is_parity_valid']
    worst_table = worst_cases[cols_to_show].round(4).to_markdown(index=False)
    
    # 4. Construct Summary String
    summary_md = (
        f"### Parity Statistics\n"
        f"- **Total Pairs Analyzed**: {total_pairs}\n"
        f"- **Pass Rate**: {pass_rate:.2f}% ({num_valid}/{total_pairs})\n"
        f"- **Mean Deviation**: {mean_diff:.4f}\n"
        f"- **Max Deviation**: {max_diff:.4f}\n"
        f"- **Flagged Arbitrage Opps**: {num_arbs}\n\n"
        f"### Top 5 Worst Violations (Largest Diff)\n"
        f"{worst_table}"
    )
    
    return df, summary_md



def create_parity_charts(df, asset_name):
    """
    Generates 2 charts: 
    1. LHS vs RHS Scatter (The Parity Line)
    2. Strike vs Deviation (Error Distribution)
    """
    if df is None or df.empty:
        return None, None

    # Chart 1: Parity Regression (LHS vs RHS)
    plt.figure(figsize=(5, 4))
    
    # Plot perfect parity line
    max_val = max(df['lhs'].max(), df['rhs'].max())
    min_val = min(df['lhs'].min(), df['rhs'].min())
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', alpha=0.5, label='Theoretical Parity')
    
    # Scatter plot of actual data
    # Color by validity (Green=Good, Red=Bad)
    colors = df['is_parity_valid'].map({True: 'green', False: 'red'})
    plt.scatter(df['lhs'], df['rhs'], c=colors, s=15, alpha=0.6, label='Observed Pairs')
    
    plt.title(f"{asset_name}: Put-Call Parity Regression", fontsize=10)
    plt.xlabel("LHS (Call + PV(K))")
    plt.ylabel("RHS (Put + Spot)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    img_regression = io.BytesIO()
    plt.savefig(img_regression, format='png', dpi=150)
    plt.close()
    img_regression.seek(0)
    
    # Chart 2: Deviation by Strike
    plt.figure(figsize=(5, 4))
    plt.scatter(df['K'], df['abs_diff'], color='purple', s=15, alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.8)
    
    plt.title(f"{asset_name}: Parity Deviation by Strike", fontsize=10)
    plt.xlabel("Strike Price (K)")
    plt.ylabel("Absolute Difference ($)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    img_deviation = io.BytesIO()
    plt.savefig(img_deviation, format='png', dpi=150)
    plt.close()
    img_deviation.seek(0)
    
    return img_regression, img_deviation


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