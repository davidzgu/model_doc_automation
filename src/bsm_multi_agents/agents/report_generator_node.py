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
    pricing_results_path = f"{output_dir}/{p.stem}_greeks_results{p.suffix}"
    validate_results_path = f"{output_dir}/{p.stem}_greeks_results_validate_results{p.suffix}"
    stress_test_results_path = f"{output_dir}/{p.stem}_greeks_results_stress_test_results{p.suffix}"
    gamma_positivity_test_results_path = f"{output_dir}/{p.stem}_greeks_results_gamma_positivity_test_results{p.suffix}"
    final_report_path = state.get("final_report_path")

    # Load DataFrames
    df_pricing = pd.read_csv(pricing_results_path)
    stress_test_results = pd.read_csv(stress_test_results_path)
    gamma_positivity_test_results = pd.read_csv(gamma_positivity_test_results_path)

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
    for asset in ["FX", "Equity", "Commodity"]:
        section_ordering += 1
        print(f">>>>>>>>> [Report Generator Agent] Compiling section 2.{section_ordering}: {asset}...")

        df_pricing_sub = df_pricing[df_pricing["asset_class"] == asset]
        df_gamma_positivity_test_results_sub = gamma_positivity_test_results[gamma_positivity_test_results["asset_class"] == asset]
        df_stress_test_results_sub = stress_test_results[stress_test_results["asset_class"] == asset]
        
        print(f">>>>>>>>>>>> [Report Generator Agent] Compiling section 2.{section_ordering}: {asset} pricing summary...")
        _generate_pricing_summary(doc, llm, asset, df_pricing_sub, section_ordering)
        
        print(f">>>>>>>>>>>> [Report Generator Agent] Compiling section 2.{section_ordering}: {asset} gamma positivity summary...")
        _generate_gamma_summary(doc, llm, asset, df_gamma_positivity_test_results_sub, section_ordering)
        
        print(f">>>>>>>>>>>> [Report Generator Agent] Compiling section 2.{section_ordering}: {asset} stress test summary...")
        _generate_stress_test_summary(doc, llm, asset, df_stress_test_results_sub, section_ordering)

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
    
    df = df_pricing_sub.sort_values("T").dropna(subset=['BSM_price','T'])
    df_call = df[df["option_type"] == "call"]
    df_put = df[df["option_type"] == "put"]

    # Generate Plot
    fig, ax = plt.subplots()
    ax.plot(df_call["T"], df_call["BSM_price"], label="call")
    ax.plot(df_put["T"], df_put["BSM_price"], label="put")
    ax.set_xlabel("Time to Maturity (T)")
    ax.set_ylabel("Option Price (BSM)")
    ax.set_title(f"Option Pricing Curve – {asset}")  
    ax.legend()

    img_stream = BytesIO()
    fig.savefig(img_stream, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    img_stream.seek(0)

    # LLM Summary
    num_calls = len(df_call)
    num_puts = len(df_put)

    system_prompt = (
        "You are a Strict Data Analyst generating a technical report.\n"
        "Your goal is to analyze the provided option pricing data and produce a structured statistical summary.\n"
        "The data will contain the following columns:\n"
        "- **Date:** The date on which the option price is calculated.\n"
        "- **S:** Current price of the underlying asset.\n"
        "- **K:** Exercise price of the option.\n"
        "- **T:** Time remaining until option expiration, expressed in years.\n"
        "- **r:** Annualized risk-free interest rate, used for discounting.\n"
        "- **sigma:** Annualized standard deviation of the underlying asset’s returns.\n"
        "- **option_type:** Call or Put.\n"
        "- **asset_class:** Classification of the underlying asset (e.g., equity, index).\n"
        "- **BSM_price:** Option price calculated using the Black-Scholes-Merton model.\n"
        "- **error:** Error when calculating the BSM price.\n"
        "The report should include the following paragraph:\n"
        f"1. **Overall Data Quality**: State that there are exactly {num_calls} Call options and {num_puts} Put options (I have counted them for you). Mention column names present and any missing values observed.\n"
        "2. **Pricing Level**: Report the Min, Max, and Average BSM_price for the asset class.\n"
        "3. **Term Structure**: Describe the range of maturity and the relationship between Time (T) and Price. (e.g., 'Longer maturity options show higher prices...').\n"
        "4. **Call vs Put Behavior**: Compare the average price of Calls vs Puts. Which is more expensive on average?\n"
        "5. **Model Consistency**: Confirm if price > 0 for all rows. Check if prices monotonically increase with T (generally).\n"
        "6. **Key Takeaway**: Summary of the portfolio's pricing health.\n\n"
    )
    user_prompt = (
        f"Analyze the following dataframe for **{asset}** options.\n\n"
        "**Call Options:**\n"
        f"{df_call.to_markdown(index=False)}\n\n"
        "**Put Options:**\n"
        f"{df_put.to_markdown(index=False)}\n\n"
        "Generate the 6-paragraph statistical report now."
    )

    ai_msg = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    for block in ai_msg.content.split("\n\n"):
        if block.strip():
            doc.add_paragraph(block.strip())

    # Add Figure
    doc.add_heading(f"2.{section_ordering}.2 Visualization of Pricing for {asset}", level=3)
    doc.add_picture(img_stream, width=Inches(6.5))


def _generate_gamma_summary(doc, llm, asset: str, df: pd.DataFrame, section_ordering: int):
    """Generate Gamma Positivity Summary"""
    doc.add_heading(f"2.{section_ordering}.3 Summary of Gamma Positivity Testing for {asset}", level=3)
    
    system_prompt = (
        "Please summmarize the gamma positivity results from the tables with only summarizing quantitative statistics and quantitative explanation into two paragraphs."
        "Also, we have some annotation for you to understand table columns"
        "Parameters:"
        "option_type (str): 'call' for call option, 'put' for put option"
        "S (float): current stock price"
        "K (float): option strike price"
        "T (float): time to expiration in years"
        "r (float): risk-free interest rate (annualized)"
        "sigma (float): volatility of the underlying stock (annualized)"
        "gamma_positivity: True if gamma positivity holds, False otherwise"
    )
    user_prompt = (
        "Here is the raw testing result tables that should become the option pricing output gamma positivity testing summary section."
        "The title should have this asset class name."
        "Please refine it as described:\n\n"
        f"{df}"
    )

    ai_msg = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    for block in ai_msg.content.split("\n\n"):
        if block.strip():
            doc.add_paragraph(block.strip())


def _generate_stress_test_summary(doc, llm, asset: str, df_stress: pd.DataFrame, section_ordering: int):
    """Generate Stress Test Summary"""
    doc.add_heading(f"2.{section_ordering}.4 Summary of Stress Testing for {asset}", level=3)
    
    stress_results_call = df_stress[
        (df_stress['option_type'] == 'call')
    ]
    stress_results_put = df_stress[
        (df_stress['option_type'] == 'put')
    ]

    system_prompt = (
        "You are a Risk Manager analyzing stress test results for an option portfolio. "
        "The provided data contains P&L impact (as % of price) under various historical and hypothetical extreme scenarios "
        "(e.g., 'Black Monday (1987)', '2008 Financial Crisis').\n"
        "Your task is to summarize the portfolio's resilience and vulnerabilities.\n"
        "Focus on:\n"
        "1. **Worst-Case Scenarios**: Which scenario causes the largest loss for Calls vs Puts?\n"
    )
    user_prompt = (
        f"Here are the stress test results for **{asset}** options.\n"
        "The table includes columns like '{Scenario}_PnL%' and 'worst_case_pnl_pct'.\n\n"
        "**Call Options Data:**\n"
        f"{stress_results_call.to_markdown(index=False)}\n\n"
        "**Put Options Data:**\n"
        f"{stress_results_put.to_markdown(index=False)}\n\n"
        "Please provide a concise risk summary, analyzing the Call and Put profiles separately if their risk sensitivities differ significantly."
    )

    ai_msg = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    for block in ai_msg.content.split("\n\n"):
        if block.strip():
            doc.add_paragraph(block.strip())


