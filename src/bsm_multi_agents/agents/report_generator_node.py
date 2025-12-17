import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from typing import Optional, List
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches
from langchain_core.messages import HumanMessage, SystemMessage

from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.config.llm_config import get_llm

def report_generator_node(state: WorkflowState) -> WorkflowState:
    """
    LangGraph node: Generates the final Word report.
    
    1. Reads `csv_file_path` and `summary_docx_path` from state.
    2. Uses LLM to refine summary and analyze pricing data.
    3. Generates charts and tables.
    4. Saves 'final_OPA_report.docx' to `output_dir`.
    5. Updates state with `final_report_path`.
    """
    errors = state.get("errors", [])
    
    # --- Input Validation ---
    csv_path = state.get("csv_file_path")
    # Assuming 'summary_docx_path' is passed in state, or we default to a known location?
    # Based on previous context, user was passing it. 
    # If not in state, we might need to look for it or fail.
    # Let's check state first.
    summary_docx_path = state.get("summary_docx_path") 
    output_dir = state.get("output_dir")

    if not csv_path or not os.path.exists(csv_path):
        errors.append("report_generator_node: csv_file_path missing or invalid")
        state["errors"] = errors
        return state

    if not summary_docx_path or not os.path.exists(summary_docx_path):
        # Fallback or Error? Let's Error for robustness as this is the input source.
        errors.append("report_generator_node: summary_docx_path missing or invalid")
        state["errors"] = errors
        return state
        
    if not output_dir:
        errors.append("report_generator_node: output_dir missing")
        state["errors"] = errors
        return state

    try:
        # --- Execution ---
        
        # 1. Initialize LLM
        llm = get_llm()
        
        # 2. Read Summary
        raw_summary = _read_summary_docx(summary_docx_path)
        
        # 3. Generate Section 2 Content
        section2_text = _generate_section2_content(llm, raw_summary)
        
        # 4. Build Document
        output_filename = "final_OPA_report.docx"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        doc = Document()
        _add_title_page(doc)
        _add_table_of_contents(doc)
        _add_section1(doc)
        _add_section2(doc, section2_text)
        _add_section3_pricing(doc, llm, csv_path)
        
        # 5. Save
        doc.save(output_path)
        abs_path = os.path.abspath(output_path)
        
        # Update State
        state["final_report_path"] = abs_path
        
        # Optional: Append a message log
        # state["messages"].append(...) # If we want to log the success

    except Exception as e:
        errors.append(f"report_generator_node: Error generating report: {e}")

    state["errors"] = errors
    return state


# --- Helper Functions (Internal) ---

def _read_summary_docx(path: str) -> str:
    doc = Document(path)
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(parts).strip()
    return text

def _invoke_llm(llm, system_prompt: str, user_prompt: str) -> str:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    response = llm.invoke(messages)
    return response.content.strip()

def _generate_section2_content(llm, raw_text: str) -> str:
    system_prompt = (
        "You are a quantitative finance expert and technical writer. "
        "Given a rough summary about a quantitative analysis, rewrite it into a clear, "
        "well-structured 'Section 2: Summary of Analysis' for a professional report. "
        "Use concise English, with logical flow and numbered or bulleted lists where helpful. "
        "Return ONLY the narrative text."
    )
    user_prompt = f"Raw Summary Text:\n\n{raw_text}"
    return _invoke_llm(llm, system_prompt, user_prompt)

def _generate_asset_analysis(llm, asset: str, df_pricing: pd.DataFrame) -> str:
    df_str = df_pricing.to_string(index=False)
    system_prompt = (
        "You are a quantitative analyst. Analyze the provided option pricing results. "
        "Cover these topics:\n"
        "1. Overall Data Quality\n"
        "2. Pricing Level by Asset Class\n"
        "3. Term Structure (Price vs Maturity)\n"
        "4. Call vs Put Behavior\n"
        "5. Model Consistency\n"
        "6. Key Takeaway\n"
    )
    user_prompt = (
        f"Here is the pricing data for Asset Class: {asset}.\n\n"
        f"{df_str}\n\n"
        "Please provide a detailed analysis summary for this asset class."
    )
    return _invoke_llm(llm, system_prompt, user_prompt)

# --- Document Building Helpers ---

def _add_title_page(doc, 
                    title="Ongoing Monitoring Analysis Report",
                    model_name="Option Pricing, BSM",
                    author="John Doe", 
                    group="Front Desk Modeling and Analytics",
                    version="v1.0"):
    p = doc.add_paragraph()
    run = p.add_run(title)
    run.bold = True
    run.font.size = doc.styles["Title"].font.size
    p.alignment = 1

    p = doc.add_paragraph()
    run = p.add_run(model_name)
    run.bold = True
    p.alignment = 1
    
    doc.add_paragraph("") 

    for label, val in [
        ("Author", author),
        ("Group", group),
        ("Report Date", datetime.date.today().strftime("%B %d, %Y")),
        ("Document Version", version)
    ]:
        p = doc.add_paragraph()
        p.add_run(f"{label}: ").bold = True
        p.add_run(val)
        p.alignment = 1

    doc.add_page_break()

def _add_table_of_contents(doc):
    doc.add_heading("Table of Contents", level=1)
    p = doc.add_paragraph()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), 'TOC \\o "1-3" \\h \\z \\u')
    p._p.append(fld)
    doc.add_page_break()

def _add_section1(doc):
    doc.add_heading("1. Introduction", level=1)
    doc.add_paragraph(
        "This section provides contextual background, objectives, and relevant "
        "considerations for the ongoing monitoring analysis. Subsequent sections "
        "expand on methodology, insights, and results."
    )

def _add_section2(doc, content):
    doc.add_heading("2. Summary of Analysis", level=1)
    for block in content.split("\n\n"):
        if block.strip():
            doc.add_paragraph(block.strip())

def _add_section3_pricing(doc, llm, csv_path):
    doc.add_heading("3. Pricing Analysis", level=1)
    
    df_pricing = pd.read_csv(csv_path)
    required_cols = ["asset_class", "option_type", "T", "BSM_Price"]
    if not all(c in df_pricing.columns for c in required_cols):
         doc.add_paragraph("Error: CSV missing required columns for analysis.")
         return

    assets = df_pricing["asset_class"].unique()
    
    for asset in assets:
        df_asset = df_pricing[df_pricing["asset_class"] == asset].copy()
        if df_asset.empty:
            continue
        
        doc.add_heading(f"Pricing Output: {asset}", level=2)
        doc.add_paragraph(f"The pricing output for {asset} is listed below.")

        analysis_text = _generate_asset_analysis(llm, asset, df_asset)
        for block in analysis_text.split("\n\n"):
            if block.strip():
                doc.add_paragraph(block.strip())
        
        _add_plot(doc, df_asset, asset)
        _add_datatable(doc, df_asset)

def _add_plot(doc, df, asset_name):
    df = df.sort_values("T")
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for otype in ["call", "put"]:
        subset = df[df["option_type"] == otype]
        if not subset.empty:
            ax.plot(subset["T"], subset["BSM_Price"], label=otype, marker='o')
    
    ax.set_xlabel("Time to Maturity (T)")
    ax.set_ylabel("Option Price (BSM)")
    ax.set_title(f"Option Pricing Curve – {asset_name}")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    img_stream = BytesIO()
    fig.savefig(img_stream, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    img_stream.seek(0)
    
    doc.add_picture(img_stream, width=Inches(6.0))

def _add_datatable(doc, df):
    table = doc.add_table(rows=len(df) + 1, cols=len(df.columns))
    table.style = "Table Grid"
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr_cells[i].text = str(col)
        hdr_cells[i].paragraphs[0].runs[0].bold = True
    for i, row in enumerate(df.itertuples(index=False)):
        cells = table.rows[i+1].cells
        for j, val in enumerate(row):
            cells[j].text = str(val)
