import os
from typing import Optional
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from anthropic import Anthropic
from bsm_multi_agents.config.llm_config_claude import DEFAULT_ANTHROPIC_API_KEY
import datetime 

def summary_to_word(
    summary_docx_path: str,
    output_docx_path: str = "final_OPA_report.docx",
    title: str = "Ongoing Monitoring Analysis Report",
    model_name: str = "Option Pricing, BSM",
    author_name: str = "John Doe",
    group_name: str = "Front Desk Modeling and Analytics",
    version: str = "v1.0",
    section1_heading: str = "1. Introduction",
    section1_paragraph: Optional[str] = None,
    section2_heading: str = "2. Testing",
    section2_paragraph: Optional[str] = None,
) -> str:
    """
    Create a formatted Word report with:
    - Title page
    - Table of contents (Word-updatable)
    - Section 1: static intro paragraph
    - Section 2: refined summary text from an existing Word file via OpenAI

    Args:
        summary_docx_path: Path to the existing .docx file containing the raw summary text.
        output_docx_path: Path for the generated formatted report (.docx).
        title: Title used on the title page and as document main title.
        section1_heading: Heading text for Section 1 (level-1 heading).
        section1_paragraph: Optional custom intro paragraph for Section 1.
                            If omitted, a default intro is used.
        section2_heading: Heading text for Section 2 (level-1 heading).
        section2_paragraph: Optional testing analysis paragraph for Section 2.

    Returns:
        Absolute path to the generated Word document.
    """

    # --------- 0. Basic checks ---------
    if not os.path.exists(summary_docx_path):
        raise FileNotFoundError(f"Summary file not found: {summary_docx_path}")

    api_key = os.getenv("API_KEY", DEFAULT_ANTHROPIC_API_KEY)
    if not api_key:
        raise RuntimeError(
            "API_KEY is not set in the environment. "
            "Please export it before calling this tool."
        )

    # --------- 1. Load raw summary text from the source Word file ---------
    src_doc = Document(summary_docx_path)
    raw_summary_parts = []
    for p in src_doc.paragraphs:
        text = p.text.strip()
        if text:
            raw_summary_parts.append(text)
    raw_summary_text = "\n\n".join(raw_summary_parts).strip()

    if not raw_summary_text:
        raise ValueError("No non-empty text found in the summary document.")

    # --------- 2. Refine summary via Claude API ---------
    client = Anthropic(api_key=api_key)

    system_prompt = (
        "You are a quantitative finance analyst who conducts model performance ongoing monitoring and writes the monitoring report. "
        "Given a rough summary about a quantitative analysis and or test, you can present it into a clear, well-structured Section 2 of a professional report. "
        "You will first generate a clear and concise narrative based on the provided raw summary. "
        "You will then format the narrative into a professional and clear section, with headings, subheadings, and bullet points as needed. "
        "Then, you will present the raw summary into tables and plots, for each asset class, generate a table and a plot to show change over time."
        "Do NOT include a title page or table of contents; only the Section 2 narrative itself."
        "After generating all contents of Section, you will examine all contents and improve on the format of Section 2 so that the report is clear and clean, without unnecessary symbols. Do make sure to include the tables and plots."
    )

    user_prompt = (
        "Here is the raw summary text that should become Section 2 of the report. "
        "Please refine it as described:\n\n"
        f"{raw_summary_text}"
    )

    # Using Claude API with system prompt
    completion = client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=2048,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    refined_summary = completion.content[0].text.strip()

    # --------- 3. Create new Word document ---------
    doc = Document()

    # Title
    title_run = doc.add_paragraph().add_run(title)
    title_run.bold = True
    title_run.font.size = doc.styles["Title"].font.size
    doc.paragraphs[-1].alignment = 1  # 0=left, 1=center

    # Subtitle (Model Name)
    subtitle_para = doc.add_paragraph()
    subtitle_run = subtitle_para.add_run(model_name)
    subtitle_para.alignment = 1
    subtitle_run.bold = True

    doc.add_paragraph("")  # spacer

    # Author
    author_para = doc.add_paragraph()
    author_para.add_run("Author: ").bold = True
    author_para.add_run(author_name).italic = False
    author_para.alignment = 1

    # Group
    group_para = doc.add_paragraph()
    group_para.add_run("Group: ").bold = True
    group_para.add_run(group_name)
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
    version_para.add_run(version)
    version_para.alignment = 1

    # Page break after title page
    doc.add_page_break()

    # ============================
    # TABLE OF CONTENTS
    # ============================
    doc.add_heading("Table of Contents", level=1)

    toc_para = doc.add_paragraph()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), 'TOC \\o "1-3" \\h \\z \\u')
    toc_para._p.append(fld)

    doc.add_page_break()

    # ============================
    # SECTION 1
    # ============================
    if not section1_paragraph:
        section1_paragraph = (
            "This section provides contextual background, objectives, and relevant "
            "considerations for the ongoing monitoring analysis. Subsequent sections "
            "expand on methodology, insights, and results."
        )

    doc.add_heading(section1_heading, level=1)
    doc.add_paragraph(section1_paragraph)

    # ============================
    # SECTION 2 – Refined Summary
    # ============================
    doc.add_heading("2. Summary of Analysis", level=1)

    for block in refined_summary.split("\n\n"):
        block = block.strip()
        if block:
            doc.add_paragraph(block)


# add the pricing table, by asset class, 
# price's time series make into figures using LLM, add here
# adjust formating and update the prompt
# pending: add testing, rn only >0.
    # --------- 4. Save ---------
    doc.save(output_docx_path)
    return os.path.abspath(output_docx_path)