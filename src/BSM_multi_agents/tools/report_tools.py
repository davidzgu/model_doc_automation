# -*- coding: utf-8 -*-
"""
Report generation and assembly tools.

Creates HTML reports combining text summaries and charts.
"""
import json
import os
from typing import Union, List, Dict, Any
from pathlib import Path
from langchain_core.tools import tool
from datetime import datetime


@tool
def load_template(template_name: str) -> str:
    """
    Load a template file from the templates directory.

    Args:
        template_name: Name of the template file (e.g., "summary_template.md", "report_template.html")

    Returns:
        Template content as string, or default template if file not found
    """
    try:
        template_path = Path(__file__).parent / "templates" / template_name

        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Return default template based on extension
            if template_name.endswith('.md'):
                return get_default_markdown_template()
            elif template_name.endswith('.html'):
                return get_default_html_template()
            else:
                return f"Template {template_name} not found"
    except Exception as e:
        return f"Error loading template: {str(e)}"


def get_default_markdown_template() -> str:
    """Return default markdown template for summary"""
    return """# Option Analysis Summary

## Overview
{overview}

## Calculation Results
{calculation_results}

## Test Results
{test_results}

## Key Findings
{key_findings}

---
*Generated on {timestamp}*
"""


def get_default_html_template() -> str:
    """Return default HTML template for final report"""
    return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Option Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .section {{
            background-color: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ margin: 0; }}
        h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        .chart {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            text-align: right;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Option Analysis Report</h1>
        <p>{subtitle}</p>
    </div>

    <div class="section">
        <h2>Summary</h2>
        {summary_content}
    </div>

    <div class="section">
        <h2>Visualizations</h2>
        {charts_content}
    </div>

    <div class="timestamp">
        Generated on {timestamp}
    </div>
</body>
</html>
"""


@tool
def assemble_html_report(
    summary_text: str,
    chart_paths: Union[str, List[str]],
    output_path: str = "outputs/report.html"
) -> str:
    """
    Assemble a complete HTML report from summary text and charts.

    Args:
        summary_text: Markdown or plain text summary
        chart_paths: JSON string or list of chart file paths
        output_path: Path to save the HTML report

    Returns:
        JSON string with report path and status
    """
    try:
        # Parse chart paths
        if isinstance(chart_paths, str):
            try:
                paths = json.loads(chart_paths)
            except:
                paths = [chart_paths]
        else:
            paths = chart_paths

        # Get HTML template
        template = get_default_html_template()

        # Convert markdown summary to HTML (basic conversion)
        html_summary = summary_text.replace('\n## ', '<h3>').replace('##', '</h3>')
        html_summary = html_summary.replace('\n### ', '<h4>').replace('###', '</h4>')
        html_summary = html_summary.replace('\n', '<br>\n')

        # Build charts HTML
        charts_html = ""
        for path in paths:
            if isinstance(path, dict):
                path = path.get('chart_path', path.get('path', ''))

            if path and os.path.exists(path):
                rel_path = os.path.relpath(path, os.path.dirname(output_path))
                charts_html += f'<div class="chart"><img src="{rel_path}" alt="Chart"><p>{os.path.basename(path)}</p></div>\n'

        # Fill template
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_html = template.format(
            subtitle="Comprehensive BSM Option Pricing Analysis",
            summary_content=html_summary,
            charts_content=charts_html if charts_html else "<p>No charts generated</p>",
            timestamp=timestamp
        )

        # Create output directory
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_html)

        return json.dumps({
            "status": "success",
            "report_path": str(output_file.absolute()),
            "message": f"Report generated successfully at {output_file}"
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error assembling report: {str(e)}"
        })
