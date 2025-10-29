@tool
def generate_summary(
    calculation_results: str,
    test_results: Union[str, Dict],
    csv_data: Union[str, Dict]
) -> str:
    """
    Generate summary report based on calculation and test results.
    
    Uses template from src/templates/summary_template.md if available,
    otherwise generates a structured summary automatically.
    """
    try:
        # 1. Load template
        template = _load_template()
        
        # 2. Parse inputs
        test_data = json.loads(test_results) if isinstance(test_results, str) else test_results
        data = json.loads(csv_data) if isinstance(csv_data, str) else csv_data
        
        # 3. Extract information
        total_options = len(data.get('S', {})) if isinstance(data, dict) else len(data)
        option_types = _count_option_types(data)
        
        # 4. Parse test results
        test_summary = _format_test_results(test_data)
        
        # 5. Extract key findings
        key_findings = _extract_key_findings(test_data, calculation_results)
        
        # 6. Generate recommendations
        recommendations = _generate_recommendations(test_data)
        
        # 7. Fill template
        if template:
            summary = template.format(
                executive_summary=_generate_executive_summary(test_data),
                total_options=total_options,
                option_types=option_types,
                analysis_date=datetime.now().strftime("%Y-%m-%d"),
                calculation_results=calculation_results,
                greeks_analysis=_analyze_greeks(test_data),
                test_results=test_summary,
                key_findings=key_findings,
                recommendations=recommendations,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        else:
            # Fallback: generate summary without template
            summary = _generate_fallback_summary(...)
        
        return summary
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"
    

def _load_template() -> Optional[str]:
    """Load template from src/templates/summary_template.md"""
    template_path = Path(__file__).parent.parent.parent / "templates" / "summary_template.md"
    if template_path.exists():
        return template_path.read_text(encoding='utf-8')
    return None

def _format_test_results(test_data: Dict) -> str:
    """Format test results into markdown"""
    tests_run = test_data.get('tests_run', [])
    formatted = "### Validation Tests\n\n"
    for test in tests_run:
        name = test.get('test_name', 'Unknown')
        result = test.get('result', {})
        status = result.get('status', 'unknown')
        formatted += f"- **{name}**: {status.upper()}\n"
        # Add details...
    return formatted

def _extract_key_findings(test_data: Dict, calc_results: str) -> str:
    """Extract key findings from test and calculation results"""
    findings = []
    
    # Check validation results
    tests_run = test_data.get('tests_run', [])
    for test in tests_run:
        result = test.get('result', {})
        if result.get('status') == 'failed':
            findings.append(f"- Validation issues found in {test.get('test_name')}")
    
    if not findings:
        findings.append("- All validation tests passed successfully")
    
    return "\n".join(findings)