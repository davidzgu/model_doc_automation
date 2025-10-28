# Multi-Agent Option Analysis System

A comprehensive 6-agent LangGraph workflow for automated Black-Scholes-Merton option pricing analysis, testing, and report generation.

## 🎯 Overview

This system uses LangGraph to orchestrate 6 specialized AI agents that work sequentially to:
1. Load option data from CSV
2. Calculate BSM prices and Greeks
3. Run validation tests
4. Generate textual summaries
5. Create visualizations
6. Assemble a final HTML report

## 🏗️ Architecture

### Agent Pipeline

```
Input CSV
    ↓
┌─────────────────────┐
│ Agent 1: Data Loader│  → Load CSV data
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Agent 2: Calculator │  → Calculate BSM prices & Greeks
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Agent 3: Tester     │  → Run validation tests
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Agent 4: Summary    │  → Generate text summary
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Agent 5: Charts     │  → Create visualizations
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Agent 6: Assembler  │  → Create final HTML report
└─────────────────────┘
    ↓
Final Report (HTML)
```

### Project Structure

```
document-automation-app/
├── src/
│   ├── multi_agent/
│   │   ├── __init__.py
│   │   ├── state.py              # State definition
│   │   ├── agents.py             # 6 agent implementations
│   │   ├── tools.py              # Tool allocation
│   │   └── workflow.py           # LangGraph workflow
│   ├── templates/
│   │   ├── summary_template.md   # Summary template
│   │   └── report_template.html  # Report template
│   ├── bsm_utils.py              # BSM calculation tools
│   ├── test_tools.py             # Testing tools
│   ├── chart_tools.py            # Visualization tools
│   ├── report_tools.py           # Report assembly tools
│   ├── llm.py                    # LLM configuration
│   └── run_multi_agent.py        # Main entry point
├── tests/
│   ├── test_greeks.py
│   └── test_sensitivity.py
├── inputs/
│   └── dummy_options.csv          # Sample input data
├── outputs/
│   ├── charts/                    # Generated charts
│   └── report.html                # Final report
└── requirements.txt
```

## 📦 Installation

### 1. Install Dependencies

```bash
cd document-automation-app
pip install -r requirements.txt
pip install tabulate matplotlib scipy
```

### 2. Required Packages

- `langchain` - LangChain framework
- `langchain-core` - Core LangChain functionality
- `langchain-ollama` - Ollama LLM integration
- `langgraph` - Multi-agent workflow
- `pandas` - Data processing
- `numpy` - Numerical computations
- `scipy` - Scientific computing (for BSM calculations)
- `matplotlib` - Charting
- `tabulate` - Table formatting

### 3. Setup Ollama

Make sure you have Ollama installed with a tool-calling capable model:

```bash
ollama pull qwen2.5:7b
```

## 🚀 Usage

### Basic Usage

Run the multi-agent workflow with default input:

```bash
cd document-automation-app
python -m src.run_multi_agent
```

### Custom CSV Input

Specify a custom CSV file:

```bash
python -m src.run_multi_agent path/to/your/options.csv
```

### Expected CSV Format

Your CSV file should have the following columns:

```csv
date,S,K,T,r,sigma,option_type
2025-09-01,100,105,1.0,0.05,0.2,call
2025-09-02,102,106,0.9,0.045,0.19,put
...
```

**Columns:**
- `S` - Spot price
- `K` - Strike price
- `T` - Time to maturity (years)
- `r` - Risk-free rate
- `sigma` - Volatility
- `option_type` - "call" or "put"

## 📊 Output

### Generated Files

1. **Calculation Results** - Markdown table with BSM prices
2. **Charts** - PNG images in `outputs/charts/`:
   - `option_prices.png` - Price visualization
   - `greeks_sensitivity.png` - Greeks sensitivity analysis
3. **Final Report** - `outputs/report.html` - Complete HTML report

### Example Output

```
🚀 MULTI-AGENT OPTION ANALYSIS WORKFLOW
================================================================================

📁 Input CSV: /path/to/inputs/dummy_options.csv
📊 Workflow: 6 agents in sequence

Agent Pipeline:
  1️⃣  Data Loader      → Load CSV data
  2️⃣  Calculator       → Calculate BSM prices & Greeks
  3️⃣  Tester          → Run validation tests
  4️⃣  Summary Writer   → Generate text summary
  5️⃣  Chart Generator  → Create visualizations
  6️⃣  Report Assembler → Create final HTML report

================================================================================

▶️  Starting workflow execution...

[Agent execution logs...]

✅ Workflow execution completed!

================================================================================
WORKFLOW STATUS SUMMARY
================================================================================
✅ Agent 1 - Data Loader          completed
✅ Agent 2 - Calculator           completed
✅ Agent 3 - Tester               completed
✅ Agent 4 - Summary Writer       completed
✅ Agent 5 - Chart Generator      completed
✅ Agent 6 - Report Assembler     completed

Overall Status: completed
================================================================================

📋 RESULTS SUMMARY
================================================================================

📄 Final Report:
   /path/to/outputs/report.html

   Open in browser: file:///path/to/outputs/report.html

================================================================================
✨ Analysis complete!
================================================================================
```

## 🔧 Customization

### Modify Agent Behavior

Edit agent prompts in `src/multi_agent/agents.py`:

```python
class Agent2_Calculator:
    def __call__(self, state: OptionAnalysisState) -> Dict[str, Any]:
        task_message = HumanMessage(content=f"""
        [Customize your agent's task here]
        """)
```

### Add New Tools

1. Create tool in appropriate file (`bsm_utils.py`, `test_tools.py`, etc.)
2. Add tool to agent's tool list in `src/multi_agent/tools.py`
3. Use tool in agent implementation

### Customize Templates

Edit templates in `src/templates/`:
- `summary_template.md` - Text summary format
- `report_template.html` - Final report HTML/CSS

## 🧪 Testing

Run individual components:

```python
# Test data loading
from src.multi_agent.agents import Agent1_DataLoader
from src.llm import get_llm

agent1 = Agent1_DataLoader(get_llm())
result = agent1({
    "csv_file_path": "inputs/dummy_options.csv",
    "messages": [],
    # ... other state fields
})
print(result)
```

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'scipy'**
```bash
pip install scipy
```

**2. ModuleNotFoundError: No module named 'tabulate'**
```bash
pip install tabulate
```

**3. Charts not generating**
```bash
pip install matplotlib
# Make sure outputs/charts/ directory exists
mkdir -p outputs/charts
```

**4. LLM not responding**
- Check Ollama is running: `ollama list`
- Verify model is installed: `ollama pull qwen2.5:7b`
- Check model supports tool calling (qwen2.5, mistral, llama3.1+)

## 📚 Technical Details

### State Management

The workflow uses `OptionAnalysisState` (defined in `state.py`) which is a TypedDict containing:
- Input fields (CSV path)
- Agent outputs (csv_data, calculation_results, test_results, etc.)
- Status fields (agent1_status, agent2_status, etc.)
- Error tracking

Data flows sequentially through agents via state updates.

### Tool Distribution

Each agent has access to specific tools:
- Agent 1: `csv_loader`
- Agent 2: `batch_bsm_calculator`, `greeks_calculator`, `sensitivity_test`
- Agent 3: `run_greeks_validation_test`, `run_sensitivity_analysis_test`
- Agent 4: `load_template`
- Agent 5: `create_option_price_chart`, `create_greeks_chart`
- Agent 6: `assemble_html_report`

### Workflow Execution

LangGraph's `StateGraph` manages:
- Sequential edge routing (Agent1 → Agent2 → ... → Agent6)
- State persistence via `MemorySaver`
- Error handling and recovery

## 🤝 Contributing

To add a new agent:

1. Define agent class in `agents.py`
2. Create tools in appropriate tool file
3. Add tool allocation in `tools.py`
4. Add node to workflow in `workflow.py`
5. Update state schema if needed

## 📄 License

MIT License

## 🙋 Support

For issues or questions:
1. Check this README
2. Review error messages in console output
3. Check `outputs/` directory for partial results
4. Verify all dependencies are installed

---

**Generated with ❤️ using LangGraph Multi-Agent Framework**
