from mcp.server.fastmcp import FastMCP
from bsm_multi_agents.mcp.data_loader import validate_input_file
from bsm_multi_agents.mcp.pricing_calculator import calculate_bsm_to_file
from bsm_multi_agents.mcp.pricing_calculator import calculate_greeks_to_file
from bsm_multi_agents.mcp.pricing_validator import validate_greeks_to_file, run_sensitivity_test_to_file, run_stress_test_to_file, run_pnl_test_to_file
from bsm_multi_agents.mcp.summary_generator import csv_to_summary
from bsm_multi_agents.mcp.report_generator import summary_to_word

# Initialize the MCP Server
mcp = FastMCP("Quant Tools Server")

# Register tools
mcp.add_tool(validate_input_file)
mcp.add_tool(calculate_bsm_to_file)
mcp.add_tool(calculate_greeks_to_file)
mcp.add_tool(validate_greeks_to_file)
mcp.add_tool(run_sensitivity_test_to_file)
mcp.add_tool(run_stress_test_to_file)   
mcp.add_tool(run_pnl_test_to_file)
mcp.add_tool(csv_to_summary)
mcp.add_tool(summary_to_word)

if __name__ == "__main__":
    mcp.run()