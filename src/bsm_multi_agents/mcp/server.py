from mcp.server.fastmcp import FastMCP
# from bsm_multi_agents.mcp.data_loader import validate_input_file
from bsm_multi_agents.mcp.option_pricer import calculate_option_analytics
from bsm_multi_agents.mcp.test_parity import verify_put_call_parity
from bsm_multi_agents.mcp.risk_analytics_engine import (
    run_sensitivity_analysis, 
    run_stress_analysis, 
    PnL_attribution_analysis
)

# Initialize the MCP Server
mcp = FastMCP("Quant Tools Server")

# Register tools
# mcp.add_tool(validate_input_file)
mcp.add_tool(calculate_option_analytics)
mcp.add_tool(verify_put_call_parity)
mcp.add_tool(run_sensitivity_analysis)
mcp.add_tool(run_stress_analysis)
mcp.add_tool(PnL_attribution_analysis)

if __name__ == "__main__":
    mcp.run()
