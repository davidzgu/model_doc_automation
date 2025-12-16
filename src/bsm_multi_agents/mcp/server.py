from mcp.server.fastmcp import FastMCP
from bsm_multi_agents.mcp.data_loader import validate_input_file
from bsm_multi_agents.mcp.pricing_calculator import calculate_bsm_to_file
from bsm_multi_agents.mcp.pricing_calculator import calculate_greeks_to_file
from bsm_multi_agents.mcp.pricing_validator import validate_greeks_to_file

# Initialize the MCP Server
mcp = FastMCP("Quant Tools Server")

# Register tools
# mcp.add_tool(validate_input_file)
# mcp.add_tool(calculate_bsm_to_file)
mcp.add_tool(calculate_greeks_to_file)
mcp.add_tool(validate_greeks_to_file)

if __name__ == "__main__":
    mcp.run()
