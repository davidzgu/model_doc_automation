from langchain.tools import Tool
import subprocess

# Example: Tool to run a Python script
def run_script(script_path: str):
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    return result.stdout

python_tool = Tool(
    name="PythonExecutor",
    func=run_script,
    description="Executes a Python script and returns the output."
)

