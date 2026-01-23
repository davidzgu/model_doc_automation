import streamlit as st
import pandas as pd
import os
import io
from pathlib import Path
import time
import contextlib
from io import StringIO
from bsm_multi_agents.graph.agent_graph import build_app
from bsm_multi_agents.graph.state import WorkflowState

def main():
    st.set_page_config(page_title="BSM Multi-Agent System", layout="wide")
    
    # Split into two columns
    left_col, right_col = st.columns([0.618, 1])
    
    with left_col:
        st.header("Model Development Ongoing Performance Analysis Report Generator")
        
        st.header("Purpose of the Tool")
        header_content = """
        This AI Agent workflow automates OPA report drafting process. 
        The Agent connects to model script, data within timeframe as well as testing script inventory as input. 
        Based on the model type, Agent selects unique sets of testing script for performance monitoring. 
        Workflow outpus an OPA report draft in-line with MRM published template, with summary and charts generated automatically.
        """
        st.write(header_content)
        
        st.header("Workflow Graph")
        try:
            st.image("src/bsm_multi_agents/figures/workflow.png", caption="Agent Workflow")
        except Exception:
            st.warning("Workflow image not found at src/bsm_multi_agents/figures/workflow.png")

    with right_col:
        # 1. File Upload
        uploaded_file = st.file_uploader("Upload Option Trades CSV", type=['csv'])
        
        csv_path = None
        if uploaded_file is not None:
            input_dir = "data/input"
            os.makedirs(input_dir, exist_ok=True)
            csv_path = os.path.join(input_dir, "uploaded_trades.csv")
            
            with open(csv_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File uploaded and saved to: `{csv_path}`")
                
            df = pd.read_csv(csv_path)
            # File preview window
            st.subheader("📋 File Preview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
            
            st.dataframe(df.head(10), width='stretch')
        else:
            st.info("Please upload a CSV file to proceed.")
            return

        # Configuration
        st.subheader("⚙️ Configuration")
        output_dir = st.text_input("Output Directory", value="data/output")
        
        # 2. Generate Report Button
        if st.button("Generate Report", type="primary"):

            os.makedirs(output_dir, exist_ok=True)
            
            # Container for logs
            log_container = st.container()
            with log_container:
                st.subheader("🔄 Agent Execution Log")
                progress_bar = st.progress(0)
                status_text = st.empty()
                log_placeholder = st.empty()
            
            logs = []
            
            # 3. Build and Run App
            app = build_app()
            
            # Initial state with the uploaded CSV path
            project_path = Path.cwd()
            output_dir = str(project_path / "data/cache")
            server_path = str(project_path / "src/bsm_multi_agents/mcp/server.py")
            local_tool_folder_path = os.path.join(project_path, "src/bsm_multi_agents/tools")
            final_report_path = str(project_path / "data/output/final_report.docx")

            initial_state = WorkflowState(
                csv_file_path=csv_path,
                output_dir=output_dir,
                server_path=server_path,
                local_tool_folder_path=local_tool_folder_path,
                final_report_path=final_report_path,
                errors=[],
                messages=[],
                # "remaining_steps": 10,
            )
            
            try:
                # Stream the graph execution
                # We use thread_id for persistence if MemorySaver is used
                config = {"configurable": {"thread_id": "streamlit_session"}}
                
                step_count = 0
                # Just a heuristic for progress bar
                total_estimated_steps = 15 
                
                # Capture stdout
                log_buffer = StringIO()
                
                with contextlib.redirect_stdout(log_buffer):
                    for output in app.stream(initial_state, config=config):
                        for node_name, state_update in output.items():
                            step_count += 1
                            
                            # Update status
                            status_text.text(f"Currently processing: {node_name}...")
                            
                            # Update progress
                            current_progress = min(step_count / total_estimated_steps, 0.95)
                            progress_bar.progress(current_progress)
                            
                            # Display captured logs dynamically
                            current_logs = log_buffer.getvalue()
                            log_placeholder.code(current_logs, language="text")
                
                progress_bar.progress(100)
                status_text.text("Workflow Finished!")
                final_logs = log_buffer.getvalue() + "\n** Workflow Finished!**"
                log_placeholder.code(final_logs, language="text")
                
                # 4. Retrieve Final State and Report
                final_output = app.get_state(config=config).values
                report_path = final_output.get("final_report_path")
                
                st.divider()
                st.subheader("🎉 Result")
                
                if report_path and os.path.exists(report_path):
                    st.success(f"Report generated successfully at: `{report_path}`")
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="Download Report (DOCX)",
                            data=f,
                            file_name=os.path.basename(report_path),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                else:
                    st.warning("Workflow finished, but no 'final_report_path' was found in the final state.")
                    st.json(final_output)
                    
            except Exception as e:
                st.error(f"Error during graph execution: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()