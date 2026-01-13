import streamlit as st
import pandas as pd
from demo import generate_report
import io
from contextlib import redirect_stdout
from bsm_multi_agents.agents.utils import print_resp

def main():
    st.title("BSM Multi-Agent System")
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return
        
        # File preview window
        st.subheader("📋 File Preview")
        # Show basic file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("File Size", f"{uploaded_file.size} bytes")
        
        # Show column names
        st.write("**Columns:**", ", ".join(df.columns.tolist()))
        
        # Preview data with options
        preview_rows = st.slider("Preview rows", min_value=1, max_value=min(100, len(df)), value=10)
        st.dataframe(df.head(preview_rows), width="stretch")
        
        # Generate report and display print output
        if st.button("Generate Report", type="primary"):
            with st.spinner("Generating BSM report..."):
                # output_placeholder = st.empty()
                # for line in your_function_stream():
                #     output_placeholder.code(line, language="text")
                # f = io.StringIO()
                # with redirect_stdout(f):
                #     final_path = generate_report("dummy_options.csv")
                # output = f.getvalue()
                log_placeholder = st.empty()
                logs = ""
                # for log_line in print_resp():
                #     if log_line:
                #         logs += log_line + "\n"
                #         log_placeholder.code(logs, language="text")
                for log_line in generate_report("dummy_options.csv"):
                    logs += log_line + "\n"
                    log_placeholder.code(logs, language="text")
                    # If your generator yields the file path as a special log line, capture it:
                    if log_line.startswith("REPORT_PATH:"):
                        final_path = log_line.split("REPORT_PATH:")[1].strip()
                st.subheader("Report Generation Log")
                st.success(f"Report generated at: {final_path}")
                if final_path:
                    with open(final_path, "rb") as f:
                        st.download_button(
                            label="Download Report",
                            data=f,
                            file_name=final_path.split("/")[-1],
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
# TODO: add window for word preview
if __name__ == "__main__":
    main()