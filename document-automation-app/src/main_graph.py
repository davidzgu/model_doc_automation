from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import StructuredTool
from langchain_utils.BSM_tests import test_gamma_positivity, digital_options_test
from langchain_utils.func_tools import  read_csv
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

class State(BaseModel):
    user_prompt: str | None = None
    test_results_from_agent: str | None = None
    report: str | None = None


builder = StateGraph(State)


def agent_node(state: State) -> State:
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
    )

    system_prompt = """
    You are a quantitative testing coordinator.
    Based on the user's request on **tests** and available tools, read the test input from given file path and then run the corresponding tests with input.
    **Note: your job is only doing the tests and output the useful information. Ignore any user's requests on reporting part.**
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])


    csv_tool = StructuredTool.from_function(read_csv)
    digital_opt_tool = StructuredTool.from_function(digital_options_test)
    gamma_tool = StructuredTool.from_function(test_gamma_positivity)
    tools = [csv_tool, digital_opt_tool, gamma_tool]

    agent = create_tool_calling_agent(llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = executor.invoke({"input": state.user_prompt})
    output_text = response.get("output", "")
    state.test_results_from_agent = output_text
    return state

def report_node(state: State) -> State:
    """Generate a final report based on the test results."""
    print("Generating report ...")

    llm = ChatOpenAI(model="gpt-4o-mini")

    # Construct structured context for LLM
    context = {
        "user_prompt": state.user_prompt,
        "tests_results": state.test_results_from_agent,
    }

    prompt = f"""
    You are a senior quantitative analyst. Leverage the tests results from another analysis and follow the user's original request:

    Context:
    {context}
    """

    response = llm.invoke(prompt)
    report = response.content
    state.report = report
    return state


builder.add_node("agent", agent_node)
builder.add_node("report_node", report_node)

builder.add_edge(START, "agent")
builder.add_edge("agent", "report_node")
builder.add_edge("report_node", END)

app = builder.compile()



if __name__ == "__main__":
    user_prompt = (
        "Read the CSV file at 'D:\ML_Experiment\model_doc_automation\document-automation-app\src\input\data.csv' (only 3 rows),"
        "then run the gamma positive test and digital options test, "
        "based on the results, generate a better analyze report. Divide the 2 tests into 2 sections and write a comprehensive summary at end of report. Also add an executive summary at the beginning of report."
    )
    result = app.invoke({"user_prompt": user_prompt})
    print("\nFinal Output:\n", result['report'])
