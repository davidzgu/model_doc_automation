# -*- coding: utf-8 -*-
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pathlib import Path
from src.llm import get_llm
from src.tools import get_tools
from src.prompts import READ_AND_CALC
from src.agent import build_agent


def main():
    llm = get_llm()
    tools = get_tools()
    agent = build_agent(llm, tools)

    csv_path = Path(__file__).resolve().parents[2] / "inputs" / "dummy_options.csv"
    print(f"\n{'='*80}")
    print(f"CSV path: {csv_path}")
    print(f"{'='*80}\n")

    msg = HumanMessage(content=READ_AND_CALC.format(path=str(csv_path)))
    print(f"Prompt:\n{msg.content}\n")
    print(f"{'='*80}\n")

    resp = agent.invoke(
        {"messages": [msg]},
        config={
            "recursion_limit": 10,
            "configurable": {"thread_id": "run-1"}
        }
    )

    
    print(f"\n{'='*80}")
    print(f"Workflow:")
    print(f"{'='*80}\n")

    step_num = 1
    for message in resp["messages"]:
        if isinstance(message, HumanMessage):
            print(f"Step {step_num} - inputs:")
            print(f"   {message.content[:200]}..." if len(message.content) > 200 else f"   {message.content}")
            print()
            step_num += 1

        elif isinstance(message, AIMessage):
            if hasattr(message, 'tool_calls') and message.tool_calls:
                # Agent 决定调用工具
                print(f"Step {step_num} - Agent decide tools used:")
                for tool_call in message.tool_calls:
                    print(f"   Tool name: {tool_call['name']}")
                    print(f"   Tool parameters: {tool_call['args']}")
                print()
                step_num += 1
            elif message.content:
                print(f"Step {step_num} - Agent outputs:")
                print(f"   {message.content}")
                print()
                step_num += 1

        elif isinstance(message, ToolMessage):
            print(f"Step {step_num} - outputs:")
            print(f"   Tool name: {message.name}")
            result_preview = message.content[:300] + "..." if len(message.content) > 300 else message.content
            print(f"   Outputs: {result_preview}")
            print()
            step_num += 1

    print(f"\n{'='*80}")
    print(f"Final outputs:")
    print(f"{'='*80}\n")
    print(resp["messages"][-1].content)

if __name__ == "__main__":
    main()