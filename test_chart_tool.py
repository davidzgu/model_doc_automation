#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 chart_generator 工具调用
"""
import json
from pathlib import Path

# 测试数据
test_bsm_results = [
    {"BSM_Price": 8.021, "K": 105, "S": 100, "T": 1, "date": "2025-09-01", "option_type": "call", "r": 0.05, "sigma": 0.2},
    {"BSM_Price": 7.214, "K": 106, "S": 102, "T": 0.9, "date": "2025-09-02", "option_type": "put", "r": 0.045, "sigma": 0.19},
]

test_greeks_results = [
    {"BSM_Price": 8.021, "K": 105, "S": 100, "T": 1, "date": "2025-09-01", "delta": 0.5, "gamma": 0.05, "option_type": "call", "r": 0.05, "rho": 10, "sigma": 0.2, "theta": -8, "vega": 20},
    {"BSM_Price": 7.214, "K": 106, "S": 102, "T": 0.9, "date": "2025-09-02", "delta": -0.5, "gamma": 0.05, "option_type": "put", "r": 0.045, "rho": -9, "sigma": 0.19, "theta": -7, "vega": 18},
]

# 测试1: 直接调用工具
print("=" * 80)
print("测试1: 直接调用 create_summary_charts 工具")
print("=" * 80)

from src.bsm_multi_agents.tools.chart_generator_tools import create_summary_charts

output_dir = str(Path.cwd() / "data" / "output")
bsm_json = json.dumps(test_bsm_results, ensure_ascii=False)
greeks_json = json.dumps(test_greeks_results, ensure_ascii=False)

print(f"\n输出目录: {output_dir}")
print(f"\nbsm_results 类型: {type(bsm_json)}")
print(f"bsm_results 前100字符: {bsm_json[:100]}...")
print(f"\ngreeks_results 类型: {type(greeks_json)}")
print(f"greeks_results 前100字符: {greeks_json[:100]}...")

try:
    result = create_summary_charts.invoke({
        "bsm_results": bsm_json,
        "greeks_results": greeks_json,
        "output_dir": output_dir
    })
    print(f"\n✅ 工具调用成功!")
    result_dict = json.loads(result)
    print(f"结果: {json.dumps(result_dict, indent=2, ensure_ascii=False)}")
except Exception as e:
    print(f"\n❌ 工具调用失败!")
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

# 测试2: 测试 Agent 调用
print("\n" + "=" * 80)
print("测试2: 通过 Agent 调用工具")
print("=" * 80)

from langchain_core.messages import HumanMessage
from src.bsm_multi_agents.agents.agent_factory import built_graph_agent_by_role
from src.bsm_multi_agents.prompts.loader import load_prompt

agent_role = "chart_generator"
default_system = "You are a chart generation agent. Call tools immediately without explanation."

print(f"\n创建 agent...")
agent = built_graph_agent_by_role(agent_role, default_system=default_system)

prompt_path = Path.cwd() / "src" / "bsm_multi_agents" / "prompts" / "chart_generator_prompts.txt"
user_prompt = load_prompt(prompt_path).format(
    bsm_results=bsm_json,
    greeks_results=greeks_json,
    output_dir=output_dir,
)

print(f"\n用户提示词长度: {len(user_prompt)} 字符")
print(f"用户提示词前200字符:\n{user_prompt[:200]}...\n")

try:
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_prompt)]},
        config={"recursion_limit": 10}
    )
    print(f"\n✅ Agent 调用成功!")
    print(f"\n消息数量: {len(result['messages'])}")
    for i, msg in enumerate(result['messages']):
        print(f"\n消息 {i+1}: {type(msg).__name__}")
        if hasattr(msg, 'content'):
            print(f"  内容: {msg.content[:100] if msg.content else '(空)'}...")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"  工具调用: {msg.tool_calls}")
except Exception as e:
    print(f"\n❌ Agent 调用失败!")
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
