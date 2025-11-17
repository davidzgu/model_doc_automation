#!/usr/bin/env python3
"""
简单测试：使用少量数据测试 summary_generator_node
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# 测试数据 - 只用1条记录
validate_results = [{
    "K": 105,
    "S": 100,
    "T": 1.0,
    "date": "2025-09-01",
    "delta": 0.5422283336,
    "gamma": 0.0198352619,
    "option_type": "call",
    "price": 8.0213522351,
    "validations_result": "passed",
    "validations_details": []
}]

bsm_results = [{
    "K": 105,
    "S": 100,
    "T": 1.0,
    "date": "2025-09-01",
    "option_type": "call",
    "BSM_Price": "8.021352235143176"
}]

greeks_results = [{
    "K": 105,
    "S": 100,
    "T": 1.0,
    "date": "2025-09-01",
    "option_type": "call",
    "delta": 0.5422283336,
    "gamma": 0.0198352619,
}]

print("=" * 80)
print("测试 1: 检查工具是否注册")
print("=" * 80)

try:
    from bsm_multi_agents.tools import get_tools_for_role
    tools = get_tools_for_role("summary_generator")
    print(f"✅ 找到 {len(tools)} 个工具:")
    for tool in tools:
        print(f"   - {tool.name}")
except Exception as e:
    print(f"❌ 获取工具失败: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("测试 2: 直接调用 generate_summary 工具")
print("=" * 80)

try:
    from bsm_multi_agents.tools.summary_generator_tools import generate_summary

    # 直接调用工具
    result = generate_summary(
        validate_results=validate_results,
        bsm_results=bsm_results,
        greeks_results=greeks_results,
        template_path=str(Path(__file__).parent / "src" / "bsm_multi_agents" / "templates" / "summary_template.md"),
        save_md=False  # 不保存文件，只生成内容
    )

    result_dict = json.loads(result)
    if "state_update" in result_dict and "report_md" in result_dict["state_update"]:
        print("✅ 工具调用成功")
        print(f"   生成的报告预览 (前200字符):")
        print(f"   {result_dict['state_update']['report_md'][:200]}")
    else:
        print(f"❌ 工具返回格式不对: {result}")
except Exception as e:
    print(f"❌ 工具调用失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("测试 3: 测试 agent 能否识别工具")
print("=" * 80)

try:
    from bsm_multi_agents.agents.agent_factory import built_graph_agent_by_role
    from langchain_core.messages import HumanMessage

    agent = built_graph_agent_by_role("summary_generator", default_system="You are a reporting agent.")

    # 创建简单的测试消息
    test_prompt = """Use the `generate_summary` tool with this small test data:

validate_results: [{"validations_result": "passed"}]
bsm_results: []
greeks_results: []
template_path: "/tmp/test.md"
"""

    result = agent.invoke(
        {"messages": [HumanMessage(content=test_prompt)]},
        config={"recursion_limit": 10, "configurable": {"thread_id": "test-1"}}
    )

    if isinstance(result, dict) and "messages" in result:
        for msg in result["messages"]:
            msg_type = type(msg).__name__
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"✅ Agent 调用了工具!")
                for tc in msg.tool_calls:
                    print(f"   工具名: {tc['name']}")
            elif msg_type == "ToolMessage":
                print(f"✅ 收到 ToolMessage")
            elif msg_type == "AIMessage":
                content_preview = msg.content[:100] if msg.content else "(empty)"
                print(f"⚠️  收到 AIMessage (可能没调用工具): {content_preview}...")
    else:
        print(f"❌ Agent 返回格式异常: {type(result)}")

except Exception as e:
    print(f"❌ Agent 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)