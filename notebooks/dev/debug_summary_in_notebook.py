#!/usr/bin/env python3
"""
在类似 notebook 的环境中调试 summary_generator_node
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# 模拟完整的 workflow
print("=" * 80)
print("模拟完整 Workflow - 调试 Summary Generator")
print("=" * 80)

from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.agents.data_loader_agent import data_loader_node
from bsm_multi_agents.agents.calculator_agent import calculator_node
from bsm_multi_agents.agents.pricing_validator_agent import validator_node
from bsm_multi_agents.agents.summary_generator_agent import summary_generator_node

# 1. 执行前面的节点获取数据
csv_path = str(Path(__file__).parent / "data" / "input" / "dummy_options.csv")
state = WorkflowState(csv_file_path=csv_path)

print("\n1. Running data_loader_node...")
out = data_loader_node(state)
state = {**state, **out}
print(f"   ✅ csv_data: {len(state.get('csv_data', []))} records")

print("\n2. Running calculator_node...")
out = calculator_node(state)
state = {**state, **out}
print(f"   ✅ bsm_results: {len(state.get('bsm_results', []))} records")
print(f"   ✅ greeks_results: {len(state.get('greeks_results', []))} records")

print("\n3. Running validator_node...")
out = validator_node(state)
state = {**state, **out}
print(f"   ✅ validate_results: {len(state.get('validate_results', []))} records")

# 2. 检查 summary_generator_node 的输入
print("\n" + "=" * 80)
print("检查 summary_generator_node 的输入")
print("=" * 80)

required_fields = ['validate_results', 'bsm_results', 'greeks_results']
for field in required_fields:
    if field in state and state[field]:
        print(f"✅ {field}: {len(state[field])} records")
    else:
        print(f"❌ {field}: 缺失")

# 3. 检查生成的 prompt
print("\n" + "=" * 80)
print("检查生成的 Prompt")
print("=" * 80)

from bsm_multi_agents.prompts.loader import load_prompt

validate_results_str = json.dumps(state["validate_results"], ensure_ascii=False)
bsm_results_str = json.dumps(state.get("bsm_results", []), ensure_ascii=False)
greeks_results_str = json.dumps(state.get("greeks_results", []), ensure_ascii=False)
template_path = str(Path(__file__).parent / "src" / "bsm_multi_agents" / "templates" / "summary_template.md")

prompt_path = Path(__file__).parent / "src" / "bsm_multi_agents" / "prompts" / "summary_generator_prompts.txt"
user_prompt = load_prompt(prompt_path).format(
    validate_results=validate_results_str,
    bsm_results=bsm_results_str,
    greeks_results=greeks_results_str,
    template_path=template_path
)

print(f"Prompt 长度: {len(user_prompt)} 字符")
print(f"Prompt 前 500 字符:")
print("-" * 80)
print(user_prompt[:500])
print("-" * 80)

# 4. 检查 Agent 的工具
print("\n" + "=" * 80)
print("检查 Agent 配置")
print("=" * 80)

from bsm_multi_agents.tools import get_tools_for_role
from bsm_multi_agents.agents.agent_factory import built_graph_agent_by_role

tools = get_tools_for_role("summary_generator")
print(f"✅ 'summary_generator' role 拥有 {len(tools)} 个工具:")
for tool in tools:
    print(f"   - {tool.name}")

agent_role = "summary_generator"
default_system = """
You are a reporting agent specialized in generating summary reports.
"""
agent = built_graph_agent_by_role(agent_role, default_system=default_system)
print(f"✅ Agent 创建成功: {type(agent)}")

# 5. 实际运行 summary_generator_node
print("\n" + "=" * 80)
print("运行 summary_generator_node")
print("=" * 80)

out = summary_generator_node(state)

# 6. 分析输出
print("\n" + "=" * 80)
print("分析输出")
print("=" * 80)

if "messages" in out:
    print(f"总共 {len(out['messages'])} 条消息:")

    has_tool_call = False
    has_tool_message = False

    for i, msg in enumerate(out["messages"]):
        msg_type = type(msg).__name__
        print(f"\n消息 {i+1}: {msg_type}")

        if msg_type == "AIMessage":
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                has_tool_call = True
                print(f"   ✅ AI 调用了工具:")
                for tc in msg.tool_calls:
                    print(f"      - 工具: {tc['name']}")
                    print(f"      - 参数: {list(tc['args'].keys())}")
            else:
                content_preview = msg.content[:200] if msg.content else "(empty)"
                print(f"   内容: {content_preview}...")

        elif msg_type == "ToolMessage":
            has_tool_message = True
            print(f"   ✅ 工具名: {msg.name}")
            try:
                content = json.loads(msg.content)
                if "state_update" in content:
                    print(f"   ✅ state_update 包含: {list(content['state_update'].keys())}")
            except:
                print(f"   内容预览: {msg.content[:200]}...")

    print("\n" + "-" * 80)
    if has_tool_call and has_tool_message:
        print("✅ Agent 成功调用了工具并收到结果")
    elif has_tool_call and not has_tool_message:
        print("⚠️  Agent 调用了工具但没有收到 ToolMessage")
    else:
        print("❌ Agent 没有调用工具")

# 7. 检查 state 更新
print("\n" + "=" * 80)
print("检查 State 更新")
print("=" * 80)

if "report_md" in out:
    print(f"✅ report_md 存在 ({len(out['report_md'])} 字符)")
    print(f"   预览: {out['report_md'][:200]}...")
else:
    print("❌ report_md 不存在")

if "report_path" in out:
    print(f"✅ report_path: {out['report_path']}")
    if Path(out['report_path']).exists():
        print(f"   ✅ 文件已保存")
    else:
        print(f"   ❌ 文件不存在")
else:
    print("❌ report_path 不存在")

if "errors" in out and out["errors"]:
    print(f"❌ 发现错误:")
    for error in out["errors"]:
        print(f"   - {error}")

print("\n" + "=" * 80)
print("调试完成")
print("=" * 80)