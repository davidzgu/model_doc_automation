#!/usr/bin/env python3
"""
诊断 summary_generator_node 为什么只产生 AIMessage 而不是 ToolMessage
"""
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bsm_multi_agents.tools import get_tools_for_role
from bsm_multi_agents.agents.agent_factory import built_graph_agent_by_role

def diagnose_summary_generator():
    print("=" * 80)
    print("诊断 Summary Generator Agent")
    print("=" * 80)

    # 1. 检查工具注册
    print("\n1. 检查 'summary_generator' role 的工具注册:")
    tools = get_tools_for_role("summary_generator")
    if tools:
        print(f"   ✅ 找到 {len(tools)} 个工具:")
        for tool in tools:
            print(f"      - {tool.name}: {tool.description[:100]}...")
    else:
        print("   ❌ 没有找到任何工具注册到 'summary_generator' role")
        print("   尝试查看其他可用的 roles...")

        # 尝试其他可能的 role
        for test_role in ["generator", "summary", "reporting"]:
            test_tools = get_tools_for_role(test_role)
            if test_tools:
                print(f"\n   发现工具注册到 '{test_role}' role:")
                for tool in test_tools:
                    print(f"      - {tool.name}")

    # 2. 检查 agent 能否访问工具
    print("\n2. 检查 agent 实例化:")
    try:
        agent_role = "summary_generator"
        default_system = "You are a reporting agent specialized in generating summary reports."
        agent = built_graph_agent_by_role(agent_role, default_system=default_system)

        # 获取 agent 的工具列表
        if hasattr(agent, 'tools'):
            agent_tools = agent.tools
        elif hasattr(agent, '_tools'):
            agent_tools = agent._tools
        else:
            # 尝试从 graph 中获取
            agent_tools = None

        if agent_tools:
            print(f"   ✅ Agent 成功创建，拥有 {len(agent_tools)} 个工具:")
            for tool in agent_tools:
                print(f"      - {tool.name if hasattr(tool, 'name') else tool}")
        else:
            print(f"   ⚠️  Agent 创建成功，但无法直接访问工具列表")
            print(f"      Agent type: {type(agent)}")

    except Exception as e:
        print(f"   ❌ Agent 创建失败: {e}")

    # 3. 测试 prompt 格式
    print("\n3. 检查 prompt 模板:")
    try:
        from bsm_multi_agents.prompts.loader import load_prompt
        prompt_path = Path(__file__).parent / "src" / "bsm_multi_agents" / "prompts" / "summary_generator_prompts.txt"

        if prompt_path.exists():
            prompt_template = load_prompt(prompt_path)
            print(f"   ✅ Prompt 文件存在:")
            print(f"      路径: {prompt_path}")
            print(f"      内容预览:")
            print("      " + "-" * 60)
            for line in prompt_template.split('\n')[:10]:
                print(f"      {line}")
            print("      " + "-" * 60)

            # 检查是否包含必要的占位符
            required_placeholders = ['{validate_results}', '{bsm_results}', '{greeks_results}', '{template_path}']
            for placeholder in required_placeholders:
                if placeholder in prompt_template:
                    print(f"   ✅ 包含占位符: {placeholder}")
                else:
                    print(f"   ❌ 缺少占位符: {placeholder}")
        else:
            print(f"   ❌ Prompt 文件不存在: {prompt_path}")

    except Exception as e:
        print(f"   ❌ 检查 prompt 失败: {e}")

    # 4. 检查模板文件
    print("\n4. 检查模板文件:")
    template_path = Path(__file__).parent / "src" / "bsm_multi_agents" / "templates" / "summary_template.md"
    if template_path.exists():
        print(f"   ✅ 模板文件存在: {template_path}")
        print(f"      文件大小: {template_path.stat().st_size} bytes")
    else:
        print(f"   ❌ 模板文件不存在: {template_path}")

    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)

if __name__ == "__main__":
    diagnose_summary_generator()
