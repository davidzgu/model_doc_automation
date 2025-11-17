#!/usr/bin/env python3
"""
验证 multi-agent graph 是否成功运行

测试流程：
1. data_loader_node 加载 CSV 数据
2. calculator_node 计算 BSM 价格和 Greeks
3. 验证最终状态包含所有必需的数据
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from bsm_multi_agents.graph.agent_graph import build_app
from bsm_multi_agents.graph.state import WorkflowState
from langchain_core.messages import HumanMessage


def print_section(title: str):
    """打印分节标题"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def verify_graph():
    """验证 graph 运行"""

    print_section("步骤 1: 构建 Graph")
    try:
        app = build_app()
        print("✅ Graph 构建成功")
    except Exception as e:
        print(f"❌ Graph 构建失败: {e}")
        return False

    print_section("步骤 2: 准备初始状态")
    csv_path = project_root / "data" / "input" / "dummy_options.csv"

    if not csv_path.exists():
        print(f"❌ CSV 文件不存在: {csv_path}")
        return False

    print(f"CSV 文件路径: {csv_path}")

    init_state: WorkflowState = {
        "csv_file_path": str(csv_path),
        "messages": [HumanMessage(content=f"Load and process options data from {csv_path}")],
    }
    print("✅ 初始状态准备完成")

    print_section("步骤 3: 运行 Graph")
    try:
        final_state = app.invoke(
            init_state,
            config={"configurable": {"thread_id": "verify-1"}}
        )
        print("✅ Graph 执行完成")
    except Exception as e:
        print(f"❌ Graph 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print_section("步骤 4: 验证结果")

    # 检查必需的字段
    required_fields = {
        "csv_data": "CSV 数据",
        "bsm_results": "BSM 价格计算结果",
        "greeks_results": "Greeks 计算结果",
    }

    all_passed = True

    for field, description in required_fields.items():
        if field in final_state and final_state[field]:
            data = final_state[field]
            count = len(data) if isinstance(data, list) else 1
            print(f"✅ {description}: {count} 条记录")

            # 显示第一条数据作为示例
            if isinstance(data, list) and len(data) > 0:
                print(f"   示例: {list(data[0].keys())}")
        else:
            print(f"❌ {description}: 缺失")
            all_passed = False

    # 检查错误
    if "errors" in final_state and final_state["errors"]:
        print(f"\n⚠️  发现错误:")
        for error in final_state["errors"]:
            print(f"   - {error}")
        all_passed = False
    else:
        print(f"\n✅ 无错误")

    # 显示消息统计
    messages = final_state.get("messages", [])
    print(f"\n📨 消息数量: {len(messages)}")

    # 统计消息类型
    message_types = {}
    for msg in messages:
        msg_type = type(msg).__name__
        message_types[msg_type] = message_types.get(msg_type, 0) + 1

    for msg_type, count in message_types.items():
        print(f"   - {msg_type}: {count}")

    print_section("步骤 5: 详细数据检查")

    # 检查 BSM 结果
    if "bsm_results" in final_state and final_state["bsm_results"]:
        bsm = final_state["bsm_results"]
        if isinstance(bsm, list) and len(bsm) > 0:
            print("BSM 结果第一条:")
            first_bsm = bsm[0]
            for key, value in first_bsm.items():
                print(f"   {key}: {value}")

    # 检查 Greeks 结果
    if "greeks_results" in final_state and final_state["greeks_results"]:
        greeks = final_state["greeks_results"]
        if isinstance(greeks, list) and len(greeks) > 0:
            print("\nGreeks 结果第一条:")
            first_greeks = greeks[0]
            for key, value in first_greeks.items():
                print(f"   {key}: {value}")

    print_section("总结")

    if all_passed:
        print("🎉 所有验证通过！Graph 运行成功！")
        print("\n✅ 数据流:")
        print("   CSV 加载 → BSM 计算 → Greeks 计算 → 完成")
        return True
    else:
        print("❌ 验证失败，请检查上述错误信息")
        return False


if __name__ == "__main__":
    success = verify_graph()
    sys.exit(0 if success else 1)