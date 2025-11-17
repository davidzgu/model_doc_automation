"""
快速验证脚本 - 可以在 notebook 或命令行中使用
"""

def verify_graph_result(final_state):
    """
    验证 graph 执行结果

    Args:
        final_state: graph.invoke() 返回的最终状态

    Returns:
        bool: 是否通过验证
    """
    print("="*80)
    print("验证 Graph 执行结果")
    print("="*80)

    # 1. 检查必需字段
    checks = {
        "csv_data": "✅ CSV 数据已加载",
        "bsm_results": "✅ BSM 价格计算完成",
        "greeks_results": "✅ Greeks 计算完成",
    }

    all_passed = True

    for field, success_msg in checks.items():
        if field in final_state and final_state[field]:
            data = final_state[field]
            count = len(data) if isinstance(data, list) else 1
            print(f"{success_msg} ({count} 条记录)")
        else:
            print(f"❌ {field} 缺失或为空")
            all_passed = False

    # 2. 检查错误
    if "errors" in final_state and final_state["errors"]:
        print(f"\n❌ 发现错误:")
        for error in final_state["errors"]:
            print(f"   {error}")
        all_passed = False

    # 3. 显示数据样例
    if all_passed:
        print("\n" + "="*80)
        print("数据样例")
        print("="*80)

        # BSM 结果
        if "bsm_results" in final_state and final_state["bsm_results"]:
            bsm = final_state["bsm_results"][0]
            print(f"\nBSM 结果 (第1条):")
            print(f"  期权类型: {bsm.get('option_type')}")
            print(f"  标的价格 S: {bsm.get('S')}")
            print(f"  行权价 K: {bsm.get('K')}")
            print(f"  BSM 价格: {bsm.get('BSM_Price', 'N/A')}")

        # Greeks 结果
        if "greeks_results" in final_state and final_state["greeks_results"]:
            greeks = final_state["greeks_results"][0]
            print(f"\nGreeks 结果 (第1条):")
            print(f"  Delta: {greeks.get('delta')}")
            print(f"  Gamma: {greeks.get('gamma')}")
            print(f"  Vega: {greeks.get('vega')}")
            print(f"  Theta: {greeks.get('theta')}")
            print(f"  Rho: {greeks.get('rho')}")

    # 4. 总结
    print("\n" + "="*80)
    if all_passed:
        print("🎉 验证通过！Graph 成功运行！")
    else:
        print("❌ 验证失败，请检查上述错误")
    print("="*80)

    return all_passed


# 使用示例（在 notebook 中）：
"""
from bsm_multi_agents.graph.agent_graph import build_app
from langchain_core.messages import HumanMessage
from pathlib import Path

# 运行 graph
app = build_app()
csv_path = Path.cwd().parents[1] / "data" / "input" / "dummy_options.csv"
init_state = {
    "csv_file_path": str(csv_path),
    "messages": [HumanMessage(content=f"Process {csv_path}")],
}
final_state = app.invoke(init_state, config={"configurable": {"thread_id": "test-1"}})

# 验证结果
from quick_verify import verify_graph_result
verify_graph_result(final_state)
"""