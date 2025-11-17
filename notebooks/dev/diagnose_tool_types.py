"""
诊断工具参数类型的辅助函数
"""

def diagnose_agent_messages(result, verbose=True):
    """
    诊断 agent 返回的消息中的工具调用参数类型

    Args:
        result: agent.invoke() 的返回值
        verbose: 是否显示详细信息

    Returns:
        dict: 工具调用信息统计
    """
    messages = result.get("messages", [])
    tool_calls_info = {}

    print("="*80)
    print("Agent 消息诊断")
    print("="*80)

    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__

        if verbose:
            print(f"\n消息 {i+1}: {msg_type}")

        # 检查 AIMessage 中的 tool_calls
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for call in msg.tool_calls:
                tool_name = call.get('name', 'unknown')
                args = call.get('args', {})

                if tool_name not in tool_calls_info:
                    tool_calls_info[tool_name] = []

                call_info = {}

                print(f"\n  🔧 工具调用: {tool_name}")

                for param_name, param_value in args.items():
                    param_type = type(param_value).__name__

                    param_info = {
                        "type": param_type,
                        "value_preview": None
                    }

                    print(f"    📦 参数: {param_name}")
                    print(f"       类型: {param_type}")

                    # 根据类型显示不同的信息
                    if isinstance(param_value, list):
                        print(f"       长度: {len(param_value)}")
                        if param_value:
                            first_type = type(param_value[0]).__name__
                            print(f"       元素类型: {first_type}")

                            # 如果是字典列表，显示键
                            if isinstance(param_value[0], dict):
                                keys = list(param_value[0].keys())
                                print(f"       字典键: {keys}")
                                param_info["dict_keys"] = keys

                            param_info["length"] = len(param_value)
                            param_info["element_type"] = first_type
                            param_info["value_preview"] = param_value[0] if len(param_value) > 0 else None

                    elif isinstance(param_value, dict):
                        keys = list(param_value.keys())
                        print(f"       字典键: {keys}")
                        param_info["dict_keys"] = keys
                        param_info["value_preview"] = param_value

                    elif isinstance(param_value, str):
                        preview = param_value[:100] + "..." if len(param_value) > 100 else param_value
                        print(f"       值预览: {preview}")
                        param_info["value_preview"] = preview

                    else:
                        print(f"       值: {param_value}")
                        param_info["value_preview"] = param_value

                    call_info[param_name] = param_info

                tool_calls_info[tool_name].append(call_info)

        # 检查 ToolMessage
        if msg_type == "ToolMessage":
            tool_name = getattr(msg, 'name', 'unknown')
            content = getattr(msg, 'content', None)

            print(f"\n  ✅ 工具返回: {tool_name}")

            if content:
                # 尝试解析 JSON
                try:
                    import json
                    data = json.loads(content) if isinstance(content, str) else content

                    if isinstance(data, dict):
                        print(f"     返回类型: dict")
                        print(f"     键: {list(data.keys())}")

                        # 检查是否有 state_update
                        if "state_update" in data:
                            state_keys = list(data["state_update"].keys())
                            print(f"     state_update 键: {state_keys}")
                    else:
                        print(f"     返回类型: {type(data).__name__}")

                except:
                    preview = content[:100] + "..." if len(content) > 100 else content
                    print(f"     内容预览: {preview}")

    print("\n" + "="*80)
    print("工具调用汇总")
    print("="*80)

    for tool_name, calls in tool_calls_info.items():
        print(f"\n工具: {tool_name}")
        print(f"  调用次数: {len(calls)}")

        if calls:
            first_call = calls[0]
            print(f"  参数:")
            for param_name, param_info in first_call.items():
                print(f"    - {param_name}: {param_info['type']}")

    print("\n" + "="*80)

    return tool_calls_info


def check_tool_signature(tool_func):
    """
    检查工具函数的参数签名

    Args:
        tool_func: 工具函数对象
    """
    import inspect

    print("="*80)
    print(f"工具签名: {tool_func.name if hasattr(tool_func, 'name') else 'unknown'}")
    print("="*80)

    # 获取函数签名
    sig = inspect.signature(tool_func.func if hasattr(tool_func, 'func') else tool_func)

    print(f"\n参数列表:")
    for param_name, param in sig.parameters.items():
        annotation = param.annotation
        default = param.default

        print(f"\n  参数: {param_name}")
        print(f"    类型注解: {annotation}")

        if default != inspect.Parameter.empty:
            print(f"    默认值: {default}")

    # 显示文档字符串
    if tool_func.__doc__:
        print(f"\n文档:")
        print(f"  {tool_func.__doc__.strip()}")

    print("\n" + "="*80)


# 使用示例
"""
# 在 notebook 中使用：

from diagnose_tool_types import diagnose_agent_messages, check_tool_signature

# 1. 诊断 agent 消息
result = agent.invoke(...)
tool_calls_info = diagnose_agent_messages(result)

# 2. 检查工具签名
from bsm_multi_agents.tools import batch_bsm_calculator
check_tool_signature(batch_bsm_calculator)
"""
