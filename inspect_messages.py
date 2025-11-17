"""
检查 agent 返回的消息结构
"""
# 添加这段代码到你的 notebook 中运行

def inspect_agent_messages(result):
    """打印 agent 返回的所有消息"""
    messages = result.get("messages", [])

    print(f"\n{'='*80}")
    print(f"总共 {len(messages)} 条消息")
    print(f"{'='*80}\n")

    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        print(f"\n--- 消息 {i+1}: {msg_type} ---")

        # 检查是否是 ToolMessage
        if hasattr(msg, 'name'):
            print(f"Tool name: {msg.name}")

        # 显示内容（截断）
        if hasattr(msg, 'content'):
            content = msg.content
            if isinstance(content, str):
                if len(content) > 300:
                    print(f"Content (前300字符): {content[:300]}...")
                else:
                    print(f"Content: {content}")

        # 检查是否有 tool_calls
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"Tool calls: {len(msg.tool_calls)}")
            for tc in msg.tool_calls:
                print(f"  - {tc.get('name', 'unknown')}")

    print(f"\n{'='*80}\n")

# 使用示例：
# inspect_agent_messages(result)