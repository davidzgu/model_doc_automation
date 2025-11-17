import json
import re
from typing import Dict, Any, Iterable

def merge_state_update_from_tool_messages(
    result: Dict[str, Any],
    out: Dict[str, Any],
    tool_names: Iterable[str],
) -> None:
    """从 result['messages'] 中找到指定工具的 ToolMessage，解析 JSON 并合并 state_update。
    支持多个工具，会按相反顺序处理所有匹配的工具消息。
    """
    msgs = result.get("messages", []) or []
    tool_names_set = set(tool_names)

    for msg in reversed(msgs):
        name = getattr(msg, "name", None) or getattr(msg, "tool", None)
        if name in tool_names_set:
            content = getattr(msg, "content", None)
            if not content:
                continue
            try:
                env = json.loads(content) if isinstance(content, str) else content
                if isinstance(env, dict) and isinstance(env.get("state_update"), dict):
                    out.update(env["state_update"])
            except Exception as e:
                out.setdefault("errors", []).append(
                    f"merge_state_update: failed to parse content from {name}: {e}"
                )


def merge_state_update_from_AI_messages(
    result: Dict[str, Any],
    out: Dict[str, Any],
) -> None:
    """从 result['messages'] 中的 AIMessage 提取 JSON 数据并合并到 out。

    遍历消息，从最后一条 AIMessage 的 content 中提取 JSON 数据。
    支持两种格式：
    1. 纯 JSON 字符串
    2. Markdown 代码块中的 JSON (```json ... ```)

    Args:
        result: Agent 的返回结果，包含 messages
        out: 输出字典，用于存储提取的状态
    """
    msgs = result.get("messages", []) or []

    for msg in reversed(msgs):
        content = getattr(msg, "content", None)
        if not content or not isinstance(content, str):
            continue

        data = None

        # 尝试直接解析 JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # 尝试从 markdown 代码块中提取 JSON
            json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                except:
                    continue

        # 如果成功解析了 JSON，合并到 out（排除 messages 字段）
        if data and isinstance(data, dict):
            for key, value in data.items():
                if key != "messages":
                    out[key] = value
            # 找到第一个有效的 JSON 就停止
            break