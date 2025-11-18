# Report Generator Agent 使用示例

## 在 Jupyter Notebook 中使用

```python
from pathlib import Path
from bsm_multi_agents.agents.report_generator_agent import report_generator_node

# 准备 state（假设你已经运行了 summary_generator 和 chart_generator）
state = {
    "messages": [],

    # Markdown 文件路径（从 summary_generator_node 获得）
    "report_md": [{
        "file_path": "/Users/yifanli/Github/model_doc_automation/data/output/validation_summary_20251117_184530.md"
    }],

    # 图表路径列表（从 chart_generator_node 获得）
    "chart_results": [
        {
            "chart_path": "/Users/yifanli/Github/model_doc_automation/data/output/option_prices.png",
            "description": "Option prices visualization"
        },
        {
            "chart_path": "/Users/yifanli/Github/model_doc_automation/data/output/greeks_sensitivity.png",
            "description": "Greeks sensitivity analysis"
        }
    ]
}

# 调用 agent
result = report_generator_node(state)

# 检查结果
if "report_path" in result:
    print(f"✅ Word 报告已生成: {result['report_path']}")
else:
    print(f"❌ 生成失败")
    if "errors" in result:
        for error in result["errors"]:
            print(f"  错误: {error}")
```

## 直接调用工具（不使用 agent）

```python
from bsm_multi_agents.tools.report_generator_tools import create_word_report
import json

result_json = create_word_report.invoke({
    "markdown_path": "/path/to/summary.md",
    "chart_paths": json.dumps([
        "/path/to/chart1.png",
        "/path/to/chart2.png"
    ]),
    "output_path": "/path/to/output/report.docx",
    "title": "Option Portfolio Analysis Report"
})

result = json.loads(result_json)
print(result)
```

## 完整工作流示例

```python
# 1. 生成摘要
summary_result = summary_generator_node(state)
state.update(summary_result)

# 2. 生成图表
chart_result = chart_generator_node(state)
state.update(chart_result)

# 3. 生成 Word 报告
report_result = report_generator_node(state)
state.update(report_result)

# 查看最终报告路径
print(f"最终报告: {state.get('report_path')}")
```

## 依赖安装

```bash
pip install python-docx markdown beautifulsoup4
```

## 输出文档包含：

1. **标题页**：报告标题和生成时间
2. **摘要内容**：从 markdown 转换的格式化内容
   - 标题层级
   - 段落
   - 列表（有序/无序）
   - 表格
3. **图表章节**：所有图表的清晰展示
   - 每个图表带标题
   - 图片自动适应页面宽度
4. **专业格式**：使用 OPA 风格的颜色和排版
