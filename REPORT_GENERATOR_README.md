# Report Generator Agent - 完整实现

## 📋 概述

为你创建了一个完整的 `report_generator` 组件，遵循你项目中 **Agent + Prompt + Tool** 的架构模式。该组件能够：

✅ 整合 Markdown 总结文档
✅ 嵌入可视化图表
✅ 生成专业的 Word OPA 文档
✅ 自动格式化和排版

---

## 📁 文件结构

### 1. **Tool** - `report_generator_tools.py`
**路径**: `src/bsm_multi_agents/tools/report_generator_tools.py`

**功能**:
- `create_word_report()`: 核心工具函数
  - 读取 Markdown 文件
  - 解析并转换为 Word 格式（标题、段落、列表、表格）
  - 嵌入图表图片
  - 应用 OPA 专业样式

**特性**:
- 自动 HTML/Markdown 解析
- 表格格式保留
- 图片自适应页面宽度
- 专业配色方案（深蓝色系）

### 2. **Prompt** - `report_generator_prompts.txt`
**路径**: `src/bsm_multi_agents/prompts/report_generator_prompts.txt`

简洁明确的提示词，指导 LLM 调用工具生成报告。

### 3. **Agent** - `report_generator_agent.py`
**路径**: `src/bsm_multi_agents/agents/report_generator_agent.py`

**功能**:
- `report_generator_node()`: 工作流节点函数
  - 从 state 提取 markdown 路径和图表路径
  - 直接调用 `create_word_report` 工具
  - 更新 state（添加 `report_path`）

**设计**:
- 直接调用工具（不依赖 LLM，确保稳定性）
- 智能路径提取（支持多种 state 格式）
- 自动时间戳命名

---

## 🔧 State 更新

已更新 `src/bsm_multi_agents/graph/state.py`:

```python
class WorkflowState(TypedDict, total=False):
    # ... 现有字段 ...
    chart_results: List[Dict[str, Any]]  # 新增：图表信息
    report_path: str                      # 新增：最终报告路径
```

---

## 📦 依赖安装

```bash
pip install python-docx markdown beautifulsoup4
```

---

## 🚀 使用方法

### 方式 1: 在工作流中使用

```python
from bsm_multi_agents.agents.report_generator_agent import report_generator_node

# state 应包含:
# - report_md: markdown 文件路径
# - chart_results: 图表信息列表

result = report_generator_node(state)

# 结果中包含:
# - report_path: Word 文档路径
# - messages: 执行消息
```

### 方式 2: 直接调用工具

```python
from bsm_multi_agents.tools.report_generator_tools import create_word_report
import json

result_json = create_word_report.invoke({
    "markdown_path": "data/output/summary.md",
    "chart_paths": json.dumps([
        "data/output/option_prices.png",
        "data/output/greeks_sensitivity.png"
    ]),
    "output_path": "data/output/OPA_Report.docx",
    "title": "Option Portfolio Analysis Report"
})

result = json.loads(result_json)
print(f"Report: {result['document_path']}")
```

### 方式 3: 完整工作流

```python
# 步骤 1: 生成 Markdown 摘要
summary_result = summary_generator_node(state)
state.update(summary_result)

# 步骤 2: 生成图表
chart_result = chart_generator_node(state)
state.update(chart_result)

# 步骤 3: 生成 Word 报告
report_result = report_generator_node(state)
state.update(report_result)

print(f"✅ 最终报告: {state['report_path']}")
```

---

## 📄 生成的文档结构

1. **封面**
   - 报告标题
   - 生成时间戳

2. **摘要内容** (从 Markdown 转换)
   - 多级标题
   - 格式化段落
   - 项目列表
   - 数据表格

3. **图表章节**
   - 每个图表独立展示
   - 带描述标题
   - 高清图片（300 DPI）

4. **专业格式**
   - OPA 企业配色
   - 一致的字体和间距
   - 自动页边距和布局

---

## 🧪 测试

### 运行测试脚本:

```bash
python notebooks/dev/test_report_generator.py
```

### Jupyter Notebook:

参考 `notebooks/dev/example_report_generator.md` 中的完整示例。

---

## 🎨 自定义

### 修改样式

在 `report_generator_tools.py` 的 `_setup_document_styles()` 函数中:

```python
# 修改标题颜色
h1.font.color.rgb = RGBColor(0, 51, 102)  # 深蓝色

# 修改字体大小
h1.font.size = Pt(16)
```

### 添加更多元素

在 `_add_markdown_content()` 函数中扩展支持:
- 代码高亮
- 引用块
- 脚注
- 等等

---

## ✅ 优势

1. **可靠**: 直接调用工具，不依赖 LLM 工具调用能力
2. **灵活**: 支持多种输入格式
3. **专业**: 银行级 OPA 文档风格
4. **完整**: 遵循项目架构模式
5. **可维护**: 清晰的代码结构和注释

---

## 📝 下一步

1. 安装依赖:
   ```bash
   pip install python-docx markdown beautifulsoup4
   ```

2. 测试工具:
   ```bash
   python notebooks/dev/test_report_generator.py
   ```

3. 集成到工作流:
   - 在 `agent_graph.py` 中添加 `report_generator_node`
   - 连接到工作流最后一步

---

## 🙋 需要帮助?

如果需要:
- 自定义文档样式
- 添加更多功能（如目录、页码）
- 集成到现有工作流
- 处理特殊的 Markdown 格式

随时告诉我！
