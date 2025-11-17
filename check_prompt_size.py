#!/usr/bin/env python3
"""
检查 prompt 的大小
"""
import json

# 模拟 10 条数据
sample_validate_results = [{
    "K": 105,
    "S": 100,
    "T": 1.0,
    "date": "2025-09-01",
    "delta": 0.5422283336,
    "gamma": 0.0198352619,
    "option_type": "call",
    "price": 8.0213522351,
    "r": 0.05,
    "rho": 46.2014811233,
    "sigma": 0.2,
    "theta": -6.277126437,
    "vega": 39.6705238084,
    "validations_result": "passed",
    "validations_details": []
}] * 10  # 10 条相同的数据

sample_bsm_results = [{
    "K": 105,
    "S": 100,
    "T": 1.0,
    "date": "2025-09-01",
    "option_type": "call",
    "r": 0.05,
    "sigma": 0.2,
    "BSM_Price": "8.021352235143176"
}] * 10

sample_greeks_results = [{
    "K": 105,
    "S": 100,
    "T": 1.0,
    "date": "2025-09-01",
    "option_type": "call",
    "r": 0.05,
    "sigma": 0.2,
    "price": 8.0213522351,
    "delta": 0.5422283336,
    "gamma": 0.0198352619,
    "vega": 39.6705238084,
    "rho": 46.2014811233,
    "theta": -6.277126437
}] * 10

# 转换为 JSON 字符串
validate_results_str = json.dumps(sample_validate_results, ensure_ascii=False)
bsm_results_str = json.dumps(sample_bsm_results, ensure_ascii=False)
greeks_results_str = json.dumps(sample_greeks_results, ensure_ascii=False)

print("=" * 80)
print("检查 Prompt 大小")
print("=" * 80)

print(f"\nvalidate_results JSON 长度: {len(validate_results_str)} 字符")
print(f"bsm_results JSON 长度: {len(bsm_results_str)} 字符")
print(f"greeks_results JSON 长度: {len(greeks_results_str)} 字符")

# 模拟 prompt
prompt_template = """Use the `generate_summary` tool to create a summary report.

Pass the following JSON strings as-is to the tool:
- validate_results: {validate_results}
- bsm_results: {bsm_results}
- greeks_results: {greeks_results}
- template_path: "/path/to/template.md"

After calling the tool, your task is complete. DO NOT reformat results."""

full_prompt = prompt_template.format(
    validate_results=validate_results_str,
    bsm_results=bsm_results_str,
    greeks_results=greeks_results_str
)

print(f"\n完整 Prompt 长度: {len(full_prompt)} 字符")
print(f"约 {len(full_prompt.split())} 个词")
print(f"约 {len(full_prompt) / 4:.0f} tokens (粗略估计)")

print("\n" + "=" * 80)
print("Prompt 预览 (前 1000 字符):")
print("=" * 80)
print(full_prompt[:1000])
print("...")
print("=" * 80)
print("Prompt 预览 (最后 500 字符):")
print("=" * 80)
print(full_prompt[-500:])
print("=" * 80)

if len(full_prompt) > 10000:
    print("\n⚠️  警告: Prompt 非常长 (>10k 字符)")
    print("   这可能导致某些 LLM 无法正确解析工具调用")
    print("\n建议: 不要在 prompt 中展示完整的 JSON 数据")