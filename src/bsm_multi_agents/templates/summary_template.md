# 📊 BSM Model Ongoing Performance Analysis

**Date:** {analysis_date} | **Total Options:** {total_options} | **Status:** {validation_status}

## 1. Executive Summary

| Metric | Value |
| :--- | :--- |
| **Total Processed** | {total_options} |
| **Pass Rate** | {pass_rate}% |
| **Failed** | {failed_count} |
| **Option Types** | {option_types_inline} |

## 2. Market Data Overview

| Metric | Range | Average |
| :--- | :--- | :--- |
| **Spot Price (S)** | {spot_range} | {spot_avg} |
| **Strike Price (K)** | {strike_range} | {strike_avg} |
| **Maturity (T)** | {maturity_range} | {maturity_avg} |
| **Volatility (σ)** | {volatility_range} | {volatility_avg} |

## 3. Pricing & Greeks Analysis

### BSM Pricing
{bsm_pricing_summary}

### Greeks Profile
| Greek | Average | Range | Expected |
| :--- | :--- | :--- | :--- |
| **Delta** | {delta_avg} | {delta_range} | [-1, 1] |
| **Gamma** | {gamma_avg} | {gamma_range} | ≥ 0 |
| **Vega** | {vega_avg} | {vega_range} | ≥ 0 |
| **Theta** | {theta_avg} | {theta_range} | ≤ 0 |
| **Rho** | {rho_avg} | {rho_range} | Varies |

## 4. Validation Details
**Critical Issues:**
{critical_issues}

## 5. Recommendations
{recommendations}

---
*Generated at {timestamp}*