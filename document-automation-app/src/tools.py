# -*- coding: utf-8 -*-
from typing import List
from langchain_core.tools import BaseTool
from src.bsm_utils import (
    bsm_calculator,
    csv_loader,
    greeks_calculator,
    sensitivity_test,
    batch_bsm_calculator,
)

def get_tools() -> List[BaseTool]:
    return [
        csv_loader,
        batch_bsm_calculator,  # 批量计算工具（推荐）
        bsm_calculator,        # 单次计算工具（备用）
        greeks_calculator,
        sensitivity_test,
    ]
