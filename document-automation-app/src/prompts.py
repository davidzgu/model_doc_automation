# -*- coding: utf-8 -*-
from langchain_core.prompts import PromptTemplate

READ_AND_CALC = PromptTemplate(
    input_variables=["path"],
    template=(
        "Read ONLY the first 5 rows from file in \n\n"
        "Path:{path}\n\n"
        "using csv_loader tool. After the data is successfully read from the file, "
        "then calculate the Black-Scholes-Merton option price for ALL data points in ONE call. "
        "\n\n"
        "IMPORTANT: Use batch_bsm_calculator tool with the JSON data returned by csv_loader. "
        "DO NOT call bsm_calculator multiple times for each row. "
        "The batch_bsm_calculator will efficiently process all rows at once and return a formatted table."
        "\n\n"
        "Expected workflow:\n"
        "1. Call csv_loader to read the CSV file\n"
        "2. Pass the JSON result directly to batch_bsm_calculator\n"
        "3. Display the markdown table returned by batch_bsm_calculator"
    ),
)