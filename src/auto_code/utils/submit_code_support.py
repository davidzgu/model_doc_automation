from pathlib import Path
import ast
import textwrap

def wrap_row_process_function(row_process_function_code:str,
                              row_process_function_docstring:str,
                              code_name:str):
    file_path = Path(__file__).resolve().parent
    template_file = file_path/"tool_function_wrapper_template.txt"
    with open(template_file, "r", encoding="utf-8") as f:
        wrapper_template = f.read()
    raw_code = textwrap.dedent(row_process_function_code).strip()
    indented_code = textwrap.indent(raw_code, " " * 4)
    # print(indented_code)
    after_wrapping = wrapper_template.replace("<<<CODE_NAME>>>", code_name)
    after_wrapping = after_wrapping.replace("<<<PROCESS_ROW_FUNCTION_DOCSTRING>>>", row_process_function_docstring)
    after_wrapping = after_wrapping.replace("<<<PROCESS_ROW_FUNCTION_CODE>>>", indented_code)
    return after_wrapping


def get_function_docstring(function_code:str) -> str:
    module = ast.parse(function_code)

    funcs = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "process_row"
    ]

    if len(funcs) != 1:
        raise ValueError("Expected exactly one process_row function")

    func = funcs[0]

    docstring = ast.get_docstring(func)
    if docstring is None:
        docstring = ''
    return docstring





