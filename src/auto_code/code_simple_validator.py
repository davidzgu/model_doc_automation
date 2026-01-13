from typing import Optional, Tuple
import ast

class SecurityValidator:
    """Static analysis to detect dangerous patterns."""
    
    FORBIDDEN_IMPORTS = {
        'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests',
        'pickle', 'marshal', 'shelve', 'pty', 'fcntl', 'termios'
    }
    
    FORBIDDEN_BUILTINS = {
        '__import__', 'eval', 'exec', 'compile', 'open', 
        'input', 'raw_input', 'execfile'
    }
    
    @classmethod
    def validate_code(cls, code: str) -> Tuple[bool, str]:
        """Validate code using AST analysis."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    if module in cls.FORBIDDEN_IMPORTS:
                        return False, f"Forbidden import: {alias.name}"
            
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split('.')[0]
                    if module in cls.FORBIDDEN_IMPORTS:
                        return False, f"Forbidden import: {node.module}"
            
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in cls.FORBIDDEN_BUILTINS:
                        return False, f"Forbidden function: {node.func.id}"
        
        return True, ""
    
    @classmethod
    def extract_function_name(cls, code: str) -> Optional[str]:
        """Extract the main function name from code."""
        try:
            tree = ast.parse(code)
            functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            return functions[0] if functions else None
        except:
            return None

