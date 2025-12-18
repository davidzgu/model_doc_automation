import docker
import json
import time
import tempfile
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from auto_code.code_simple_validator import SecurityValidator


class ExecutionStatus(Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    SECURITY_VIOLATION = "security_violation"
    CONTAINER_ERROR = "container_error"


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    output: Any = None
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    memory_used: int = 0
    error_message: str = ""
    exit_code: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for agent consumption."""
        return {
            'status': self.status.value,
            'output': self.output,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'execution_time': self.execution_time,
            'error_message': self.error_message,
            'success': self.status == ExecutionStatus.SUCCESS
        }
    
    def get_feedback(self) -> str:
        """Generate natural language feedback for the agent."""
        if self.status == ExecutionStatus.SUCCESS:
            return f"✓ Code executed successfully in {self.execution_time:.3f}s. Output: {self.output}"
        elif self.status == ExecutionStatus.SYNTAX_ERROR:
            return f"✗ Syntax Error: {self.error_message}"
        elif self.status == ExecutionStatus.RUNTIME_ERROR:
            return f"✗ Runtime Error: {self.error_message}\nStderr: {self.stderr}"
        elif self.status == ExecutionStatus.TIMEOUT:
            return f"✗ Timeout: {self.error_message}"
        elif self.status == ExecutionStatus.SECURITY_VIOLATION:
            return f"✗ Security Violation: {self.error_message}"
        else:
            return f"✗ Error ({self.status.value}): {self.error_message}"


class DockerSandbox:
    """Execute code in isolated Docker containers - designed as a tool for agents."""
    
    DEFAULT_IMAGE = "python:3.11-alpine"
    
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        timeout_seconds: int = 30,
        memory_limit: str = "256m",
        enable_network: bool = False,
        preinstall_packages: Optional[List[str]] = None,
    ):
        # base image the user requested (e.g. "python:3.11-slim")
        self.base_image = image
        # if preinstall_packages provided, we'll build a derived image tag
        self.preinstall_packages = preinstall_packages or []
        self.image = (
            f"{self.base_image}-with-pkgs" if self.preinstall_packages else self.base_image
        )
        self.timeout_seconds = timeout_seconds
        self.memory_limit = memory_limit
        self.enable_network = enable_network
        
        try:
            self.client = docker.from_env()
            self._ensure_image()
        except docker.errors.DockerException as e:
            raise RuntimeError(f"Docker not available: {e}")
    
    def _ensure_image(self):
            """Ensure the Docker image is available. If preinstall_packages is set, build a derived image."""
            try:
                self.client.images.get(self.image)
            except docker.errors.ImageNotFound:
                if self.preinstall_packages:
                    print(f"Building image {self.image} from {self.base_image} with packages {self.preinstall_packages}...")
                    temp_dir = tempfile.mkdtemp()
                    try:
                        dockerfile = (
                            f"FROM {self.base_image}\n"
                            "ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\n"
                            "RUN pip install --no-cache-dir " + " ".join(self.preinstall_packages) + "\n"
                        )
                        with open(os.path.join(temp_dir, "Dockerfile"), "w", encoding="utf-8") as df:
                            df.write(dockerfile)
                        # build the image and tag it as self.image
                        self.client.images.build(path=temp_dir, tag=self.image, rm=True, pull=True)
                    finally:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                else:
                    print(f"Pulling image {self.base_image}...")
                    self.client.images.pull(self.base_image)
    
    def _create_execution_script(
        self, 
        code: str, 
        function_name: str,
        args: tuple,
        kwargs: dict
    ) -> str:
        """Create a Python script that executes the function."""
        args_json = json.dumps(args)
        kwargs_json = json.dumps(kwargs)
        
        script = f"""
import json
import sys
import traceback
import time

# User-provided code
{code}

def execute():
    try:
        start_time = time.time()
        args = json.loads('''{args_json}''')
        kwargs = json.loads('''{kwargs_json}''')
        
        result = {function_name}(*args, **kwargs)
        execution_time = time.time() - start_time
        
        output = {{
            'status': 'success',
            'result': result,
            'execution_time': execution_time
        }}
        print("__RESULT_START__")
        print(json.dumps(output))
        print("__RESULT_END__")
        
    except Exception as e:
        output = {{
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc()
        }}
        print("__RESULT_START__")
        print(json.dumps(output))
        print("__RESULT_END__")
        sys.exit(1)

if __name__ == '__main__':
    execute()
"""
        return script
    
    def _extract_result(self, output: str) -> Optional[dict]:
        """Extract JSON result from container output."""
        try:
            start = output.index("__RESULT_START__") + len("__RESULT_START__")
            end = output.index("__RESULT_END__")
            return json.loads(output[start:end].strip())
        except:
            return None
    
    def test_code(
        self, 
        code: str, 
        *args,
        skip_validation: bool = False,
        **kwargs
    ) -> ExecutionResult:
        """
        Test code execution - primary interface for agents.
        
        This is the tool function that agents call to test their generated code.
        
        Args:
            code: Python code containing a function to test
            *args: Arguments to pass to the function
            skip_validation: Skip security validation (for testing)
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            ExecutionResult with detailed feedback
        """
        
        # Validate code
        if not skip_validation:
            is_valid, error_msg = SecurityValidator.validate_code(code)
            if not is_valid:
                return ExecutionResult(
                    status=ExecutionStatus.SECURITY_VIOLATION,
                    error_message=error_msg
                )
        
        # Extract function name
        function_name = SecurityValidator.extract_function_name(code)
        if not function_name:
            return ExecutionResult(
                status=ExecutionStatus.SYNTAX_ERROR,
                error_message="No function definition found in code"
            )
        
        # Create execution script
        script = self._create_execution_script(code, function_name, args, kwargs)
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        script_path = os.path.join(temp_dir, "execute.py")
        
        try:
            with open(script_path, 'w') as f:
                f.write(script)
            
            # Container configuration
            container_config = {
                'image': self.image,
                'command': ['python', '/workspace/execute.py'],
                'volumes': {temp_dir: {'bind': '/workspace', 'mode': 'ro'}},
                'working_dir': '/workspace',
                'mem_limit': self.memory_limit,
                'network_disabled': not self.enable_network,
                'detach': True,
                'remove': False,
                'security_opt': ['no-new-privileges'],
                'cap_drop': ['ALL'],
                'read_only': True,
            }
            
            # Run container
            start_time = time.time()
            container = self.client.containers.run(**container_config)
            
            try:
                result = container.wait(timeout=self.timeout_seconds)
                execution_time = time.time() - start_time
                logs = container.logs(stdout=True, stderr=True).decode('utf-8')
                
                parsed = self._extract_result(logs)
                stats = container.stats(stream=False)
                memory_used = stats['memory_stats'].get('usage', 0)
                
                container.remove(force=True)
                
                if parsed:
                    if parsed['status'] == 'success':
                        return ExecutionResult(
                            status=ExecutionStatus.SUCCESS,
                            output=parsed.get('result'),
                            stdout=logs,
                            execution_time=parsed.get('execution_time', execution_time),
                            memory_used=memory_used,
                            exit_code=result['StatusCode']
                        )
                    else:
                        return ExecutionResult(
                            status=ExecutionStatus.RUNTIME_ERROR,
                            error_message=f"{parsed.get('error_type', 'Error')}: {parsed.get('error', 'Unknown')}",
                            stderr=parsed.get('traceback', ''),
                            stdout=logs,
                            execution_time=execution_time,
                            exit_code=result['StatusCode']
                        )
                else:
                    return ExecutionResult(
                        status=ExecutionStatus.RUNTIME_ERROR,
                        error_message="Failed to parse execution result",
                        stdout=logs,
                        execution_time=execution_time,
                        exit_code=result['StatusCode']
                    )
                    
            except Exception as e:
                container.stop(timeout=1)
                container.remove(force=True)
                
                if "timeout" in str(e).lower():
                    return ExecutionResult(
                        status=ExecutionStatus.TIMEOUT,
                        error_message=f"Execution exceeded {self.timeout_seconds}s"
                    )
                else:
                    return ExecutionResult(
                        status=ExecutionStatus.CONTAINER_ERROR,
                        error_message=str(e)
                    )
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_sandbox = DockerSandbox()
    test_code = """
def fibonacci(n):
    if n <= 0:
        return 0
    a, b = 0, 1
    for _ in range(n-1):
        a, b = b, a + b
    return b
"""
    test_cases = [
        {'input': {'n': 0}, 'expected': 0},
        {'input': {'n': 1}, 'expected': 1},
        {'input': {'n': 5}, 'expected': 5},
        {'input': {'n': 10}, 'expected': 55}
    ]
    inputs = test_cases[1].get('input', {})
    res = test_sandbox.test_code(test_code, **inputs)
    print(res)