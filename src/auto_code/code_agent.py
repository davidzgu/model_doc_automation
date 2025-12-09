"""
Self-Improving Code Generation System
Agent uses Docker sandbox as a tool to iteratively generate and test code until it works.
"""


import json
from typing import Any, Dict, List, Optional, Tuple, Callable
from auto_code.docker_sandbox import DockerSandbox, ExecutionStatus



class CodeGenerationAgent:
    """
    Agent that uses the Docker sandbox to iteratively generate correct code.
    This simulates how an LLM agent would use the sandbox as a tool.
    """
    
    def __init__(self, sandbox: DockerSandbox, llm_function: Optional[Callable] = None):
        """
        Args:
            sandbox: DockerSandbox instance to use for testing
            llm_function: Function that takes (prompt, conversation_history) and returns code
                         If None, uses a mock implementation
        """
        self.sandbox = sandbox
        self.llm_function = llm_function
        self.conversation_history = []
        self.max_iterations = 5
    
 
    
    def generate_code_with_testing(
        self,
        task_description: str,
        test_cases: List[Dict[str, Any]],
        requirements: Optional[str] = None
    ) -> Tuple[Optional[str], List[Dict]]:
        """
        Generate code using the sandbox as a testing tool.
        
        Args:
            task_description: Natural language description of what the code should do
            test_cases: List of test cases with 'input' and 'expected' keys
            requirements: Additional requirements or constraints
            
        Returns:
            (final_code, iteration_log) - None if failed after max iterations
        """
        
        iteration_log = []
        self.conversation_history = []
        
        # Initial prompt
        initial_prompt = f"""
Generate Python code for the following task:

Task: {task_description}

Requirements:
{requirements or 'None'}

Test cases that your code must pass:
{json.dumps(test_cases, indent=2)}

Generate a single Python function that solves this task.
Return only the function code, no explanations.
"""
        
        for iteration in range(self.max_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{self.max_iterations}")
            print(f"{'='*60}")
            
            # Generate code
            if iteration == 0:
                code = self.llm_function(initial_prompt, self.conversation_history)
            else:
                # Use feedback from previous iteration
                code = self.llm_function(
                    f"Previous attempt failed. Please fix the code based on the feedback above.",
                    self.conversation_history
                )
            
            print(f"Generated code:\n{code}\n")
            
            # Test all test cases
            all_passed = True
            test_results = []
            
            for i, test_case in enumerate(test_cases):
                inputs = test_case.get('input', {})
                expected = test_case.get('expected')
                
                print(f"Testing case {i+1}: {inputs} -> expecting {expected}")
                
                # Use sandbox as a tool to test the code
                if isinstance(inputs, dict):
                    result = self.sandbox.test_code(code, **inputs)
                else:
                    result = self.sandbox.test_code(code, *inputs)
                
                test_results.append({
                    'test_case': i + 1,
                    'input': inputs,
                    'expected': expected,
                    'actual': result.output,
                    'passed': result.status == ExecutionStatus.SUCCESS and result.output == expected,
                    'feedback': result.get_feedback()
                })
                
                print(f"  {result.get_feedback()}")
                
                if result.status != ExecutionStatus.SUCCESS:
                    all_passed = False
                    break
                
                if expected is not None and result.output != expected:
                    all_passed = False
                    print(f"  ✗ Expected {expected}, got {result.output}")
                    break
            
            # Log iteration
            iteration_log.append({
                'iteration': iteration + 1,
                'code': code,
                'test_results': test_results,
                'all_passed': all_passed
            })
            
            # Update conversation history for LLM
            self.conversation_history.append({
                'role': 'assistant',
                'content': code
            })
            
            if all_passed:
                print(f"\n✓ All tests passed! Code generation successful.")
                return code, iteration_log
            
            # Prepare feedback for next iteration
            feedback = self._generate_feedback(test_results)
            self.conversation_history.append({
                'role': 'user',
                'content': feedback
            })
            
            print(f"\nFeedback for next iteration:\n{feedback}")
        
        print(f"\n✗ Failed to generate correct code after {self.max_iterations} iterations")
        return None, iteration_log
    
    def _generate_feedback(self, test_results: List[Dict]) -> str:
        """Generate feedback message for the LLM based on test results."""
        feedback = "The code failed the tests. Here's what went wrong:\n\n"
        
        for result in test_results:
            if not result['passed']:
                feedback += f"Test Case {result['test_case']}:\n"
                feedback += f"  Input: {result['input']}\n"
                feedback += f"  Expected: {result['expected']}\n"
                feedback += f"  Actual: {result.get('actual', 'N/A')}\n"
                feedback += f"  Feedback: {result['feedback']}\n\n"
        
        feedback += "Please fix the code to pass all test cases."
        return feedback


