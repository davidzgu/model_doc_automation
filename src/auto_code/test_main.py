from auto_code.docker_sandbox import DockerSandbox
from auto_code.code_agent import CodeGenerationAgent
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

def extract_code_from_response(text: str) -> str:
    """If the model returned a fenced code block, extract it; otherwise return full text."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    return text.strip()
    
def llm_chat_function(prompt: str, conversation_history: list) -> str:
    """
    Adapter that matches CodeGenerationAgent's expected signature:
        (prompt, conversation_history) -> code string
    Uses LangChain ChatOpenAI. Expects OPENAI_API_KEY in environment.
    """
    # Fallback mock if LangChain/OpenAI not available
    if ChatOpenAI is None:
        # Return a simple correct fibonacci function as a safe fallback
        return (
            "def fibonacci(n):\n"
            "    if n <= 0:\n"
            "        return 0\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n-1):\n"
            "        a, b = b, a + b\n"
            "    return b\n"
        )

    # Build messages from conversation_history
    messages = []
    for msg in conversation_history:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if role == 'system':
            messages.append(SystemMessage(content=content))
        elif role == 'assistant':
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    # Append current prompt as the latest user message
    messages.append(HumanMessage(content=prompt))

    # Initialize chat model (ensure OPENAI_API_KEY is set in env)
    chat = ChatOpenAI(model_name="gpt-5-mini", temperature=0)
    response = chat(messages)  # returns an AIMessage-like object
    text = getattr(response, "content", str(response))

    # Extract code if wrapped in fences
    return extract_code_from_response(text)


if __name__ == "__main__":    
    print("="*60)
    print("Self-Improving Code Generation with Docker Sandbox")
    print("="*60)
    
    # Initialize sandbox
    sandbox = DockerSandbox(
        timeout_seconds=10,
        memory_limit="256m"
    )
    
    # Initialize agent
    agent = CodeGenerationAgent(sandbox)
    
    # Example: Generate fibonacci function
    task = "Calculate the nth Fibonacci number"
    test_cases = [
        {'input': {'n': 0}, 'expected': 0},
        {'input': {'n': 1}, 'expected': 1},
        {'input': {'n': 5}, 'expected': 5},
        {'input': {'n': 10}, 'expected': 55}
    ]
    
    final_code, log = agent.generate_code_with_testing(
        task_description=task,
        test_cases=test_cases
    )
    
    if final_code:
        print("\n" + "="*60)
        print("FINAL WORKING CODE:")
        print("="*60)
        print(final_code)
        print("\nIteration Summary:")
        for entry in log:
            status = "✓ PASSED" if entry['all_passed'] else "✗ FAILED"
            print(f"  Iteration {entry['iteration']}: {status}")
    
    # Clean up
    print("\nCleaning up containers...")
    containers = sandbox.client.containers.list(all=True, filters={'ancestor': sandbox.image})
    for container in containers:
        try:
            container.remove(force=True)
        except:
            pass
    print("Done!")