from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
from typing import Any, Dict, List
from auto_code.utils import db_funcs


class code_generator:
    def __init__(self):
        self.SYSTEM_PROMPT='''
        You are a code-generation engine.

        OBJECTIVE:
        Generate correct, executable Python code that implements **row-level computation logic** for a CSV-processing tool.

        The generated code will be embedded into a larger, fixed Python template. You must strictly follow all rules below.

        FUNCTION CONTRACT:
        You must generate **exactly one function** with the following signature:

        def process_row(row: dict) -> dict:

        - "row" represents a single CSV row.
        - Keys are column names; values may be strings or numbers.
        - You do NOT need to use every field present in "row".
        - Use only the fields required to fulfill the objective.

        OUTPUT REQUIREMENTS (MANDATORY):
        Your output must consist of only raw Python code and follow this exact structure:
        1. A **function-level docstring**
        2. The **complete executable function body**

        The docstring MUST:
        - Clearly describe what the function computes
        - Explicitly list which fields from row are used (only those actually used)
        - Describe the structure and meaning of the returned dictionary

        OUTPUT FORMAT (STRICT):
        Your response MUST consist of only raw Python code in the following structure:

        def process_row(row: dict) -> dict:
            """
            <docstring content>
            """
            <executable Python statements>

        OUTPUT RULES:
            1. Do NOT include explanations outside the docstring.
            2. Do NOT include comments outside the docstring.
            3. Do NOT include example usage.
            4. Do NOT include markdown, code fences, or markers of any kind.
            5. Do NOT define any additional functions or classes.
            6. Do NOT import any modules.
            7. Do NOT perform file I/O.
            8. Do NOT perform network or system calls.
            9. The function must be deterministic and side-effect free.
            10. The function MUST return a dictionary.
            11. Output ONLY raw Python code.

        SAFETY & SANDBOX RULES:
        - Use only core Python syntax and built-in types.
        - No imports of any kind.
        - No access to the file system.
        - No environment variables.
        - No reflection, exec, or eval.

        INPUT EXAMPLE (FOR REFERENCE ONLY):
        The CSV row may contain fields similar to the following:
        {}
        You are NOT required to use all fields in this example.

        FIELD DESCRIPTIONS (FOR REFERENCE ONLY):
        {}

        Your response MUST contain only the Python function code that satisfies all requirements above.
        '''
    
    def insert_test_input(self, test_input:str=None, tset_input_desc:str=None):
        self.SYSTEM_PROMPT = self.SYSTEM_PROMPT.format(test_input, tset_input_desc)


    def history_dict2lc(self, chat_history:List[dict]) -> List[Any]:

        conversation_history_lc = []
        for msg in chat_history:
            role = msg.get('role', 'human')
            content = msg.get('content', '')
            if role == "human":
                conversation_history_lc.append(HumanMessage(content=content))
            elif role == "ai":
                conversation_history_lc.append(AIMessage(content=content))
            elif role == "system":
                conversation_history_lc.append(SystemMessage(content=content))
            else:
                raise ValueError(f"Unknown role: {role}")
        return conversation_history_lc

 
    def generate_code_through_ai(self, prompt:List[tuple], chat_history:str) -> str:
        """
        Adapter that matches CodeGenerationAgent's expected signature:
            (prompt, conversation_history) -> code string
        Uses LangChain ChatOpenAI. Expects OPENAI_API_KEY in environment.
        """

        # load historical conversation
        lc_chat_history = self.history_dict2lc(chat_history)

        load_dotenv()  # loads .env
        api_key = os.getenv("OPENAI_API_KEY")

        # Initialize chat model (ensure OPENAI_API_KEY is set in env)
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
 
        messages = [("system", self.SYSTEM_PROMPT), 
                    MessagesPlaceholder("history")]
        messages += prompt

        prompt_template = ChatPromptTemplate.from_messages(messages)

        agent = prompt_template | llm
        print("<sys>: \nAI is working on it")
        agent_ouput = agent.invoke({'history': lc_chat_history})
        code_text = agent_ouput.content
        
        return code_text






