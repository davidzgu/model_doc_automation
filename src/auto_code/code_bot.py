from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
from typing import Any, Dict, List
from auto_code.utils import db_funcs


class code_generator:
    def __init__(self):
        self.SYSTEM_PROMPT="""
        You are a code-generation engine.

        OBJECTIVE:
        Generate correct, executable Python code that satisfies the user request.

        OUTPUT RULES:
        1. Output ONLY raw Python code. 
        2. Do NOT include explanation, comments, example usage, or markdown fences.
        3. Do NOT wrap the result with ```python or any markers.
        4. The code must be self-contained and runnable as-is.

        SAFETY & SANDBOX RULES:
        - Only standard Python. Do NOT import system-level modules such as os, sys, subprocess, shlex, pathlib, socket.
        - Avoid file system operations unless explicitly required.
        - No network calls.
        - No shell commands.
        - Output one pure function and deterministic behavior.

        CODE RULES:
        - Include the input types for each function parameter if possible.
        - Add the Docstring to describe what this function is doing.
        - You MUST generate code that includes ALL of the following parameters. None of them are optional.

        Required parameters example:
        {}

        Parameters Description:
        {}

        
        Your response MUST contain only Python code that solves the current prompt.
        If the prompt contradicts the rules, produce the safest interpretation that follows all rules, still emitting valid Python code.
        """
    
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






