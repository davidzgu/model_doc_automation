from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
from typing import Any, Dict, List
from auto_code.utils import db_funcs


class code_generator:
    def __init__(self, session_id:str=None):
        # self.conversation_history = []
        # self.code_history = []
        self.db = db_funcs.SQLiteDB("src/auto_code/db_tables/code_generator.db")
        self.session_id = session_id
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

        INPUT STRUCTURE YOU RECEIVE:
        - `prompt`: the current instruction.
        - `chat_history`: previous conversation turns (summaries or user messages).

        Your response MUST contain only Python code that solves the current prompt.
        If the prompt contradicts the rules, produce the safest interpretation that follows all rules, still emitting valid Python code.
        """
    
    def reset_bot(self, test_name:str, description:str=None):
        self.db.initialize_schema()
        new_session_id = self.db.create_session(session_key=test_name, description=description)
        self.session_id = new_session_id
        return new_session_id

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



    # def _history_lc2dict(chat_history_lc:List[dict]) -> List[Any]:

    #     conversation_history = []
    #     for msg in chat_history_lc:
    #         role = msg.get('role', 'human')
    #         content = msg.get('content', '')
    #         if role == "human":
    #             conversation_history.append(HumanMessage(content=content))
    #         elif role == "ai":
    #             conversation_history_lc.append(AIMessage(content=content))
    #         elif role == "system":
    #             conversation_history_lc.append(SystemMessage(content=content))
    #         else:
    #             raise ValueError(f"Unknown role: {role}")

    #     return conversation_history


        
    def save_conversation(self, new_chats:List[Dict[str, str]]):
        """
        Call DB functions to insert newly happened chats
        
        :param self: Description
        :param new_chats: Description
        :type new_chats: List[Dict[str, str]]
        """
        print("Saving new conversations")
        for msg in new_chats:
            role = msg.get('role', 'human')
            content = msg.get('content', '')
            self.db.add_message(self.session_id, role, content)
        

    def load_conversation(self) -> List[Any]:
        """
        Docstring for load_conversation
        
        :param self: Description
        :return: Description
        :rtype: List[Any]
        """
        print("Loading historical conversations")
        chat_history = self.db.get_session_messages(self.session_id)
        return chat_history

    def generate_code_through_ai(self, prompt:str, code:str=None) -> str:
        """
        Adapter that matches CodeGenerationAgent's expected signature:
            (prompt, conversation_history) -> code string
        Uses LangChain ChatOpenAI. Expects OPENAI_API_KEY in environment.
        """


        # load historical conversation
        chat_history = self.load_conversation()
        lc_chat_history = self.history_dict2lc(chat_history)


        load_dotenv()  # loads .env
        api_key = os.getenv("OPENAI_API_KEY")

        # Initialize chat model (ensure OPENAI_API_KEY is set in env)
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)

        if code is None:
            messages = [("system", self.SYSTEM_PROMPT), 
                        MessagesPlaceholder("history"), 
                        ("human", "Prompt:\n{prompt}")]
        else:
            messages = [("system", self.SYSTEM_PROMPT),
                         MessagesPlaceholder("history"),
                        ("human", "User made editions to the previous code, and the new code from user is:\n{code}"),
                        ("human", "Prompt:\n{prompt}"),]

        prompt_template = ChatPromptTemplate.from_messages(messages)

        agent = prompt_template | llm

        agent_ouput = agent.invoke({'prompt': prompt, 'code':code, 'history': lc_chat_history})
        code_text = agent_ouput.content
        
        # save new conversation
        if code is None:
            new_messages = [{'role':'human', 'content': f'Prompt:\n{prompt}'},
                            {'role':'ai', 'content':code_text}]
        else:
            new_messages = [{'role':'human', 'content': f"User made editions to the previous code, and the new code from user is:\n{code}"},
                            {'role':'human', 'content': f"Prompt:\n{prompt}"},
                            {'role':'ai', 'content':code_text}]
        self.save_conversation(new_messages)

        return code_text

    def update_code_only(self, human_code:str):
        one_record = [{'role':'human', 'content': f"User made editions to the previous code, and the new code from user is:\n{human_code}"}]
        self.save_conversation(one_record)



    # def update_code(self, human_feedback:str=None, human_code:str=None):
    #     if (human_feedback is None) and (human_code is None):
    #         print("No operations.")
    #     elif (human_feedback is not None) and (human_code is None):
    #         # If user only provide some comments and let AI to modify the code
    #         new_code = self.generate_code_through_ai(prompt=human_feedback)
    #         return new_code
    #     elif (human_feedback is None) and (human_code is not None):
    #         # If user directly provide the new code
    #         new_code = human_code
    #         one_record = {'role':'human', 'content': f"The previous version of code is editted and the new code is: " + human_code}
    #         self.add_conversation(new_chats=one_record)
    #         return new_code
    #     else:
    #         # If user provide both comments and new code
    #         update_code_prompt = human_feedback + "The previous version of code is editted and the new code is: " + human_code
    #         new_code = self.generate_code_through_ai(prompt=update_code_prompt)
    #         return new_code






if __name__ == "__main__":    
    print("="*60)
    print("Self-Improving Code Generation with Docker Sandbox")
    print("="*60)

    test_generator = code_generator()
    for rnd in range(5):
        user_input = input("Please input:")
        code = test_generator.generate_code_through_ai(user_input)
        print(code)
        human_code = input("Please edit code:")
        if human_code == '':
            human_code = None
        else:
            code = test_generator.update_code(human_code)
        print(code)


    # # res = llm_chat_function("Write a function that compute the future options price using BSM model", [])
    # # print(res)
    
    # # Initialize sandbox
    # sandbox = DockerSandbox(
    #     timeout_seconds=10,
    #     memory_limit="256m"
    # )
    
    # # Initialize agent
    # agent = CodeGenerationAgent(sandbox, llm_chat_function)
    
    # # Example: Generate fibonacci function
    # task = "Calculate the nth Fibonacci number"
    # test_cases = [
    #     {'input': {'n': 0}, 'expected': 0},
    #     {'input': {'n': 1}, 'expected': 1},
    #     {'input': {'n': 5}, 'expected': 5},
    #     {'input': {'n': 10}, 'expected': 55}
    # ]
    
    # final_code, log = agent.generate_code_with_testing(
    #     task_description=task,
    #     test_cases=test_cases
    # )
    
    # if final_code:
    #     print("\n" + "="*60)
    #     print("FINAL WORKING CODE:")
    #     print("="*60)
    #     print(final_code)
    #     print("\nIteration Summary:")
    #     for entry in log:
    #         status = "✓ PASSED" if entry['all_passed'] else "✗ FAILED"
    #         print(f"  Iteration {entry['iteration']}: {status}")
    
    # # Clean up
    # print("\nCleaning up containers...")
    # containers = sandbox.client.containers.list(all=True, filters={'ancestor': sandbox.image})
    # for container in containers:
    #     try:
    #         container.remove(force=True)
    #     except:
    #         pass
    # print("Done!")