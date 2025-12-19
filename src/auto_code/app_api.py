from auto_code.utils import db_funcs
from auto_code.code_bot import code_generator
from typing import Any, Dict, List
from pathlib import Path
import pandas as pd
from auto_code.docker_sandbox import ExecutionResult, ExecutionStatus, DockerSandbox

class ChatSession:
    def __init__(self,
                 db:db_funcs.SQLiteDB, 
                 code_bot:code_generator, 
                 sandbox:DockerSandbox, 
                 test_name:str,
                 test_data_path:str=None,
                 test_description:str=None,
                 test_data_description:str=None):
        self.db = db
        self.code_bot = code_bot
        self.sandbox = sandbox
        self.test_name = test_name
        # Set up session in db
        print("<sys>: Fetching the existing session")
        existing_session = self.db.get_session_from_key(session_key=test_name)
        if existing_session is None:
            print("<sys>: No session found")
            new_session_id = self.db.create_session(session_key=test_name, description=test_description)
            self.session_id = new_session_id
            print(f"<sys>: Session ID: {self.session_id}")
            print("<sys>: Pass test case example to AI prompt")
            print(test_data_path)
            self.upload_test_data(test_data_path, test_data_description)
        else:
            print("<sys>: \n", existing_session)
            self.session_id = existing_session['id']
            # Pass the test case exmple to AI
            print("<sys>: Pass test case example to AI prompt")
            test_data = self.fetch_test_data()
            first_test_data = test_data[0]
            test_example_str = ", ".join(f"{k}={v}" for k, v in first_test_data.items())
            self.code_bot.insert_test_input(test_example_str, test_data_description)
            
        # Path info for code
        self.BASE_DIR = Path(__file__).resolve().parent
        self.code_name = f"{self.test_name}_code.py"
        
    def get_session_id(self) -> int:
        return self.session_id
    
    def save_conversation(self, new_chats:List[Dict[str, str]]):
        """
        Call DB functions to insert newly happened chats

        :param self: Description
        :param new_chats: Description
        :type new_chats: List[Dict[str, str]]
        """
        print("<sys>: Saving new conversations")
        for msg in new_chats:
            role = msg.get('role', 'human')
            content = msg.get('content', '')
            self.db.add_message(self.session_id, role, content)
        
    
    def print_chat_history(self):
        print("<sys>: \nLoading historical conversations")
        history_dict = self.db.get_session_messages(self.session_id)
        if len(history_dict) == 0:
            print("<sys>: \nNo history found")
        for msg in history_dict:
            role = msg.get('role', 'human')
            content = msg.get('content', '')
            print(f"<{role}>: \n{content}\n")


    def upload_test_data(self, test_file_path:str, description:str=None):
        # register the test data into db
        print("<sys>: Uploading test cases")
        self.db.set_test_data_location(self.session_id, self.test_name, test_file_path, description)
        # Pass the test case exmple to AI
        print("<sys>: Pass test case example to AI prompt")
        test_data = self.fetch_test_data()
        first_test_data = test_data[0]
        test_example_str = ", ".join(f"{k}={v}" for k, v in first_test_data.items())
        self.code_bot.insert_test_input(test_example_str, description)
        pass


    def chat_with_bot(self, human_message:str, human_code:str=None):
        # load history
        chat_history = self.db.get_session_messages(self.session_id)
        # clean up user input
        message_to_bot = []
        if human_code is None:
            message_to_bot.append(("human", human_message))
        else:
            message_to_bot.append(("human", f"User made editions to the previous code, and the new code from user is:\n{human_code}"))
            message_to_bot.append(("human", human_message))
        # record user input
        messages_to_db = [{'role':msg[0], 'content': msg[1]} for msg in message_to_bot]
        self.save_conversation(messages_to_db)
        # Talk to AI
        ai_code = self.code_bot.generate_code_through_ai(message_to_bot, chat_history)
        # recrods AI output into chat history
        self.save_conversation([{'role':'ai', 'content':ai_code}])
        # recrods AI code into code repo
        print("<sys>: Saving Code")
        self.db.add_code_entry(self.session_id, self.code_name, ai_code)
        return ai_code
    
    def update_code(self, human_code:str):
        one_record = [{'role':'human', 'content': f"User made editions to the previous code, and the new code from user is:\n{human_code}"}]
        self.save_conversation(one_record)
        self.db.add_code_entry(self.session_id, self.code_name, human_code)

    
    def fetch_test_data(self):
        test_data_file_info = self.db.get_test_data_locations(self.session_id)
        test_data_file_path = test_data_file_info['file_path']
        test_data_df = pd.read_csv(test_data_file_path)
        test_inputs = test_data_df.to_dict("records")
        return test_inputs

    def exectue_code(self):
        # Save code into local
        current_code_info = self.db.get_last_code_version(self.session_id, self.code_name)
        current_code = current_code_info['content']
        code_path = self.BASE_DIR/"code_storage"/self.code_name
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(current_code, encoding="utf-8")
        code_path_str = str(code_path)
        print("<sys>: \nFound test code, saved at: ", code_path)

        # Get test inputs
        test_inputs = self.fetch_test_data()
        print("<sys>: \nFound test data: ", test_inputs)

        # Run each test cases and get results
        test_results = []
        all_passed = True
        for i, one_input in enumerate(test_inputs):
            print('-'*60)
            print(f"<sys>: \nTesting case {i+1}: {one_input}")
            
            # Use sandbox as a tool to test the code
            if isinstance(one_input, dict):
                result = self.sandbox.test_code(current_code, **one_input)
            else:
                result = self.sandbox.test_code(current_code, *one_input)
            
            test_results.append({
                'test_case': i + 1,
                'input': one_input,
                'output': result.output,
                'passed': result.status == ExecutionStatus.SUCCESS,
                'sandbox_summary_str': result.get_feedback()
            })

            if result.status != ExecutionStatus.SUCCESS:
                    all_passed = False

            print(f"<sandboox>: \n{result.get_feedback()}")
                
        # return test_results

        # records results into conversation history and code history
        # Records conversation
        if all_passed:
            sandbox_message = [{'role':'human', 'content':'Code exectution sucessfully in the Sandbox.'}]
        else:
            test_feedback = "The code failed the tests. Here's what went wrong:\n\n"
            for result in test_results:
                if not result['passed']:
                    test_feedback += f"Test Case {result['test_case']}:\n"
                    test_feedback += f"  Input: {result['input']}\n"
                    test_feedback += f"  Output: {result.get('output', 'N/A')}\n"
                    test_feedback += f"  Sanbox Feedback: {result['sandbox_summary_str']}\n\n"
            test_feedback += "Please fix the code to pass all test cases."
            sandbox_message = [{'role':'human', 'content':test_feedback}]
        self.save_conversation(sandbox_message)
        # Records output into code history


    def submit_code(self):
        print("Submitting the Code")
        pass




if __name__ == "__main__":
    from prompt_toolkit import prompt


    # test_data = pd.read_csv(r"D:\ML_Experiment\model_doc_automation\src\auto_code\test_data_storage\dummy_options_greeks_results.csv")
    # print(test_data)

    print("="*60)
    print("Self-Improving Code Generation with Docker Sandbox")
    print("="*60)

    test_name = "greeks_test"#input("test name: ")
    desc = "some"#input("test descriptions: ")
    test_file_path=r"D:\ML_Experiment\model_doc_automation\src\auto_code\test_data_storage\dummy_options_greeks_results.csv" 
    test_data_description="The file contains the market data of options, BSM prices and greeks are provided by the pricing model."
    # Initialize
    db_connection = db_funcs.SQLiteDB("src/auto_code/db_tables/code_generator.db")
    one_bot = code_generator()
    sandbox = DockerSandbox(preinstall_packages=['scipy'])
    user_session = ChatSession(db_connection, one_bot, sandbox, test_name, test_file_path, desc, test_data_description)
    # sid = user_session.get_session_id()
    user_session.print_chat_history()


    while True:
        user_input = input("<Human>:")
        human_code = code = prompt("Edit code (Ctrl-D to finish or Esc + Enter for Windows):\n", multiline=True)
        if human_code == '':
            human_code = None
        code = user_session.chat_with_bot(user_input, human_code)
        print(f"<AI>: \n{code}")
        is_execute = input("Execute [Yes]/[No]?")
        if is_execute == 'Yes':
            user_session.exectue_code()
        else:
            continue
        approve = input(f"<sys>: \n Submit the last version of code [Yes]/[No]?")
        if approve == 'Yes':
            user_session.submit_code()
            break
