from auto_code.utils import db_funcs
from auto_code.code_bot import code_generator
from typing import Any, Dict, List

class ChatSession:
    def __init__(self, db:db_funcs.SQLiteDB, code_bot, sandbox, session_id:int=None):
        self.db = db
        self.code_bot = code_bot
        self.sandbox = sandbox
        self.session_id = session_id
    
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
        
    
    def print_chat_history(self):
        print("<sys>: \nLoading historical conversations")
        history_dict = self.db.get_session_messages(self.session_id)
        if len(history_dict) == 0:
            print("<sys>: \nNo history found")
        for msg in history_dict:
            role = msg.get('role', 'human')
            content = msg.get('content', '')
            print(f"{role}: \n{content}\n")

    def upload_test_data(session_id:int, test_name:str, target_path:str, description:str=None):
        # register the test data into db
        # return test cases
        db = db_funcs.SQLiteDB("src/auto_code/db_tables/code_generator.db")
        db.set_test_data_location(session_id, test_name, target_path, description)
        pass

    def fetch_test_data(session_id:int):
        pass


    def initialize_session(self, test_name:str, description:str=None) -> int:
        print("Fetching the existing session")
        existing_session = self.db.get_session_from_key(session_key=test_name)
        if existing_session is None:
            print("No session found")
            new_session_id = self.db.create_session(session_key=test_name, description=description)
            self.session_id = new_session_id
            self.code_bot.reset_bot(new_session_id)
            print(f"Session ID: {self.session_id}")
        else:
            print(existing_session)
            self.session_id = existing_session['id']
        return self.session_id


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
        # recrods AI output
        self.save_conversation([{'role':'ai', 'content':ai_code}])


        return ai_code
    
    def update_code(self, human_code:str):
        one_record = [{'role':'human', 'content': f"User made editions to the previous code, and the new code from user is:\n{human_code}"}]
        self.save_conversation(one_record)

    # def update_code(session_id:int, human_code:str):
    #     # update the code and record in db
    #     # update the chat history in db
    #     code_bot = code_generator(session_id)
    #     code_bot.update_code_only(human_code)

    def exectue_code():
        pass

    def submit_code():
        pass




if __name__ == "__main__":
    print("="*60)
    print("Self-Improving Code Generation with Docker Sandbox")
    print("="*60)

    test_name = "gamma_test"#input("test name: ")
    desc = "some"#input("test descriptions: ")
    db_connection = db_funcs.SQLiteDB("src/auto_code/db_tables/code_generator.db")
    one_bot = code_generator(db_connection)
    user_session = ChatSession(db_connection, one_bot, None, None)
    sid = user_session.initialize_session(test_name, desc)
    user_session.print_chat_history()
    for rnd in range(5):

        user_input = input("Human:")
        human_code = input("Please edit ai code:")
        if human_code == '':
            human_code = None
        code = user_session.chat_with_bot(user_input, human_code)
        print(f"AI: \n{code}")
