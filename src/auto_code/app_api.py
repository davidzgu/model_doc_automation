import pandas as pd
from auto_code.utils import db_funcs
from auto_code.code_bot import code_generator


def upload_test_data(session_id:int, test_name:str, target_path:str, description:str=None):
    # register the test data into db
    # return test cases
    db = db_funcs.SQLiteDB("src/auto_code/db_tables/code_generator.db")
    db.set_test_data_location(session_id, test_name, target_path, description)
    pass

def fetch_test_data(session_id:int):
    pass


def initialize_session(test_name:str, description:str=None) -> int:
    db = db_funcs.SQLiteDB("src/auto_code/db_tables/code_generator.db")
    existing_session = db.get_session(session_key=test_name)
    print(existing_session)
    if existing_session is None:
        code_bot = code_generator()
        session_id = code_bot.reset_bot(test_name, description)
    else:
        session_id = existing_session['id']
    return session_id

def print_chat_history(session_id:int):
    code_bot = code_generator(session_id)
    history_dict = code_bot.load_conversation()
    for msg in history_dict:
        role = msg.get('role', 'human')
        content = msg.get('content', '')
        print(f"{role}: \n{content}\n")


def chat_with_bot(session_id:int, human_message:str, human_code:str=None):
    # get bot's response based on human message, human code and hisotry chat
    code_bot = code_generator(session_id)
    ai_code = code_bot.generate_code_through_ai(human_message, human_code)
    return ai_code

def update_code(session_id:int, human_code:str):
    # update the code and record in db
    # update the chat history in db
    code_bot = code_generator(session_id)
    code_bot.update_code_only(human_code)

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
    sid = initialize_session(test_name, desc)
    for rnd in range(5):
        print_chat_history(sid)
        user_input = input("Human:")
        human_code = input("Please edit ai code:")
        if human_code == '':
            human_code = None
        else:
            code = update_code(sid, human_code)
            print(code)
        ai_code = chat_with_bot(sid, user_input, human_code)