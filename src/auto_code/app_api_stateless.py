# ...existing code...
from auto_code.utils import db_funcs
from auto_code.code_bot import code_generator
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd
import json
from auto_code.docker_sandbox import ExecutionResult, ExecutionStatus, DockerSandbox


def init_session(
    db: db_funcs.SQLiteDB,
    code_bot: code_generator,
    test_name: str,
    test_data_path: Optional[str] = None,
    test_description: Optional[str] = None,
    test_data_description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create or fetch a session and ensure test data is registered and test example is inserted
    Returns a context dict that is stateless and can be passed to other functions.
    """
    existing_session = db.get_session_from_key(session_key=test_name)
    if existing_session is None:
        new_session_id = db.create_session(session_key=test_name, description=test_description)
        if test_data_path:
            db.set_test_data_location(new_session_id, test_name, test_data_path, test_data_description)
            # pass test example to code_bot
            df = pd.read_csv(test_data_path)
            first = df.to_dict("records")[0]
            example = ", ".join(f"{k}={v}" for k, v in first.items())
            code_bot.insert_test_input(example, test_data_description)
        session_id = new_session_id
    else:
        session_id = existing_session["id"]
        # pass test example to code_bot (from registered test data)
        file_info = db.get_test_data_locations(session_id)
        file_path = file_info.get("file_path")
        if file_path:
            df = pd.read_csv(file_path)
            first = df.to_dict("records")[0]
            example = ", ".join(f"{k}={v}" for k, v in first.items())
            code_bot.insert_test_input(example, test_data_description)

    base_dir = Path(__file__).resolve().parent
    code_name = f"{test_name}_code.py"
    return {"session_id": session_id, "base_dir": base_dir, "code_name": code_name}


def save_conversation(db: db_funcs.SQLiteDB, session_id: int, new_chats: List[Dict[str, str]]):
    for msg in new_chats:
        role = msg.get("role", "human")
        content = msg.get("content", "")
        db.add_message(session_id, role, content)


def get_chat_history(db: db_funcs.SQLiteDB, session_id: int) -> List[Dict[str, str]]:
    return db.get_session_messages(session_id)


def upload_test_data(
    db: db_funcs.SQLiteDB,
    code_bot: code_generator,
    session_id: int,
    test_name: str,
    test_file_path: str,
    description: Optional[str] = None,
):
    db.set_test_data_location(session_id, test_name, test_file_path, description)
    # pass example to bot
    df = pd.read_csv(test_file_path)
    first = df.to_dict("records")[0]
    example = ", ".join(f"{k}={v}" for k, v in first.items())
    code_bot.insert_test_input(example, description)


def chat_with_bot(
    db: db_funcs.SQLiteDB,
    code_bot: code_generator,
    session_id: int,
    code_name: str,
    human_message: str,
    human_code: Optional[str] = None,
) -> str:
    chat_history = db.get_session_messages(session_id)
    message_to_bot: List[tuple] = []
    if human_code is None:
        message_to_bot.append(("human", human_message))
    else:
        message_to_bot.append(("human", f"User made editions to the previous code, and the new code from user is:\n{human_code}"))
        message_to_bot.append(("human", human_message))

    messages_to_db = [{"role": m[0], "content": m[1]} for m in message_to_bot]
    save_conversation(db, session_id, messages_to_db)

    ai_code = code_bot.generate_code_through_ai(message_to_bot, chat_history)

    save_conversation(db, session_id, [{"role": "ai", "content": ai_code}])
    db.add_code_entry(session_id, code_name, ai_code)
    return ai_code


def update_code(db: db_funcs.SQLiteDB, session_id: int, code_name: str, human_code: str):
    save_conversation(db, session_id, [{"role": "human", "content": f"User made editions to the previous code, and the new code from user is:\n{human_code}"}])
    db.add_code_entry(session_id, code_name, human_code)


def fetch_test_data(db: db_funcs.SQLiteDB, session_id: int) -> List[Dict[str, Any]]:
    file_info = db.get_test_data_locations(session_id)
    test_data_file_path = file_info["file_path"]
    df = pd.read_csv(test_data_file_path)
    return df.to_dict("records")


def execute_code(
    db: db_funcs.SQLiteDB,
    sandbox: DockerSandbox,
    session_id: int,
    code_name: str,
    base_dir: Path,
) -> List[Dict[str, Any]]:
    current_code_info = db.get_current_version_code(session_id, code_name)
    current_code = current_code_info["content"]
    code_path = base_dir / "code_storage" / code_name
    code_path.parent.mkdir(parents=True, exist_ok=True)
    code_path.write_text(current_code, encoding="utf-8")

    test_inputs = fetch_test_data(db, session_id)

    test_results: List[Dict[str, Any]] = []
    all_passed = True

    for i, one_input in enumerate(test_inputs):
        if isinstance(one_input, dict):
            result: ExecutionResult = sandbox.test_code(current_code, **one_input)
        else:
            result = sandbox.test_code(current_code, *one_input)

        passed = result.status == ExecutionStatus.SUCCESS
        test_results.append({
            "test_case": i + 1,
            "input": one_input,
            "output": result.output,
            "passed": passed,
            "sandbox_summary_str": result.get_feedback()
        })

        if not passed:
            all_passed = False

    if all_passed:
        sandbox_message = [{"role": "human", "content": "Code execution successfully in the Sandbox."}]
    else:
        test_feedback = "The code failed the tests. Here's what went wrong:\n\n"
        for r in test_results:
            if not r["passed"]:
                test_feedback += f"Test Case {r['test_case']}:\n"
                test_feedback += f"  Input: {r['input']}\n"
                test_feedback += f"  Output: {r.get('output', 'N/A')}\n"
                test_feedback += f"  Sandbox Feedback: {r['sandbox_summary_str']}\n\n"
        test_feedback += "Please fix the code to pass all test cases."
        sandbox_message = [{"role": "human", "content": test_feedback}]

    save_conversation(db, session_id, sandbox_message)
    db.record_code_execution(session_id, code_name, json.dumps(test_results))
    return test_results


def submit_code(
    db: db_funcs.SQLiteDB,
    session_id: int,
    code_name: str,
    base_dir: Path,
):
    current_code_info = db.get_current_version_code(session_id, code_name)
    current_code = current_code_info["content"]
    current_version = current_code_info["version"]
    db.record_code_approval(session_id, code_name, current_version)
    code_path = base_dir / "../bsm_multi_agents/tools" / code_name
    code_path.parent.mkdir(parents=True, exist_ok=True)
    code_path.write_text(current_code, encoding="utf-8")