import json
import re
import time
import subprocess
import tempfile
import os
from typing import List, Dict, Any, Tuple
from query_llm import YandexGPTApi
import dotenv

dotenv.load_dotenv()

folder_id = os.getenv("FOLDER_ID")
secret_key = os.getenv("SECRET_KEY")
model_uri = f"gpt://{folder_id}/llama/rc"
llm = YandexGPTApi(api_key=secret_key, model_uri=model_uri)


def parse_steps(text: str, patterns=None) -> List[str]:
    if patterns is None:
        patterns = ["Step", "Шаг", "Part"]
    separator_pattern = "|".join(re.escape(p) for p in patterns)
    regex = f"({separator_pattern})\\s*\\d+"
    parts = re.split(regex, text)
    return [
        part.strip()
        for part in parts
        if part.strip() and not re.match(f"{separator_pattern}\\s*\\d*", part)
    ]


def verify_step(step_text: str, context: str) -> Tuple[bool, float, str, str]:
    start_time = time.time()

    system_prompt = """You are a math problem verification assistant. Your task is to generate precise and executable Python code to validate the correctness of each step in a mathematical solution.

    Your output must be a single Python script that:
    0. Includes necessary imports (e.g., sympy, numpy, math).
    1. Defines all variables used in this step or referenced from previous steps.
    2. Performs calculations, symbolic manipulations, or comparisons as described.
    3. Returns only 'True' if the current step is correct, and 'False' otherwise.
    4. Does not include any explanations — only ready-to-run code.

    If the step contains an expected result (e.g., "the answer should be x=2"), compare your computed result against it.
    If no expected value is given, use logical validation (e.g., equation simplification, domain constraints).

    Use SymPy for symbolic math when possible. Avoid floating point precision errors by using exact expressions or tolerance thresholds where appropriate."""

    user_prompt = f"""
    Previous steps:
    \"\"\"
    {context}
    \"\"\"

    Current step:
    \"\"\"
    {step_text}
    \"\"\"

    Write code to check the current step, taking into account the previous ones if needed.
    """

    try:
        code_snippet = llm.send_prompt(system_text=system_prompt, user_text=user_prompt)
        code_snippet = re.sub(r"^\s*```(?:python)?", "", code_snippet, flags=re.MULTILINE).strip()
    except Exception as e:
        return False, 0, str(e), "Language Model Error"

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
        tmpfile.write(code_snippet.encode())
        tmpfile_path = tmpfile.name

    try:
        result = subprocess.run(
            ["python3", tmpfile_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        correct = "True" in stdout
        output = stdout + "\n" + stderr
    except subprocess.TimeoutExpired:
        correct = False
        output = "Error: Execution timed out"
    finally:
        os.remove(tmpfile_path)

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    error_type = None
    if not correct:
        if "Error: Execution timed out" in output:
            error_type = "Timeout/Error Execution"
        elif "Traceback" in output or "SyntaxError" in output or "NameError" in output:
            error_type = "Syntax/Language Error"
        elif "False" in output:
            error_type = "Calculation Error"
        else:
            error_type = "Logical Error"

    return correct, duration, output, error_type


def extract_final_answer(problem_content: str) -> bool:
    return "\\boxed{" in problem_content


def process_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
    problem_id = problem["problem_id"]
    content = problem["content"]

    steps = parse_steps(content)
    results = []
    total_duration = 0

    for i, step in enumerate(steps):
        context = "\n".join(steps[max(0, i - 3):i])
        correct, duration, output, error_type = verify_step(step, context)

        results.append({
            "step": step,
            "correct": correct,
            "output": output,
            "duration": duration,
            "error_type": error_type
        })

        total_duration += duration

    final_answer_correct = extract_final_answer(content)

    return {
        "problem_id": problem_id,
        "total_steps": len(steps),
        "total_duration": round(total_duration, 2),
        "steps": results,
        "final_answer_correct": final_answer_correct
    }


def compute_statistics(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = {
        "total_problems": len(all_results),
        "total_steps": 0,
        "correct_steps": 0,
        "incorrect_steps": 0,
        "total_duration": 0,
        "false_negatives_by_type": {
            "Timeout/Error Execution": 0,
            "Syntax/Language Error": 0,
            "Calculation Error": 0,
            "Logical Error": 0,
        },
        "final_answer_correct_count": 0
    }

    for result in all_results:
        stats["total_steps"] += result["total_steps"]
        stats["total_duration"] += result["total_duration"]
        for step in result["steps"]:
            if step["correct"]:
                stats["correct_steps"] += 1
            else:
                stats["incorrect_steps"] += 1
                et = step["error_type"]
                stats["false_negatives_by_type"][et] = stats["false_negatives_by_type"].get(et, 0) + 1
        if result["final_answer_correct"]:
            stats["final_answer_correct_count"] += 1

    stats["average_duration"] = round(stats["total_duration"] / stats["total_steps"], 2) if stats["total_steps"] else 0
    stats["step_level_accuracy"] = round(stats["correct_steps"] / stats["total_steps"], 4) if stats["total_steps"] else 0
    stats["number_of_correct_final_steps"] = stats["final_answer_correct_count"]

    return stats


def main(input_file: str, output_dir: str = "results"):
    print(f"Reading input from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        problems = json.load(f)

    print(f"Processing {len(problems)} problems...")
    all_results = []

    for problem in problems:
        print(f"Processing problem: {problem['problem_id']}")
        result = process_problem(problem)
        all_results.append(result)

    print("Computing statistics...")
    stats = compute_statistics(all_results)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    result_file = os.path.join(output_dir, f"results_{timestamp}.json")

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input_file": input_file
            },
            "statistics": stats,
            "problems": all_results
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {result_file}")


if __name__ == "__main__":
    INPUT_JSON_FILE = "test_40_questions.json" # Synthetic data
    main(INPUT_JSON_FILE)
