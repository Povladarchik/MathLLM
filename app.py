# app.py
from query_llm import YandexGPTApi
from flask import Flask, request, jsonify, render_template
import re
import time
import subprocess
import tempfile
import os
import json
import dotenv

dotenv.load_dotenv()


app = Flask(__name__)
folder_id = os.getenv("FOLDER_ID")
secret_key = os.getenv("SECRET_KEY")
model_uri = f"gpt://{folder_id}/llama/rc"
llm = YandexGPTApi(api_key=secret_key, model_uri=model_uri)


def parse_steps(text, patterns=None):
    if patterns is None:
        patterns = ["Шаг", "Step", "Part"]
    separator_pattern = "|".join(re.escape(p) for p in patterns)
    regex = f"({separator_pattern})\\s*\\d+"
    parts = re.split(regex, text)
    steps = [
        part.strip()
        for part in parts
        if part.strip() and not re.match(f"{separator_pattern}\\s*\\d*", part)
    ]
    return steps


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/parse-steps", methods=["POST"])
def parse_steps_route():
    data = request.get_json()
    solution_text = data.get("solution", "")
    separator_input = data.get("separator", "Шаг|Step|Part")

    try:
        custom_separators = [s.strip() for s in separator_input.split("|") if s.strip()]
        steps = parse_steps(solution_text, custom_separators)
        return jsonify({"success": True, "steps": steps})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/check-all-steps", methods=["POST"])
def check_all_steps():
    import uuid

    data = request.get_json()
    steps = data.get("steps", [])

    results = []
    total_duration = 0

    for i, step in enumerate(steps):
        start_time = time.time()

        # Формируем контекст из предыдущих трёх шагов (если они есть)
        context_steps = steps[max(0, i - 3) : i]  # Берём до трёх предыдущих шагов
        context = "\n".join(
            [f"Шаг {j + i - 3 + 1}: {steps[j]}" for j, step in enumerate(context_steps)]
        )

        system_text = """You are a math problem verification assistant. Your task is to generate precise and executable Python code to validate the correctness of each step in a mathematical solution.

        Your output must be a single Python script that:
        0. Includes necessary imports (e.g., sympy, numpy, math).
        1. Defines all variables used in this step or referenced from previous steps.
        2. Performs calculations, symbolic manipulations, or comparisons as described.
        3. Returns only 'True' if the current step is correct, and 'False' otherwise.
        4. Does not include any explanations — only ready-to-run code.

        If the step contains an expected result (e.g., "the answer should be x=2"), compare your computed result against it.
        If no expected value is given, use logical validation (e.g., equation simplification, domain constraints).

        Use SymPy for symbolic math when possible. Avoid floating point precision errors by using exact expressions or tolerance thresholds where appropriate."""

        user_text = f"""
        Previous steps:
        \"\"\"
        {context}
        \"\"\"

        Current step:
        \"\"\"
        {step}
        \"\"\"

        Write code to check the current step, taking into account the previous ones if needed.
        """

        code_snippet = llm.send_prompt(system_text=system_text, user_text=user_text)
        code_snippet = re.sub(
            r"^\s*```(?:python)?", "", code_snippet, flags=re.MULTILINE
        ).strip()

        # Сохраняем код во временный файл
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, f"code_{uuid.uuid4()}.py")
            with open(filename, "w") as f:
                f.write(code_snippet)

            # Выполняем код
            try:
                result = subprocess.run(
                    ["python3", filename],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5,  # Защита от бесконечных вычислений
                )

                stdout = result.stdout.strip()
                stderr = result.stderr.strip()

                # Анализируем вывод
                correct = "True" in stdout
                output = stdout + "\n" + stderr

            except subprocess.TimeoutExpired:
                correct = False
                output = "Error: Execution timed out"

        end_time = time.time()
        duration = round(end_time - start_time, 2)
        total_duration += duration

        error_type = None
        if not correct:
            if "Error: Execution timed out" in output:
                error_type = "Timeout/Error Execution"
            elif (
                "Traceback" in output
                or "SyntaxError" in output
                or "NameError" in output
            ):
                error_type = "Syntax/Language Error"
            elif "False" in output:
                error_type = "Calculation Error"
            else:
                error_type = "Logical Error"

        results.append(
            {
                "step": step,
                "code": code_snippet.strip(),
                "correct": correct,
                "output": output,
                "duration": duration,
                "error_type": error_type,
            }
        )

    results_summary = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_steps": len(steps),
            "total_duration": round(total_duration, 2),
        },
        "results": results,
    }

    # Save to JSON
    os.makedirs("results", exist_ok=True)
    timestamp = int(time.time())
    filename = f"results/results_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    return jsonify({"success": True, "results": results_summary})


if __name__ == "__main__":
    app.run(debug=True)
