import sys
import json
import os
import platform
import subprocess
import time
import urllib.request

import yaml

from src.parser import (
    parse_rule_answer,
    parse_yes_no_answer
)

from datetime import datetime, timezone

MODEL_NAME = "gemma4_12b"
OLLAMA_MODEL = "gemma4:12b-it-q4_K_M"
SUPPORTS_THINK = False

DATASET_FILES = [
    "data/generated/experiment_1_general.jsonl",
    "data/generated/experiment_2_nearmatch.jsonl",
    "data/generated/experiment_3_independent.jsonl",
    "data/generated/experiment_3_linked.jsonl",
    "data/generated/experiment_4_filtering.jsonl"
]

def load_model_entry(model_name):
    with open("config/models.yaml") as file:
        entries = yaml.safe_load(file)["models"]

    for entry in entries:
        if entry["name"] == model_name:
            return entry

    raise SystemExit(
        f"Unknown model name: {model_name}"
    )

def load_config(model_entry):
    with open("config/inference.yaml") as file:
        config = yaml.safe_load(file)

    for key in ("num_ctx", "num_predict"):
        if key in model_entry:
            config[key] = model_entry[key]

    return config


def load_dataset():
    items = []

    for filename in DATASET_FILES:
        with open(filename) as file:
            for line in file:
                items.append(json.loads(line))

    return items


def load_completed_ids(filename):
    completed_ids = set()

    if not os.path.exists(filename):
        return completed_ids

    with open(filename) as file:
        for line in file:
            if line.strip() == "":
                continue

            item = json.loads(line)
            completed_ids.add(item["id"])

    return completed_ids


def get_command_output(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

        return result.stdout.strip()

    except Exception:
        return "unknown"


def get_git_commit():
    return get_command_output(
        ["git", "rev-parse", "HEAD"]
    )


def get_ollama_version():
    return get_command_output(
        ["ollama", "--version"]
    )


def get_model_id():
    output = get_command_output(
        ["ollama", "list"]
    )

    for line in output.splitlines():
        parts = line.split()

        if (
            len(parts) >= 2
            and parts[0] == OLLAMA_MODEL
        ):
            return parts[1]

    return "unknown"


def verify_model_digest(expected_digest):
    actual = get_model_id()

    if not actual.startswith(expected_digest):
        raise SystemExit(
            f"DIGEST MISMATCH for {OLLAMA_MODEL}: "
            f"expected {expected_digest}, "
            f"got {actual}. "
            "Wrong model pulled. Aborting."
        )

    return actual


def create_manifest(config, result_directory, actual_digest):
    manifest_file = os.path.join(
        result_directory,
        "run_manifest.json"
    )

    if os.path.exists(manifest_file):
        return

    manifest = {
        "model_name": MODEL_NAME,
        "ollama_model": OLLAMA_MODEL,
        "model_id": get_model_id(),
        "model_digest": actual_digest,
        "supports_think": SUPPORTS_THINK,
        "ollama_version": get_ollama_version(),
        "dataset_freeze_commit": "652d697fceeb2d9b3132fd48efc2e108a640b793",
        "code_commit": get_git_commit(),
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "inference_config": config
    }

    with open(manifest_file, "w") as file:
        json.dump(
            manifest,
            file,
            indent=4
        )


def ask_model(prompt, config):
    request_data = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": config["stream"],
        "keep_alive": config["keep_alive"],
        "options": {
            "temperature": config["temperature"],
            "seed": config["seed"],
            "repeat_penalty": config["repeat_penalty"],
            "num_ctx": config["num_ctx"],
            "num_predict": config["num_predict"]
        }
    }

    if SUPPORTS_THINK:
        request_data["think"] = config["thinking"]

    data = json.dumps(
        request_data
    ).encode("utf-8")

    request = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=data,
        headers={
            "Content-Type": "application/json"
        }
    )

    start_time = time.time()

    with urllib.request.urlopen(
        request,
        timeout=3600
    ) as response:
        result = json.loads(
            response.read().decode("utf-8")
        )

    end_time = time.time()

    return result, end_time - start_time


def ask_model_with_retry(prompt, config):
    attempts = 3

    for attempt in range(1, attempts + 1):
        try:
            return ask_model(
                prompt,
                config
            )

        except Exception as error:
            print(
                "Request failed:",
                error
            )

            if attempt == attempts:
                raise

            print(
                "Retrying in 10 seconds..."
            )

            time.sleep(10)


def parse_answer(item, response_text):
    if item["experiment"] == 3:
        return parse_yes_no_answer(
            response_text
        )

    return parse_rule_answer(
        response_text
    )


def check_response_integrity(result):
    message = result["message"]

    content = message.get("content") or ""
    thinking = message.get("thinking") or ""

    problems = []

    if (
        "<think>" in content
        or "</think>" in content
    ):
        problems.append(
            "reasoning-trace tag found in content"
        )

    if thinking.strip():
        problems.append(
            "message.thinking is non-empty"
        )

    if problems:
        raise SystemExit(
            "PROTOCOL VIOLATION — run aborted: "
            + "; ".join(problems)
        )

def main():
    global MODEL_NAME
    global OLLAMA_MODEL
    global SUPPORTS_THINK

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m src.run_model <model_name>"
        )

    model_entry = load_model_entry(
        sys.argv[1]
    )

    MODEL_NAME = model_entry["name"]
    OLLAMA_MODEL = model_entry["ollama_model"]
    SUPPORTS_THINK = model_entry["supports_think"]

    actual_digest = verify_model_digest(
        model_entry["expected_digest"]
    )

    config = load_config(model_entry)
    dataset = load_dataset()

    assert len(dataset) == 1700

    result_directory = os.path.join(
        "results",
        "raw",
        MODEL_NAME
    )

    os.makedirs(
        result_directory,
        exist_ok=True
    )

    response_file = os.path.join(
        result_directory,
        "responses.jsonl"
    )

    create_manifest(
        config,
        result_directory,
        actual_digest
    )

    completed_ids = load_completed_ids(
        response_file
    )

    print("MODEL:", OLLAMA_MODEL)
    print("TOTAL DATASET ITEMS:", len(dataset))
    print("ALREADY COMPLETED:", len(completed_ids))
    print(
        "REMAINING:",
        len(dataset) - len(completed_ids)
    )
    print()

    session_times = []
    session_completed = 0

    with open(
        response_file,
        "a"
    ) as output_file:

        for index in range(len(dataset)):
            item = dataset[index]

            if item["id"] in completed_ids:
                continue

            print("=" * 70)
            print(
                "ITEM:",
                index + 1,
                "/",
                len(dataset)
            )

            print("ID:", item["id"])
            print("GOLD:", item["gold_answer"])

            result, seconds = ask_model_with_retry(
                item["prompt"],
                config
            )
      
            check_response_integrity(result)

            response_text = result[
                "message"
            ][
                "content"
            ]

            parsed_answer, fallback_used = parse_answer(
                item,
                response_text
            )

            correct = (
                parsed_answer
                == item["gold_answer"]
            )

            done_reason = result.get(
                "done_reason"
            )

            truncated = (
                done_reason == "length"
            )

            unparsable = (
                parsed_answer is None
            )

            result_item = {
                "id": item["id"],
                "experiment": item["experiment"],
                "model_name": MODEL_NAME,
                "ollama_model": OLLAMA_MODEL,
                "thinking_field": (
                    result["message"].get("thinking") or ""
                ),
                "model_digest": actual_digest,
                "gold_answer": item["gold_answer"],
                "parsed_answer": parsed_answer,
                "correct": correct,
                "unparsable": unparsable,
                "fallback_used": fallback_used,
                "truncated": truncated,
                "done_reason": done_reason,
                "raw_response": response_text,

                "prompt_eval_count": result.get(
                    "prompt_eval_count"
                ),

                "eval_count": result.get(
                    "eval_count"
                ),

                "total_duration": result.get(
                    "total_duration"
                ),

                "load_duration": result.get(
                    "load_duration"
                ),

                "prompt_eval_duration": result.get(
                    "prompt_eval_duration"
                ),

                "eval_duration": result.get(
                    "eval_duration"
                ),

                "wall_time_seconds": round(
                    seconds,
                    2
                )  
            }

            output_file.write(
                json.dumps(result_item)
                + "\n"
            )

            output_file.flush()
            os.fsync(
                output_file.fileno()
            )

            session_completed += 1
            session_times.append(seconds)

            average_time = (
                sum(session_times)
                / len(session_times)
            )

            remaining_items = (
                len(dataset)
                - len(completed_ids)
                - session_completed
            )

            estimated_hours = (
                remaining_items
                * average_time
                / 3600
            )

            print(
                "ANSWER:",
                parsed_answer
            )

            print(
                "CORRECT:",
                correct
            )

            print(
                "TIME:",
                round(seconds, 2),
                "seconds"
            )

            print(
                "TOKENS:",
                result.get("eval_count")
            )

            print(
                "DONE:",
                done_reason
            )

            print(
                "SESSION COMPLETED:",
                session_completed
            )

            print(
                "ESTIMATED REMAINING:",
                round(
                    estimated_hours,
                    1
                ),
                "hours"
            )

            print()

if __name__ == "__main__":
    main()
