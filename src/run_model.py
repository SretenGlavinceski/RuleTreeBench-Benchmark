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

MODEL_NAME = "gemma3_4b"
OLLAMA_MODEL = "gemma3:4b"

DATASET_FILES = [
    "data/experiment_1_general.jsonl",
    "data/experiment_2_nearmatch.jsonl",
    "data/experiment_3_independent.jsonl",
    "data/experiment_3_linked.jsonl",
    "data/experiment_4_filtering.jsonl"
]


def load_config():
    with open("config/inference.yaml") as file:
        return yaml.safe_load(file)


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
        if line.startswith(OLLAMA_MODEL):
            parts = line.split()

            if len(parts) >= 2:
                return parts[1]

    return "unknown"


def create_manifest(config, result_directory):
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
        "think": config["thinking"],
        "keep_alive": config["keep_alive"],
        "options": {
            "temperature": config["temperature"],
            "seed": config["seed"],
            "repeat_penalty": config["repeat_penalty"],
            "num_ctx": config["num_ctx"],
            "num_predict": config["num_predict"]
        }
    }

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


def main():
    config = load_config()
    dataset = load_dataset()

    assert len(dataset) == 1700

    result_directory = os.path.join(
        "results",
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
        result_directory
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

        for index in range(3):
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
