from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

DATASET_PATH = HERE / "generated" / "rulearena_airline_300.jsonl"
RESULTS_ROOT = REPO_ROOT / "results" / "external" / "rulearena_airline"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

NUM_CTX = 8192
NUM_PREDICT = 4096
TEMPERATURE = 0
SEED = 42
REPEAT_PENALTY = 1.0
KEEP_ALIVE = "30m"
REQUEST_TIMEOUT_SECONDS = 3600

EXPECTED_DATASET_SHA256 = (
    "e4cae947a8fad80eb47cada63805a5a0144a394ea08d6030ba4acc346673779e"
)

ENGINEERING_TEST_IDS = [
    "rulearena_airline_L1_000",
    "rulearena_airline_L2_000",
    "rulearena_airline_L3_000",
]

MODELS = {
    "qwen3_4b": {
        "display_name": "Qwen3 4B",
        "ollama_model": "qwen3:4b-q4_K_M",
        "expected_id_prefix": "2bfd38a7daaf",
        "supports_think_option": True,
    },
    "qwen3_8b": {
        "display_name": "Qwen3 8B",
        "ollama_model": "qwen3:8b-q4_K_M",
        "expected_id_prefix": "500a1f067a9f",
        "supports_think_option": True,
    },
    "gemma3_4b": {
        "display_name": "Gemma 3 4B",
        "ollama_model": "gemma3:4b",
        "expected_id_prefix": "a2af6cc3eb7f",
        "supports_think_option": False,
    },
    "gemma4_12b": {
        "display_name": "Gemma 4 12B",
        "ollama_model": "gemma4:12b-it-q4_K_M",
        "expected_id_prefix": "4eb23ef187e2",
        "supports_think_option": True,
    },
}

FINAL_COST_RE = re.compile(
    r"The\s+total\s+cost\s+is\s+\$"
    r"\s*([0-9][0-9,]*(?:\.[0-9]+)?)"
    r"\s*\.?\s*$",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc
            rows.append(row)
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        json.dump(row, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


def post_json(
    endpoint: str,
    payload: dict[str, Any],
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    url = f"{OLLAMA_BASE_URL}{endpoint}"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama HTTP error {exc.code}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running."
        ) from exc

    return json.loads(raw)


def get_json(endpoint: str, timeout: int = 30) -> dict[str, Any]:
    url = f"{OLLAMA_BASE_URL}{endpoint}"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_BASE_URL}."
        ) from exc
    return json.loads(raw)


def normalize_digest(value: str) -> str:
    value = value.strip()
    if value.startswith("sha256:"):
        value = value[len("sha256:") :]
    return value


def verify_model(model_config: dict[str, Any]) -> str:
    tags = get_json("/api/tags")
    target_name = model_config["ollama_model"]
    expected_prefix = model_config["expected_id_prefix"]

    for model in tags.get("models", []):
        model_name = model.get("name") or model.get("model") or ""
        if model_name != target_name:
            continue
        digest = normalize_digest(str(model.get("digest", "")))
        if not digest.startswith(expected_prefix):
            raise RuntimeError(
                f"Model {target_name} is installed, but its digest is {digest!r}; "
                f"expected prefix {expected_prefix!r}."
            )
        return digest

    raise RuntimeError(f"Required model is not installed: {target_name}")


def get_ollama_version() -> str:
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    combined = result.stdout.strip() or result.stderr.strip()
    return combined or "unknown"


def get_git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def get_git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def normalize_response_for_parser(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = cleaned.replace("`", "")
    return cleaned.strip()


def parse_final_cost(raw_response: str) -> tuple[str, int | float | None]:
    cleaned = normalize_response_for_parser(raw_response)
    match = FINAL_COST_RE.search(cleaned)
    if match is None:
        return "parse_failure", None

    number_text = match.group(1).replace(",", "")
    try:
        value = Decimal(number_text)
    except InvalidOperation:
        return "parse_failure", None

    if value == value.to_integral_value():
        return "parsed", int(value)
    return "parsed", float(value)


def costs_equal(
    parsed_cost: int | float | None,
    gold_cost: int | float,
) -> bool:
    if parsed_cost is None:
        return False
    try:
        return Decimal(str(parsed_cost)) == Decimal(str(gold_cost))
    except InvalidOperation:
        return False


def build_chat_payload(
    example: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_config["ollama_model"],
        "messages": [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user", "content": example["user_prompt"]},
        ],
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "repeat_penalty": REPEAT_PENALTY,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
    }

    if model_config["supports_think_option"]:
        payload["think"] = False

    return payload


def select_examples(
    rows: list[dict[str, Any]],
    engineering_test: bool,
) -> list[dict[str, Any]]:
    if not engineering_test:
        return rows

    by_id = {row["id"]: row for row in rows}
    missing = [example_id for example_id in ENGINEERING_TEST_IDS if example_id not in by_id]
    if missing:
        raise RuntimeError(
            "Engineering-test IDs missing from dataset: " + ", ".join(missing)
        )
    return [by_id[example_id] for example_id in ENGINEERING_TEST_IDS]


def result_directory(model_key: str, engineering_test: bool) -> Path:
    if engineering_test:
        return RESULTS_ROOT / "_engineering" / model_key
    return RESULTS_ROOT / model_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen RuleArena Airline external evaluation through Ollama."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--engineering-test",
        action="store_true",
        help="Run only L1_000, L2_000, and L3_000.",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Run all 300 frozen Airline examples.",
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODELS),
        help="Model configuration to run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing response file before running.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip IDs already present in an existing response file.",
    )

    args = parser.parse_args()
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume cannot be used together")
    return args


def main() -> None:
    args = parse_args()

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Frozen dataset not found: {DATASET_PATH}")

    dataset_sha = sha256_file(DATASET_PATH)
    if dataset_sha != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "Frozen dataset SHA256 mismatch.\n"
            f"Expected: {EXPECTED_DATASET_SHA256}\n"
            f"Found:    {dataset_sha}"
        )

    all_rows = load_jsonl(DATASET_PATH)
    if len(all_rows) != 300:
        raise RuntimeError(
            f"Expected exactly 300 frozen examples, found {len(all_rows)}."
        )

    examples = select_examples(all_rows, engineering_test=args.engineering_test)
    model_config = MODELS[args.model]
    installed_digest = verify_model(model_config)

    output_dir = result_directory(args.model, engineering_test=args.engineering_test)
    responses_path = output_dir / "responses.jsonl"
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        if responses_path.exists():
            responses_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    existing_ids: set[str] = set()
    if responses_path.exists():
        if not args.resume:
            raise RuntimeError(
                f"{responses_path} already exists. Use --resume to continue it or "
                "--overwrite to replace it."
            )
        existing_rows = load_jsonl(responses_path)
        existing_ids = {row["id"] for row in existing_rows}

    run_started_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "benchmark": "RuleArena",
        "domain": "airline",
        "run_type": "engineering_test" if args.engineering_test else "official_full",
        "model_key": args.model,
        "model_display_name": model_config["display_name"],
        "ollama_model": model_config["ollama_model"],
        "expected_model_id_prefix": model_config["expected_id_prefix"],
        "installed_model_digest": installed_digest,
        "ollama_version": get_ollama_version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_head": get_git_head(),
        "git_worktree_dirty": get_git_dirty(),
        "dataset_path": str(DATASET_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "dataset_sha256": dataset_sha,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "num_examples_selected": len(examples),
        "selected_ids": [row["id"] for row in examples],
        "inference_settings": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "repeat_penalty": REPEAT_PENALTY,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "think": False if model_config["supports_think_option"] else "not_requested",
        },
        "started_at_utc": run_started_at,
    }
    write_json(manifest_path, manifest)

    completed_this_run = 0

    for position, example in enumerate(examples, start=1):
        example_id = example["id"]
        if example_id in existing_ids:
            print(f"[{position}/{len(examples)}] SKIP {example_id}")
            continue

        print(f"[{position}/{len(examples)}] RUN  {example_id}")
        payload = build_chat_payload(example, model_config)
        response = post_json("/api/chat", payload)

        message = response.get("message", {})
        raw_response = str(message.get("content", ""))
        parse_status, parsed_cost = parse_final_cost(raw_response)
        gold_cost = example["gold_total_cost"]
        correct = costs_equal(parsed_cost, gold_cost)

        prompt_eval_count = response.get("prompt_eval_count")
        eval_count = response.get("eval_count")
        token_total = None
        if isinstance(prompt_eval_count, int) and isinstance(eval_count, int):
            token_total = prompt_eval_count + eval_count

        result_row = {
            "id": example_id,
            "difficulty_level": example["difficulty_level"],
            "source_complexity": example["source_complexity"],
            "source_index": example["source_index"],
            "gold_total_cost": gold_cost,
            "raw_response": raw_response,
            "parse_status": parse_status,
            "parsed_total_cost": parsed_cost,
            "correct": correct,
            "done": response.get("done"),
            "done_reason": response.get("done_reason"),
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "prompt_plus_output_tokens": token_total,
            "total_duration": response.get("total_duration"),
            "load_duration": response.get("load_duration"),
            "prompt_eval_duration": response.get("prompt_eval_duration"),
            "eval_duration": response.get("eval_duration"),
            "created_at": response.get("created_at"),
            "ollama_model": response.get("model"),
            "message_role": message.get("role"),
        }

        append_jsonl(responses_path, result_row)
        completed_this_run += 1

        print(
            "    "
            f"gold={gold_cost} "
            f"parsed={parsed_cost} "
            f"correct={correct} "
            f"done_reason={result_row['done_reason']} "
            f"prompt_tokens={prompt_eval_count} "
            f"output_tokens={eval_count} "
            f"total_tokens={token_total}"
        )

    final_rows = load_jsonl(responses_path) if responses_path.exists() else []
    manifest["completed_this_run"] = completed_this_run
    manifest["responses_in_file"] = len(final_rows)
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    if responses_path.exists():
        manifest["responses_sha256"] = sha256_file(responses_path)
    write_json(manifest_path, manifest)

    print()
    print("Run finished.")
    print(f"Responses: {responses_path}")
    print(f"Manifest:  {manifest_path}")
    print(f"Rows:      {len(final_rows)}")


if __name__ == "__main__":
    main()
