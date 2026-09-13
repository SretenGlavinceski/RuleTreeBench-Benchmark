import json
import time
import urllib.request


MODEL = "qwen3:4b-q4_K_M"

FILES = [
    "data/experiment_1_general.jsonl",
    "data/experiment_2_nearmatch.jsonl",
    "data/experiment_3_independent.jsonl",
    "data/experiment_3_linked.jsonl",
    "data/experiment_4_filtering.jsonl"
]


def load_items(filename, number):
    items = []

    with open(filename) as file:
        for line in file:
            items.append(json.loads(line))

            if len(items) == number:
                break

    return items


def ask_model(prompt):
    request_data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "seed": 42,
            "repeat_penalty": 1.0,
            "num_ctx": 8192,
            "num_predict": 4096
        }
    }

    data = json.dumps(request_data).encode("utf-8")

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


def check_response(result):
    content = result["message"].get("content") or ""
    thinking = result["message"].get("thinking") or ""

    prompt_tokens = result.get("prompt_eval_count") or 0
    output_tokens = result.get("eval_count") or 0

    problems = []

    if "<think>" in content or "</think>" in content:
        problems.append("THINK TAG FOUND")

    if thinking.strip():
        problems.append("THINKING FIELD NOT EMPTY")

    if prompt_tokens + output_tokens >= 8192:
        problems.append("CONTEXT LIMIT REACHED")

    return problems


items = []

# Take two prompts from every dataset.
for filename in FILES:
    items += load_items(filename, 2)


print("MODEL:", MODEL)
print("TOTAL SMOKE PROMPTS:", len(items))
print()


for i in range(len(items)):
    item = items[i]

    print("=" * 70)
    print(
        "PROMPT",
        i + 1,
        "/",
        len(items)
    )

    print("ID:", item["id"])
    print("GOLD:", item["gold_answer"])

    result, seconds = ask_model(
        item["prompt"]
    )

    response_text = result["message"]["content"]

    problems = check_response(result)

    print()
    print("RESPONSE:")
    print(response_text)

    print()
    print(
        "TIME:",
        round(seconds, 2),
        "seconds"
    )

    print(
        "PROMPT TOKENS:",
        result.get("prompt_eval_count")
    )

    print(
        "OUTPUT TOKENS:",
        result.get("eval_count")
    )

    print(
        "DONE REASON:",
        result.get("done_reason")
    )

    print(
        "THINKING FIELD:",
        repr(
            result["message"].get("thinking") or ""
        )
    )

    if problems:
        print(
            "INTEGRITY PROBLEMS:",
            problems
        )
    else:
        print(
            "INTEGRITY:",
            "OK"
        )

    print()