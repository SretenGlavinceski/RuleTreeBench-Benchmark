import json
import time
import urllib.request


MODEL = "gemma3:4b"

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
            "num_ctx": 4096,
            "num_predict": 1024
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

    with urllib.request.urlopen(request) as response:
        result = json.loads(
            response.read().decode("utf-8")
        )

    end_time = time.time()

    return result, end_time - start_time


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
        "TOKENS:",
        result.get("eval_count")
    )

    print(
        "DONE REASON:",
        result.get("done_reason")
    )

    print()
