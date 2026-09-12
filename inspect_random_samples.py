import json
import random


def load_jsonl(filename):
    items = []

    with open(filename) as file:
        for line in file:
            items.append(json.loads(line))

    return items


def print_samples(name, items, number_of_samples):
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    samples = random.sample(items, number_of_samples)

    for item in samples:
        print()
        print("-" * 80)
        print("ID:", item["id"])
        print("GOLD:", item["gold_answer"])
        print("-" * 80)
        print(item["prompt"])


random.seed(123)


exp1 = load_jsonl(
    "data/experiment_1_general.jsonl"
)

exp2 = load_jsonl(
    "data/experiment_2_nearmatch.jsonl"
)

exp3_independent = load_jsonl(
    "data/experiment_3_independent.jsonl"
)

exp3_linked = load_jsonl(
    "data/experiment_3_linked.jsonl"
)

exp4 = load_jsonl(
    "data/experiment_4_filtering.jsonl"
)


print_samples(
    "EXPERIMENT 1",
    exp1,
    6
)

print_samples(
    "EXPERIMENT 2",
    exp2,
    6
)

print_samples(
    "EXPERIMENT 3 - INDEPENDENT",
    exp3_independent,
    4
)

print_samples(
    "EXPERIMENT 3 - LINKED",
    exp3_linked,
    4
)

print_samples(
    "EXPERIMENT 4",
    exp4,
    4
)
