import json

from src.models import Query, Rule
from src.render import render_applicability_task
from src.solver import applies, failed_conditions, specificity


def dict_to_query(query_data):
    return Query(query_data)


def dict_to_rule(rule_data):
    return Rule(
        rule_data["label"],
        rule_data["conditions"]
    )


def rule_to_dict(rule):
    return {
        "label": rule.label,
        "conditions": rule.conditions
    }


def build_exp3_linked_dataset():
    dataset = []

    with open("data/generated/experiment_2_nearmatch.jsonl") as file:
        for line in file:
            item = json.loads(line)

            # We only want the N1 item from each family.
            if item["level"] != 1:
                continue

            query = dict_to_query(item["query"])

            # Find the exact winner.
            winner_rule = None

            for rule_data in item["rules"]:
                if rule_data["label"] == item["gold_answer"]:
                    winner_rule = dict_to_rule(rule_data)
                    break

            if winner_rule is None:
                raise Exception(
                    "Could not find winner for " + item["id"]
                )

            # Reuse the exact N1 near match.
            near_match_rule = dict_to_rule(
                item["first_near_match"]
            )

            winner_item = {
                "id": item["family_id"] + "_linked_winner",
                "experiment": 3,
                "part": "linked",
                "probe_type": "winner",
                "source_family_id": item["family_id"],
                "source_exp2_id": item["id"],
                "query": item["query"],
                "rule": rule_to_dict(winner_rule),
                "specificity": specificity(winner_rule),
                "failed_conditions": failed_conditions(
                    winner_rule,
                    query
                ),
                "gold_answer": "YES",
                "prompt": render_applicability_task(
                    query,
                    winner_rule
                )
            }

            dataset.append(winner_item)

            near_match_item = {
                "id": item["family_id"] + "_linked_near_match",
                "experiment": 3,
                "part": "linked",
                "probe_type": "near_match",
                "source_family_id": item["family_id"],
                "source_exp2_id": item["id"],
                "query": item["query"],
                "rule": rule_to_dict(near_match_rule),
                "specificity": specificity(near_match_rule),
                "failed_conditions": failed_conditions(
                    near_match_rule,
                    query
                ),
                "gold_answer": "NO",
                "prompt": render_applicability_task(
                    query,
                    near_match_rule
                )
            }

            dataset.append(near_match_item)

    return dataset


def save_dataset(dataset, filename):
    with open(filename, "w") as file:
        for item in dataset:
            file.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    dataset = build_exp3_linked_dataset()

    save_dataset(
        dataset,
        "data/generated/experiment_3_linked.jsonl"
    )

    print(
        "Saved",
        len(dataset),
        "Experiment 3 linked prompts"
    )
