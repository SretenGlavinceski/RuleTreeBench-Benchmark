import random

from src.models import Query, Rule
from src.generator_exp1 import (
    ATTRIBUTES,
    VALUES,
    generate_query,
    generate_winner,
    get_different_value,
    generate_unique_invalid_rule,
    rule_exists,
    copy_rule
)


def generate_near_matches(query, winner):
    near_matches = []

    winner_attributes = list(winner.conditions.keys())

    for attribute in winner_attributes:
        conditions = winner.conditions.copy()

        conditions[attribute] = get_different_value(
            query.values[attribute]
        )

        rule = Rule("NEAR_MATCH", conditions)
        near_matches.append(rule)

    return near_matches

def generate_exp2_family():
    query = generate_query()
    winner = generate_winner(query)
    near_matches = generate_near_matches(query, winner)

    random.shuffle(near_matches)

    rules = []
    rules.append(winner)

    replaceable_rules = []

    for i in range(4):
        rule = generate_unique_invalid_rule(
            query,
            4,
            rules
        )

        replaceable_rules.append(rule)
        rules.append(rule)

    additional_spec4_rules = []

    for i in range(3):
        rule = generate_unique_invalid_rule(
            query,
            4,
            rules
        )

        additional_spec4_rules.append(rule)
        rules.append(rule)

    rule = generate_unique_invalid_rule(
        query,
        2,
        rules
    )
    rules.append(rule)

    rule = generate_unique_invalid_rule(
        query,
        3,
        rules
    )
    rules.append(rule)

    random.shuffle(rules)

    for i in range(len(rules)):
        rules[i].label = "R" + str(i + 1)

    return {
        "query": query,
        "winner": winner,
        "near_matches": near_matches,
        "replaceable_rules": replaceable_rules,
        "rules": rules
    }

def generate_exp2_variants(family):
    levels = [0, 1, 2, 4]

    near_matches = family["near_matches"]
    replaceable_rules = family["replaceable_rules"]
    base_rules = family["rules"]

    variants = {}

    for level in levels:
        variant_rules = []

        for base_rule in base_rules:
            replacement_index = -1

            for i in range(len(replaceable_rules)):
                if base_rule is replaceable_rules[i]:
                    replacement_index = i
                    break

            if replacement_index != -1 and replacement_index < level:
                near_match = near_matches[replacement_index]

                replacement = Rule(
                    base_rule.label,
                    near_match.conditions.copy()
                )

                variant_rules.append(replacement)

            else:
                variant_rules.append(copy_rule(base_rule))

        variants[level] = variant_rules

    return variants
