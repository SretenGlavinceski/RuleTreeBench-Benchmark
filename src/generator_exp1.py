import random

from src.models import Query, Rule


ATTRIBUTES = ["a1", "a2", "a3", "a4", "a5", "a6"]
VALUES = ["v1", "v2", "v3"]


def generate_query():
    values = {}

    for attribute in ATTRIBUTES:
        values[attribute] = random.choice(VALUES)

    return Query(values)


def generate_winner(query):
    selected_attributes = random.sample(ATTRIBUTES, 4)

    conditions = {}

    for attribute in selected_attributes:
        conditions[attribute] = query.values[attribute]

    return Rule("WINNER", conditions)


def generate_general_rules(winner):
    general_rules = []

    winner_attributes = list(winner.conditions.keys())

    for removed_attribute in winner_attributes:
        conditions = {}

        for attribute in winner_attributes:
            if attribute != removed_attribute:
                conditions[attribute] = winner.conditions[attribute]

        rule = Rule("GENERAL", conditions)
        general_rules.append(rule)

    return general_rules

def get_different_value(current_value):
    possible_values = []

    for value in VALUES:
        if value != current_value:
            possible_values.append(value)

    return random.choice(possible_values)


def generate_invalid_rule(query, rule_specificity):
    selected_attributes = random.sample(ATTRIBUTES, rule_specificity)

    conditions = {}

    # First two conditions will deliberately be wrong
    wrong_attributes = selected_attributes[:2]

    for attribute in selected_attributes:
        if attribute in wrong_attributes:
            conditions[attribute] = get_different_value(
                query.values[attribute]
            )
        else:
            conditions[attribute] = query.values[attribute]

    return Rule("DISTRACTOR", conditions)

def rules_are_same(rule1, rule2):
    return rule1.conditions == rule2.conditions


def rule_exists(rule, rules):
    for existing_rule in rules:
        if rules_are_same(rule, existing_rule):
            return True

    return False


def generate_unique_invalid_rule(query, rule_specificity, existing_rules):
    while True:
        rule = generate_invalid_rule(query, rule_specificity)

        if not rule_exists(rule, existing_rules):
            return rule

def generate_exp1_family():
    query = generate_query()
    winner = generate_winner(query)
    general_rules = generate_general_rules(winner)
    random.shuffle(general_rules)

    rules = []
    rules.append(winner)

    replaceable_rules = []

    for i in range(4):
        rule = generate_unique_invalid_rule(
            query,
            3,
            rules
        )

        replaceable_rules.append(rule)
        rules.append(rule)

    invalid_spec4_rules = []

    for i in range(3):
        rule = generate_unique_invalid_rule(
            query,
            4,
            rules
        )

        invalid_spec4_rules.append(rule)
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
        "general_rules": general_rules,
        "replaceable_rules": replaceable_rules,
        "rules": rules
    }

def copy_rule(rule):
    new_conditions = {}

    for attribute in rule.conditions:
        new_conditions[attribute] = rule.conditions[attribute]

    return Rule(rule.label, new_conditions)

def generate_exp1_variants(family):
    levels = [0, 1, 2, 4]

    general_rules = family["general_rules"]
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
                general_rule = general_rules[replacement_index]

                replacement = Rule(
                    base_rule.label,
                    general_rule.conditions.copy()
                )

                variant_rules.append(replacement)

            else:
                variant_rules.append(copy_rule(base_rule))

        variants[level] = variant_rules

    return variants

