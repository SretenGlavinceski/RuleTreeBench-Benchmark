from src.models import Query, Rule


def applies(rule, query):
    for attribute in rule.conditions:
        expected_value = rule.conditions[attribute]
        actual_value = query.values[attribute]

        if expected_value != actual_value:
            return False

    return True


def failed_conditions(rule, query):
    failed = 0

    for attribute in rule.conditions:
        expected_value = rule.conditions[attribute]
        actual_value = query.values[attribute]

        if expected_value != actual_value:
            failed += 1

    return failed


def specificity(rule):
    return len(rule.conditions)


def solve(query, rules):
    applicable_rules = []

    for rule in rules:
        if applies(rule, query):
            applicable_rules.append(rule)

    if len(applicable_rules) == 0:
        return []

    highest_specificity = 0

    for rule in applicable_rules:
        rule_specificity = specificity(rule)

        if rule_specificity > highest_specificity:
            highest_specificity = rule_specificity

    winners = []

    for rule in applicable_rules:
        if specificity(rule) == highest_specificity:
            winners.append(rule)

    return winners
