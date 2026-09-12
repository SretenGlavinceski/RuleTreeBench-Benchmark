from src.models import Query, Rule
from src.solver import applies, failed_conditions, specificity, solve


def make_query():
    return Query({
        "a1": "v1",
        "a2": "v2",
        "a3": "v3",
        "a4": "v1",
        "a5": "v2",
        "a6": "v3"
    })


def test_single_winner():
    query = make_query()

    winner = Rule("R1", {
        "a1": "v1",
        "a2": "v2",
        "a3": "v3",
        "a4": "v1"
    })

    distractor = Rule("R2", {
        "a1": "v2",
        "a2": "v2",
        "a3": "v1",
        "a4": "v1"
    })

    result = solve(query, [winner, distractor])

    assert len(result) == 1
    assert result[0].label == "R1"


def test_near_match_does_not_apply():
    query = make_query()

    near_match = Rule("R1", {
        "a1": "v1",
        "a2": "v2",
        "a3": "v3",
        "a4": "v2"
    })

    assert applies(near_match, query) == False
    assert failed_conditions(near_match, query) == 1


def test_two_wrong_conditions():
    query = make_query()

    rule = Rule("R1", {
        "a1": "v2",
        "a2": "v2",
        "a3": "v1",
        "a4": "v1"
    })

    assert applies(rule, query) == False
    assert failed_conditions(rule, query) == 2


def test_specificity():
    rule = Rule("R1", {
        "a1": "v1",
        "a2": "v2",
        "a3": "v3"
    })

    assert specificity(rule) == 3


def test_more_specific_rule_wins():
    query = make_query()

    general_rule = Rule("R1", {
        "a1": "v1",
        "a2": "v2",
        "a3": "v3"
    })

    specific_rule = Rule("R2", {
        "a1": "v1",
        "a2": "v2",
        "a3": "v3",
        "a4": "v1"
    })

    result = solve(query, [general_rule, specific_rule])

    assert len(result) == 1
    assert result[0].label == "R2"


def test_all_four_general_rules_lose_to_winner():
    query = make_query()

    winner = Rule("R1", {
        "a1": "v1",
        "a2": "v2",
        "a3": "v3",
        "a4": "v1"
    })

    general1 = Rule("R2", {
        "a1": "v1",
        "a2": "v2",
        "a3": "v3"
    })

    general2 = Rule("R3", {
        "a1": "v1",
        "a2": "v2",
        "a4": "v1"
    })

    general3 = Rule("R4", {
        "a1": "v1",
        "a3": "v3",
        "a4": "v1"
    })

    general4 = Rule("R5", {
        "a2": "v2",
        "a3": "v3",
        "a4": "v1"
    })

    rules = [winner, general1, general2, general3, general4]

    result = solve(query, rules)

    assert len(result) == 1
    assert result[0].label == "R1"


def test_no_rule_applies():
    query = make_query()

    rule1 = Rule("R1", {
        "a1": "v2",
        "a2": "v3"
    })

    rule2 = Rule("R2", {
        "a3": "v1",
        "a4": "v2"
    })

    result = solve(query, [rule1, rule2])

    assert len(result) == 0


def test_tie_returns_both_rules():
    query = make_query()

    rule1 = Rule("R1", {
        "a1": "v1",
        "a2": "v2",
        "a3": "v3",
        "a4": "v1"
    })

    rule2 = Rule("R2", {
        "a1": "v1",
        "a2": "v2",
        "a3": "v3",
        "a5": "v2"
    })

    result = solve(query, [rule1, rule2])

    assert len(result) == 2
