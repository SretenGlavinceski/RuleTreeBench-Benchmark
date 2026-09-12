import random

from src.generator_exp2 import (
    generate_near_matches,
    generate_exp2_family,
    generate_exp2_variants
)

from src.generator_exp1 import (
    generate_query,
    generate_winner
)

from src.solver import (
    applies,
    failed_conditions,
    specificity,
    solve
)


def test_four_near_matches_are_generated():
    random.seed(42)

    query = generate_query()
    winner = generate_winner(query)

    near_matches = generate_near_matches(query, winner)

    assert len(near_matches) == 4


def test_near_matches_have_one_error():
    random.seed(42)

    query = generate_query()
    winner = generate_winner(query)

    near_matches = generate_near_matches(query, winner)

    for rule in near_matches:
        assert specificity(rule) == 4
        assert applies(rule, query) == False
        assert failed_conditions(rule, query) == 1


def test_near_matches_are_different():
    random.seed(42)

    query = generate_query()
    winner = generate_winner(query)

    near_matches = generate_near_matches(query, winner)

    for i in range(len(near_matches)):
        for j in range(i + 1, len(near_matches)):
            assert near_matches[i].conditions != near_matches[j].conditions


def test_exp2_family_has_ten_rules():
    random.seed(42)

    family = generate_exp2_family()

    assert len(family["rules"]) == 10


def test_exp2_has_four_levels():
    random.seed(42)

    family = generate_exp2_family()
    variants = generate_exp2_variants(family)

    assert 0 in variants
    assert 1 in variants
    assert 2 in variants
    assert 4 in variants


def test_every_exp2_variant_has_ten_rules():
    random.seed(42)

    family = generate_exp2_family()
    variants = generate_exp2_variants(family)

    for level in variants:
        assert len(variants[level]) == 10


def test_near_match_count_per_level():
    random.seed(42)

    family = generate_exp2_family()
    variants = generate_exp2_variants(family)

    query = family["query"]

    expected = {
        0: 0,
        1: 1,
        2: 2,
        4: 4
    }

    for level in [0, 1, 2, 4]:
        near_match_count = 0

        for rule in variants[level]:
            if (
                specificity(rule) == 4
                and failed_conditions(rule, query) == 1
            ):
                near_match_count += 1

        assert near_match_count == expected[level]


def test_winner_is_unique_at_every_level():
    random.seed(42)

    family = generate_exp2_family()
    variants = generate_exp2_variants(family)

    query = family["query"]

    for level in [0, 1, 2, 4]:
        result = solve(query, variants[level])

        assert len(result) == 1
        assert result[0].label == family["winner"].label


def test_rule_labels_stay_in_same_positions():
    random.seed(42)

    family = generate_exp2_family()
    variants = generate_exp2_variants(family)

    labels_n0 = []
    labels_n1 = []
    labels_n2 = []
    labels_n4 = []

    for rule in variants[0]:
        labels_n0.append(rule.label)

    for rule in variants[1]:
        labels_n1.append(rule.label)

    for rule in variants[2]:
        labels_n2.append(rule.label)

    for rule in variants[4]:
        labels_n4.append(rule.label)

    assert labels_n0 == labels_n1
    assert labels_n0 == labels_n2
    assert labels_n0 == labels_n4


def test_exp2_variants_are_reproducible():
    random.seed(42)

    family = generate_exp2_family()

    first = generate_exp2_variants(family)
    second = generate_exp2_variants(family)

    for level in [0, 1, 2, 4]:
        for i in range(10):
            assert first[level][i].label == second[level][i].label
            assert first[level][i].conditions == second[level][i].conditions

def test_first_near_match_stays_across_levels():
    random.seed(42)

    family = generate_exp2_family()
    variants = generate_exp2_variants(family)

    query = family["query"]

    n1_near_match = None

    for rule in variants[1]:
        if failed_conditions(rule, query) == 1:
            n1_near_match = rule
            break

    assert n1_near_match is not None

    found_in_n2 = False
    found_in_n4 = False

    for rule in variants[2]:
        if (
            rule.label == n1_near_match.label
            and rule.conditions == n1_near_match.conditions
        ):
            found_in_n2 = True

    for rule in variants[4]:
        if (
            rule.label == n1_near_match.label
            and rule.conditions == n1_near_match.conditions
        ):
            found_in_n4 = True

    assert found_in_n2 == True
    assert found_in_n4 == True
