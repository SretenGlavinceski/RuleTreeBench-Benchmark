import random

from src.generator_exp2 import generate_exp2_family, generate_exp2_variants
from src.render import render_rule_task
from src.solver import solve, failed_conditions


random.seed(42)

family = generate_exp2_family()
variants = generate_exp2_variants(family)

query = family["query"]

for level in [0, 1, 2, 4]:
    rules = variants[level]

    print()
    print("=" * 70)
    print("EXPERIMENT 2 - N" + str(level))
    print("=" * 70)
    print()

    print(render_rule_task(query, rules))

    result = solve(query, rules)

    print()
    print("GOLD ANSWER:", result[0].label)

    print()
    print("NEAR MATCHES PRESENT:")

    for rule in rules:
        if failed_conditions(rule, query) == 1:
            print(rule.label, rule.conditions)
