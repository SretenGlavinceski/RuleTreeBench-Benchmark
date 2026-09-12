import random

from src.generator_exp1 import generate_exp1_family, generate_exp1_variants
from src.render import render_rule_task
from src.solver import solve


random.seed(42)

family = generate_exp1_family()
variants = generate_exp1_variants(family)

query = family["query"]

for level in [0, 1, 2, 4]:
    rules = variants[level]

    print()
    print("=" * 70)
    print("EXPERIMENT 1 - L" + str(level))
    print("=" * 70)
    print()

    print(render_rule_task(query, rules))

    result = solve(query, rules)

    print()
    print("GOLD ANSWER:", result[0].label)
