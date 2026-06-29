import json
import random


def arithmetic(n_terms):
    start = random.randint(1, 50)
    diff = random.randint(1, 20)
    seq = [start + i * diff for i in range(n_terms + 1)]
    return seq[:-1], seq[-1]


def geometric(n_terms):
    start = random.randint(1, 10)
    ratio = random.randint(2, 5)
    seq = [start * (ratio**i) for i in range(n_terms + 1)]
    return seq[:-1], seq[-1]


def squares(n_terms):
    start = random.randint(1, 10)
    seq = [(start + i) ** 2 for i in range(n_terms + 1)]
    return seq[:-1], seq[-1]


def cubes(n_terms):
    start = random.randint(1, 5)
    seq = [(start + i) ** 3 for i in range(n_terms + 1)]
    return seq[:-1], seq[-1]


def fibonacci(n_terms):
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    seq = [a, b]
    for _ in range(n_terms - 1):
        seq.append(seq[-1] + seq[-2])
    return seq[:-1], seq[-1]


def mixed(n_terms):
    start = random.randint(1, 20)
    seq = [start]
    for i in range(n_terms):
        if i % 2 == 0:
            seq.append(seq[-1] * 2)  # even steps → multiply by 2
        else:
            seq.append(seq[-1] + 3)  # odd steps  → add 3
    return seq[:-1], seq[-1]


def polynomial(n_terms):
    start = random.randint(1, 5)
    seq = [(start + i) ** 2 + (start + i) for i in range(n_terms + 1)]
    return seq[:-1], seq[-1]


def generate_dataset(n_samples=100000, n_terms=5):
    rules = [
        arithmetic,
        geometric,
        squares,
        cubes,
        fibonacci,
        mixed,
        polynomial,
    ]
    dataset = []

    for _ in range(n_samples):
        # pick a random rule each iteration
        rule_fn = random.choice(rules)

        inputs, target = rule_fn(n_terms)

        # skip if any number exceeds 1000
        if max(inputs) >= 1000 or target >= 1000:
            continue

        dataset.append(
            {
                "input": inputs,
                "target": target,
            }
        )

    return dataset


if __name__ == "__main__":
    print("Generating dataset...")
    data = generate_dataset(n_samples=100000, n_terms=5)
    print(f"Generated {len(data)} samples")

    # save to a json file
    with open("data/dataset.json", "w") as f:
        json.dump(data, f)

    print("Saved to data/dataset.json")

    # print 5 examples so we can verify
    print("\nSample examples:")
    for i in range(5):
        d = data[i]
        print(f"  input: {d['input']}  →  target: {d['target']}")
