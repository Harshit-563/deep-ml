import math
from collections import Counter


def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    total = len(labels)
    counts = Counter(labels)

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def calculate_information_gain(
    examples: list[dict], attr: str, target_attr: str
) -> float:
    """Calculate the information gain of splitting on attr."""
    total_labels = [ex[target_attr] for ex in examples]
    parent_entropy = calculate_entropy(total_labels)

    total = len(examples)
    weighted_entropy = 0.0

    attr_values = set(ex[attr] for ex in examples)

    for value in attr_values:
        subset = [ex for ex in examples if ex[attr] == value]
        subset_labels = [ex[target_attr] for ex in subset]

        weighted_entropy += (
            len(subset) / total
        ) * calculate_entropy(subset_labels)

    return parent_entropy - weighted_entropy


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    counts = Counter(ex[target_attr] for ex in examples)

    max_count = max(counts.values())
    candidates = [label for label, cnt in counts.items() if cnt == max_count]

    return sorted(candidates)[0]


def learn_decision_tree(
    examples: list[dict], attributes: list[str], target_attr: str
) -> dict:
    """Build a decision tree using the ID3 algorithm."""

    labels = [ex[target_attr] for ex in examples]

    # All examples have same class
    if len(set(labels)) == 1:
        return labels[0]

    # No attributes left
    if not attributes:
        return majority_class(examples, target_attr)

    # Choose attribute with maximum information gain
    best_attr = max(
        attributes,
        key=lambda attr: calculate_information_gain(
            examples, attr, target_attr
        ),
    )

    tree = {best_attr: {}}

    remaining_attrs = [a for a in attributes if a != best_attr]

    for value in sorted(set(ex[best_attr] for ex in examples)):
        subset = [ex for ex in examples if ex[best_attr] == value]

        if not subset:
            tree[best_attr][value] = majority_class(
                examples, target_attr
            )
        else:
            tree[best_attr][value] = learn_decision_tree(
                subset, remaining_attrs, target_attr
            )

    return tree