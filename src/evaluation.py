
import math


def precision_at_k(recommended, relevant, k):
    if k <= 0:
        raise ValueError("k must be positive")

    top_k = recommended[:k]
    if not top_k:
        return 0.0

    hits = len(set(top_k) & set(relevant))
    return hits / len(top_k)


def recall_at_k(recommended, relevant, k):
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & set(relevant)) / len(set(relevant))


def dcg_at_k(recommended, relevant, k):
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant)
    return sum(
        1 / math.log2(rank + 1)
        for rank, item in enumerate(recommended[:k], start=1)
        if item in relevant
    )


def ndcg_at_k(recommended, relevant, k):
    if k <= 0:
        raise ValueError("k must be positive")
    ideal_hits = min(len(set(relevant)), k)
    if ideal_hits == 0:
        return 0.0

    ideal_dcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )
    return dcg_at_k(recommended, relevant, k) / ideal_dcg


def average_precision_at_k(recommended, relevant, k):
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant)
    if not relevant:
        return 0.0

    hits = 0
    score = 0.0

    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank

    return score / min(len(relevant), k)


def reciprocal_rank(recommended, relevant):
    relevant = set(relevant)
    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            return 1 / rank
    return 0.0
