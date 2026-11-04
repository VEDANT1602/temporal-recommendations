"""Full-catalog, common-population ranking evaluation with explicit denominators."""
from __future__ import annotations

import math
import numpy as np
from .data import history


def top_items(scores, items, excluded, k=10):
    eligible = [(float(score), int(item)) for score, item in zip(scores, items)
                if int(item) not in excluded and np.isfinite(score)]
    # Deterministic tie-break, rather than storage order or hidden randomness.
    eligible.sort(key=lambda pair: (-pair[0], pair[1]))
    return [item for _, item in eligible[:k]]


def ranking_metrics(recommendations: list[int], relevant: set[int], k=10):
    if k != 10:
        raise ValueError("This experiment's metric contract fixes k at 10")
    if not relevant:
        raise ValueError("Recall is undefined for an empty relevant set")
    dcg = sum(1 / math.log2(rank + 2) for rank, item in enumerate(recommendations[:k]) if item in relevant)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return {"recall_at_10": len(set(recommendations[:k]) & relevant) / len(relevant),
            "ndcg_at_10": dcg / ideal}


def evaluate(heldout, prior_rows, train_users, items, scorer, k=10, allow_cold_users=False):
    seen = history(prior_rows)
    item_set, warm_users = set(items), set(train_users)
    positive = heldout[heldout["rating"] >= 4]
    actual = history(positive)
    recommended_items = set()
    records = []
    total_positive, supported_positive = len(positive), 0
    excluded_cold_users = excluded_no_available_positive = cold_users_evaluated = 0
    for user, target in sorted(actual.items()):
        if user not in warm_users and not allow_cold_users:
            excluded_cold_users += 1
            continue
        relevant = (target & item_set) - seen.get(user, set())
        supported_positive += len(relevant)
        if not relevant:
            excluded_no_available_positive += 1
            continue
        cold_users_evaluated += int(user not in warm_users)
        ranked = top_items(scorer(user), items, seen.get(user, set()), k)
        recommended_items.update(ranked)
        # A neutral row number avoids redistributing original identifiers.
        records.append({"evaluation_user": len(records) + 1,
                        "relevant_count": len(relevant),
                        "candidate_count": len(item_set - seen.get(user, set())),
                        **ranking_metrics(ranked, relevant, k)})
    if not records:
        raise ValueError("No warm users with eligible held-out positive ratings")
    return {"users_evaluated": len(records),
            "cold_users_evaluated": cold_users_evaluated,
            "heldout_users_with_positive_ratings": len(actual),
            "excluded_cold_users": excluded_cold_users,
            "excluded_users_without_available_positive": excluded_no_available_positive,
            "heldout_positive_ratings": total_positive,
            "supported_positive_ratings": supported_positive,
            "positive_target_coverage": supported_positive / total_positive,
            "training_catalog_items": len(items),
            "recommended_catalog_items": len(recommended_items),
            "catalog_coverage_at_10": len(recommended_items) / len(items),
            "recall_at_10": float(np.mean([r["recall_at_10"] for r in records])),
            "ndcg_at_10": float(np.mean([r["ndcg_at_10"] for r in records]))}, records


def paired_interval(first, second, seed=42, draws=1000):
    if [r["evaluation_user"] for r in first] != [r["evaluation_user"] for r in second]:
        raise ValueError("Bootstrap requires the same evaluation population")
    difference = np.array([a["ndcg_at_10"] - b["ndcg_at_10"] for a, b in zip(first, second)])
    rng = np.random.default_rng(seed)
    samples = np.array([rng.choice(difference, len(difference), replace=True).mean() for _ in range(draws)])
    return {"mean_difference": float(difference.mean()),
            "lower_95": float(np.quantile(samples, .025)), "upper_95": float(np.quantile(samples, .975)),
            "draws": draws, "unit": "heldout user", "seed": seed}
