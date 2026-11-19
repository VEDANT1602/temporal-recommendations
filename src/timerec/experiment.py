"""The committed report is generated from this experiment's numerical evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import DATA_URL, EXPECTED_SHA256, load_ratings, sha256, temporal_split
from .evaluate import evaluate, paired_interval
from .model import popularity, train_bpr

OFFICIAL_RATINGS_SHA256 = "06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490"


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def as_percent(value):
    return f"{100 * value:.2f}%"


def chart(result, path):
    """Small dependency-free SVG, using the saved numerical values verbatim."""
    labels = ["Popularity", "BPR matrix factorization"]
    values = [result["test"]["popularity"]["ndcg_at_10"], result["test"]["bpr"]["ndcg_at_10"]]
    maximum = max(.10, max(values) * 1.2)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="820" height="310" viewBox="0 0 820 310" role="img" aria-labelledby="title desc">',
             '<title id="title">Full-catalog test NDCG at 10</title>',
             '<desc id="desc">A zero-baseline bar chart of saved offline metrics. Not an online business outcome.</desc>',
             '<rect width="820" height="310" fill="#f6f8fc"/>',
             '<text x="32" y="34" font-family="sans-serif" font-size="23" fill="#132b45">What does personalization add after a time split?</text>',
             '<text x="32" y="62" font-family="sans-serif" font-size="15" fill="#476079">Macro NDCG@10, warm users with eligible positive ratings</text>']
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 102 + index * 80
        width = value / maximum * 450
        parts.extend([f'<text x="32" y="{y + 23}" font-family="sans-serif" font-size="16">{label}</text>',
                      f'<rect x="275" y="{y}" width="{width:.2f}" height="34" rx="4" fill="{["#68809b", "#1a827f"][index]}"/>',
                      f'<text x="{285 + width:.2f}" y="{y + 23}" font-family="sans-serif" font-size="17">{value:.4f}</text>'])
    parts.extend(['<text x="32" y="283" font-family="sans-serif" font-size="14" fill="#476079">Source: metrics.json. Ratings are not impressions; no CTR or revenue claim.</text>', '</svg>'])
    path.write_text("\n".join(parts), encoding="utf-8")


def coverage_chart(result, path):
    values = [result["test"]["bpr"]["positive_target_coverage"],
              result["secondary_system_test"]["hybrid"]["positive_target_coverage"]]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="820" height="300" viewBox="0 0 820 300" role="img" aria-labelledby="title desc">',
             '<title id="title">A fallback changes who can be scored</title>',
             '<desc id="desc">The share of all positive test ratings scorable under each population policy. Different scopes, not model lift.</desc>',
             '<rect width="820" height="300" fill="#f6f8fc"/>',
             '<text x="32" y="34" font-family="sans-serif" font-size="23" fill="#132b45">A warm-only result hides a cold-start service problem</text>',
             '<text x="32" y="62" font-family="sans-serif" font-size="15" fill="#476079">Scorable positive ratings / all positive test ratings; fixed training catalog</text>']
    for index, (label, value) in enumerate(zip(["Warm-user model only", "Model + popularity fallback"], values)):
        y = 102 + index * 80
        width = value * 440
        parts.extend([f'<text x="32" y="{y + 23}" font-family="sans-serif" font-size="16">{label}</text>',
                      f'<rect x="275" y="{y}" width="{width:.2f}" height="34" rx="4" fill="{["#68809b", "#1a827f"][index]}"/>',
                      f'<text x="{285 + width:.2f}" y="{y + 23}" font-family="sans-serif" font-size="17">{100 * value:.2f}%</text>'])
    parts.extend(['<text x="32" y="280" font-family="sans-serif" font-size="14" fill="#476079">This is population coverage, not improved recommendation quality.</text>', '</svg>'])
    path.write_text("\n".join(parts), encoding="utf-8")


def narrative(result):
    baseline = result["test"]["popularity"]
    bpr = result["test"]["bpr"]
    paired = result["paired_ndcg_difference"]
    system = result["secondary_system_test"]
    hybrid = system["hybrid"]
    fallback_pair = system["paired_ndcg_difference"]
    dataset = result["dataset"]["name"]
    if result["dataset"]["kind"] != "official_movielens_100k":
        scope = "These are supplied ratings or a smoke fixture. Interpret the values only for this input."
    else:
        scope = "MovieLens ratings record preferences among rated items. They provide no impression logs or online engagement outcomes."
    if paired["lower_95"] > 0:
        interval_reading = "The interval favors BPR in this window. A second time window and independent training seeds would test whether the advantage repeats."
    elif paired["upper_95"] < 0:
        interval_reading = "The interval favors popularity in this window. The extra training cost of BPR has no ranking benefit here."
    else:
        interval_reading = "The interval includes zero. These users and this split leave the ranking advantage unresolved."
    return f"""# Popularity and personalized ranking on {dataset}

{scope}

The first comparison covers {bpr['users_evaluated']:,} warm users. It scores {bpr['supported_positive_ratings']:,} of {bpr['heldout_positive_ratings']:,} held-out positive ratings. That second count matters when choosing a serving policy: a strong list for a known user does little for someone with no trained user factor.

## Ranking the same catalog

One global clock partitions the ratings into {result['rows']['train']:,} training, {result['rows']['validation']:,} validation and {result['rows']['test']:,} test rows. Rows at the same timestamp stay together. Each test list comes from the {bpr['training_catalog_items']:,} training-catalog items, excluding titles that user rated during training or validation. Both models score the full eligible catalog. Four- and five-star ratings count as positive.

Popularity ranks by positive training count. Bayesian personalized ranking (BPR) learns user and item factors from pairwise comparisons during training. Validation NDCG@10 selected {result['selection']['factors']} factors. Its weights stay fixed for the test window.

| Warm-user test measure | Popularity | BPR |
| --- | ---: | ---: |
| NDCG@10 | {baseline['ndcg_at_10']:.4f} | {bpr['ndcg_at_10']:.4f} |
| Recall@10 | {as_percent(baseline['recall_at_10'])} | {as_percent(bpr['recall_at_10'])} |
| Share of catalog appearing in top-ten lists | {as_percent(baseline['catalog_coverage_at_10'])} | {as_percent(bpr['catalog_coverage_at_10'])} |

Each user contributes equally to recall and NDCG. The paired difference in NDCG@10 (BPR minus popularity) was **{paired['mean_difference']:+.4f}**, with a 95% user-bootstrap interval of **[{paired['lower_95']:+.4f}, {paired['upper_95']:+.4f}]**. {interval_reading} Resampling users holds the fitted weights and the time window fixed. [Paired scores](per_user_test.csv)

![Full-catalog ranking on the warm-user population](ranking.svg)

## Serving a user with no positive training rating

The warm-user evaluation excluded {bpr['excluded_cold_users']:,} users with no positive training signal and {bpr['excluded_users_without_available_positive']:,} additional warm users whose positive targets were unavailable. A second pass uses the same BPR weights for warm users and a popularity list for cold users. New titles outside the training catalog remain ineligible.

| System comparison | Saved result |
| --- | ---: |
| Users with an eligible test target | {hybrid['users_evaluated']:,} |
| Of those, users served by the cold-user fallback | {hybrid['cold_users_evaluated']:,} |
| Positive ratings eligible under the hybrid | {hybrid['supported_positive_ratings']:,} / {hybrid['heldout_positive_ratings']:,} ({as_percent(hybrid['positive_target_coverage'])}) |
| Popularity NDCG@10 on this population | {system['popularity']['ndcg_at_10']:.4f} |
| Hybrid NDCG@10 on this population | {hybrid['ndcg_at_10']:.4f} |
| Paired hybrid minus popularity, 95% interval | {fallback_pair['mean_difference']:+.4f} [{fallback_pair['lower_95']:+.4f}, {fallback_pair['upper_95']:+.4f}] |

The change from {as_percent(bpr['positive_target_coverage'])} to {as_percent(hybrid['positive_target_coverage'])} describes **coverage of held-out positive ratings**. Ranking quality needs a comparison on matched users. This broader pass was added after inspecting the original coverage, so its ranking result is exploratory. [Per-user system comparison](per_user_system_test.csv)

![Share of positive ratings that each population policy can score](coverage.svg)

The [saved metrics](metrics.json) contain the timestamp cutoffs, input digest, validation runs and training losses. The study uses the [GroupLens MovieLens protocol and data terms](https://grouplens.org/datasets/movielens/100k/) for the public dataset and the [BPR objective](https://arxiv.org/abs/1205.2618). [Build Your Own X](https://github.com/codecrafters-io/build-your-own-x) provided the original learning-by-building reference. This is an independent offline experiment by Vedant Thaker.
"""


def run(data_path: Path, output: Path):
    started = time.perf_counter()
    rows = load_ratings(data_path)
    train, validation, test, cutoffs = temporal_split(rows)
    items = sorted(set(map(int, train["item"])))
    # An observed low rating alone does not train a BPR user factor.
    # Such users must take the same explicit fallback as entirely new users.
    users = sorted(set(map(int, train[train["rating"] >= 4]["user"])))
    baseline_scores = popularity(train, items)
    baseline_validation, _ = evaluate(validation, train, users, items, lambda _: baseline_scores)
    candidates = []
    models = []
    for factors in [16, 32]:
        tick = time.perf_counter()
        model, losses = train_bpr(train, factors=factors)
        metrics, _ = evaluate(validation, train, users, items, model.scores)
        candidates.append({"factors": factors, "epochs": 16, "seed": 42, "validation": metrics,
                           "training_seconds": time.perf_counter() - tick, "training_loss": losses})
        models.append(model)
    choice = max(range(len(candidates)), key=lambda i: (candidates[i]["validation"]["ndcg_at_10"], -candidates[i]["factors"]))
    selected = models[choice]
    prior = np.concatenate([train, validation])
    baseline_test, baseline_records = evaluate(test, prior, users, items, lambda _: baseline_scores)
    selected_test, model_records = evaluate(test, prior, users, items, selected.scores)
    system_popularity, system_popularity_records = evaluate(test, prior, users, items, lambda _: baseline_scores, allow_cold_users=True)
    hybrid_test, hybrid_records = evaluate(test, prior, users, items,
                                         lambda u: selected.scores(u) if u in selected.trained_users else baseline_scores,
                                         allow_cold_users=True)
    ratings_digest = sha256(data_path)
    # Exact MovieLens recognition is by the observed full-file SHA, configured below.
    official = ratings_digest == OFFICIAL_RATINGS_SHA256
    result = {"schema_version": 1, "run_kind": "executed_offline_research",
              "dataset": {"name": "MovieLens 100K" if official else "Supplied ratings / original fixture",
                          "kind": "official_movielens_100k" if official else "supplied_or_fixture",
                          "source": DATA_URL if official else data_path.name,
                          "archive_sha256": EXPECTED_SHA256 if official else None, "ratings_sha256": ratings_digest,
                          "raw_data_redistributed": False},
              "rows": {"all": len(rows), "train": len(train), "validation": len(validation), "test": len(test)},
              "split": cutoffs,
              "protocol": {"positive_rating_threshold": 4, "k": 10, "training_fraction": .70,
                           "validation_end_fraction": .85, "evaluation_negative_sampling": False,
                           "test_model_refit": False, "test_history_includes_validation": True,
                           "warm_user_definition": "at least one positive training rating",
                           "selection_metric": "validation_macro_ndcg_at_10", "torch_threads": 2},
              "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                              "platform": platform.platform(), "device": "cpu"},
              "validation_popularity": baseline_validation, "candidate_experiments": candidates,
              "selection": {"factors": candidates[choice]["factors"], "criterion": "validation NDCG@10, smaller factor count on tie"},
              "test": {"popularity": baseline_test, "bpr": selected_test},
              "paired_ndcg_difference": paired_interval(model_records, baseline_records),
              "secondary_system_test": {"status": "audit_driven_diagnostic_no_retuning",
                                        "popularity": system_popularity, "hybrid": hybrid_test,
                                        "paired_ndcg_difference": paired_interval(hybrid_records, system_popularity_records)},
              "runtime_seconds": time.perf_counter() - started}
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    paired = [{**{"evaluation_user": a["evaluation_user"], "relevant_count": a["relevant_count"], "candidate_count": a["candidate_count"]},
               "popularity_recall_at_10": a["recall_at_10"], "popularity_ndcg_at_10": a["ndcg_at_10"],
               "bpr_recall_at_10": b["recall_at_10"], "bpr_ndcg_at_10": b["ndcg_at_10"]}
              for a, b in zip(baseline_records, model_records)]
    write_csv(output / "per_user_test.csv", paired)
    system_paired = [{"evaluation_user": a["evaluation_user"], "relevant_count": a["relevant_count"], "candidate_count": a["candidate_count"],
                      "popularity_recall_at_10": a["recall_at_10"], "popularity_ndcg_at_10": a["ndcg_at_10"],
                      "hybrid_recall_at_10": b["recall_at_10"], "hybrid_ndcg_at_10": b["ndcg_at_10"]}
                     for a, b in zip(system_popularity_records, hybrid_records)]
    write_csv(output / "per_user_system_test.csv", system_paired)
    chart(result, output / "ranking.svg")
    coverage_chart(result, output / "coverage.svg")
    (output / "case-study.md").write_text(narrative(result), encoding="utf-8")
    print(json.dumps({"selected_factors": result["selection"]["factors"], "test": result["test"],
                      "paired_difference": result["paired_ndcg_difference"], "runtime_seconds": result["runtime_seconds"]}, indent=2))
