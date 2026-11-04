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
    baseline, personalized = result["test"]["popularity"], result["test"]["bpr"]
    difference = result["paired_ndcg_difference"]
    system = result["secondary_system_test"]
    fixture_notice = ("**Smoke fixture only:** These values use invented fixture ratings, not MovieLens. They validate execution only.\n\n"
                      if result["dataset"]["kind"] != "official_movielens_100k" else "")
    decision = ("Advance BPR to a broader offline validation, keeping popularity as the fallback."
                if difference["lower_95"] > 0 else
                "Keep popularity as the default while collecting stronger evidence for personalization.")
    return f'''# Temporal Recommendations: earn the right to personalize

{fixture_notice}

**Independent portfolio research by Vedant Thaker.** This implementation was built for this public portfolio. It is not a reconstruction of any employer or client system, and its metrics do not substantiate historical workplace claims.

## A ranked list is a decision, not just a model score

Recommendation systems turn a long catalog into a short list. Matrix factorization is one way to personalize that list. Before adding that complexity, I wanted to answer a narrower question: does a model trained only on the past rank later positive ratings better than a simple popularity baseline?

The recommendation from this run is: **{decision}**

Three pieces of calculated evidence frame that decision:

1. On the same {baseline['users_evaluated']} eligible test users, popularity reached NDCG@10 **{baseline['ndcg_at_10']:.4f}** and BPR reached **{personalized['ndcg_at_10']:.4f}**.
2. Recall@10 was **{as_percent(baseline['recall_at_10'])}** for popularity and **{as_percent(personalized['recall_at_10'])}** for BPR. These are macro averages over users, not the fraction of all ratings recovered.
3. The training catalog and warm-user requirement covered only **{as_percent(baseline['positive_target_coverage'])}** of positive test ratings. An audit-driven popularity fallback made **{as_percent(system['hybrid']['positive_target_coverage'])}** of positive test ratings scorable under the same training-catalog policy. These are different evaluation populations; that coverage change is not a model-quality lift.

![Offline ranking comparison](ranking.svg)

## The conventional shortcut hides two different problems

A random row split can let a model learn from an interaction that occurred after the event it is meant to predict. Sampling a few easy negatives at evaluation can also turn a large-catalog ranking problem into a much easier test. Those are protocol risks, not claims that every conventional recommender makes those mistakes.

I addressed them directly: one global clock splits all users' events into train, validation and test; evaluation scores the full eligible training catalog. I do not compare these temporal results with an easier random-split or sampled-negative number. That comparison would mix model quality with evaluation design.

## My framework: clock, catalog, candidates, comparison

**Clock.** The observed training window ends at Unix timestamp {result['split']['train_end']}; validation ends at {result['split']['validation_end']}. Equal timestamps remain together. The partitions contain {result['rows']['train']:,}, {result['rows']['validation']:,} and {result['rows']['test']:,} ratings. The model is frozen after training; validation selects between two factor counts. Test is evaluated after that selection.

**Catalog.** Every candidate item existed in training. Users without positive training signal and test-only items are measured as coverage limitations, rather than silently granted embeddings learned from future rows.

**Candidates.** A positive means a rating of 4 or 5; a warm user has at least one positive training rating. Users with only low training ratings take the fallback because BPR has not learned their user factor. At validation, every prior training rating is excluded from recommendations. At test, training and validation histories are excluded. Every model sees the same candidate universe and eligible users. There are no randomly sampled negatives in evaluation. During BPR training only, one unobserved/non-positive training-catalog item is sampled per positive per epoch; future ratings are not consulted. A missing rating is not evidence of dislike.

**Comparison.** Popularity counts positive training ratings. The PyTorch model learns user/item factors and item bias using pairwise BPR loss. I predeclared 16 and 32 factors, 16 epochs, learning rate 0.02 and seed 42. Validation NDCG@10 selected **{result['selection']['factors']} factors**; no test metric selected the model. Both experiments and their loss curves are in `metrics.json`.

```mermaid
flowchart LR
  A[Official local ratings] --> B[Validate grain and duplicates]
  B --> C[Global time split]
  C --> D[Train popularity and BPR]
  D --> E[Choose factors on validation]
  E --> F[Freeze model and candidate policy]
  F --> G[Full-catalog test ranking]
  G --> H[Metrics, coverage, paired uncertainty]
```

## The result includes the downside

| Offline measure | Popularity | BPR |
|---|---:|---:|
| NDCG@10 | {baseline['ndcg_at_10']:.4f} | {personalized['ndcg_at_10']:.4f} |
| Recall@10 | {as_percent(baseline['recall_at_10'])} | {as_percent(personalized['recall_at_10'])} |
| Recommended catalog coverage | {as_percent(baseline['catalog_coverage_at_10'])} | {as_percent(personalized['catalog_coverage_at_10'])} |
| Evaluated users | {baseline['users_evaluated']} | {personalized['users_evaluated']} |

The paired bootstrap difference in NDCG@10 (BPR minus popularity) is **{difference['mean_difference']:+.4f}**, with a user-resampling 95% interval of **[{difference['lower_95']:+.4f}, {difference['upper_95']:+.4f}]**. This interval describes variation across the observed eligible users. It does not include retraining uncertainty, time-window selection, future drift or generalization to another product.

This run excluded {baseline['excluded_cold_users']} test users with positive ratings because they had no positive training signal, plus {baseline['excluded_users_without_available_positive']} warm users without an eligible positive item. I would retain popularity for cold-start fallback, then evaluate that fallback on an explicitly defined new-user population before rollout.

## The denominator audit changed the system I needed to evaluate

The first warm-only result was too narrow to describe a whole recommendation service. That was an observed finding, not an assumed benefit of personalization. I added a secondary evaluation after that coverage audit: keep the selected BPR model frozen, and return the training popularity ranking for users without a learned user factor. No factor count, epoch count, learning rate, or model weight was retuned using test results.

In this offline pass, the hybrid ranked candidates for {system['hybrid']['users_evaluated']} eligible test users, including {system['hybrid']['cold_users_evaluated']} cold users. Its NDCG@10 was **{system['hybrid']['ndcg_at_10']:.4f}**, against **{system['popularity']['ndcg_at_10']:.4f}** for popularity on that same broader population. The paired NDCG difference was **{system['paired_ndcg_difference']['mean_difference']:+.4f}**, with a user-bootstrap 95% interval **[{system['paired_ndcg_difference']['lower_95']:+.4f}, {system['paired_ndcg_difference']['upper_95']:+.4f}]**. Treat this as a labeled diagnostic system comparison, not an independent confirmatory experiment. The next run should declare this complete system policy before looking at its test window.

![Coverage audit](coverage.svg)

The fallback makes more requests scorable; it does not establish that recommendations helped those users. Test-only items remain unavailable. This distinction is why I report ranking quality and population coverage separately.

## What impact can I defend?

The implemented value is a reproducible release decision: a baseline, a personalized alternative, consistent candidates, a temporal boundary, and evidence that can be rerun. The benchmark measures offline ranking on historical MovieLens ratings. It does **not** measure online CTR, session time, retention, revenue, fraud capture or a causal business effect. MovieLens ratings are selected observations, not exposure logs; unrated movies are not verified negatives.

The principal tradeoff is added training and maintenance complexity against the observed ranking difference. The next checkpoint is another predeclared time window, more training seeds, cold-start evaluation and, if a real product is available, a guarded randomized online experiment with engagement and coverage metrics. I would change the decision if that evidence reverses the observed advantage or exposes unacceptable cold-start or catalog-concentration behavior.

## Evidence and sources

- All quoted experiment values are calculated from `metrics.json`; `per_user_test.csv` contains the paired per-user measures with neutral row IDs. No original rating rows are redistributed.
- The original problem and implementation are authored for this portfolio. [Build Your Own X](https://github.com/codecrafters-io/build-your-own-x) supplied the learning-by-building direction; no tutorial code or published tutorial result was copied.
- GroupLens Research, [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) and [usage terms](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt).
- F. Maxwell Harper and Joseph A. Konstan (2015), *The MovieLens Datasets: History and Context*, [doi:10.1145/2827872](https://doi.org/10.1145/2827872).
- Steffen Rendle et al. (2009), *BPR: Bayesian Personalized Ranking from Implicit Feedback*, [arXiv:1205.2618](https://arxiv.org/abs/1205.2618). The implementation here is original.
'''


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
