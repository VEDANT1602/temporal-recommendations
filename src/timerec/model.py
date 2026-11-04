"""BPR matrix factorization: pairwise training without future interaction access."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from .data import history


class FactorRanker(nn.Module):
    def __init__(self, users: list[int], items: list[int], factors: int):
        super().__init__()
        self.users, self.items = users, items
        self.user_map = {v: k for k, v in enumerate(users)}
        self.item_map = {v: k for k, v in enumerate(items)}
        self.trained_users = set()
        self.user_embedding = nn.Embedding(len(users), factors)
        self.item_embedding = nn.Embedding(len(items), factors)
        self.item_bias = nn.Embedding(len(items), 1)
        nn.init.normal_(self.user_embedding.weight, std=.05)
        nn.init.normal_(self.item_embedding.weight, std=.05)
        nn.init.zeros_(self.item_bias.weight)

    def pair_scores(self, users, items):
        return (self.user_embedding(users) * self.item_embedding(items)).sum(-1) + self.item_bias(items).squeeze(-1)

    def scores(self, user: int) -> np.ndarray:
        if user not in self.trained_users:
            raise ValueError("User has no positive training signal; use the explicit fallback")
        with torch.no_grad():
            uid = torch.tensor([self.user_map[user]])
            values = self.item_embedding.weight @ self.user_embedding(uid).squeeze(0)
            return (values + self.item_bias.weight.squeeze(-1)).numpy()


def train_bpr(rows, factors: int = 32, epochs: int = 16, seed: int = 42):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    users = sorted(set(map(int, rows["user"])))
    items = sorted(set(map(int, rows["item"])))
    ranker = FactorRanker(users, items, factors)
    positives = rows[rows["rating"] >= 4]
    positive_history = history(positives)
    ranker.trained_users = set(positive_history)
    pools = {
        user: np.array([ranker.item_map[item] for item in items if item not in seen], dtype=np.int64)
        for user, seen in positive_history.items()
    }
    valid = [row for row in positives if len(pools[int(row["user"])])]
    if not valid:
        raise ValueError("BPR needs positive ratings and at least one non-positive candidate per training user")
    uid = np.array([ranker.user_map[int(row["user"])] for row in valid])
    iid = np.array([ranker.item_map[int(row["item"])] for row in valid])
    original_users = np.array([int(row["user"]) for row in valid])
    optimizer = torch.optim.Adam(ranker.parameters(), lr=.02)
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(valid))
        total = 0.0
        for start in range(0, len(order), 1024):
            index = order[start:start + 1024]
            negative = np.array([rng.choice(pools[int(u)]) for u in original_users[index]])
            u, p, n = (torch.from_numpy(arr).long() for arr in (uid[index], iid[index], negative))
            difference = ranker.pair_scores(u, p) - ranker.pair_scores(u, n)
            # Regularize only sampled factors, with an explicit per-example scale.
            regularization = (ranker.user_embedding(u).square().mean() +
                              ranker.item_embedding(p).square().mean() +
                              ranker.item_embedding(n).square().mean())
            loss = nn.functional.softplus(-difference).mean() + .001 * regularization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        losses.append(total / len(valid))
    return ranker, losses


def popularity(rows, items: list[int]) -> np.ndarray:
    counts = {item: 0 for item in items}
    for row in rows[rows["rating"] >= 4]:
        counts[int(row["item"])] += 1
    return np.array([counts[item] for item in items], dtype=float)
