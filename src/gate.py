"""Case-level fusion of an ontology ranking and an LLM differential."""
from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lopo import HPOAIndex


CASE_FEATURES = (
    "n_observed", "n_excluded", "log_n_observed", "frac_excluded",
    "ic_mean", "ic_max", "ic_min", "ic_std", "depth_mean", "depth_max",
    "ic_mean_excluded", "has_onset", "has_age", "sex_known",
)
COMPONENT_FEATURES = (
    "log_list_length", "score_margin_norm", "score_entropy10", "score_std_norm",
    "resnik_top1", "resnik_mean3", "resnik_mean10", "resnik_max10",
    "resnik_std10", "resnik_best_rank10",
    "nb_top1", "nb_mean3", "nb_mean10", "nb_max10", "nb_std10", "nb_best_rank10",
    "profile_log_terms_top1", "profile_log_terms_mean10",
    "other_rr_top1", "other_rr_mean3", "other_rr_mean10",
    "overlap1", "overlap3", "overlap5", "overlap10",
)
FEATURE_NAMES = CASE_FEATURES + COMPONENT_FEATURES


def _curie(iri: str) -> str:
    return iri.rsplit("/", 1)[-1].replace("_", ":")


class HPOGraph:
    def __init__(self, hp_json: str | Path):
        graph = json.loads(Path(hp_json).read_text())["graphs"][0]
        self.terms: set[str] = set()
        self.obsolete: set[str] = set()
        self.parents: dict[str, set[str]] = defaultdict(set)
        self.alt: dict[str, str] = {}

        for node in graph["nodes"]:
            if not node["id"].startswith("http://purl.obolibrary.org/obo/HP_"):
                continue
            term = _curie(node["id"])
            self.terms.add(term)
            meta = node.get("meta", {})
            if meta.get("deprecated"):
                self.obsolete.add(term)
            for item in meta.get("basicPropertyValues", []) or []:
                if item["pred"].endswith("hasAlternativeId"):
                    self.alt[item["val"]] = term
                elif item["pred"].endswith("IAO_0100001"):
                    self.alt[term] = _curie(item["val"])

        for edge in graph["edges"]:
            if edge["pred"] == "is_a":
                child, parent = _curie(edge["sub"]), _curie(edge["obj"])
                if child.startswith("HP:") and parent.startswith("HP:"):
                    self.parents[child].add(parent)

        self._ancestors: dict[str, frozenset[str]] = {}
        self.depth: dict[str, int] = {}
        for term in self.terms:
            self._depth(term, ())

    def canon(self, term: str) -> str | None:
        if term in self.terms and term not in self.obsolete:
            return term
        replacement = self.alt.get(term)
        if replacement in self.terms:
            return replacement
        return term if term in self.terms else None

    def ancestors(self, term: str) -> frozenset[str]:
        if term not in self._ancestors:
            found, stack = set(), [term]
            while stack:
                current = stack.pop()
                if current in found:
                    continue
                found.add(current)
                stack.extend(self.parents.get(current, ()))
            self._ancestors[term] = frozenset(found)
        return self._ancestors[term]

    def _depth(self, term: str, seen: tuple[str, ...]) -> int:
        if term in self.depth:
            return self.depth[term]
        if term in seen or not self.parents.get(term):
            self.depth[term] = 0
            return 0
        value = 1 + min(self._depth(parent, seen + (term,)) for parent in self.parents[term])
        self.depth[term] = value
        return value


class OntologyEvidence:
    """Fixed-IC phenotype evidence with a case-specific LOPO profile."""

    def __init__(self, graph: HPOGraph, hpoa: HPOAIndex):
        self.graph = graph
        self.hpoa = hpoa
        self.diseases = sorted(hpoa.direct)
        self.n_diseases = len(self.diseases)
        self.propagated = {
            disease: self._propagate(profile)
            for disease, profile in hpoa.direct.items()
        }
        counts: dict[str, int] = defaultdict(int)
        for profile in self.propagated.values():
            for term in profile:
                counts[term] += 1
        self.term_count = dict(counts)
        self.ic = {
            term: -math.log(count / self.n_diseases)
            for term, count in counts.items()
        }
        self.max_ic = max(self.ic.values(), default=0.0)
        self._cache: dict[tuple[str, str, tuple[str, ...], tuple[str, ...]], tuple[float, float, float]] = {}

    def information_content(self, term: str) -> float:
        return self.ic.get(term, self.max_ic)

    def _propagate(self, direct: dict[str, float]) -> dict[str, float]:
        propagated: dict[str, float] = {}
        for term, frequency in direct.items():
            canonical = self.graph.canon(term)
            if canonical is None:
                continue
            for ancestor in self.graph.ancestors(canonical):
                propagated[ancestor] = max(propagated.get(ancestor, 0.0), frequency)
        return propagated

    def _resnik(self, observed: tuple[str, ...], profile: dict[str, float]) -> float:
        if not observed:
            return 0.0
        propagated = self._propagate(profile)
        scores = []
        for query in observed:
            scores.append(max(
                (self.information_content(term) for term in self.graph.ancestors(query)
                 if term in propagated),
                default=0.0,
            ))
        return float(np.mean(scores))

    def _naive_bayes(
        self,
        observed: tuple[str, ...],
        excluded: tuple[str, ...],
        profile: dict[str, float],
    ) -> float:
        propagated = self._propagate(profile)
        score = 0.0
        for query in observed:
            query_ic = self.information_content(query)
            background = max(
                self.term_count.get(query, 1) / self.n_diseases,
                1.0 / self.n_diseases,
            )
            match = next(
                (term for term in sorted(
                    self.graph.ancestors(query),
                    key=lambda term: (-self.information_content(term), term),
                ) if term in propagated),
                None,
            )
            likelihood = 0.005 if match is None else max(
                propagated[match] * math.exp(-(query_ic - self.information_content(match))),
                0.005,
            )
            score += math.log(likelihood) - math.log(background)

        for term in excluded:
            background = max(
                self.term_count.get(term, 1) / self.n_diseases,
                1.0 / self.n_diseases,
            )
            likelihood = max(1.0 - propagated[term], 0.02) if term in propagated else 1.0
            score += math.log(likelihood) - math.log(max(1.0 - background, 1e-6))
        return score

    def score(
        self,
        publication: str,
        disease: str,
        observed: tuple[str, ...],
        excluded: tuple[str, ...],
    ) -> tuple[float, float, float]:
        key = publication, disease, observed, excluded
        if key not in self._cache:
            if disease not in self.hpoa.direct:
                value = 0.0, -100.0, 0.0
            else:
                profile = self.hpoa.profile(disease, publication)
                value = (
                    self._resnik(observed, profile),
                    self._naive_bayes(observed, excluded, profile),
                    float(len(profile)),
                )
            self._cache[key] = value
        return self._cache[key]


@dataclass(frozen=True)
class Case:
    case_id: str
    publication: str
    gold: str
    observed: tuple[str, ...]
    excluded: tuple[str, ...] = ()
    onset: str | None = None
    age: str | None = None
    sex: str | None = None


Ranking = list[list[str | float]]


def _safe_stats(values: list[float], n: int = 10) -> tuple[float, float, float, float, float]:
    array = np.asarray(values[:n], dtype=np.float64)
    if not len(array):
        return 0.0, 0.0, 0.0, 0.0, 1.0
    return (
        float(array[0]), float(array.mean()), float(array.max()),
        float(array.std()), float(np.argmax(array) + 1) / n,
    )


def _score_geometry(ranking: Ranking) -> tuple[float, float, float]:
    scores = np.asarray([float(score) for _, score in ranking if np.isfinite(score)])
    if not len(scores):
        return 0.0, 0.0, 0.0
    low, high = float(scores.min()), float(scores.max())
    normalized = (scores - low) / (high - low) if high > low else np.zeros_like(scores)
    margin = float(normalized[0] - normalized[1]) if len(normalized) > 1 else 1.0
    head = normalized[:10]
    probabilities = np.exp(head - head.max())
    probabilities /= max(probabilities.sum(), 1e-12)
    entropy = float(
        -(probabilities * np.log(probabilities + 1e-12)).sum()
        / max(math.log(len(head)), 1.0)
    )
    return margin, entropy, float(normalized.std())


class FeatureBuilder:
    def __init__(self, graph: HPOGraph, evidence: OntologyEvidence):
        self.graph = graph
        self.evidence = evidence

    def _terms(self, case: Case) -> tuple[tuple[str, ...], tuple[str, ...]]:
        observed = tuple(filter(None, (self.graph.canon(term) for term in case.observed)))
        excluded = tuple(filter(None, (self.graph.canon(term) for term in case.excluded)))
        return observed, excluded

    def patient(self, case: Case) -> np.ndarray:
        observed, excluded = self._terms(case)
        ic = np.asarray([self.evidence.information_content(term) for term in observed])
        depth = np.asarray([self.graph.depth.get(term, 0) for term in observed])
        excluded_ic = np.asarray([self.evidence.information_content(term) for term in excluded])
        if not len(ic):
            ic = np.zeros(1)
        if not len(depth):
            depth = np.zeros(1)
        return np.asarray([
            len(observed), len(excluded), np.log1p(len(observed)),
            len(excluded) / max(len(observed) + len(excluded), 1),
            ic.mean(), ic.max(), ic.min(), ic.std(), depth.mean(), depth.max(),
            excluded_ic.mean() if len(excluded_ic) else 0.0,
            float(bool(case.onset)), float(bool(case.age)),
            float(case.sex in {"MALE", "FEMALE"}),
        ], dtype=np.float32)

    def component(self, case: Case, ranking: Ranking, other: Ranking) -> np.ndarray:
        ids = [str(disease) for disease, _ in ranking]
        other_ids = [str(disease) for disease, _ in other]
        observed, excluded = self._terms(case)
        margin, entropy, score_std = _score_geometry(ranking)
        evidence = [
            self.evidence.score(case.publication, disease, observed, excluded)
            for disease in ids[:10]
        ]
        resnik = [item[0] for item in evidence]
        nb = [item[1] for item in evidence]
        profile_size = [math.log1p(item[2]) for item in evidence]
        r1, rmean10, rmax, rstd, rargmax = _safe_stats(resnik)
        n1, nmean10, nmax, nstd, nargmax = _safe_stats(nb)
        rmean3 = float(np.mean(resnik[:3])) if resnik else 0.0
        nmean3 = float(np.mean(nb[:3])) if nb else 0.0
        other_position = {disease: rank + 1 for rank, disease in enumerate(other_ids)}
        other_rr = [1.0 / other_position[disease] if disease in other_position else 0.0
                    for disease in ids[:10]]

        def overlap(k: int) -> float:
            left, right = set(ids[:k]), set(other_ids[:k])
            return len(left & right) / max(len(left | right), 1)

        return np.asarray([
            math.log1p(len(ids)), margin, entropy, score_std,
            r1, rmean3, rmean10, rmax, rstd, rargmax,
            n1, nmean3, nmean10, nmax, nstd, nargmax,
            profile_size[0] if profile_size else 0.0,
            float(np.mean(profile_size)) if profile_size else 0.0,
            other_rr[0] if other_rr else 0.0,
            float(np.mean(other_rr[:3])) if other_rr else 0.0,
            float(np.mean(other_rr)) if other_rr else 0.0,
            overlap(1), overlap(3), overlap(5), overlap(10),
        ], dtype=np.float32)

    def pair(self, case: Case, tool: Ranking, llm: Ranking) -> tuple[np.ndarray, np.ndarray]:
        patient = self.patient(case)
        tool_vector = np.r_[patient, self.component(case, tool, llm)]
        llm_vector = np.r_[patient, self.component(case, llm, tool)]
        if len(tool_vector) != 39 or len(llm_vector) != 39:
            raise AssertionError("the feature contract must contain 39 values")
        return tool_vector, llm_vector


@dataclass
class PairDataset:
    case_ids: list[str]
    tool_features: np.ndarray
    llm_features: np.ndarray
    candidates: list[list[str]]
    tool_rr: list[np.ndarray]
    llm_rr: list[np.ndarray]
    gold_index: np.ndarray

    def take(self, indices: np.ndarray) -> "PairDataset":
        indices = np.asarray(indices, dtype=np.int64)
        return PairDataset(
            [self.case_ids[i] for i in indices],
            self.tool_features[indices], self.llm_features[indices],
            [self.candidates[i] for i in indices],
            [self.tool_rr[i] for i in indices], [self.llm_rr[i] for i in indices],
            self.gold_index[indices],
        )


def candidate_pool(tool: Ranking, llm: Ranking) -> tuple[list[str], np.ndarray, np.ndarray]:
    tool_ids = [str(disease) for disease, _ in tool]
    llm_ids = [str(disease) for disease, _ in llm]
    candidates = list(dict.fromkeys(tool_ids + llm_ids))
    tool_rank = {disease: rank + 1 for rank, disease in enumerate(tool_ids)}
    llm_rank = {disease: rank + 1 for rank, disease in enumerate(llm_ids)}
    tool_rr = np.asarray([1.0 / tool_rank[d] if d in tool_rank else 0.0 for d in candidates])
    llm_rr = np.asarray([1.0 / llm_rank[d] if d in llm_rank else 0.0 for d in candidates])
    return candidates, tool_rr, llm_rr


def build_dataset(
    cases: list[Case],
    tool_rankings: dict[str, Ranking],
    llm_rankings: dict[str, Ranking],
    features: FeatureBuilder,
) -> PairDataset:
    case_ids, tool_x, llm_x, candidates, tool_rr, llm_rr, gold_index = [], [], [], [], [], [], []
    for case in cases:
        if case.case_id not in tool_rankings or case.case_id not in llm_rankings:
            continue
        tool = tool_rankings[case.case_id]
        llm = llm_rankings[case.case_id]
        if not tool and not llm:
            continue
        tx, lx = features.pair(case, tool, llm)
        pool, trr, lrr = candidate_pool(tool, llm)
        case_ids.append(case.case_id)
        tool_x.append(tx)
        llm_x.append(lx)
        candidates.append(pool)
        tool_rr.append(trr)
        llm_rr.append(lrr)
        gold_index.append(pool.index(case.gold) if case.gold in pool else -1)
    return PairDataset(
        case_ids, np.asarray(tool_x, dtype=np.float32), np.asarray(llm_x, dtype=np.float32),
        candidates, tool_rr, llm_rr, np.asarray(gold_index, dtype=np.int64),
    )


class SharedGate:
    """One scorer applied to both component representations."""

    def __init__(self, n_features: int = 39, seed: int = 0):
        import torch

        self.torch = torch
        torch.manual_seed(seed)
        self.seed = seed
        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_features, 48), torch.nn.ReLU(),
            torch.nn.Linear(48, 24), torch.nn.ReLU(),
            torch.nn.Linear(24, 1),
        )
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.best_epoch = -1

    def _standardize(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("the gate has not been fitted")
        return (values - self.mean) / self.std

    @staticmethod
    def _padded(dataset: PairDataset, indices: np.ndarray | None = None):
        indices = np.arange(len(dataset.case_ids)) if indices is None else np.asarray(indices)
        width = max((len(dataset.candidates[i]) for i in indices), default=1)
        tool = np.zeros((len(indices), width), dtype=np.float32)
        llm = np.zeros((len(indices), width), dtype=np.float32)
        mask = np.full((len(indices), width), -1e9, dtype=np.float32)
        for row, index in enumerate(indices):
            length = len(dataset.candidates[index])
            tool[row, :length] = dataset.tool_rr[index]
            llm[row, :length] = dataset.llm_rr[index]
            mask[row, :length] = 0.0
        return tool, llm, mask

    def fit(
        self,
        train: PairDataset,
        validation: PairDataset,
        epochs: int = 100,
        batch_size: int = 512,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
        patience: int = 12,
        temperature: float = 8.0,
    ) -> "SharedGate":
        torch = self.torch
        train_keep = np.flatnonzero(train.gold_index >= 0)
        val_keep = np.flatnonzero(validation.gold_index >= 0)
        if not len(train_keep) or not len(val_keep):
            raise ValueError("no gold diagnosis occurs in the candidate unions")

        both = np.r_[train.tool_features, train.llm_features]
        self.mean = both.mean(axis=0)
        self.std = both.std(axis=0) + 1e-6
        train_tool = torch.tensor(self._standardize(train.tool_features), dtype=torch.float32)
        train_llm = torch.tensor(self._standardize(train.llm_features), dtype=torch.float32)
        val_tool = torch.tensor(self._standardize(validation.tool_features[val_keep]), dtype=torch.float32)
        val_llm = torch.tensor(self._standardize(validation.llm_features[val_keep]), dtype=torch.float32)
        trr, lrr, mask = self._padded(train)
        vtrr, vlrr, vmask = self._padded(validation, val_keep)
        trr, lrr, mask = map(torch.tensor, (trr, lrr, mask))
        vtrr, vlrr, vmask = map(torch.tensor, (vtrr, vlrr, vmask))
        gold = torch.tensor(train.gold_index, dtype=torch.long)
        val_gold = torch.tensor(validation.gold_index[val_keep], dtype=torch.long)

        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )
        rng = np.random.default_rng(self.seed)
        best_loss, bad_epochs, best_state = float("inf"), 0, None
        for epoch in range(epochs):
            self.network.train()
            order = rng.permutation(train_keep)
            for start in range(0, len(train_keep), batch_size):
                batch = torch.tensor(order[start:start + batch_size])
                optimizer.zero_grad()
                weight = torch.sigmoid(
                    self.network(train_tool[batch]) - self.network(train_llm[batch])
                )
                scores = temperature * (weight * trr[batch] + (1.0 - weight) * lrr[batch]) + mask[batch]
                loss = torch.nn.functional.cross_entropy(scores, gold[batch])
                loss.backward()
                optimizer.step()

            self.network.eval()
            with torch.no_grad():
                weight = torch.sigmoid(self.network(val_tool) - self.network(val_llm))
                scores = temperature * (weight * vtrr + (1.0 - weight) * vlrr) + vmask
                val_loss = float(torch.nn.functional.cross_entropy(scores, val_gold))
            if val_loss < best_loss - 1e-5:
                best_loss, bad_epochs = val_loss, 0
                best_state = copy.deepcopy(self.network.state_dict())
                self.best_epoch = epoch
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                break

        if best_state is None:
            raise RuntimeError("training did not produce a checkpoint")
        self.network.load_state_dict(best_state)
        return self

    def weights(self, tool_features: np.ndarray, llm_features: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.network.eval()
        with torch.no_grad():
            tool = torch.tensor(self._standardize(tool_features), dtype=torch.float32)
            llm = torch.tensor(self._standardize(llm_features), dtype=torch.float32)
            return torch.sigmoid(self.network(tool) - self.network(llm)).squeeze(-1).numpy()


def fuse(dataset: PairDataset, weights: np.ndarray, top_k: int = 100) -> dict[str, Ranking]:
    output: dict[str, Ranking] = {}
    for case_id, candidates, tool_rr, llm_rr, weight in zip(
        dataset.case_ids, dataset.candidates, dataset.tool_rr, dataset.llm_rr, weights,
    ):
        scores = float(weight) * tool_rr + (1.0 - float(weight)) * llm_rr
        order = np.argsort(-scores, kind="stable")[:top_k]
        output[case_id] = [[candidates[i], float(scores[i])] for i in order]
    return output
