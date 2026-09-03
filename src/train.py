"""Train one leave-one-backbone-family-out fusion gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate import (
    FEATURE_NAMES, Case, FeatureBuilder, HPOGraph, OntologyEvidence, PairDataset,
    SharedGate, build_dataset, fuse,
)
from lopo import HPOAIndex


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle]


def read_cases(cases_path: str | Path, views_path: str | Path) -> dict[str, Case]:
    cases = {row["case_uid"]: row for row in read_jsonl(cases_path)}
    views = {row["case_uid"]: row for row in read_jsonl(views_path)}
    output = {}
    for case_id in cases.keys() & views.keys():
        record, view = cases[case_id], views[case_id]
        disease_ids = record.get("disease_ids") or []
        if not disease_ids:
            continue
        output[case_id] = Case(
            case_id=case_id,
            publication=str(record["pmid"]),
            gold=disease_ids[0],
            observed=tuple(item["id"] if isinstance(item, dict) else item
                           for item in view.get("observed", [])),
            excluded=tuple(item["id"] if isinstance(item, dict) else item
                           for item in view.get("excluded", [])),
            onset=record.get("onset"), age=record.get("age_last_encounter"),
            sex=record.get("sex"),
        )
    return output


def concatenate(parts: list[PairDataset]) -> PairDataset:
    return PairDataset(
        [case_id for part in parts for case_id in part.case_ids],
        np.concatenate([part.tool_features for part in parts]),
        np.concatenate([part.llm_features for part in parts]),
        [candidates for part in parts for candidates in part.candidates],
        [values for part in parts for values in part.tool_rr],
        [values for part in parts for values in part.llm_rr],
        np.concatenate([part.gold_index for part in parts]),
    )


def balanced_supervised_sample(
    parts: list[PairDataset], budget: int, seed: int,
) -> tuple[PairDataset, list[int]]:
    eligible = [part.take(np.flatnonzero(part.gold_index >= 0)) for part in parts]
    base, remainder = divmod(budget, len(eligible))
    allocation = [base + int(i < remainder) for i in range(len(eligible))]
    if any(count > len(part.case_ids) for count, part in zip(allocation, eligible)):
        raise ValueError("the requested balanced budget exceeds the eligible source rows")
    sampled = []
    for i, (part, count) in enumerate(zip(eligible, allocation)):
        rng = np.random.default_rng(seed + 1009 * i)
        indices = np.sort(rng.choice(len(part.case_ids), count, replace=False))
        sampled.append(part.take(indices))
    return concatenate(sampled), allocation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hp", required=True, help="HPO hp.json")
    parser.add_argument("--hpoa", required=True, help="HPO phenotype.hpoa")
    parser.add_argument("--cases", required=True, help="case labels and source PMIDs, JSONL")
    parser.add_argument("--views", required=True, help="phenotype-only case inputs, JSONL")
    parser.add_argument("--splits", required=True)
    parser.add_argument("--tool-ranking", required=True)
    parser.add_argument("--models", required=True, help="JSON model/family/ranking manifest")
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-budget", type=int, default=7024)
    parser.add_argument("--validation-budget", type=int, default=1094)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.models)
    manifest = json.loads(manifest_path.read_text())["models"]
    if args.target not in manifest:
        raise ValueError(f"target {args.target!r} is absent from the model manifest")
    target_family = manifest[args.target]["family"]
    source_models = [
        name for name, record in manifest.items()
        if record["family"] != target_family
    ]
    if not source_models:
        raise ValueError("holding out the target family leaves no source models")

    def ranking_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else manifest_path.parent / path

    rankings = {
        name: json.loads(ranking_path(record["ranking"]).read_text())
        for name, record in manifest.items()
    }
    tool_rankings = json.loads(Path(args.tool_ranking).read_text())
    cases = read_cases(args.cases, args.views)
    split = json.loads(Path(args.splits).read_text())
    evaluable = set(split.get("evaluable", cases))
    split_cases = {
        name: [cases[case_id] for case_id in split["splits"][name]
               if case_id in cases and case_id in evaluable]
        for name in ("train", "val", "test")
    }
    required = {case.case_id for rows in split_cases.values() for case in rows}
    missing_tool = required - tool_rankings.keys()
    if missing_tool:
        raise ValueError(f"the ontology ranking is missing {len(missing_tool)} required cases")
    for model, model_rankings in rankings.items():
        missing = required - model_rankings.keys()
        if missing:
            raise ValueError(f"{model} is missing {len(missing)} required cases")

    graph = HPOGraph(args.hp)
    hpoa = HPOAIndex(args.hpoa)
    feature_builder = FeatureBuilder(graph, OntologyEvidence(graph, hpoa))
    datasets = {
        model: {
            split_name: build_dataset(split_cases[split_name], tool_rankings,
                                      rankings[model], feature_builder)
            for split_name in ("train", "val", "test")
        }
        for model in manifest
    }
    train, train_allocation = balanced_supervised_sample(
        [datasets[model]["train"] for model in source_models], args.train_budget, args.seed,
    )
    validation, validation_allocation = balanced_supervised_sample(
        [datasets[model]["val"] for model in source_models],
        args.validation_budget, args.seed + 17,
    )
    gate = SharedGate(seed=args.seed).fit(train, validation)
    test = datasets[args.target]["test"]
    weights = gate.weights(test.tool_features, test.llm_features)
    fused = fuse(test, weights)
    correct = np.mean([
        bool(fused[case.case_id]) and fused[case.case_id][0][0] == case.gold
        for case in split_cases["test"] if case.case_id in fused
    ])

    import torch
    torch.save({
        "state_dict": gate.network.state_dict(),
        "mean": gate.mean,
        "std": gate.std,
        "features": FEATURE_NAMES,
        "target": args.target,
        "held_out_family": target_family,
        "source_models": source_models,
        "train_allocation": dict(zip(source_models, train_allocation)),
        "validation_allocation": dict(zip(source_models, validation_allocation)),
        "seed": args.seed,
        "best_epoch": gate.best_epoch,
    }, output / "gate.pt")
    (output / "ranking.json").write_text(json.dumps(fused))
    (output / "metrics.json").write_text(json.dumps({
        "target": args.target,
        "held_out_family": target_family,
        "n_test": len(fused),
        "recall@1": float(correct),
    }, indent=2))
    print(json.dumps({"target": args.target, "recall@1": float(correct)}, indent=2))


if __name__ == "__main__":
    main()
