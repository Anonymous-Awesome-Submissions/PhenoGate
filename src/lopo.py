"""Publication-level provenance filtering for HPOA.

For a case drawn from publication p, LOPO removes a disease--phenotype
relation only when every PMID recorded for that relation is p. Relations with
independent publication support are left unchanged.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


FREQUENCY_TERMS = {
    "HP:0040280": 1.00,
    "HP:0040281": 0.90,
    "HP:0040282": 0.55,
    "HP:0040283": 0.17,
    "HP:0040284": 0.025,
    "HP:0040285": 0.00,
}
_FRACTION = re.compile(r"^(\d+)/(\d+)$")
_PERCENT = re.compile(r"^([\d.]+)%$")


def parse_frequency(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    if value in FREQUENCY_TERMS:
        return FREQUENCY_TERMS[value]
    if match := _FRACTION.match(value):
        numerator, denominator = map(int, match.groups())
        return numerator / denominator if denominator else None
    if match := _PERCENT.match(value):
        return float(match.group(1)) / 100.0
    return None


def pmids(reference: str) -> frozenset[str]:
    return frozenset(
        item.strip()[5:]
        for item in reference.split(";")
        if item.strip().startswith("PMID:")
    )


class HPOAIndex:
    """Direct HPOA profiles and their publication provenance."""

    def __init__(self, path: str | Path, prefixes: tuple[str, ...] = ("OMIM",)):
        self.path = Path(path)
        self.prefixes = prefixes
        self.direct: dict[str, dict[str, float]] = defaultdict(dict)
        self.references: dict[str, dict[str, frozenset[str]]] = defaultdict(dict)
        self.pmid_to_diseases: dict[str, set[str]] = defaultdict(set)
        self._read()

    def _read(self) -> None:
        accumulated_refs: dict[tuple[str, str], set[str]] = defaultdict(set)
        accumulated_freqs: dict[tuple[str, str], list[float]] = defaultdict(list)

        with self.path.open() as handle:
            columns = None
            for line in handle:
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if fields[0] == "database_id":
                    columns = {name: i for i, name in enumerate(fields)}
                    continue
                if columns is None:
                    raise ValueError("phenotype.hpoa header not found")

                disease = fields[columns["database_id"]]
                if not disease.startswith(self.prefixes):
                    continue
                if fields[columns["aspect"]] != "P":
                    continue
                if fields[columns["qualifier"]] == "NOT":
                    continue

                term = fields[columns["hpo_id"]]
                key = disease, term
                accumulated_refs[key].update(pmids(fields[columns["reference"]]))
                frequency = parse_frequency(fields[columns["frequency"]])
                accumulated_freqs[key].append(0.55 if frequency is None else frequency)

        for (disease, term), refs in accumulated_refs.items():
            self.direct[disease][term] = max(accumulated_freqs[disease, term])
            frozen = frozenset(refs)
            self.references[disease][term] = frozen
            for publication in frozen:
                self.pmid_to_diseases[publication].add(disease)

    def overlaps(self, publication: str, disease: str) -> bool:
        return any(
            publication in refs
            for refs in self.references.get(disease, {}).values()
        )

    def profile(self, disease: str, omit_publication: str | None = None) -> dict[str, float]:
        """Return a direct disease profile, optionally under exact LOPO."""
        profile = self.direct.get(disease, {})
        if omit_publication is None:
            return dict(profile)
        return {
            term: frequency
            for term, frequency in profile.items()
            if self.references[disease][term] != frozenset({omit_publication})
        }

    def affected_profiles(self, publication: str) -> dict[str, dict[str, float]]:
        """LOPO profiles for diseases whose annotations cite ``publication``."""
        return {
            disease: self.profile(disease, publication)
            for disease in self.pmid_to_diseases.get(publication, set())
        }

    def write_filtered(self, publication: str, output: str | Path) -> int:
        """Write an HPOA file with source-exclusive relations removed.

        The output preserves the original file byte-for-byte except for rows
        belonging to a removed disease--phenotype relation. The return value is
        the number of relations removed.
        """
        removable = {
            (disease, term)
            for disease, terms in self.references.items()
            for term, refs in terms.items()
            if refs == frozenset({publication})
        }
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open() as source, output.open("w") as target:
            columns = None
            for line in source:
                fields = line.rstrip("\n").split("\t")
                if line.startswith("#"):
                    target.write(line)
                    continue
                if fields[0] == "database_id":
                    columns = {name: i for i, name in enumerate(fields)}
                    target.write(line)
                    continue
                if columns is None:
                    raise ValueError("phenotype.hpoa header not found")
                key = fields[columns["database_id"]], fields[columns["hpo_id"]]
                positive_phenotype = (
                    key[0].startswith(self.prefixes)
                    and fields[columns["aspect"]] == "P"
                    and fields[columns["qualifier"]] != "NOT"
                )
                if not positive_phenotype or key not in removable:
                    target.write(line)
        return len(removable)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one publication-filtered HPOA file")
    parser.add_argument("--hpoa", required=True)
    parser.add_argument("--pmid", required=True, help="PMID without the 'PMID:' prefix")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    index = HPOAIndex(args.hpoa)
    removed = index.write_filtered(args.pmid, args.output)
    print(f"removed {removed} source-exclusive relations")


if __name__ == "__main__":
    main()
