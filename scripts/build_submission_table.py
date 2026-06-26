#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    source = ROOT / "results" / "speciesnet_overlap" / "application_safe_results.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    out = ROOT / "results" / "speciesnet_overlap" / "rebuilt_submission_table.md"
    lines = [
        "# Rebuilt SpeciesNet-overlap submission table",
        "",
        "| Model route | Runs | Events | Species | Species accuracy | Raw TIS | Adjusted TIS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {runs} | {events} | {species} | {sp}% | {raw}% | {adj}% |".format(
                model=row["model_route"],
                runs=row["runs"],
                events=row["event_count"],
                species=row["ground_truth_species_count"],
                sp=row["species_accuracy_pct_mean"],
                raw=row["raw_tis_pct_mean"],
                adj=row["adjusted_tis_pct_mean"] or "n/a",
            )
        )
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
