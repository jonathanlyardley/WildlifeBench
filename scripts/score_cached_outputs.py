#!/usr/bin/env python3
"""Re-aggregate the cached scored event table without API credentials."""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    rows = list(csv.DictReader((ROOT / "results" / "speciesnet_overlap" / "application_safe_results_events.csv").open(encoding="utf-8")))
    by_route = defaultdict(list)
    for row in rows:
        by_route[row["model_route"]].append(row)
    out_rows = []
    for route, route_rows in sorted(by_route.items()):
        species_hits = sum(1 for row in route_rows if row.get("ground_truth_species") and row.get("ground_truth_species") == row.get("prediction_species"))
        tis_values = [num(row.get("tis_points")) for row in route_rows]
        adj_values = [num(row.get("adjusted_tis_points")) for row in route_rows]
        tis_values = [value for value in tis_values if value is not None]
        adj_values = [value for value in adj_values if value is not None]
        out_rows.append(
            {
                "model_route": route,
                "event_rows": len(route_rows),
                "species_accuracy_pct": round(100 * species_hits / len(route_rows), 1),
                "raw_tis_pct": round(100 * sum(tis_values) / (len(route_rows) * 5), 1) if tis_values else "",
                "adjusted_tis_pct": round(100 * sum(adj_values) / (len(route_rows) * 5), 1) if adj_values else "",
                "runs_present": len({row["run_index"] for row in route_rows}),
            }
        )
    out = ROOT / "results" / "speciesnet_overlap" / "cached_output_reaggregate.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
