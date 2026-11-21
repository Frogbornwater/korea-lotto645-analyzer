# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v84_event_portfolio import (
    OUTCOME_COUNT,
    V81_REFERENCE_TICKETS,
    V84_VERSION,
    exact_random_distinct_portfolio_baseline,
    generate_v84_candidates,
    portfolio_structure,
)

OUTPUT_DIR = ROOT / "artifacts" / "v84"
EVALUATOR_SOURCE = ROOT / "scripts" / "v84_exact_evaluator.cpp"


def _ticket_mask(ticket: list[int] | tuple[int, ...]) -> int:
    return sum(1 << (int(number) - 1) for number in ticket)


def _compile_evaluator(binary: Path) -> None:
    subprocess.run(
        ["g++", "-O3", "-std=c++17", str(EVALUATOR_SOURCE), "-o", str(binary)],
        check=True,
    )


def _exact_scores(portfolios: list[list[list[int]]]) -> list[dict[str, int]]:
    with tempfile.TemporaryDirectory(prefix="lotto-v84-") as temporary:
        temp = Path(temporary)
        binary = temp / "v84_exact_evaluator"
        input_path = temp / "portfolios.txt"
        output_path = temp / "scores.csv"
        _compile_evaluator(binary)
        lines = [str(len(portfolios))]
        for portfolio in portfolios:
            if len(portfolio) != 10:
                raise ValueError("Exact evaluator requires ten tickets per portfolio.")
            lines.append(" ".join(str(_ticket_mask(ticket)) for ticket in portfolio))
        input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        subprocess.run([str(binary), str(input_path), str(output_path)], check=True)
        with output_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    if len(rows) != len(portfolios):
        raise RuntimeError("Exact evaluator returned an unexpected portfolio count.")
    return [
        {
            "portfolio_index": int(row["portfolio_id"]),
            "any_3_plus_outcomes": int(row["any3"]),
            "any_4_plus_outcomes": int(row["any4"]),
            "any_5_plus_outcomes": int(row["any5"]),
            "jackpot_outcomes": int(row["any6"]),
            "total_outcomes": int(row["total_outcomes"]),
        }
        for row in rows
    ]


def _probabilities(score: dict[str, int]) -> dict[str, float]:
    total = score["total_outcomes"]
    return {
        "any_3_plus_probability": score["any_3_plus_outcomes"] / total,
        "any_4_plus_probability": score["any_4_plus_outcomes"] / total,
        "any_5_plus_probability": score["any_5_plus_outcomes"] / total,
        "jackpot_probability": score["jackpot_outcomes"] / total,
    }


def main() -> None:
    candidates = generate_v84_candidates(seed=8_452_026, target_unique_graphs=19)
    portfolios: list[list[list[int]]] = [
        [list(ticket) for ticket in V81_REFERENCE_TICKETS]
    ] + [candidate["tickets"] for candidate in candidates]  # type: ignore[list-item]
    scores = _exact_scores(portfolios)

    baseline = exact_random_distinct_portfolio_baseline(ticket_count=10)
    entries = []
    for index, (portfolio, score) in enumerate(zip(portfolios, scores)):
        structure = portfolio_structure(portfolio)
        probabilities = _probabilities(score)
        entry = {
            "name": "V8.1 frozen reference" if index == 0 else candidates[index - 1]["candidate_id"],
            "tickets": portfolio,
            "structure": structure,
            "exact_outcome_counts": score,
            "exact_probabilities": probabilities,
        }
        if index > 0:
            entry["graph_edges"] = candidates[index - 1]["graph_edges"]
            entry["graph_fingerprint"] = candidates[index - 1]["graph_fingerprint"]
        entries.append(entry)

    # Primary objective: exact batch 3+ coverage. Secondary: exact batch 4+.
    # 5+ and jackpot are reported as guardrails, not tuned.
    v81 = entries[0]
    best = max(
        entries[1:],
        key=lambda item: (
            item["exact_outcome_counts"]["any_3_plus_outcomes"],
            item["exact_outcome_counts"]["any_4_plus_outcomes"],
        ),
    )
    v81_p = v81["exact_probabilities"]
    best_p = best["exact_probabilities"]
    delta = {key: best_p[key] - v81_p[key] for key in best_p}
    report = {
        "research_version": V84_VERSION,
        "status": "exact_combinatorial_batch_structure_research_only",
        "manual_history_used": False,
        "official_lottery_sources_accessed": False,
        "production_model_changed": False,
        "forward_lock_created": False,
        "existing_draw_1252_lock_modified": False,
        "outcome_space_size": OUTCOME_COUNT,
        "candidate_graph_fingerprints_tested": len(candidates),
        "construction": {
            "description": (
                "Ten tickets are represented as vertices of a simple 3-regular graph. "
                "Each of the 15 graph edges maps to one number shared by its two endpoint "
                "tickets; each vertex receives three private numbers. This guarantees "
                "coverage of all 45 numbers and pairwise ticket overlap at most one."
            ),
            "candidate_seed": 8_452_026,
            "primary_objective": "maximize exact probability of at least one 3+ ticket",
            "secondary_objective": "maximize exact probability of at least one 4+ ticket",
            "guardrail_metrics": ["at least one 5+ ticket", "jackpot"],
        },
        "uniform_random_distinct_10_ticket_baseline": baseline,
        "v81_reference": v81,
        "v84_best": best,
        "v84_minus_v81_probability_delta": delta,
        "candidate_results": entries[1:],
        "evidence_gate": {
            "batch_structure_improved_vs_v81": (
                best_p["any_3_plus_probability"] > v81_p["any_3_plus_probability"]
            ),
            "single_ticket_probability_changed": False,
            "predictive_superiority_proven": False,
            "production_model_change_allowed": False,
            "prospective_validation_required": True,
        },
        "interpretation": (
            "The reported probabilities are exact counts over all C(45,6) equally likely "
            "draw outcomes for the fixed ten-ticket portfolio. This optimizes only the "
            "chance that at least one ticket in the batch reaches a hit threshold. It does "
            "not increase the probability of any individual fixed ticket or establish that "
            "history predicts future winning numbers."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "v84_exact_portfolio_report.json"
    csv_path = OUTPUT_DIR / "v84_exact_portfolio_candidates.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name", "coverage", "max_overlap", "mean_overlap",
                "any_3_plus_probability", "any_4_plus_probability",
                "any_5_plus_probability", "jackpot_probability",
            ],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow({
                "name": entry["name"],
                "coverage": entry["structure"]["number_coverage"],
                "max_overlap": entry["structure"]["max_pairwise_overlap"],
                "mean_overlap": entry["structure"]["mean_pairwise_overlap"],
                **entry["exact_probabilities"],
            })

    print(json.dumps({
        "status": report["status"],
        "outcome_space_size": OUTCOME_COUNT,
        "tested_graph_fingerprints": len(candidates),
        "v81_structure": v81["structure"],
        "v81_probabilities": v81_p,
        "v84_best_name": best["name"],
        "v84_best_tickets": best["tickets"],
        "v84_best_structure": best["structure"],
        "v84_best_probabilities": best_p,
        "probability_delta_vs_v81": delta,
        "random_distinct_10_ticket_baseline": baseline,
        "evidence_gate": report["evidence_gate"],
        "report": str(json_path),
        "csv": str(csv_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
