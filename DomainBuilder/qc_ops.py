from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .common import quantiles


def _reason_counter(domains: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for d in domains:
        for r in d.get("qc_reject_reasons", []):
            counts[str(r)] += 1
    return {k: int(v) for k, v in sorted(counts.items(), key=lambda x: x[0])}


def build_program_domain_summary_table(
    program_summary_rows: list[dict],
    domains: list[dict],
) -> pd.DataFrame:
    by_program: dict[str, dict] = {str(r["program_id"]): dict(r) for r in program_summary_rows}
    for d in domains:
        pid = str(d["program_seed_id"])
        row = by_program.setdefault(pid, {"program_id": pid})
        row["domain_count_total"] = int(row.get("domain_count_total", 0)) + 1
        if bool(d.get("qc_pass", False)):
            row["domain_count_pass"] = int(row.get("domain_count_pass", 0)) + 1
        else:
            row["domain_count_reject"] = int(row.get("domain_count_reject", 0)) + 1

    df = pd.DataFrame(list(by_program.values()))
    if df.empty:
        return pd.DataFrame(
            columns=[
                "program_id",
                "basin_count",
                "screening_pass_count",
                "domain_count_total",
                "domain_count_pass",
                "domain_count_reject",
                "no_domain_program",
            ]
        )

    for col in ["domain_count_total", "domain_count_pass", "domain_count_reject"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(np.int32)
    df["no_domain_program"] = df["domain_count_pass"] <= 0
    return df.sort_values("program_id").reset_index(drop=True)


def build_qc_report(
    sample_id: str,
    n_spots: int,
    n_programs: int,
    segmentation_summary: dict,
    domains: list[dict],
    program_summary_table: pd.DataFrame,
    merge_summary: dict,
    acceptance_gates: dict,
) -> dict:
    spot_counts = np.asarray([float(x.get("spot_count", 0)) for x in domains], dtype=np.float32)
    peak_vals = np.asarray([float(x.get("prog_peak_value", 0.0)) for x in domains], dtype=np.float32)
    prominences = np.asarray([float(x.get("prog_prominence", 0.0)) for x in domains], dtype=np.float32)
    mass_vals = np.asarray([float(x.get("prog_seed_sum", 0.0)) for x in domains], dtype=np.float32)
    enrich_ratio_vals = np.asarray([float(x.get("prog_mean_enrichment_ratio", 0.0)) for x in domains], dtype=np.float32)
    enrich_delta_vals = np.asarray([float(x.get("prog_mean_enrichment_delta", 0.0)) for x in domains], dtype=np.float32)

    passed = [x for x in domains if bool(x.get("qc_pass", False))]
    passed_spot_counts = np.asarray([float(x.get("spot_count", 0)) for x in passed], dtype=np.float32)
    passed_peak = np.asarray([float(x.get("prog_peak_value", 0.0)) for x in passed], dtype=np.float32)
    passed_prom = np.asarray([float(x.get("prog_prominence", 0.0)) for x in passed], dtype=np.float32)
    passed_enrich_ratio = np.asarray([float(x.get("prog_mean_enrichment_ratio", 0.0)) for x in passed], dtype=np.float32)
    passed_enrich_delta = np.asarray([float(x.get("prog_mean_enrichment_delta", 0.0)) for x in passed], dtype=np.float32)

    accepted_program_count = 0
    no_domain_program_count = 0
    if not program_summary_table.empty and "no_domain_program" in program_summary_table.columns:
        no_domain_program_count = int(np.sum(program_summary_table["no_domain_program"].to_numpy(dtype=bool)))
        accepted_program_count = int(program_summary_table.shape[0] - no_domain_program_count)

    return {
        "sample_id": sample_id,
        "inputs_summary": {
            "n_spots": int(n_spots),
            "n_programs": int(n_programs),
        },
        "segmentation_summary": segmentation_summary,
        "candidate_domain_summary": {
            "candidate_domain_count": int(len(domains)),
            "candidate_spot_count_quantiles": quantiles(spot_counts),
            "candidate_peak_quantiles": quantiles(peak_vals),
            "candidate_prominence_quantiles": quantiles(prominences),
            "candidate_mass_quantiles": quantiles(mass_vals),
            "candidate_mean_enrichment_ratio_quantiles": quantiles(enrich_ratio_vals),
            "candidate_mean_enrichment_delta_quantiles": quantiles(enrich_delta_vals),
            "qc_pass_domain_count": int(len(passed)),
            "qc_pass_spot_count_quantiles": quantiles(passed_spot_counts),
            "qc_pass_peak_quantiles": quantiles(passed_peak),
            "qc_pass_prominence_quantiles": quantiles(passed_prom),
            "qc_pass_mean_enrichment_ratio_quantiles": quantiles(passed_enrich_ratio),
            "qc_pass_mean_enrichment_delta_quantiles": quantiles(passed_enrich_delta),
        },
        "filter_summary": {
            "reject_reason_counts": _reason_counter(domains),
        },
        "program_domain_summary": {
            "accepted_program_count": int(accepted_program_count),
            "no_domain_program_count": int(no_domain_program_count),
            "qc_pass_domains_per_program_quantiles": quantiles(
                program_summary_table["domain_count_pass"].to_numpy(dtype=np.float32)
                if ("domain_count_pass" in program_summary_table.columns and not program_summary_table.empty)
                else np.zeros(0, dtype=np.float32)
            ),
        },
        "merge_summary": merge_summary,
        "acceptance": acceptance_gates,
    }
