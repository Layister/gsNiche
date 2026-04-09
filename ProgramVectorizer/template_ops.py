from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

from .common import adjusted_rand_index, jaccard, quantiles
from .schema import ProgramBootstrapConfig


@dataclass
class TemplateDiscoveryConfig:
    evidence_row_quantile: float = 0.80
    evidence_min_value: float = 1e-4
    evidence_min_genes_per_spot: int = 8
    nmf_k_grid: tuple[int, ...] = (12, 16, 24)
    nmf_max_iter: int = 400
    nmf_alpha_w: float = 0.0
    nmf_alpha_h: float = 1e-4
    nmf_l1_ratio: float = 0.3
    candidate_gene_weight_quantile: float = 0.80
    candidate_gene_fraction: float = 0.20
    candidate_gene_min_count: int = 15
    candidate_max_gene_count: int = 300
    candidate_activation_quantile: float = 0.75
    candidate_min_support_spots: int = 20
    candidate_min_support_frac: float = 0.01
    merge_activation_corr: float = 0.85
    merge_gene_jaccard: float = 0.30
    merge_activation_corr_strict: float = 0.92
    consensus_core_gene_min_run_frac: float = 0.60
    consensus_support_gene_min_run_frac: float = 0.40
    consensus_edge_gene_min_run_frac: float = 0.25
    min_template_run_support_frac: float = 0.20
    min_program_size_genes: int = 20
    max_program_gene_frac_warn: float = 0.20


def _safe_positive_corr(a: np.ndarray, b: np.ndarray) -> float:
    xa = np.asarray(a, dtype=np.float32).reshape(-1)
    xb = np.asarray(b, dtype=np.float32).reshape(-1)
    if xa.size == 0 or xb.size == 0 or xa.size != xb.size:
        return 0.0
    if float(np.std(xa)) <= 1e-8 or float(np.std(xb)) <= 1e-8:
        return 0.0
    corr = float(np.corrcoef(xa, xb)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, 0.0, 1.0))


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    total = float(np.sum(arr))
    if total <= 0:
        return np.zeros(arr.shape[0], dtype=np.float32)
    return (arr / total).astype(np.float32, copy=False)


def _normalized_entropy(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    pos = arr[arr > 0]
    if pos.size == 0:
        return 0.0
    probs = pos / max(float(np.sum(pos)), 1e-8)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    max_entropy = float(np.log(max(2, probs.size)))
    return float(entropy / max(max_entropy, 1e-8))


def _top_weight_concentration(values: np.ndarray, top_n: int = 10) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    pos = arr[arr > 0]
    if pos.size == 0:
        return 0.0
    ranked = np.sort(pos)[::-1]
    return float(np.sum(ranked[: min(int(top_n), ranked.size)]) / max(float(np.sum(pos)), 1e-8))


def _row_l1_normalize_csr(matrix: sp.csr_matrix) -> sp.csr_matrix:
    mat = matrix.tocsr().astype(np.float32, copy=True)
    if mat.nnz == 0:
        return mat
    row_sums = np.asarray(mat.sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
    safe = np.maximum(row_sums, 1e-8)
    inv = 1.0 / safe
    for row_idx in range(mat.shape[0]):
        start = int(mat.indptr[row_idx])
        end = int(mat.indptr[row_idx + 1])
        if start < end:
            mat.data[start:end] *= float(inv[row_idx])
    return mat


def encode_template_evidence(
    active_strength: sp.csr_matrix,
    cfg: TemplateDiscoveryConfig,
) -> tuple[sp.csr_matrix, dict]:
    mat = active_strength.tocsr().astype(np.float32, copy=False)
    out_indptr = [0]
    out_indices: list[int] = []
    out_data: list[float] = []
    nnz_per_row: list[int] = []
    row_sums: list[float] = []
    row_saturation_medians: list[float] = []
    row_pre_norm_dynamic_ranges: list[float] = []

    for row_idx in range(mat.shape[0]):
        start = int(mat.indptr[row_idx])
        end = int(mat.indptr[row_idx + 1])
        idx = mat.indices[start:end]
        vals = mat.data[start:end].astype(np.float32, copy=False)
        if vals.size == 0:
            out_indptr.append(len(out_indices))
            nnz_per_row.append(0)
            row_sums.append(0.0)
            row_saturation_medians.append(0.0)
            row_pre_norm_dynamic_ranges.append(0.0)
            continue

        c = float(np.median(vals))
        if c <= 0:
            c = float(np.mean(vals)) if vals.size > 0 else 1.0
        c = max(c, 1e-8)
        vals_sat = (vals / (vals + c)).astype(np.float32, copy=False)
        q = float(np.clip(cfg.evidence_row_quantile, 0.0, 1.0))
        thr = float(max(cfg.evidence_min_value, np.quantile(vals_sat, q)))
        keep = vals_sat >= thr
        min_keep = min(vals_sat.size, max(1, int(cfg.evidence_min_genes_per_spot)))
        if int(np.count_nonzero(keep)) < min_keep:
            order = np.argsort(-vals_sat)[:min_keep]
            keep = np.zeros(vals.size, dtype=bool)
            keep[order] = True

        kept_idx = idx[keep]
        kept_vals = vals_sat[keep]
        kept_sum = float(np.sum(kept_vals))
        if kept_sum > 0:
            kept_vals = (kept_vals / kept_sum).astype(np.float32, copy=False)

        out_indices.extend([int(x) for x in kept_idx.tolist()])
        out_data.extend([float(x) for x in kept_vals.tolist()])
        out_indptr.append(len(out_indices))
        nnz_per_row.append(int(kept_idx.size))
        row_sums.append(float(np.sum(kept_vals)))
        row_saturation_medians.append(float(c))
        row_pre_norm_dynamic_ranges.append(
            float(np.max(vals_sat) - np.min(vals_sat)) if vals_sat.size > 0 else 0.0
        )

    encoded = sp.csr_matrix(
        (
            np.asarray(out_data, dtype=np.float32),
            np.asarray(out_indices, dtype=np.int32),
            np.asarray(out_indptr, dtype=np.int32),
        ),
        shape=mat.shape,
    )
    summary = {
        "matrix_shape": [int(encoded.shape[0]), int(encoded.shape[1])],
        "row_nnz_quantiles": quantiles(np.asarray(nnz_per_row, dtype=np.float32)),
        "row_sum_quantiles": quantiles(np.asarray(row_sums, dtype=np.float32)),
        "row_saturation_median_quantiles": quantiles(np.asarray(row_saturation_medians, dtype=np.float32)),
        "row_pre_norm_dynamic_range_quantiles": quantiles(
            np.asarray(row_pre_norm_dynamic_ranges, dtype=np.float32)
        ),
        "nonzero_frac": float(encoded.nnz / max(1, encoded.shape[0] * encoded.shape[1])),
        "encoding_mode": "row_saturated_salient_gss_template_evidence",
    }
    return encoded, summary


def _select_component_genes(
    weights: np.ndarray,
    cfg: TemplateDiscoveryConfig,
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    pos_idx = np.flatnonzero(w > 0)
    if pos_idx.size == 0:
        return np.empty((0,), dtype=np.int64)
    pos_vals = w[pos_idx]
    thr = float(np.quantile(pos_vals, float(np.clip(cfg.candidate_gene_weight_quantile, 0.0, 1.0))))
    keep = pos_idx[pos_vals >= thr]
    min_keep = min(
        w.size,
        max(
            int(cfg.candidate_gene_min_count),
            int(np.ceil(float(cfg.candidate_gene_fraction) * max(1, pos_idx.size))),
        ),
    )
    if keep.size < min_keep:
        ranked = pos_idx[np.argsort(-w[pos_idx])]
        keep = ranked[:min_keep]
    if int(cfg.candidate_max_gene_count) > 0 and keep.size > int(cfg.candidate_max_gene_count):
        keep = keep[np.argsort(-w[keep])[: int(cfg.candidate_max_gene_count)]]
    return np.asarray(sorted(set(int(x) for x in keep.tolist())), dtype=np.int64)


def _component_support_mask(values: np.ndarray, cfg: TemplateDiscoveryConfig) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    pos = arr[arr > 0]
    if pos.size == 0:
        return np.zeros(arr.shape[0], dtype=bool)
    thr = float(np.quantile(pos, float(np.clip(cfg.candidate_activation_quantile, 0.0, 1.0))))
    return arr >= thr


def _run_nmf_templates(
    evidence_matrix: sp.csr_matrix,
    n_components: int,
    seed: int,
    cfg: TemplateDiscoveryConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    alpha_h_schedule: list[float] = []
    base_alpha_h = max(0.0, float(cfg.nmf_alpha_h))
    alpha_h_schedule.append(float(base_alpha_h))
    if base_alpha_h > 0:
        cur = float(base_alpha_h)
        while cur > 1e-6:
            cur *= 0.1
            alpha_h_schedule.append(float(cur))
        alpha_h_schedule.append(0.0)

    seen: set[float] = set()
    alpha_h_schedule = [x for x in alpha_h_schedule if not (x in seen or seen.add(x))]

    last_W = None
    last_H = None
    last_alpha_h = float(base_alpha_h)
    for alpha_h in alpha_h_schedule:
        model = NMF(
            n_components=int(n_components),
            init="nndsvda",
            random_state=int(seed),
            solver="cd",
            beta_loss="frobenius",
            max_iter=int(cfg.nmf_max_iter),
            alpha_W=float(cfg.nmf_alpha_w),
            alpha_H=float(alpha_h),
            l1_ratio=float(cfg.nmf_l1_ratio),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            W = np.asarray(model.fit_transform(evidence_matrix), dtype=np.float32)
        H = np.asarray(model.components_, dtype=np.float32)
        last_W = W
        last_H = H
        last_alpha_h = float(alpha_h)
        if int(np.count_nonzero(H > 0)) > 0:
            return W, H, float(alpha_h)

    assert last_W is not None
    assert last_H is not None
    return last_W, last_H, float(last_alpha_h)


def discover_template_programs(
    active_strength: sp.csr_matrix,
    gene_ids: np.ndarray,
    support_spots: np.ndarray,
    idf: np.ndarray,
    blacklist_factor: np.ndarray,
    cfg: TemplateDiscoveryConfig,
    bootstrap_cfg: ProgramBootstrapConfig,
    random_seed: int,
) -> dict:
    encoded_matrix, encoding_summary = encode_template_evidence(active_strength=active_strength, cfg=cfg)
    feature_weights = np.asarray(idf, dtype=np.float32).reshape(-1) * np.asarray(blacklist_factor, dtype=np.float32).reshape(-1)
    feature_weights = np.clip(feature_weights, 0.0, None).astype(np.float32, copy=False)
    if feature_weights.shape[0] != encoded_matrix.shape[1]:
        raise ValueError("feature weight length does not match encoded template evidence matrix")
    encoded_matrix = (encoded_matrix @ sp.diags(feature_weights.astype(np.float32, copy=False), format="csr")).tocsr()
    encoded_matrix = _row_l1_normalize_csr(encoded_matrix)
    n_spots, n_genes = encoded_matrix.shape
    total_runs = max(1, int(bootstrap_cfg.bootstrap_B))
    k_cap = max(2, min(int(n_spots), int(n_genes)))
    k_grid = tuple(sorted({int(k) for k in cfg.nmf_k_grid if 1 < int(k) <= k_cap}))
    if not k_grid:
        k_grid = (min(8, k_cap),)

    raw_candidates: list[dict] = []
    component_qc_records: list[dict] = []
    consensus_qc_records: list[dict] = []
    repeat_program_summaries: list[list[dict]] = []
    run_labels: list[np.ndarray] = []
    stable_consecutive = 0
    prev_labels: np.ndarray | None = None
    runs_done = 0

    for run_idx in range(total_runs):
        k = int(k_grid[run_idx % len(k_grid)])
        W, H, alpha_h_used = _run_nmf_templates(
            evidence_matrix=encoded_matrix,
            n_components=k,
            seed=int(random_seed + 17 * run_idx),
            cfg=cfg,
        )
        run_candidates: list[dict] = []
        run_scores = np.zeros((n_spots, 0), dtype=np.float32)

        for comp_idx in range(k):
            h_row = np.asarray(H[comp_idx], dtype=np.float32).reshape(-1)
            gene_idx = _select_component_genes(h_row, cfg=cfg)
            support_mask = _component_support_mask(W[:, comp_idx], cfg=cfg)
            support_count = int(np.count_nonzero(support_mask))
            min_support = max(int(cfg.candidate_min_support_spots), int(np.ceil(cfg.candidate_min_support_frac * n_spots)))
            component_reject_reason = ""
            component_birth_reason = ""
            if gene_idx.size == 0:
                component_reject_reason = "empty_gene_panel_after_weight_filter"
            elif support_count < min_support:
                component_reject_reason = "support_spots_below_min"
            else:
                component_birth_reason = "passed_component_gene_and_usage_filters"
            component_qc_records.append(
                {
                    "run_idx": int(run_idx),
                    "k": int(k),
                    "component_idx": int(comp_idx),
                    "nmf_alpha_h_used": float(alpha_h_used),
                    "component_h_sparsity": float(1.0 - (np.count_nonzero(h_row > 0) / max(1, h_row.size))),
                    "component_entropy": float(_normalized_entropy(h_row)),
                    "top_gene_weight_concentration": float(_top_weight_concentration(h_row)),
                    "selected_gene_count": int(gene_idx.size),
                    "usage_support_spot_count": int(support_count),
                    "usage_support_spot_frac": float(support_count / max(1, n_spots)),
                    "candidate_birth_reason": str(component_birth_reason),
                    "candidate_reject_reason": str(component_reject_reason),
                }
            )
            if component_reject_reason:
                continue

            weights = _normalize(h_row[gene_idx] * idf[gene_idx] * blacklist_factor[gene_idx])
            ranked_gene_indices = [int(gene_idx[pos]) for pos in np.argsort(-weights)]
            gene_set = {str(gene_ids[int(g)]) for g in ranked_gene_indices}
            top_sets = {}
            for n in tuple(sorted({int(x) for x in bootstrap_cfg.top_ns if int(x) > 0})) or (20, 50):
                top_sets[int(n)] = {
                    str(gene_ids[int(g)]) for g in ranked_gene_indices[: min(len(ranked_gene_indices), int(n))]
                }

            spot_support_frac = float(support_count / max(1, n_spots))
            focus = float(np.sum(weights**2))
            weight_map = {int(gene_idx[pos]): float(weights[pos]) for pos in range(gene_idx.size)}
            raw_candidate = {
                "run_idx": int(run_idx),
                "k": int(k),
                "nmf_alpha_h_used": float(alpha_h_used),
                "component_idx": int(comp_idx),
                "gene_indices": set(int(x) for x in ranked_gene_indices),
                "ranked_gene_indices": ranked_gene_indices,
                "weights_full": weight_map,
                "usage_values": np.asarray(W[:, comp_idx], dtype=np.float32).reshape(-1),
                "support_mask": np.asarray(support_mask, dtype=bool),
                "spot_support_frac": float(spot_support_frac),
                "focus": float(focus),
                "gene_set": gene_set,
                "ranked_gene_ids": [str(gene_ids[int(g)]) for g in ranked_gene_indices],
                "top_sets": top_sets,
            }
            raw_candidates.append(raw_candidate)
            run_candidates.append(raw_candidate)

        if run_candidates:
            usage_mat = np.column_stack([cand["usage_values"] for cand in run_candidates]).astype(np.float32, copy=False)
            labels = np.argmax(usage_mat, axis=1).astype(np.int32, copy=False)
        else:
            labels = np.full(n_spots, -1, dtype=np.int32)
        run_labels.append(labels)
        repeat_program_summaries.append(
            [
                {
                    "gene_set": set(item["gene_set"]),
                    "ranked_gene_ids": list(item["ranked_gene_ids"]),
                    "top_sets": {int(k): set(v) for k, v in item["top_sets"].items()},
                }
                for item in run_candidates
            ]
        )
        runs_done += 1
        if prev_labels is not None:
            ari = float(adjusted_rand_index(prev_labels, labels))
            stable_consecutive = stable_consecutive + 1 if ari >= float(bootstrap_cfg.early_stop_label_match) else 0
        prev_labels = labels
        if (
            bool(bootstrap_cfg.early_stop_enabled)
            and runs_done >= max(2, int(bootstrap_cfg.early_stop_min_rounds))
            and stable_consecutive >= max(1, int(bootstrap_cfg.early_stop_consecutive_rounds))
        ):
            break

    if not raw_candidates:
        return {
            "candidate_program_payload": [],
            "repeat_program_summaries": repeat_program_summaries,
            "assignment_label_stability": {"metric": "ARI", "quantiles": quantiles(np.asarray([], dtype=np.float32))},
            "template_summary": {
                "discovery_mode": "gss_template_learning",
                "encoded_evidence": encoding_summary,
                "feature_weighting_mode": "column_downweight_before_nmf",
                "runs_done": int(runs_done),
                "raw_candidate_count": 0,
                "consensus_template_count": 0,
                "stability_mode": "multi_run_consensus",
                "assignment_label_stability": {"metric": "ARI", "quantiles": quantiles(np.asarray([], dtype=np.float32))},
            },
        }

    clusters: list[list[int]] = []
    rep_indices: list[int] = []
    for cand_idx, cand in enumerate(raw_candidates):
        best_cluster = None
        best_score = -1.0
        for cluster_idx, rep_idx in enumerate(rep_indices):
            rep = raw_candidates[rep_idx]
            act_corr = _safe_positive_corr(cand["usage_values"], rep["usage_values"])
            gene_ov = jaccard(set(cand["gene_indices"]), set(rep["gene_indices"]))
            match = act_corr >= float(cfg.merge_activation_corr_strict) or (
                act_corr >= float(cfg.merge_activation_corr) and gene_ov >= float(cfg.merge_gene_jaccard)
            )
            if match:
                score = 0.65 * act_corr + 0.35 * gene_ov
                if score > best_score:
                    best_score = score
                    best_cluster = cluster_idx
        if best_cluster is None:
            rep_indices.append(cand_idx)
            clusters.append([cand_idx])
        else:
            clusters[best_cluster].append(cand_idx)

    candidate_program_payload: list[dict] = []
    cluster_run_labels = [np.zeros((n_spots,), dtype=np.float32) for _ in clusters]

    for cluster_idx, member_idx in enumerate(clusters):
        members = [raw_candidates[i] for i in member_idx]
        run_support = len({int(m["run_idx"]) for m in members})
        run_support_frac = float(run_support / max(1, runs_done))
        if run_support_frac < float(cfg.min_template_run_support_frac):
            consensus_qc_records.append(
                {
                    "cluster_idx": int(cluster_idx),
                    "member_count": int(len(members)),
                    "run_support_count": int(run_support),
                    "run_support_frac": float(run_support_frac),
                    "retained_gene_count": 0,
                    "candidate_birth_reason": "",
                    "candidate_reject_reason": "run_support_below_min",
                }
            )
            continue

        gene_counts: dict[int, int] = {}
        gene_weight_sum: dict[int, float] = {}
        avg_usage = np.zeros((n_spots,), dtype=np.float32)
        for member in members:
            avg_usage += np.asarray(member["usage_values"], dtype=np.float32)
            for g, w in member["weights_full"].items():
                gene_counts[int(g)] = gene_counts.get(int(g), 0) + 1
                gene_weight_sum[int(g)] = gene_weight_sum.get(int(g), 0.0) + float(w)
        avg_usage /= max(1, len(members))

        core_min = float(cfg.consensus_core_gene_min_run_frac)
        support_min = float(
            np.clip(
                max(cfg.consensus_support_gene_min_run_frac, cfg.consensus_edge_gene_min_run_frac),
                0.0,
                1.0,
            )
        )
        edge_min = float(cfg.consensus_edge_gene_min_run_frac)
        retained = [g for g, c in gene_counts.items() if (c / max(1, run_support)) >= edge_min]
        if not retained:
            retained = sorted(gene_weight_sum, key=lambda g: gene_weight_sum[g], reverse=True)[: int(cfg.candidate_gene_min_count)]
        retained = sorted(
            retained,
            key=lambda g: gene_weight_sum.get(int(g), 0.0) / max(1, gene_counts.get(int(g), 1)),
            reverse=True,
        )
        max_keep = int(cfg.candidate_max_gene_count)
        if max_keep > 0:
            retained = retained[:max_keep]
        if len(retained) < int(cfg.candidate_gene_min_count):
            consensus_qc_records.append(
                {
                    "cluster_idx": int(cluster_idx),
                    "member_count": int(len(members)),
                    "run_support_count": int(run_support),
                    "run_support_frac": float(run_support_frac),
                    "retained_gene_count": int(len(retained)),
                    "candidate_birth_reason": "",
                    "candidate_reject_reason": "retained_gene_count_below_min",
                }
            )
            continue

        avg_gene_weight = np.asarray(
            [gene_weight_sum[int(g)] / max(1, gene_counts.get(int(g), 1)) for g in retained],
            dtype=np.float32,
        )
        weights = _normalize(avg_gene_weight)
        ranked_gene_indices = [int(retained[pos]) for pos in np.argsort(-weights)]
        spot_support_mask = _component_support_mask(avg_usage, cfg=cfg)
        spot_support_frac = float(np.mean(spot_support_mask)) if spot_support_mask.size > 0 else 0.0
        focus = float(np.sum(weights**2))
        evidence_score = float(np.clip(0.40 * run_support_frac + 0.35 * spot_support_frac + 0.25 * focus, 0.0, 1.0))
        gene_run_support_frac = {
            int(g): float(gene_counts.get(int(g), 0) / max(1, run_support)) for g in ranked_gene_indices
        }
        scaffold_gene_indices = sorted(
            int(g) for g in ranked_gene_indices if gene_run_support_frac.get(int(g), 0.0) >= core_min
        )
        if not scaffold_gene_indices:
            scaffold_gene_indices = ranked_gene_indices[: min(len(ranked_gene_indices), int(cfg.candidate_gene_min_count))]
        scaffold_set = set(scaffold_gene_indices)
        support_gene_indices = [
            int(g)
            for g in ranked_gene_indices
            if int(g) not in scaffold_set and gene_run_support_frac.get(int(g), 0.0) >= support_min
        ]
        support_set = set(support_gene_indices)
        context_edge_gene_indices = [
            int(g) for g in ranked_gene_indices if int(g) not in scaffold_set and int(g) not in support_set
        ]
        consensus_qc_records.append(
            {
                "cluster_idx": int(cluster_idx),
                "member_count": int(len(members)),
                "run_support_count": int(run_support),
                "run_support_frac": float(run_support_frac),
                "retained_gene_count": int(len(ranked_gene_indices)),
                "candidate_birth_reason": "consensus_template_accepted",
                "candidate_reject_reason": "",
            }
        )

        candidate_program_payload.append(
            {
                "program_id": f"P{len(candidate_program_payload):04d}",
                "discovery_mode": "template",
                "gene_indices": set(int(g) for g in ranked_gene_indices),
                "ranked_gene_indices": ranked_gene_indices,
                "weights_full": {int(g): float(w) for g, w in zip(ranked_gene_indices, weights[np.argsort(-weights)].tolist())},
                "usage_values": np.asarray(avg_usage, dtype=np.float32).reshape(-1),
                "support_mask": np.asarray(spot_support_mask, dtype=bool).reshape(-1),
                "role_by_gene": {},
                "gene_role_score": {int(g): float(w) for g, w in zip(ranked_gene_indices, weights[np.argsort(-weights)].tolist())},
                "gene_attachment_score": {int(g): 0.0 for g in ranked_gene_indices},
                "gene_core_contribution": {int(g): 0.0 for g in ranked_gene_indices},
                "idf_by_gene": {int(g): float(idf[int(g)]) for g in ranked_gene_indices},
                "support_spots_by_gene": {int(g): int(support_spots[int(g)]) for g in ranked_gene_indices},
                "intra_degree_by_gene": {int(g): 0.0 for g in ranked_gene_indices},
                "program_size_genes": int(len(ranked_gene_indices)),
                "program_gene_frac": float(len(ranked_gene_indices) / max(1, n_genes)),
                "template_focus_score": float(focus),
                "template_evidence_score": float(evidence_score),
                "template_run_support_frac": float(run_support_frac),
                "template_spot_support_frac": float(spot_support_frac),
                "template_member_count": int(len(members)),
                "template_gene_run_support_frac": gene_run_support_frac,
                "template_scaffold_gene_indices": set(int(g) for g in scaffold_gene_indices),
                "template_support_gene_indices": set(int(g) for g in support_gene_indices),
                "template_context_edge_gene_indices": set(int(g) for g in context_edge_gene_indices),
                "template_scaffold_gene_count": int(len(scaffold_gene_indices)),
                "template_support_gene_count": int(len(support_gene_indices)),
                "template_context_edge_gene_count": int(len(context_edge_gene_indices)),
            }
        )

    program_count = len(candidate_program_payload)
    if program_count == 0:
        return {
            "candidate_program_payload": [],
            "repeat_program_summaries": repeat_program_summaries,
            "assignment_label_stability": {"metric": "ARI", "quantiles": quantiles(np.asarray([], dtype=np.float32))},
            "template_summary": {
                "discovery_mode": "gss_template_learning",
                "encoded_evidence": encoding_summary,
                "feature_weighting_mode": "column_downweight_before_nmf",
                "runs_done": int(runs_done),
                "raw_candidate_count": int(len(raw_candidates)),
                "consensus_template_count": 0,
                "stability_mode": "multi_run_consensus",
                "assignment_label_stability": {"metric": "ARI", "quantiles": quantiles(np.asarray([], dtype=np.float32))},
                "component_qc_records": component_qc_records,
                "consensus_qc_records": consensus_qc_records,
            },
        }

    ari_values: list[float] = []
    if len(run_labels) >= 2:
        for i in range(len(run_labels)):
            for j in range(i + 1, len(run_labels)):
                ari_values.append(float(adjusted_rand_index(run_labels[i], run_labels[j])))

    template_summary = {
        "discovery_mode": "gss_template_learning",
        "encoded_evidence": encoding_summary,
        "feature_weighting_mode": "column_downweight_before_nmf",
        "k_grid": [int(k) for k in k_grid],
        "effective_alpha_h_values": sorted(
            {
                float(item.get("nmf_alpha_h_used", float(cfg.nmf_alpha_h)))
                for item in raw_candidates
            }
        ),
        "runs_done": int(runs_done),
        "raw_candidate_count": int(len(raw_candidates)),
        "consensus_template_count": int(program_count),
        "stability_mode": "multi_run_consensus",
        "assignment_label_stability": {"metric": "ARI", "quantiles": quantiles(np.asarray(ari_values, dtype=np.float32))},
        "run_support_frac_quantiles": quantiles(
            np.asarray([float(item.get("template_run_support_frac", 0.0)) for item in candidate_program_payload], dtype=np.float32)
        ),
        "spot_support_frac_quantiles": quantiles(
            np.asarray([float(item.get("template_spot_support_frac", 0.0)) for item in candidate_program_payload], dtype=np.float32)
        ),
        "template_focus_score_quantiles": quantiles(
            np.asarray([float(item.get("template_focus_score", 0.0)) for item in candidate_program_payload], dtype=np.float32)
        ),
        "template_evidence_score_quantiles": quantiles(
            np.asarray([float(item.get("template_evidence_score", 0.0)) for item in candidate_program_payload], dtype=np.float32)
        ),
        "component_qc_records": component_qc_records,
        "consensus_qc_records": consensus_qc_records,
        "component_reject_reason_counts": {
            str(reason): int(
                sum(1 for row in component_qc_records if str(row.get("candidate_reject_reason", "")) == str(reason))
            )
            for reason in sorted(
                {str(row.get("candidate_reject_reason", "")) for row in component_qc_records if str(row.get("candidate_reject_reason", ""))}
            )
        },
        "consensus_reject_reason_counts": {
            str(reason): int(
                sum(1 for row in consensus_qc_records if str(row.get("candidate_reject_reason", "")) == str(reason))
            )
            for reason in sorted(
                {str(row.get("candidate_reject_reason", "")) for row in consensus_qc_records if str(row.get("candidate_reject_reason", ""))}
            )
        },
    }
    return {
        "candidate_program_payload": candidate_program_payload,
        "repeat_program_summaries": repeat_program_summaries,
        "assignment_label_stability": {"metric": "ARI", "quantiles": quantiles(np.asarray(ari_values, dtype=np.float32))},
        "template_summary": template_summary,
    }
