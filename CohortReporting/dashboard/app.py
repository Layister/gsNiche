from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import pandas as pd
import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_data_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir / "data"


def _build_triangulation(coords_df: pd.DataFrame) -> mtri.Triangulation | None:
    if coords_df.shape[0] < 3:
        return None
    coords = coords_df.loc[:, ["x", "y"]].to_numpy(dtype=float)
    triangulation = mtri.Triangulation(coords[:, 0], coords[:, 1])
    if triangulation.triangles.size == 0:
        return None
    analyzer = mtri.TriAnalyzer(triangulation)
    mask = analyzer.get_flat_tri_mask(min_circle_ratio=0.01)
    triangles = triangulation.triangles
    p0 = coords[triangles[:, 0]]
    p1 = coords[triangles[:, 1]]
    p2 = coords[triangles[:, 2]]
    edge_len = np.stack(
        [
            np.linalg.norm(p0 - p1, axis=1),
            np.linalg.norm(p1 - p2, axis=1),
            np.linalg.norm(p2 - p0, axis=1),
        ],
        axis=1,
    )
    max_edge = edge_len.max(axis=1)
    finite = max_edge[np.isfinite(max_edge)]
    if finite.size > 0:
        cutoff = float(np.quantile(finite, 0.92))
        mask = np.asarray(mask | (max_edge > cutoff), dtype=bool)
    triangulation.set_mask(mask)
    return triangulation


def _render_domain_spatial_viewer(data_root: Path, sample_id: str, section: dict, app_config: dict) -> None:
    import streamlit as st
    import numpy as np

    viewer = dict(section.get("domain_spatial_viewer", {}) or {})
    viewer_path = data_root / str(viewer.get("data_ref", ""))
    if not viewer_path.exists():
        st.info("Domain Spatial Viewer data is not available for this sample.")
        return
    viewer_df = pd.read_parquet(viewer_path)
    available_programs = [str(x) for x in viewer.get("available_program_ids", [])]
    default_programs = [str(x) for x in viewer.get("default_selected_program_ids", [])]
    selected_programs = st.multiselect(
        "Program selector",
        available_programs,
        default=default_programs,
        max_selections=int(viewer.get("max_program_count", app_config.get("domain_spatial_viewer_max_programs", 5))),
        key=f"domain-viewer-programs-{sample_id}",
    )
    view_mode = st.selectbox(
        "View mode",
        list(viewer.get("view_modes", ["fill", "boundary", "overlay"])),
        index=max(0, list(viewer.get("view_modes", ["fill", "boundary", "overlay"])).index(str(viewer.get("default_view_mode", "overlay")))),
        key=f"domain-viewer-mode-{sample_id}",
    )
    representative_only = st.checkbox("Representative only", value=False, key=f"domain-viewer-representative-{sample_id}")
    opacity = st.slider("Opacity", min_value=0.15, max_value=1.0, value=0.65, step=0.05, key=f"domain-viewer-opacity-{sample_id}")
    if len(selected_programs) >= int(viewer.get("dense_selection_threshold", 4)):
        st.caption(str(viewer.get("dense_selection_note", "")))
    filtered = viewer_df.copy()
    if selected_programs:
        filtered = filtered.loc[filtered["source_program_id"].astype(str).isin(selected_programs)].copy()
    if representative_only and "representative_status" in filtered.columns:
        filtered = filtered.loc[filtered["representative_status"].astype(bool)].copy()
    st.caption(str(viewer.get("footprint_note", app_config.get("domain_spatial_viewer_footprint_note", ""))))
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=180)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    color_map = {program_id: colors[idx % len(colors)] for idx, program_id in enumerate(available_programs)}
    coords_universe = viewer_df.loc[:, ["spot_id", "x", "y"]].dropna().drop_duplicates("spot_id").reset_index(drop=True)
    if not coords_universe.empty:
        ax.scatter(coords_universe["x"], coords_universe["y"], s=4.0, c="#e2e2e2", alpha=0.45, linewidths=0, zorder=1)
    tri = _build_triangulation(coords_universe) if not coords_universe.empty else None
    membership = (
        filtered.loc[:, ["spot_id", "source_program_id", "domain_key", "representative_status", "coverage_burden_share"]].copy()
        if not filtered.empty
        else pd.DataFrame(columns=["spot_id", "source_program_id", "domain_key", "representative_status", "coverage_burden_share"])
    )
    membership["spot_id"] = membership.get("spot_id", pd.Series(dtype=str)).astype(str)
    occupancy_by_program = {
        str(program_id): set(membership.loc[membership["source_program_id"].astype(str) == str(program_id), "spot_id"].astype(str).tolist())
        for program_id in (selected_programs if selected_programs else available_programs)
    }
    spot_index = {str(spot_id): idx for idx, spot_id in enumerate(coords_universe["spot_id"].astype(str).tolist())}
    for program_id in (selected_programs if selected_programs else available_programs):
        active_spots = occupancy_by_program.get(str(program_id), set())
        if not active_spots:
            continue
        color = color_map.get(str(program_id), "#666666")
        mask = np.zeros(coords_universe.shape[0], dtype=float)
        for spot_id in active_spots:
            idx = spot_index.get(str(spot_id))
            if idx is not None:
                mask[idx] = 1.0
        if tri is not None and np.count_nonzero(mask) >= 3:
            if view_mode in ("fill", "overlay"):
                try:
                    ax.tricontourf(tri, mask, levels=[0.5, 1.1], colors=[color], alpha=opacity * 0.28, zorder=2)
                except Exception:
                    pass
            try:
                ax.tricontour(tri, mask, levels=[0.5], colors=[color], linewidths=1.4, zorder=3)
            except Exception:
                pass
        active_df = coords_universe.loc[[spot_index[s] for s in active_spots if s in spot_index], :]
        ax.scatter(active_df["x"], active_df["y"], s=8.0, c=color, alpha=0.08 if view_mode == "boundary" else 0.18, linewidths=0, zorder=2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0.2)
    st.pyplot(fig, use_container_width=True)
    summary_cols = st.columns(4)
    summary_cols[0].metric("Selected programs", len(selected_programs) if selected_programs else len(available_programs))
    displayed_domains = int(filtered["domain_key"].astype(str).nunique()) if not filtered.empty and "domain_key" in filtered.columns else 0
    summary_cols[1].metric("Displayed domains", displayed_domains)
    per_program = (
        filtered.loc[:, ["source_program_id", "domain_key"]]
        .drop_duplicates()
        .groupby("source_program_id")
        .size()
        .sort_values(ascending=False)
        .astype(int)
        .to_dict()
        if not filtered.empty and {"source_program_id", "domain_key"}.issubset(filtered.columns)
        else {}
    )
    summary_cols[2].write("Per-program domain count")
    summary_cols[2].json(per_program)
    overlap_note = "Programs appear spatially separated or lightly overlapping in this filtered view."
    active_programs = [str(x) for x in (selected_programs if selected_programs else available_programs)]
    overlap_pairs: list[str] = []
    if len(active_programs) >= 2:
        overlap_counts: dict[tuple[str, str], int] = {}
        for left_idx, left_program in enumerate(active_programs):
            left_spots = occupancy_by_program.get(left_program, set())
            if not left_spots:
                continue
            for right_program in active_programs[left_idx + 1 :]:
                right_spots = occupancy_by_program.get(right_program, set())
                if not right_spots:
                    continue
                shared = left_spots & right_spots
                if shared:
                    overlap_counts[(left_program, right_program)] = len(shared)
        if overlap_counts:
            ranked = sorted(overlap_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
            overlap_pairs = [f"{left}-{right} ({count} shared spots)" for (left, right), count in ranked[:3]]
            overlap_note = "Visible overlap is present in the selected Programs; shared-spot contour proximity is strongest for: " + ", ".join(overlap_pairs) + "."
        else:
            centroids = (
                filtered.loc[:, ["source_program_id", "x", "y"]]
                .groupby("source_program_id", as_index=False)
                .mean(numeric_only=True)
            )
            if centroids.shape[0] >= 2:
                coords = centroids.loc[:, ["x", "y"]].to_numpy(dtype=float)
                distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
                upper = distances[np.triu_indices_from(distances, k=1)]
                if upper.size > 0 and float(np.nanmin(upper)) < float(np.nanmedian(upper) * 0.6):
                    overlap_note = "Programs are spatially close with limited direct overlap; boundary contact or adjacency is plausible in this filtered view."
    summary_cols[3].write("Overlap / adjacency note")
    summary_cols[3].markdown(overlap_note)
    plt.close(fig)


def _render_niche_spatial_viewer(data_root: Path, sample_id: str, atlas: dict, section: dict, app_config: dict) -> None:
    import streamlit as st

    viewer = dict(section.get("niche_spatial_viewer", {}) or {})
    viewer_path = data_root / str(viewer.get("data_ref", ""))
    if not viewer_path.exists():
        st.info("Niche Spatial Viewer data is not available for this sample.")
        return
    viewer_df = pd.read_parquet(viewer_path)
    available_niches = [str(x) for x in viewer.get("available_niche_ids", []) if str(x)]
    default_niche = str(viewer.get("default_selected_niche_id", "") or (available_niches[0] if available_niches else ""))
    selected_niche = st.selectbox(
        "Niche selector",
        available_niches,
        index=max(0, available_niches.index(default_niche)) if default_niche in available_niches else 0,
        key=f"niche-viewer-selector-{sample_id}",
    ) if available_niches else ""
    view_modes = list(viewer.get("view_modes", ["fill", "boundary", "overlay"]))
    default_view_mode = str(viewer.get("default_view_mode", app_config.get("niche_spatial_viewer_default_view_mode", "boundary")))
    view_mode = st.selectbox(
        "View mode",
        view_modes,
        index=max(0, view_modes.index(default_view_mode)) if default_view_mode in view_modes else 0,
        key=f"niche-viewer-mode-{sample_id}",
    )
    opacity = st.slider("Opacity", min_value=0.15, max_value=1.0, value=0.65, step=0.05, key=f"niche-viewer-opacity-{sample_id}")
    filtered = viewer_df.loc[viewer_df["niche_id"].astype(str) == str(selected_niche)].copy() if selected_niche else viewer_df.head(0).copy()
    st.caption(str(viewer.get("footprint_note", app_config.get("niche_spatial_viewer_footprint_note", ""))))
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=180)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    domain_section = dict((atlas.get("sections", {}) or {}).get("domain", {}) or {})
    program_order = [str(x) for x in ((domain_section.get("domain_spatial_viewer", {}) or {}).get("available_program_ids", []) or []) if str(x)]
    if not program_order:
        program_order = sorted(filtered.get("source_program_id", pd.Series(dtype=str)).astype(str).dropna().unique().tolist())
    color_map = {program_id: colors[idx % len(colors)] for idx, program_id in enumerate(program_order)}
    if "is_sample_background_spot" in viewer_df.columns:
        background_mask = viewer_df["is_sample_background_spot"].fillna(False).astype(bool)
    else:
        background_mask = pd.Series(False, index=viewer_df.index)
    background_df = viewer_df.loc[background_mask, ["spot_id", "x", "y"]].dropna().drop_duplicates("spot_id")
    coords_universe = background_df.reset_index(drop=True) if not background_df.empty else viewer_df.loc[:, ["spot_id", "x", "y"]].dropna().drop_duplicates("spot_id").reset_index(drop=True)
    if not coords_universe.empty:
        ax.scatter(coords_universe["x"], coords_universe["y"], s=4.0, c="#e2e2e2", alpha=0.35, linewidths=0, zorder=1)
    tri = _build_triangulation(coords_universe) if not coords_universe.empty else None
    spot_index = {str(spot_id): idx for idx, spot_id in enumerate(coords_universe["spot_id"].astype(str).tolist())}
    if "is_sample_background_spot" in filtered.columns:
        active_mask = ~filtered["is_sample_background_spot"].fillna(False).astype(bool)
    else:
        active_mask = pd.Series(True, index=filtered.index)
    active_members = filtered.loc[active_mask].copy()
    for domain_key, sub in active_members.groupby("domain_key", dropna=False):
        if sub.empty:
            continue
        program_id = str(sub["source_program_id"].astype(str).iloc[0]) if "source_program_id" in sub.columns else ""
        color = color_map.get(program_id, "#666666")
        active_spots = set(sub["spot_id"].astype(str).tolist())
        mask = np.zeros(coords_universe.shape[0], dtype=float)
        for spot_id in active_spots:
            idx = spot_index.get(str(spot_id))
            if idx is not None:
                mask[idx] = 1.0
        is_backbone = bool(sub.get("is_backbone_member", pd.Series([False])).astype(bool).any())
        linewidth = 1.8 if is_backbone else 1.1
        if tri is not None and np.count_nonzero(mask) >= 3:
            if view_mode in ("fill", "overlay"):
                try:
                    ax.tricontourf(tri, mask, levels=[0.5, 1.1], colors=[color], alpha=opacity * 0.28, zorder=2)
                except Exception:
                    pass
            if view_mode in ("boundary", "overlay"):
                try:
                    ax.tricontour(tri, mask, levels=[0.5], colors=[color], linewidths=linewidth, zorder=3)
                except Exception:
                    pass
        active_df = coords_universe.loc[[spot_index[s] for s in active_spots if s in spot_index], :]
        ax.scatter(active_df["x"], active_df["y"], s=7.0, c=color, alpha=0.04 if view_mode == "boundary" else 0.12, linewidths=0, zorder=2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0.2)
    left_col, right_col = st.columns([2.2, 1.0])
    with left_col:
        st.pyplot(fig, use_container_width=True)
    with right_col:
        niche_summary = filtered.iloc[0] if not filtered.empty else pd.Series(dtype=object)
        st.markdown("**Niche summary**")
        if not active_members.empty:
            niche_summary = active_members.iloc[0]
            summary_df = pd.DataFrame(
                [
                    {"field": "niche_id", "value": str(niche_summary.get("niche_id", ""))},
                    {"field": "member_count", "value": int(pd.to_numeric(niche_summary.get("member_count", 0), errors="coerce"))},
                    {"field": "confidence", "value": f"{float(pd.to_numeric(niche_summary.get('niche_confidence', 0.0), errors='coerce')):.3f}"},
                    {"field": "dominant_pair", "value": str(niche_summary.get("dominant_contact_pair", "") or "NA")},
                    {"field": "secondary_pair", "value": str(niche_summary.get("secondary_contact_pair", "") or "NA")},
                    {"field": "contact_hint", "value": str(niche_summary.get("contact_structure_hint", "") or "NA")},
                ]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True, height=245)
        st.markdown("**Member domains**")
        if not active_members.empty:
            member_df = (
                active_members.loc[
                    :,
                    [
                        "domain_id",
                        "source_program_id",
                        "leading_component_anchor",
                        "leading_role_anchor",
                        "domain_level_confidence",
                        "spot_count",
                    ],
                ]
                .drop_duplicates("domain_id")
                .sort_values(["source_program_id", "domain_id"], ascending=[True, True])
                .reset_index(drop=True)
            )
            st.dataframe(member_df, use_container_width=True, height=260)
            with st.expander("More domain details", expanded=False):
                detail_df = (
                    active_members.loc[
                        :,
                        [
                            "domain_id",
                            "source_program_id",
                            "component_strip",
                            "role_strip",
                            "geo_boundary_ratio",
                            "mixed_neighbor_fraction",
                        ],
                    ]
                    .drop_duplicates("domain_id")
                    .sort_values(["source_program_id", "domain_id"], ascending=[True, True])
                    .reset_index(drop=True)
                )
                st.dataframe(detail_df, use_container_width=True, height=220)
    plt.close(fig)


def main() -> None:
    try:
        import streamlit as st
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Streamlit is required to run the CohortReporting dashboard. "
            "Install streamlit and launch with `streamlit run app.py`."
        ) from exc

    data_root = _coerce_data_root()
    app_config = _load_json(data_root / "app_config.json")
    sample_atlas = _load_json(data_root / "sample_atlas.json")
    cross_sample_synthesis = _load_json(data_root / "cross_sample_synthesis.json")
    program_cross_sample_umap = _load_json(data_root / "cross_sample" / "program_cross_sample_umap.json") if (data_root / "cross_sample" / "program_cross_sample_umap.json").exists() else {}
    program_catalog_df = pd.read_csv(data_root / "program_cross_sample_pattern_catalog.csv")
    domain_catalog_df = pd.read_csv(data_root / "domain_cross_sample_deployment_catalog.csv")
    niche_catalog_df = pd.read_csv(data_root / "niche_cross_sample_structure_catalog.csv")
    chain_catalog_df = pd.read_csv(data_root / "cross_layer_chain_catalog.csv")
    triage_df = pd.read_csv(data_root / "cross_sample_result_triage.csv")

    st.set_page_config(page_title="Cohort Reporting", layout="wide")
    st.title("Representation Cohort Reporting")
    st.caption(str(app_config.get("description", "Read-only report browser; not analysis platform.")))
    page = st.sidebar.radio(
        "Page",
        ["Sample Atlas", "Cross-sample Synthesis"],
    )

    if page == "Sample Atlas":
        sample_options = sorted(sample_atlas.keys())
        default_sample_id = str(app_config.get("default_sample_id") or (sample_options[0] if sample_options else ""))
        sample_id = st.selectbox("Sample", sample_options, index=max(0, sample_options.index(default_sample_id)) if default_sample_id in sample_options else 0)
        atlas = sample_atlas.get(sample_id, {})
        st.subheader(f"Sample {sample_id}")
        st.caption("Primary observation unit is within-sample objects.")
        section_tabs = st.tabs(["Program", "Domain", "Niche"])
        for tab, layer_name in zip(section_tabs, ("program", "domain", "niche")):
            with tab:
                section = dict((atlas.get("sections", {}) or {}).get(layer_name, {}) or {})
                st.markdown(f"**Section question**: {section.get('section_question', '')}")
                st.markdown(f"**Section summary**: {section.get('section_summary_text', '')}")
                st.markdown(f"**Section takeaway**: {section.get('section_takeaway', '')}")
                object_records = list(section.get("object_overview_records", []) or [])
                if layer_name == "program":
                    figure = str((section.get("program_composition_overview", {}) or {}).get("figure_png", ""))
                    if figure:
                        st.image(str(Path(__file__).resolve().parent.parent / figure))
                    if object_records:
                        st.dataframe(pd.DataFrame(object_records), use_container_width=True, height=320)
                if layer_name == "domain":
                    bridge_figure = str((section.get("program_domain_deployment_matrix", {}) or {}).get("figure_png", ""))
                    if bridge_figure:
                        st.image(str(Path(__file__).resolve().parent.parent / bridge_figure))
                    _render_domain_spatial_viewer(data_root, sample_id, section, app_config)
                elif layer_name == "niche":
                    niche_figure = str((section.get("sample_level_niche_assembly_matrix", {}) or {}).get("figure_png", ""))
                    if niche_figure:
                        st.image(str(Path(__file__).resolve().parent.parent / niche_figure))
                    _render_niche_spatial_viewer(data_root, sample_id, atlas, section, app_config)
                    if object_records:
                        st.dataframe(pd.DataFrame(object_records), use_container_width=True, height=360)

    else:
        st.subheader("Cross-sample Synthesis")
        st.caption("These are synthesis-level summaries, not restored cohort-similarity views. Recurrence classes are interpreted within the current cohort scale.")
        synthesis_tabs = st.tabs(["Program Patterns", "Domain Deployment", "Niche Structures", "Cross-layer Chains", "Triage"])
        with synthesis_tabs[0]:
            st.markdown("Program comparability is used here only for visual grouping of Program objects across samples.")
            st.caption("Nearby points indicate higher cross-sample comparability between Program objects; the layout is for visual grouping only and does not define patterns or triage classes.")
            if str(program_cross_sample_umap.get("status", "skipped")) == "ok":
                figure_ref = ((cross_sample_synthesis.get("figures", {}) or {}).get("program_cross_sample_umap", {}) or {}).get("file_png", "")
                if figure_ref:
                    st.image(str(Path(__file__).resolve().parent.parent / figure_ref))
            elif program_cross_sample_umap:
                st.info(str(program_cross_sample_umap.get("skip_reason", "Program cross-sample UMAP was skipped.")))
            st.markdown(cross_sample_synthesis.get("sections", {}).get("program_patterns", {}).get("question", ""))
            st.dataframe(program_catalog_df, use_container_width=True)
        with synthesis_tabs[1]:
            st.markdown(cross_sample_synthesis.get("sections", {}).get("domain_deployment", {}).get("question", ""))
            st.dataframe(domain_catalog_df, use_container_width=True)
        with synthesis_tabs[2]:
            st.markdown(cross_sample_synthesis.get("sections", {}).get("niche_structures", {}).get("question", ""))
            st.dataframe(niche_catalog_df, use_container_width=True)
        with synthesis_tabs[3]:
            st.markdown(cross_sample_synthesis.get("sections", {}).get("cross_layer_chains", {}).get("question", ""))
            st.dataframe(chain_catalog_df, use_container_width=True)
        with synthesis_tabs[4]:
            st.dataframe(triage_df, use_container_width=True)


if __name__ == "__main__":  # pragma: no cover
    main()
