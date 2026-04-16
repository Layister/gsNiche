from __future__ import annotations

import json
import shutil
from pathlib import Path
import pandas as pd

from .dashboard_data_builder import build_dashboard_payloads
from .cross_sample_synthesis import (
    build_cross_layer_chain_catalog,
    build_program_cross_sample_umap_payload,
    build_cross_sample_result_triage,
    build_cross_sample_synthesis_payload,
    build_domain_cross_sample_deployment_catalog,
    build_niche_cross_sample_structure_catalog,
    build_program_cross_sample_pattern_catalog,
    build_sample_chain_support_table,
)
from .loaders import load_cohort_reporting_inputs
from .plotting import (
    build_sample_atlas_figures,
    plot_program_cross_sample_umap,
)
from .sample_atlas import build_sample_atlas_payload
from .schema import CohortReportingConfig
from .tables import (
    TABLE_NOTES,
    dataframe_to_markdown,
)
from .writers import (
    build_manifest,
    copy_tree_contents,
    ensure_output_dirs,
    write_csv,
    write_json,
    write_parquet,
    write_text,
)


def _build_domain_spatial_viewer_df(bundle, atlas_payload: dict, sample_chains_df: pd.DataFrame) -> pd.DataFrame:
    domain_section = dict((atlas_payload.get("sections", {}) or {}).get("domain", {}) or {})
    bridge_records = pd.DataFrame((domain_section.get("program_domain_deployment_matrix", {}) or {}).get("records", []) or [])
    default_programs = (
        list((domain_section.get("domain_spatial_viewer", {}) or {}).get("default_selected_program_ids", []) or [])
    )
    domains = bundle.domains_df.copy() if not bundle.domains_df.empty else bundle.domain_profile_df.copy()
    memberships = bundle.domain_spot_membership_df.copy()
    coords = bundle.spot_coords_df.copy()
    if memberships.empty or coords.empty:
        return pd.DataFrame(
            {
                "sample_id": [bundle.sample_id] * len(domains),
                "domain_key": domains.get("domain_key", pd.Series(dtype=str)).astype(str),
                "domain_id": domains.get("domain_id", pd.Series(dtype=str)).astype(str),
                "source_program_id": domains.get("program_seed_id", domains.get("source_program_id", pd.Series(dtype=str))).astype(str),
                "spot_id": "",
                "x": pd.to_numeric(domains.get("geo_centroid_x", 0.0), errors="coerce").fillna(0.0),
                "y": pd.to_numeric(domains.get("geo_centroid_y", 0.0), errors="coerce").fillna(0.0),
                "geo_centroid_x": pd.to_numeric(domains.get("geo_centroid_x", 0.0), errors="coerce").fillna(0.0),
                "geo_centroid_y": pd.to_numeric(domains.get("geo_centroid_y", 0.0), errors="coerce").fillna(0.0),
                "geo_area_est": pd.to_numeric(domains.get("geo_area_est", 0.0), errors="coerce").fillna(0.0),
                "representative_status": False,
                "coverage_burden_share": 0.0,
            }
        )
    if "source_program_id" not in domains.columns:
        domains["source_program_id"] = domains.get("program_seed_id", pd.Series(dtype=str)).astype(str)
    domains["domain_key"] = domains.get("domain_key", pd.Series(dtype=str)).astype(str)
    domains["domain_id"] = domains.get("domain_id", pd.Series(dtype=str)).astype(str)
    coords["spot_id"] = coords["spot_id"].astype(str)
    memberships["spot_id"] = memberships.get("spot_id", pd.Series(dtype=str)).astype(str)
    memberships["domain_key"] = memberships.get("domain_key", pd.Series(dtype=str)).astype(str)
    rep_map = (
        pd.DataFrame(domain_section.get("object_overview_records", []) or [])
        .loc[:, ["domain_key", "representative_status"]]
        .drop_duplicates("domain_key")
        if domain_section.get("object_overview_records")
        else pd.DataFrame(columns=["domain_key", "representative_status"])
    )
    coverage_map = (
        sample_chains_df.groupby("domain_key", dropna=False)["coverage_burden_share"].max().rename("coverage_burden_share").reset_index()
        if "domain_key" in sample_chains_df.columns and not sample_chains_df.empty
        else pd.DataFrame(columns=["domain_key", "coverage_burden_share"])
    )
    support_domains = set(bundle.niche_membership_df.get("domain_key", pd.Series(dtype=str)).astype(str).tolist())
    merged = memberships.merge(coords.loc[:, ["spot_id", "x", "y"]], on="spot_id", how="left")
    merged = merged.merge(
        domains.loc[:, ["domain_key", "domain_id", "source_program_id", "spot_count", "geo_area_est", "geo_elongation", "geo_boundary_ratio", "geo_centroid_x", "geo_centroid_y"]],
        on="domain_key",
        how="left",
    )
    merged = merged.merge(rep_map, on="domain_key", how="left")
    merged = merged.merge(coverage_map, on="domain_key", how="left")
    merged["sample_id"] = bundle.sample_id
    merged["representative_status"] = merged["representative_status"].fillna(False).astype(bool)
    merged["coverage_burden_share"] = pd.to_numeric(merged.get("coverage_burden_share", 0.0), errors="coerce").fillna(0.0)
    merged["linkage_support_flag"] = merged["domain_key"].astype(str).isin(support_domains)
    merged["is_default_program"] = merged["source_program_id"].astype(str).isin(default_programs)
    return merged.sort_values(["source_program_id", "domain_key", "spot_id"]).reset_index(drop=True)
def run_cohort_reporting_pipeline(
    out_root: str | Path,
    sample_ids: list[str] | tuple[str, ...] | None = None,
    cancer_type: str | None = None,
    config: CohortReportingConfig | None = None,
) -> Path:
    cfg = config or CohortReportingConfig()
    inputs = load_cohort_reporting_inputs(out_root=out_root, sample_ids=sample_ids, cancer_type=cancer_type, config=cfg)
    root = Path(out_root) / cfg.output.cohort_dirname
    if root.exists():
        shutil.rmtree(root)
    dirs = ensure_output_dirs(root, cfg)
    module_dashboard_app = Path(__file__).resolve().parent / "dashboard" / "app.py"

    if not inputs.sample_bundles:
        write_text(dirs["dashboard"] / "app.py", module_dashboard_app.read_text(encoding="utf-8"))
        write_json(
            root / cfg.output.manifest_filename,
            build_manifest(
                root,
                cfg,
                [],
                [],
                [],
                [],
                ["dashboard/app.py"],
                [],
                None,
                axis_order={"component": [], "role": []},
                sample_order_source="empty",
            ),
        )
        return root

    available = {bundle.sample_id for bundle in inputs.sample_bundles}
    sample_order = sorted(available)
    sample_order_source = "sorted_sample_id"
    written_tables: list[str] = []
    written_figures: list[str] = []

    sample_atlas_payloads: dict[str, dict] = {}
    sample_atlas_files: list[str] = []
    for bundle in inputs.sample_bundles:
        sample_chains_df = pd.DataFrame(columns=["program_id", "domain_key", "coverage_burden_share"])
        section_artifacts, atlas_payload = build_sample_atlas_payload(
            bundle,
            inputs,
            cfg,
            sample_chains_df,
            figures_root=f"{cfg.output.figures_dirname}/{cfg.output.sample_atlas_dirname}",
            tables_root=f"{cfg.output.tables_dirname}/{cfg.output.sample_atlas_dirname}",
        )
        sample_atlas_payloads[bundle.sample_id] = atlas_payload

    sample_atlas_figure_paths = build_sample_atlas_figures(sample_atlas_payloads, cfg, dirs["figures"])
    sample_atlas_files.extend(
        [
            f"{cfg.output.figures_dirname}/{cfg.output.sample_atlas_dirname}/{path.parent.name}/{path.name}"
            for path in sample_atlas_figure_paths
        ]
    )
    written_figures.extend([])

    program_catalog_df = build_program_cross_sample_pattern_catalog(inputs)
    domain_catalog_df = build_domain_cross_sample_deployment_catalog(inputs, sample_atlas_payloads)
    niche_catalog_df, niche_rows_df = build_niche_cross_sample_structure_catalog(inputs, sample_atlas_payloads)
    chain_catalog_df = build_cross_layer_chain_catalog(inputs, sample_atlas_payloads, domain_catalog_df, niche_rows_df)
    sample_chain_support_df = build_sample_chain_support_table(inputs, sample_atlas_payloads, domain_catalog_df, niche_rows_df)
    triage_df = build_cross_sample_result_triage(len(inputs.sample_bundles), program_catalog_df, domain_catalog_df, niche_catalog_df, chain_catalog_df)
    cross_sample_synthesis = build_cross_sample_synthesis_payload(
        program_catalog_df,
        domain_catalog_df,
        niche_catalog_df,
        chain_catalog_df,
        triage_df,
    )
    cross_sample_synthesis["figures"] = {}
    cross_sample_synthesis["figure_order"] = []
    program_umap_payload = build_program_cross_sample_umap_payload(inputs)

    sample_atlas_payloads = {}
    sample_atlas_files = [
        path
        for path in sample_atlas_files
        if not path.endswith("/niche_assembly_matrix.parquet")
        and not path.endswith("/domain_spatial_viewer.parquet")
        and not path.endswith("/niche_spatial_viewer.parquet")
    ]
    for bundle in inputs.sample_bundles:
        sample_chains_df = (
            sample_chain_support_df.loc[sample_chain_support_df["sample_id"].astype(str) == str(bundle.sample_id)].copy()
            if not sample_chain_support_df.empty
            else pd.DataFrame(columns=["sample_id", "program_id", "domain_key", "niche_id", "cross_layer_chain_id", "coverage_burden_share"])
        )
        section_artifacts, atlas_payload = build_sample_atlas_payload(
            bundle,
            inputs,
            cfg,
            sample_chains_df,
            figures_root=f"{cfg.output.figures_dirname}/{cfg.output.sample_atlas_dirname}",
            tables_root=f"{cfg.output.tables_dirname}/{cfg.output.sample_atlas_dirname}",
        )
        sample_atlas_payloads[bundle.sample_id] = atlas_payload
        table_root = dirs["tables"] / cfg.output.sample_atlas_dirname / str(bundle.sample_id)
        for layer_name, artifacts in section_artifacts.items():
            csv_path = table_root / f"{layer_name}_object_overview.csv"
            md_path = table_root / f"{layer_name}_object_overview.md"
            write_csv(csv_path, artifacts.overview_df)
            write_text(md_path, dataframe_to_markdown(artifacts.overview_df, note=TABLE_NOTES.get("object_overview")))
            sample_atlas_files.extend(
                [
                    f"{cfg.output.tables_dirname}/{cfg.output.sample_atlas_dirname}/{bundle.sample_id}/{csv_path.name}",
                    f"{cfg.output.tables_dirname}/{cfg.output.sample_atlas_dirname}/{bundle.sample_id}/{md_path.name}",
                ]
            )
        niche_matrix_records = pd.DataFrame(atlas_payload.get("niche_assembly_matrix_records", []) or [])
        niche_matrix_path = dirs["data"] / cfg.output.sample_atlas_dirname / str(bundle.sample_id) / "niche_assembly_matrix.parquet"
        write_parquet(niche_matrix_path, niche_matrix_records)
        sample_atlas_files.append(
            f"{cfg.output.dashboard_dirname}/{cfg.output.dashboard_data_dirname}/{cfg.output.sample_atlas_dirname}/{bundle.sample_id}/niche_assembly_matrix.parquet"
        )
        niche_viewer_records = pd.DataFrame(atlas_payload.get("niche_spatial_viewer_records", []) or [])
        niche_viewer_path = dirs["data"] / cfg.output.sample_atlas_dirname / str(bundle.sample_id) / "niche_spatial_viewer.parquet"
        write_parquet(niche_viewer_path, niche_viewer_records)
        sample_atlas_files.append(
            f"{cfg.output.dashboard_dirname}/{cfg.output.dashboard_data_dirname}/{cfg.output.sample_atlas_dirname}/{bundle.sample_id}/niche_spatial_viewer.parquet"
        )
        viewer_df = _build_domain_spatial_viewer_df(bundle, atlas_payload, sample_chains_df)
        viewer_path = dirs["data"] / cfg.output.sample_atlas_dirname / str(bundle.sample_id) / "domain_spatial_viewer.parquet"
        write_parquet(viewer_path, viewer_df)
        sample_atlas_files.append(
            f"{cfg.output.dashboard_dirname}/{cfg.output.dashboard_data_dirname}/{cfg.output.sample_atlas_dirname}/{bundle.sample_id}/domain_spatial_viewer.parquet"
        )

    sample_atlas_figure_paths = build_sample_atlas_figures(sample_atlas_payloads, cfg, dirs["figures"])
    written_figures = [
        f"{cfg.output.figures_dirname}/{cfg.output.sample_atlas_dirname}/{path.parent.name}/{path.name}"
        for path in sample_atlas_figure_paths
    ]
    table_specs = {
        "program_cross_sample_pattern_catalog": program_catalog_df,
        "domain_cross_sample_deployment_catalog": domain_catalog_df,
        "niche_cross_sample_structure_catalog": niche_catalog_df,
        "cross_layer_chain_catalog": chain_catalog_df,
        "cross_sample_result_triage": triage_df,
    }
    for stem, df in table_specs.items():
        csv_path = dirs["tables"] / f"{stem}.csv"
        md_path = dirs["tables"] / f"{stem}.md"
        dashboard_csv_path = dirs["data"] / f"{stem}.csv"
        write_csv(csv_path, df)
        write_text(md_path, dataframe_to_markdown(df))
        write_csv(dashboard_csv_path, df)
        written_tables.extend(
            [
                f"{cfg.output.tables_dirname}/{csv_path.name}",
                f"{cfg.output.tables_dirname}/{md_path.name}",
            ]
        )

    cross_sample_data_dir = dirs["data"] / "cross_sample"
    cross_sample_data_dir.mkdir(parents=True, exist_ok=True)
    write_json(cross_sample_data_dir / "program_cross_sample_umap.json", program_umap_payload)
    if str(program_umap_payload.get("status", "skipped")) == "ok":
        figure_paths = plot_program_cross_sample_umap(program_umap_payload, dirs["figures"] / "cross_sample")
        if figure_paths:
            cross_sample_synthesis["figures"]["program_cross_sample_umap"] = {
                "figure_name": "program_cross_sample_umap",
                "title": "Program cross-sample UMAP",
                "role": "program_comparability_distribution_view",
                "file_png": f"{cfg.output.figures_dirname}/cross_sample/program_cross_sample_umap.png",
                "file_pdf": f"{cfg.output.figures_dirname}/cross_sample/program_cross_sample_umap.pdf",
                "data_ref": f"{cfg.output.dashboard_dirname}/{cfg.output.dashboard_data_dirname}/cross_sample/program_cross_sample_umap.json",
                "description": "Nearby points indicate higher cross-sample comparability between Program objects; the layout is for visual grouping only and does not define patterns or triage classes.",
            }
            cross_sample_synthesis["figure_order"] = ["program_cross_sample_umap"]
            written_figures.extend(
                [
                    f"{cfg.output.figures_dirname}/cross_sample/program_cross_sample_umap.png",
                    f"{cfg.output.figures_dirname}/cross_sample/program_cross_sample_umap.pdf",
                ]
            )

    dashboard_payloads = build_dashboard_payloads(
        inputs=inputs,
        cfg=cfg,
        sample_order=sample_order,
        sample_atlas_payloads=sample_atlas_payloads,
        cross_sample_synthesis=cross_sample_synthesis,
        cross_sample_figures={},
    )
    write_json(dirs["data"] / "sample_atlas.json", dashboard_payloads["sample_atlas"])
    for sample_id, payload in dashboard_payloads["sample_atlas"].items():
        write_json(dirs["data"] / cfg.output.sample_atlas_dirname / f"{sample_id}.json", payload)
    write_json(dirs["data"] / "cross_sample_synthesis.json", dashboard_payloads["cross_sample_synthesis"])
    write_json(dirs["data"] / "app_config.json", dashboard_payloads["app_config"])
    write_text(dirs["dashboard"] / "app.py", module_dashboard_app.read_text(encoding="utf-8"))
    sample_atlas_files.extend(
        [
            f"{cfg.output.dashboard_dirname}/{cfg.output.dashboard_data_dirname}/sample_atlas.json",
            f"{cfg.output.dashboard_dirname}/{cfg.output.dashboard_data_dirname}/cross_sample_synthesis.json",
            f"{cfg.output.dashboard_dirname}/{cfg.output.dashboard_data_dirname}/cross_sample/program_cross_sample_umap.json",
            *[
                f"{cfg.output.dashboard_dirname}/{cfg.output.dashboard_data_dirname}/{stem}.csv"
                for stem in table_specs.keys()
            ],
            *[
                f"{cfg.output.dashboard_dirname}/{cfg.output.dashboard_data_dirname}/{cfg.output.sample_atlas_dirname}/{sample_id}.json"
                for sample_id in dashboard_payloads["sample_atlas"].keys()
            ],
        ]
    )

    manifest = build_manifest(
        root=root,
        cfg=cfg,
        sample_ids=[bundle.sample_id for bundle in inputs.sample_bundles],
        sample_order=sample_order,
        figures=written_figures,
        tables=written_tables,
        dashboard_files=[
            f"{cfg.output.dashboard_dirname}/app.py",
            f"{cfg.output.dashboard_dirname}/{cfg.output.dashboard_data_dirname}/sample_atlas.json",
            f"{cfg.output.dashboard_dirname}/{cfg.output.dashboard_data_dirname}/cross_sample_synthesis.json",
            f"{cfg.output.dashboard_dirname}/{cfg.output.dashboard_data_dirname}/cross_sample/program_cross_sample_umap.json",
            *[
                f"{cfg.output.dashboard_dirname}/{cfg.output.dashboard_data_dirname}/{stem}.csv"
                for stem in table_specs.keys()
            ],
            f"{cfg.output.dashboard_dirname}/{cfg.output.dashboard_data_dirname}/app_config.json",
        ],
        sample_atlas_files=sample_atlas_files,
        default_sample_id=sample_order[0] if sample_order else None,
        axis_order={
            "component": inputs.component_axes,
            "role": inputs.role_axes,
        },
        sample_order_source=sample_order_source,
    )
    write_json(root / cfg.output.manifest_filename, manifest)

    for bundle in inputs.sample_bundles:
        sample_root = bundle.representation_bundle_path / cfg.output.sample_dirname
        if sample_root.exists():
            shutil.rmtree(sample_root)
        copy_tree_contents(root, sample_root)
        sample_app_config = dict(dashboard_payloads["app_config"])
        sample_app_config["default_sample_id"] = bundle.sample_id
        write_json(
            sample_root / cfg.output.dashboard_dirname / cfg.output.dashboard_data_dirname / "app_config.json",
            sample_app_config,
        )
        sample_manifest = dict(manifest)
        sample_manifest["default_sample_id"] = bundle.sample_id
        write_json(sample_root / cfg.output.manifest_filename, sample_manifest)

    return root
