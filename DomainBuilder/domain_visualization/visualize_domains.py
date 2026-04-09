from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Keep package imports working when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".mplconfig")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = str(PROJECT_ROOT / ".cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from DomainBuilder.bundle_io import read_json
from DomainBuilder.data_prep import load_domain_visualization_inputs
from DomainBuilder.schema import DomainPipelineConfig

DEFAULT_WORK_DIR = Path("/Users/wuyang/Documents/SC-ST data")
DEFAULT_CANCER = "COAD"
DEFAULT_SAMPLE_ID = "TENX89"


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quick visualization for Domain quality inspection.")
    p.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR), type=str, help="Root directory that contains <cancer>/ST/<sample_id>.")
    p.add_argument("--cancer", default=DEFAULT_CANCER, type=str, help="Cancer cohort name used to resolve bundle paths.")
    p.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID, type=str, help="Sample ID used to resolve bundle paths.")
    p.add_argument(
        "--domain-bundle",
        default=None,
        type=str,
        help="Optional explicit path to domain_bundle directory. Overrides --work-dir/--cancer/--sample-id.",
    )
    p.add_argument(
        "--program-bundle",
        default=None,
        type=str,
        help="Optional explicit path to program_bundle directory. If omitted, inferred from sample path or domain manifest.",
    )
    p.add_argument(
        "--program-id",
        default=None,
        nargs="*",
        type=str,
        help=(
            "Program ID(s) to visualize. Supports one or many values, e.g. "
            "--program-id P0003 P0011. If omitted, visualize all program_ids listed in program_bundle/program_meta.json."
        ),
    )
    p.add_argument(
        "--out",
        default=None,
        type=str,
        help="Output image path. Default: <domain_bundle>/plot/qc_plot.<program_id>.png",
    )
    p.add_argument(
        "--max-domains",
        default=0,
        type=int,
        help="Max number of domains (largest by spot_count) to overlay.",
    )
    p.add_argument(
        "--min-spot-count",
        default=0,
        type=int,
        help="Only overlay domains with spot_count >= this value.",
    )
    p.add_argument(
        "--include-rejected",
        action="store_true",
        help="Try including qc_rejected domains. Note: only domains in membership table can be drawn.",
    )
    p.add_argument("--dpi", default=180, type=int, help="Saved figure DPI.")
    p.add_argument("--point-size", default=20.0, type=float, help="Spot point size.")
    p.add_argument("--point-alpha", default=0.6, type=float, help="Spot alpha on domain assignment panel.")
    p.add_argument(
        "--activation-source",
        default="effective",
        choices=["effective", "raw"],
        help="Activation source for heatmap. 'effective' uses confidence-weighted activation (new default).",
    )
    p.add_argument("--vmin-quantile", default=0.02, type=float, help="Activation lower clip quantile.")
    p.add_argument("--vmax-quantile", default=0.98, type=float, help="Activation upper clip quantile.")
    p.add_argument("--seed", default=2024, type=int, help="Config random seed for loading inputs.")
    return p


def _infer_program_bundle(domain_bundle: Path, manifest: dict) -> Path:
    path = manifest.get("inputs", {}).get("program_bundle_path", "")
    if path:
        p = Path(str(path))
        if p.exists():
            return p
    sibling = domain_bundle.parent / "program_bundle"
    if sibling.exists():
        return sibling
    raise FileNotFoundError(
        "Unable to infer program_bundle path from domain manifest. "
        "Please provide --program-bundle explicitly."
    )


def _resolve_domain_bundle(args: argparse.Namespace) -> Path:
    if args.domain_bundle:
        return Path(args.domain_bundle)
    return Path(args.work_dir) / str(args.cancer) / "ST" / str(args.sample_id) / "domain_bundle"


def _resolve_program_bundle(args: argparse.Namespace, domain_bundle: Path, manifest: dict) -> Path:
    if args.program_bundle:
        return Path(args.program_bundle)
    sample_program_bundle = Path(args.work_dir) / str(args.cancer) / "ST" / str(args.sample_id) / "program_bundle"
    if sample_program_bundle.exists():
        return sample_program_bundle
    return _infer_program_bundle(domain_bundle=domain_bundle, manifest=manifest)


def _normalize_program_id_args(user_program_ids: list[str] | str | None) -> list[str]:
    if not user_program_ids:
        return []
    if isinstance(user_program_ids, str):
        values = [user_program_ids]
    else:
        values = list(user_program_ids)
    out: list[str] = []
    for value in values:
        for chunk in str(value).split(","):
            for item in chunk.split():
                item = item.strip()
                if item:
                    out.append(item)
    return out


def _load_program_ids_from_meta(program_bundle: Path) -> list[str]:
    program_meta_path = program_bundle / "program_meta.json"
    if not program_meta_path.exists():
        return []
    payload = read_json(program_meta_path)
    program_ids = payload.get("program_ids", [])
    if not isinstance(program_ids, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in program_ids:
        pid = str(value).strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def _pick_program_ids(
    program_ids: np.ndarray,
    dense_activation: np.ndarray,
    user_program_ids: list[str] | None,
    default_program_ids: list[str] | None = None,
) -> list[str]:
    known = set(str(x) for x in program_ids.tolist())
    requested = _normalize_program_id_args(user_program_ids)
    if requested:
        missing = [pid for pid in requested if pid not in known]
        if missing:
            raise ValueError(f"program_id not found: {missing[:5]}. Available count={len(known)}.")
        uniq: list[str] = []
        seen: set[str] = set()
        for pid in requested:
            if pid in seen:
                continue
            seen.add(pid)
            uniq.append(pid)
        return uniq

    preferred = _normalize_program_id_args(default_program_ids)
    if preferred:
        available = [pid for pid in preferred if pid in known]
        if available:
            return available

    known_list = [str(x) for x in program_ids.tolist()]
    if known_list:
        return known_list
    if dense_activation.shape[1] == 0:
        raise ValueError("No programs found in activation matrix.")
    mass = np.sum(np.asarray(dense_activation, dtype=np.float32), axis=0)
    return [str(program_ids[int(np.argmax(mass))])]


def _assign_spot_owners(
    n_spots: int,
    program_domains: pd.DataFrame,
    by_domain_spots: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[dict], int]:
    owner = np.full(n_spots, fill_value=-1, dtype=np.int32)
    drawn_domains: list[dict] = []
    overlap_spot_count = 0

    for i, row in enumerate(program_domains.itertuples(index=False)):
        dkey = str(getattr(row, "domain_key"))
        spots = by_domain_spots.get(dkey, None)
        if spots is None or spots.size == 0:
            continue
        spots = np.asarray(spots, dtype=np.int32)
        free_mask = owner[spots] < 0
        overlap_spot_count += int(np.count_nonzero(~free_mask))
        assigned_spots = spots[free_mask]
        if assigned_spots.size == 0:
            continue
        owner[assigned_spots] = int(i)
        drawn_domains.append(
            {
                "domain_idx": int(i),
                "domain_key": dkey,
                "domain_id": str(getattr(row, "domain_id", dkey)),
                "spot_count": int(assigned_spots.size),
            }
        )
    return owner, drawn_domains, overlap_spot_count


def _resolve_output_path(
    domain_bundle: Path,
    out_arg: str | None,
    program_id: str,
    multi_program: bool,
) -> Path:
    safe_pid = str(program_id).replace("/", "_")
    if out_arg:
        out_path = Path(out_arg)
        if not multi_program:
            return out_path
        if out_path.suffix:
            return out_path.with_name(f"{out_path.stem}.{safe_pid}{out_path.suffix}")
        return out_path / f"qc_plot.{safe_pid}.png"
    return domain_bundle / "plot" / f"qc_plot.{safe_pid}.png"


def _render_program_plot(
    sample_id: str,
    domain_bundle: Path,
    domains_df: pd.DataFrame,
    membership_df: pd.DataFrame,
    conf_table: pd.DataFrame,
    coords: np.ndarray,
    dense_activation: np.ndarray,
    program_id: str,
    pid_to_col: dict[str, int],
    args: argparse.Namespace,
    activation_label: str,
    out_arg: str | None,
    multi_program: bool,
) -> None:
    col = int(pid_to_col[program_id])
    act = np.asarray(dense_activation[:, col], dtype=np.float32)

    sel = domains_df["program_seed_id"] == str(program_id)
    if not bool(args.include_rejected) and ("qc_pass" in domains_df.columns):
        sel = sel & domains_df["qc_pass"].astype(bool)
    if int(args.min_spot_count) > 0 and ("spot_count" in domains_df.columns):
        sel = sel & (domains_df["spot_count"].astype(int) >= int(args.min_spot_count))
    program_domains = domains_df.loc[sel].copy()

    if ("prog_seed_sum" in program_domains.columns) and ("spot_count" in program_domains.columns):
        program_domains = program_domains.sort_values(["prog_seed_sum", "spot_count"], ascending=[False, False])
    elif "spot_count" in program_domains.columns:
        program_domains = program_domains.sort_values("spot_count", ascending=False)
    if int(args.max_domains) > 0:
        program_domains = program_domains.head(int(args.max_domains)).copy()

    domain_keys = set(program_domains["domain_key"].astype(str).tolist()) if not program_domains.empty else set()
    mem_sel = membership_df["domain_key"].isin(domain_keys)
    prog_membership = membership_df.loc[mem_sel].copy()
    by_domain_spots = {
        str(k): np.asarray(v["spot_idx"].to_numpy(dtype=np.int32), dtype=np.int32)
        for k, v in prog_membership.groupby("domain_key")
    }

    missing = [k for k in domain_keys if k not in by_domain_spots]
    if missing:
        print(
            f"[warn] program_id={program_id} missing_membership_domains={len(missing)}; "
            "they will not be drawn."
        )

    vq0 = float(max(0.0, min(1.0, args.vmin_quantile)))
    vq1 = float(max(0.0, min(1.0, args.vmax_quantile)))
    if vq1 <= vq0:
        raise ValueError(f"Require vmax_quantile > vmin_quantile, got {vq1} <= {vq0}")
    vmin = float(np.quantile(act, vq0))
    vmax = float(np.quantile(act, vq1))
    if not np.isfinite(vmin):
        vmin = float(np.min(act))
    if not np.isfinite(vmax):
        vmax = float(np.max(act))
    if vmax <= vmin:
        vmax = vmin + 1e-8

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    ax0, ax1 = axes

    sc0 = ax0.scatter(
        coords[:, 0],
        coords[:, 1],
        c=act,
        s=float(args.point_size),
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    fig.colorbar(sc0, ax=ax0, fraction=0.046, pad=0.04, label=activation_label)
    ax0.set_title(f"{sample_id} | {program_id} activation ({activation_label})")
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xticks([])
    ax0.set_yticks([])

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    owner, drawn_domains, overlap_spot_count = _assign_spot_owners(
        n_spots=int(coords.shape[0]),
        program_domains=program_domains,
        by_domain_spots=by_domain_spots,
    )
    neutral = np.array([0.88, 0.88, 0.88, 1.0], dtype=np.float64)
    spot_colors = np.tile(neutral, (coords.shape[0], 1))
    for d in drawn_domains:
        di = int(d["domain_idx"])
        spot_colors[owner == di] = colors[di % len(colors)]

    ax1.scatter(
        coords[:, 0],
        coords[:, 1],
        c=spot_colors,
        s=float(args.point_size),
        linewidths=0,
        alpha=float(max(0.0, min(1.0, args.point_alpha))),
        zorder=1,
    )

    legend_labels: list[str] = []
    legend_handles = []
    for d in drawn_domains:
        di = int(d["domain_idx"])
        c = colors[di % len(colors)]
        d_id = str(d["domain_id"])
        n = int(d["spot_count"])
        drow = program_domains.iloc[di]
        dmass = float(drow.get("prog_seed_sum", 0.0))
        legend_labels.append(f"{d_id} (n={n}, mass={dmass:.1f})")
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=c,
                lw=1.0,
                marker="s",
                markerfacecolor=c,
                markeredgecolor=c,
                markersize=5.5,
            )
        )

    drawn = int(len(drawn_domains))
    pconf_note = ""
    if not conf_table.empty and "program_id" in conf_table.columns:
        conf_row = conf_table.loc[conf_table["program_id"].astype(str) == str(program_id)]
        if not conf_row.empty:
            pconf = float(conf_row.iloc[0].get("program_confidence_weight", np.nan))
            if np.isfinite(pconf):
                pconf_note = f", program_weight={pconf:.3f}"
    ax1.set_title(f"{sample_id} | {program_id} domain assignment (drawn={drawn}{pconf_note})")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xticks([])
    ax1.set_yticks([])

    if legend_handles:
        ax1.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
            title="Domain colors",
        )

    out_path = _resolve_output_path(
        domain_bundle=domain_bundle,
        out_arg=out_arg,
        program_id=program_id,
        multi_program=multi_program,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)

    print(f"[ok] sample={sample_id}")
    print(f"[ok] program_id={program_id}")
    print(f"[ok] activation_source={activation_label}")
    print(f"[ok] selected_domains={len(domain_keys)}, drawn_domains={drawn}")
    if overlap_spot_count > 0:
        print(f"[warn] overlap_spot_count={overlap_spot_count} (assigned by first-seen domain order)")
    print(f"[ok] output={out_path}")


def main() -> None:
    args = _build_cli().parse_args()

    domain_bundle = _resolve_domain_bundle(args)
    manifest_path = domain_bundle / "manifest.json"
    domains_path = domain_bundle / "domains.parquet"
    membership_path = domain_bundle / "domain_spot_membership.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing domain manifest: {manifest_path}")
    if not domains_path.exists():
        raise FileNotFoundError(f"Missing domains parquet: {domains_path}")
    if not membership_path.exists():
        raise FileNotFoundError(f"Missing domain membership parquet: {membership_path}")

    manifest = read_json(manifest_path)
    sample_id = str(manifest.get("sample_id", "unknown_sample"))
    program_bundle = _resolve_program_bundle(args=args, domain_bundle=domain_bundle, manifest=manifest)

    cfg = DomainPipelineConfig(random_seed=int(args.seed))
    payload = load_domain_visualization_inputs(program_bundle_path=program_bundle, cfg=cfg)
    coords = payload["coords"]
    if coords is None:
        raise ValueError(
            "No spatial coordinates available. Cannot draw spatial heatmap/contours. "
            "Check gss manifest h5ad path and spot order mapping."
        )
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[1] < 2:
        raise ValueError(f"coords must have shape [n_spots,2+], got: {coords.shape}")

    dense_activation_effective = np.asarray(payload["dense_activation"], dtype=np.float32)
    dense_activation_raw = np.asarray(payload.get("dense_activation_raw", dense_activation_effective), dtype=np.float32)
    if str(args.activation_source) == "raw":
        dense_activation = dense_activation_raw
        activation_label = "raw_activation"
    else:
        dense_activation = dense_activation_effective
        activation_label = "effective_activation"

    conf_table_path = domain_bundle / "qc_tables" / "program_confidence_weighting.parquet"
    conf_table = pd.read_parquet(conf_table_path) if conf_table_path.exists() else pd.DataFrame()
    program_ids = payload["program_ids"]
    program_ids_from_meta = _load_program_ids_from_meta(program_bundle=program_bundle)
    program_id_list = _pick_program_ids(
        program_ids=program_ids,
        dense_activation=dense_activation,
        user_program_ids=args.program_id,
        default_program_ids=program_ids_from_meta,
    )
    pid_to_col = {str(pid): i for i, pid in enumerate(program_ids.tolist())}

    domains_df = pd.read_parquet(domains_path)
    membership_df = pd.read_parquet(membership_path)
    domains_df["program_seed_id"] = domains_df["program_seed_id"].astype(str)
    membership_df["domain_key"] = membership_df["domain_key"].astype(str)
    if "program_confidence_weight" not in domains_df.columns:
        domains_df["program_confidence_weight"] = 1.0
    if "program_confidence_used" not in domains_df.columns:
        domains_df["program_confidence_used"] = domains_df["program_confidence_weight"].astype(float)
    if "prog_seed_sum" not in domains_df.columns:
        domains_df["prog_seed_sum"] = 0.0

    multi_program = len(program_id_list) > 1
    print(f"[ok] domain_bundle={domain_bundle}")
    print(f"[ok] program_bundle={program_bundle}")
    print(f"[ok] selected_program_count={len(program_id_list)}")
    for program_id in program_id_list:
        _render_program_plot(
            sample_id=sample_id,
            domain_bundle=domain_bundle,
            domains_df=domains_df,
            membership_df=membership_df,
            conf_table=conf_table,
            coords=coords,
            dense_activation=dense_activation,
            program_id=program_id,
            pid_to_col=pid_to_col,
            args=args,
            activation_label=activation_label,
            out_arg=args.out,
            multi_program=multi_program,
        )


if __name__ == "__main__":
    main()
