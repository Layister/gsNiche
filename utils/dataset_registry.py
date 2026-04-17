from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class DatasetSpec:
    dataset_family: Literal["pan_cancer", "external_dataset"]
    dataset_id: str
    sample_ids: tuple[str, ...]
    species: str = "human"
    tissue: str | None = None
    condition: str | None = None
    modality: str = "ST"
    work_dir: Path = Path("/Users/wuyang/Documents/SC-ST data")
    data_layer: str = "X"
    spatial_obsm_key: str = "spatial"
    spot_id_field: str | None = None
    gene_id_source: str = "var_names"
    expression_state: Literal["raw_counts", "normalized", "log_normalized", "unknown"] = "raw_counts"

    def h5ad_path(self, sample_id: str) -> Path:
        return self.work_dir / self.dataset_id / self.modality / f"{sample_id}.h5ad"

    def out_root(self) -> Path:
        return self.work_dir / self.dataset_id / self.modality

    def sample_dir(self, sample_id: str) -> Path:
        return self.out_root() / sample_id


DATASETS: dict[str, DatasetSpec] = {
    "COAD": DatasetSpec(
        "pan_cancer",
        "COAD",
        ("TENX89", "TENX90", "TENX91", "TENX92"),
        tissue="colon",
        condition="cancer",
    ),
    "PAAD": DatasetSpec(
        "pan_cancer",
        "PAAD",
        ("NCBI569", "NCBI570", "NCBI571", "NCBI572"),
        tissue="pancreas",
        condition="cancer",
    ),
    "IDC": DatasetSpec(
        "pan_cancer",
        "IDC",
        ("NCBI681", "NCBI682", "NCBI683", "TENX13", "TENX14"),
        tissue="breast",
        condition="cancer",
    ),
    "PRAD": DatasetSpec(
        "pan_cancer",
        "PRAD",
        ("INT25", "INT26", "INT27", "INT28", "TENX40", "TENX46"),
        tissue="prostate",
        condition="cancer",
    ),
    "EPM": DatasetSpec(
        "pan_cancer",
        "EPM",
        ("NCBI629", "NCBI630", "NCBI631", "NCBI632", "NCBI633"),
        tissue="ependymoma",
        condition="cancer",
    ),
    "DLPFC": DatasetSpec(
        "external_dataset",
        "DLPFC",
        (
            # "151507",
            # "151508",
            # "151509",
            # "151510",
            # "151669",
            # "151670",
            # "151671",
            # "151672",
            "151673",
            # "151674",
            # "151675",
            # "151676",
        ),
        tissue="dorsolateral prefrontal cortex",
        condition="normal_brain",
    ),
}


def get_dataset(dataset_id: str) -> DatasetSpec:
    key = str(dataset_id).strip().upper()
    try:
        return DATASETS[key]
    except KeyError as exc:
        available = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset_id {dataset_id!r}. Available datasets: {available}") from exc
