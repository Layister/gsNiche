import argparse
import logging
import os
import sys
import threading
import time
from collections import OrderedDict, namedtuple
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from pprint import pprint
from typing import Literal, Union, Optional
import psutil
import pyfiglet


# Global registry to hold functions
version = "1.0.0"
cli_function_registry = OrderedDict()
subcommand = namedtuple("subcommand", ["name", "func", "add_args_function", "description"])


def get_gsMap_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[{asctime}] {levelname:.5s} | {name} - {message}", style="{")
    )
    logger.addHandler(handler)
    return logger


logger = get_gsMap_logger("gsMap")


def track_resource_usage(func):
    """
    Decorator to track resource usage during function execution.
    Logs memory usage, CPU time, and wall clock time at the end of the function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the current process
        process = psutil.Process(os.getpid())

        # Initialize tracking variables
        peak_memory = 0
        cpu_percent_samples = []
        stop_thread = False

        # Function to monitor resource usage
        def resource_monitor():
            nonlocal peak_memory, cpu_percent_samples
            while not stop_thread:
                try:
                    # Get current memory usage in MB
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    peak_memory = max(peak_memory, current_memory)

                    # Get CPU usage percentage
                    cpu_percent = process.cpu_percent(interval=None)
                    if cpu_percent > 0:  # Skip initial zero readings
                        cpu_percent_samples.append(cpu_percent)

                    time.sleep(0.5)
                except Exception:  # Catching all exceptions here because... # noqa: BLE001
                    pass

        # Start resource monitoring in a separate thread
        monitor_thread = threading.Thread(target=resource_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()

        # Get start times
        start_wall_time = time.time()
        start_cpu_time = process.cpu_times().user + process.cpu_times().system

        try:
            # Run the actual function
            result = func(*args, **kwargs)
            return result
        finally:
            # Stop the monitoring thread
            stop_thread = True
            monitor_thread.join(timeout=1.0)

            # Calculate elapsed times
            end_wall_time = time.time()
            end_cpu_time = process.cpu_times().user + process.cpu_times().system

            wall_time = end_wall_time - start_wall_time
            cpu_time = end_cpu_time - start_cpu_time

            # Calculate average CPU percentage
            avg_cpu_percent = (
                sum(cpu_percent_samples) / len(cpu_percent_samples) if cpu_percent_samples else 0
            )

            # Format memory for display
            if peak_memory < 1024:
                memory_str = f"{peak_memory:.2f} MB"
            else:
                memory_str = f"{peak_memory / 1024:.2f} GB"

            # Format times for display
            if wall_time < 60:
                wall_time_str = f"{wall_time:.2f} seconds"
            elif wall_time < 3600:
                wall_time_str = f"{wall_time / 60:.2f} minutes"
            else:
                wall_time_str = f"{wall_time / 3600:.2f} hours"

            if cpu_time < 60:
                cpu_time_str = f"{cpu_time:.2f} seconds"
            elif cpu_time < 3600:
                cpu_time_str = f"{cpu_time / 60:.2f} minutes"
            else:
                cpu_time_str = f"{cpu_time / 3600:.2f} hours"

            # Log the resource usage
            import logging

            logger = logging.getLogger("gsMap")
            logger.info("Resource usage summary:")
            logger.info(f"  • Wall clock time: {wall_time_str}")
            logger.info(f"  • CPU time: {cpu_time_str}")
            logger.info(f"  • Average CPU utilization: {avg_cpu_percent:.1f}%")
            logger.info(f"  • Peak memory usage: {memory_str}")

    return wrapper


# Decorator to register functions for cli parsing
def register_cli(name: str, description: str, add_args_function: Callable) -> Callable:
    def decorator(func: Callable) -> Callable:
        @track_resource_usage  # Use enhanced resource tracking
        @wraps(func)
        def wrapper(*args, **kwargs):
            name.replace("_", " ")
            gsMap_main_logo = pyfiglet.figlet_format(
                "gsMap",
                font="doom",
                width=80,
                justify="center",
            ).rstrip()
            print(gsMap_main_logo, flush=True)
            version_number = "Version: " + version
            print(version_number.center(80), flush=True)
            print("=" * 80, flush=True)
            logger.info(f"Running {name}...")

            # Record start time for the log message
            start_time = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Started at: {start_time}")

            func(*args, **kwargs)

            # Record end time for the log message
            end_time = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Finished running {name} at: {end_time}.")

        cli_function_registry[name] = subcommand(
            name=name, func=wrapper, add_args_function=add_args_function, description=description
        )
        return wrapper

    return decorator


def add_shared_args(parser):
    parser.add_argument(
        "--workdir", type=str, required=True, help="Path to the working directory."
    )
    parser.add_argument("--sample_name", type=str, required=True, help="Name of the sample.")


def add_find_latent_representations_args(parser):
    add_shared_args(parser)
    parser.add_argument(
        "--input_hdf5_path", required=True, type=str, help="Path to the input HDF5 file."
    )
    parser.add_argument(
        "--annotation", required=True, type=str, help="Name of the annotation in adata.obs to use."
    )
    parser.add_argument(
        "--data_layer",
        type=str,
        default="counts",
        required=True,
        help='Data layer for gene expression (e.g., "count", "counts", "log1p").',
    )
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument(
        "--feat_hidden1", type=int, default=256, help="Neurons in the first hidden layer."
    )
    parser.add_argument(
        "--feat_hidden2", type=int, default=128, help="Neurons in the second hidden layer."
    )
    parser.add_argument(
        "--gat_hidden1", type=int, default=64, help="Units in the first GAT hidden layer."
    )
    parser.add_argument(
        "--gat_hidden2", type=int, default=30, help="Units in the second GAT hidden layer."
    )
    parser.add_argument("--p_drop", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--gat_lr", type=float, default=0.001, help="Learning rate for the GAT.")
    parser.add_argument("--n_neighbors", type=int, default=11, help="Number of neighbors for GAT.")
    parser.add_argument(
        "--n_comps", type=int, default=300, help="Number of principal components for PCA."
    )
    parser.add_argument(
        "--weighted_adj", action="store_true", help="Use weighted adjacency in GAT."
    )
    parser.add_argument(
        "--convergence_threshold", type=float, default=1e-4, help="Threshold for convergence."
    )
    parser.add_argument(
        "--hierarchically",
        action="store_true",
        help="Enable hierarchical latent representation finding.",
    )
    parser.add_argument(
        "--pearson_residuals", action="store_true", help="Using the pearson residuals."
    )


def filter_args_for_dataclass(args_dict, data_class: dataclass):
    return {k: v for k, v in args_dict.items() if k in data_class.__dataclass_fields__}


def get_dataclass_from_parser(args: argparse.Namespace, data_class: dataclass):
    remain_kwargs = filter_args_for_dataclass(vars(args), data_class)
    print(f"Using the following arguments for {data_class.__name__}:", flush=True)
    pprint(remain_kwargs, indent=4)
    sys.stdout.flush()
    return data_class(**remain_kwargs)


def add_latent_to_gene_args(parser):
    add_shared_args(parser)

    parser.add_argument(
        "--input_hdf5_path",
        type=str,
        default=None,
        help="Path to the input HDF5 file with latent representations, if --latent_representation is specified.",
    )
    parser.add_argument(
        "--no_expression_fraction", action="store_true", help="Skip expression fraction filtering."
    )
    parser.add_argument(
        "--latent_representation",
        type=str,
        default=None,
        help="Type of latent representation. This should exist in the h5ad obsm.",
    )
    parser.add_argument("--num_neighbour", type=int, default=21, help="Number of neighbors.")
    parser.add_argument(
        "--num_neighbour_spatial", type=int, default=101, help="Number of spatial neighbors."
    )
    parser.add_argument(
        "--homolog_file",
        type=str,
        default=None,
        help="Path to homologous gene conversion file (optional).",
    )
    parser.add_argument(
        "--annotation",
        type=str,
        default=None,
        help="Name of the annotation in adata.obs to use (optional).",
    )


def add_run_all_mode_args(parser):
    add_shared_args(parser)

    # Required paths and configurations
    parser.add_argument(
        "--gsMap_resource_dir",
        type=str,
        required=True,
        help="Directory containing gsMap resources (e.g., genome annotations, LD reference panel, etc.).",
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        required=True,
        help="Path to the input spatial transcriptomics data (H5AD format).",
    )
    parser.add_argument(
        "--annotation", type=str, required=True, help="Name of the annotation in adata.obs to use."
    )
    parser.add_argument(
        "--data_layer",
        type=str,
        default="counts",
        required=True,
        help='Data layer for gene expression (e.g., "count", "counts", "log1p").',
    )

    # GWAS Data Parameters
    parser.add_argument(
        "--trait_name",
        type=str,
        help="Name of the trait for GWAS analysis (required if sumstats_file is provided).",
    )
    parser.add_argument(
        "--sumstats_file",
        type=str,
        help="Path to GWAS summary statistics file. Either sumstats_file or sumstats_config_file is required.",
    )
    parser.add_argument(
        "--sumstats_config_file",
        type=str,
        help="Path to GWAS summary statistics config file. Either sumstats_file or sumstats_config_file is required.",
    )

    # Homolog Data Parameters
    parser.add_argument(
        "--homolog_file",
        type=str,
        help="Path to homologous gene for converting gene names from different species to human (optional, used for cross-species analysis).",
    )

    # Maximum number of processes
    parser.add_argument(
        "--max_processes",
        type=int,
        default=10,
        help="Maximum number of processes for parallel execution.",
    )

    parser.add_argument(
        "--latent_representation",
        type=str,
        default=None,
        help="Type of latent representation. This should exist in the h5ad obsm.",
    )
    parser.add_argument("--num_neighbour", type=int, default=21, help="Number of neighbors.")
    parser.add_argument(
        "--num_neighbour_spatial", type=int, default=101, help="Number of spatial neighbors."
    )
    parser.add_argument(
        "--pearson_residuals", action="store_true", help="Using the pearson residuals."
    )


def ensure_path_exists(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, Path):
            if result.suffix:
                result.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            else:  # It's a directory path
                result.mkdir(parents=True, exist_ok=True, mode=0o755)
        return result

    return wrapper


@dataclass
class ConfigWithAutoPaths:
    workdir: str
    sample_name: Optional[str]

    def __post_init__(self):
        if self.workdir is None:
            raise ValueError("workdir must be provided.")

    @property
    @ensure_path_exists
    def hdf5_with_latent_path(self) -> Path:
        return Path(
            f"{self.workdir}/{self.sample_name}/find_latent_representations/{self.sample_name}_add_latent.h5ad"
        )

    @property
    @ensure_path_exists
    def mkscore_feather_path(self) -> Path:
        return Path(
            f"{self.workdir}/{self.sample_name}/latent_to_gene/{self.sample_name}_gene_marker_score.feather"
        )


@dataclass
class FindLatentRepresentationsConfig(ConfigWithAutoPaths):
    input_hdf5_path: str
    # output_hdf5_path: str
    annotation: str = None
    data_layer: str = None

    epochs: int = 300
    feat_hidden1: int = 256
    feat_hidden2: int = 128
    feat_cell: int = 3000
    gat_hidden1: int = 64
    gat_hidden2: int = 30
    p_drop: float = 0.1
    gat_lr: float = 0.001
    gcn_decay: float = 0.01
    n_neighbors: int = 11
    label_w: float = 1
    rec_w: float = 1
    input_pca: bool = True
    n_comps: int = 300
    weighted_adj: bool = False
    nheads: int = 3
    var: bool = False
    convergence_threshold: float = 1e-4
    hierarchically: bool = False
    pearson_residuals: bool = False

    def __post_init__(self):
        # self.output_hdf5_path = self.hdf5_with_latent_path
        if self.hierarchically:
            if self.annotation is None:
                raise ValueError("annotation must be provided if hierarchically is True.")
            logger.info(
                "------Hierarchical mode is enabled. This will find the latent representations within each annotation."
            )

        # remind for not providing annotation
        if self.annotation is None:
            logger.warning(
                "annotation is not provided. This will find the latent representations for the whole dataset."
            )
        else:
            logger.info(f"------Find latent representations for {self.annotation}...")


@dataclass
class LatentToGeneConfig(ConfigWithAutoPaths):
    # input_hdf5_with_latent_path: str
    # output_feather_path: str
    input_hdf5_path: Union[str, Path] = None
    no_expression_fraction: bool = False
    latent_representation: str = None
    num_neighbour: int = 21
    num_neighbour_spatial: int = 101
    homolog_file: str = None
    annotation: str = None
    species: str = None

    def __post_init__(self):
        if self.input_hdf5_path is None:
            self.input_hdf5_path = self.hdf5_with_latent_path
            assert self.input_hdf5_path.exists(), (
                f"{self.input_hdf5_path} does not exist. Please run FindLatentRepresentations first."
            )
        else:
            assert Path(self.input_hdf5_path).exists(), f"{self.input_hdf5_path} does not exist."
            # copy to self.hdf5_with_latent_path
            import shutil

            shutil.copy2(self.input_hdf5_path, self.hdf5_with_latent_path)

        if self.latent_representation is not None:
            logger.info(f"Using the provided latent representation: {self.latent_representation}")
        else:
            self.latent_representation = "latent_GVAE"
            logger.info(f"Using default latent representation: {self.latent_representation}")


@dataclass
class RunAllModeConfig(ConfigWithAutoPaths):
    # == ST DATA PARAMETERS (用于生成潜在表征，是mk_score计算的前提) ==
    hdf5_path: str  # 输入空间转录组数据路径，用于生成潜在表征
    annotation: str  # 注释信息，潜在表征生成可能依赖
    data_layer: str = "X"  # 基因表达数据层，潜在表征生成的输入

    # == Find Latent Representation PARAMETERS (潜在表征生成参数，mk_score依赖其输出) ==
    n_comps: int = 300  # PCA主成分数，影响潜在表征
    pearson_residuals: bool = False  # 是否使用pearson残差，影响数据预处理

    # == latent 2 Gene PARAMETERS (直接参与mk_score计算的参数) ==
    latent_representation: str = None  # 潜在表征类型，mk_score计算的核心输入
    num_neighbour: int = 21  # 邻居数量，影响mk_score计算逻辑
    num_neighbour_spatial: int = 101  # 空间邻居数量，影响mk_score计算
    homolog_file: Optional[str] = None  # 同源基因文件，可选，用于基因转换

    max_processes: int = 40

    def __post_init__(self):
        super().__post_init__()  # 保留父类的workdir校验

        # check the existence of the input files
        if not Path(self.hdf5_path).exists():
            raise FileNotFoundError(f"File {self.hdf5_path} does not exist")