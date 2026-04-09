import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def convert_10x_spatial_to_h5ad(
        h5_file_path: str,
        spatial_dir: str,
        output_file: str,
        genome: str = None,
        gene_symbol_col: str = 'gene_name',  # 或 'name' 取决于HDF5文件结构
        slice_name: str = 'slice1',
        filter_cells: bool = True,
        basic_preprocessing: bool = True,
        include_hvg: bool = True,
        image_key: str = 'hires',   # 或 'lowres'
        annotation_file_path: str = None
) -> ad.AnnData:
    """
    将10x Genomics Visium空间转录组数据转换为h5ad格式

    参数:
    - h5_file_path: HDF5文件路径 (raw_feature_bc_matrix.h5 或 filtered_feature_bc_matrix.h5)
    - spatial_dir: 空间信息目录路径 (包含tissue_hires_image.png等文件)
    - output_file: 输出h5ad文件路径
    - genome: 指定基因组 (如 'GRCh38')，留空则自动检测
    - gene_symbol_col: 基因符号所在的列名
    - slice_name: 组织切片名称
    - filter_cells: 是否过滤低质量细胞
    - basic_preprocessing: 是否进行基础预处理
    - include_hvg: 是否仅保留高变基因
    - image_key: 使用的图像分辨率 ('hires' 或 'lowres')
    - annotation_file_path: 注释文件路径
    """
    # 验证输入路径
    if not os.path.exists(h5_file_path):
        raise FileNotFoundError(f"HDF5文件不存在: {h5_file_path}")

    if not os.path.exists(spatial_dir):
        raise FileNotFoundError(f"空间信息目录不存在: {spatial_dir}")

    # 读取空间转录组数据
    print(f"正在读取空间转录组数据: {h5_file_path}")

    # 使用read_visium读取空间数据，同时指定count_file为HDF5文件
    adata = sc.read_visium(
        path=spatial_dir,
        count_file=h5_file_path,
        genome=genome
    )

    # 确保变量名唯一
    adata.var_names_make_unique()

    # 设置组织切片名称
    adata.uns['spatial'][slice_name] = adata.uns['spatial'].pop(list(adata.uns['spatial'].keys())[0])

    # 设置基因名称作为变量名
    if gene_symbol_col in adata.var.columns:
        adata.var_names = adata.var[gene_symbol_col]
        # 再次确保基因名称唯一
        adata.var_names_make_unique()

    print(f"数据读取完成: {adata.shape}")
    print(f"空间坐标形状: {adata.obsm['spatial'].shape}")

    # 如果提供了注释文件路径，则添加注释信息
    if annotation_file_path:
        try:
            # 读取注释文件
            annotations = pd.read_csv(annotation_file_path, sep='\t', header=None, names=['barcode', 'annotation_type'])
            # 合并新类别
            annotations['combine_type'] = annotations['annotation_type'].apply(
                lambda x: x.rsplit('_', 1)[0] if '_' in x else x
            )
            annotations.set_index('barcode', inplace=True)
            # 将新旧注释信息添加到 adata 的 obs 数据框中
            adata.obs['annotation_type'] = annotations.loc[adata.obs_names, 'annotation_type']
            adata.obs['combine_type'] = annotations.loc[adata.obs_names, 'combine_type']
            print("注释信息已成功添加到 h5ad 文件中")
        except FileNotFoundError:
            print(f"注释文件 {annotation_file_path} 未找到，请检查路径")
        except KeyError:
            print("注释文件中的条形码与 h5ad 文件中的细胞条形码不匹配，请检查数据")

    # 基础质量控制和预处理
    # if filter_cells:
    #     print("正在进行细胞过滤...")
    #
    #     # 计算每个spot的总计数和检测到的基因数
    #     sc.pp.filter_cells(adata, min_genes=200)
    #     sc.pp.filter_genes(adata, min_cells=3)
    #
    #     # 计算线粒体基因比例
    #     adata.var['mt'] = adata.var_names.str.startswith('MT-')  # 人类线粒体基因
    #     sc.pp.calculate_qc_metrics(
    #         adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True
    #     )
    #
    #     # 过滤低质量spots
    #     # 这里的过滤阈值可以根据实际数据分布调整
    #     adata = adata[adata.obs.n_genes_by_counts < 5000, :].copy()  # 过滤高计数异常值
    #     adata = adata[adata.obs.pct_counts_mt < 20, :].copy()  # 过滤线粒体比例过高的spots
    #
    #     print(f"过滤后数据: {adata.shape}")

    # 基础预处理
    # if basic_preprocessing:
    #     print("正在进行基础预处理...")
    #
    #     # 保存原始表达值
    #     adata.raw = adata
    #
    #     # 标准化和对数转换
    #     sc.pp.normalize_total(adata, target_sum=1e4)
    #     sc.pp.log1p(adata)
    #
    #     if include_hvg:
    #         # 识别高变基因
    #         print("正在识别高变基因...")
    #         sc.pp.highly_variable_genes(
    #             adata, min_mean=0.0125, max_mean=3, min_disp=0.5
    #         )
    #
    #         # 仅保留高变基因用于降维和聚类
    #         adata = adata[:, adata.var.highly_variable].copy()
    #         print(f"保留高变基因后: {adata.shape}")
    #
    #     # 回归并缩放
    #     sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    #     sc.pp.scale(adata, max_value=10)

    # 空间降维和聚类
    print("正在进行空间降维和聚类...")

    # PCA降维
    sc.tl.pca(adata, svd_solver='arpack')

    # 构建邻域图
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

    # UMAP可视化
    sc.tl.umap(adata)

    # Leiden聚类
    # sc.tl.leiden(
    #     adata,
    #     resolution=0.5,
    #     flavor='igraph',  # 使用igraph作为后端
    #     directed=False,  # 确保与igraph后端兼容
    #     n_iterations=2  # 使用igraph的默认迭代次数
    # )

    # 保存为h5ad格式
    adata.write_h5ad(output_file)
    print(f"成功保存到: {output_file}")

    return adata


def check_duplicate_genes(adata):
    """检查并报告重复基因名"""
    gene_names = adata.var_names

    # 找出所有重复基因
    duplicates = gene_names[gene_names.duplicated(keep=False)]

    if duplicates.empty:
        print("没有发现重复基因名！！！")
        return

    # 打印重复基因信息
    print(f"\n发现 {len(duplicates)} 个重复基因名:")
    print("重复基因名及其出现次数:")
    print(gene_names.value_counts()[gene_names.value_counts() > 1].sort_index())

    # 获取包含重复基因的详细信息
    duplicate_details = adata.var.loc[duplicates.index].copy()
    duplicate_details['重复计数'] = gene_names.value_counts()[duplicates].values

    return duplicate_details


# 使用示例
if __name__ == "__main__":
    # 替换为实际路径
    work_dir = "/Users/wuyang/Documents/MyPaper/3/gsVis/data/"
    sample_id = "BRCA/"
    h5_file = work_dir + sample_id + "filtered_feature_bc_matrix.h5"
    spatial_dir = work_dir + sample_id  # 包含tissue_hires_image.png等文件的目录
    output_h5ad = work_dir + sample_id + "spatial_transcriptomics.h5ad"
    annotation_file_path = work_dir + sample_id + "truth.txt"

    # 转换空间转录组数据
    adata = convert_10x_spatial_to_h5ad(
        h5_file_path=h5_file,
        spatial_dir=spatial_dir,
        output_file=output_h5ad,
        filter_cells=True,
        basic_preprocessing=True,
        annotation_file_path=annotation_file_path
    )

    # 检查重复基因名
    duplicates = check_duplicate_genes(adata)

    # 可视化空间聚类结果
    try:
        plt.rcParams['figure.dpi'] = 300

        # 在组织切片上可视化聚类结果
        sc.pl.spatial(
            adata,
            color='combine_type',    # annotation_type
            cmap='viridis',
            size=0.8,
            alpha=0.7,
            img_key='hires',
            alpha_img=0.8,
            show=True,
            frameon=False,
            title='Spatial Clustering',
            save='_combine_spatial_clusters.png'
        )

        # UMAP可视化
        sc.pl.umap(
            adata,
            color='combine_type',   # annotation_type
            title='UMAP Clustering',
            save='_combine_clusters.png'
        )

        print("可视化结果已保存")
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        print("数据已成功转换为h5ad格式，可以后续进行可视化")