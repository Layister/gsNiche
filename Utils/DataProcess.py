import os
import argparse
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm


# work_dir = "/Users/wuyang/Documents/MyPaper/3/gsVis"
def get_self_data_dir(method, work_dir, sample_name, sample_id):
    feather_path = f'{work_dir}/data/{sample_id}/{sample_name}/latent_to_gene/{sample_name}_gene_marker_score.feather'
    h5ad_path = f'{work_dir}/data/{sample_id}/{sample_name}/find_latent_representations/{sample_name}_add_latent.h5ad'
    output_dir = f'{work_dir}/output/{sample_id}/{method}'

    return feather_path, h5ad_path, output_dir


def get_hest_data_dir(method, work_dir, dataset, species, cancer, id):
    feather_path = f'{work_dir}/data/{dataset}/{species}/{cancer}/{id}/latent_to_gene/{id}_gene_marker_score.feather'
    h5ad_path = f'{work_dir}/data/{dataset}/{species}/{cancer}/{id}/find_latent_representations/{id}_add_latent.h5ad'
    output_dir = f'{work_dir}/output/{dataset}/{species}/{cancer}/{id}/{method}'

    return feather_path, h5ad_path, output_dir


def set_chinese_font():
    """设置可用的中文字体"""
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 定义常用中文字体列表
    chinese_fonts = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Microsoft YaHei", "Arial Unicode MS"]

    # 查找第一个可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams["font.family"] = font
            print(f"已设置中文字体: {font}")
            return

    print("警告: 未找到可用的中文字体，中文可能无法正确显示")
    print(f"可用字体示例: {available_fonts[:10]}")


def read_data(feather_path, h5ad_path):
    """读取latent_to_gene.py生成的两个文件"""
    # 读取标记分数文件
    print(f"读取标记分数文件: {feather_path}")
    mk_score_df = pd.read_feather(feather_path)

    # 检查数据结构
    if pd.api.types.is_numeric_dtype(mk_score_df.index):
        print("检测到DataFrame使用数字索引，尝试从第一列获取基因名...")

        # 检查第一列是否包含字符串类型的基因名
        first_column = mk_score_df.iloc[:, 0]
        if pd.api.types.is_string_dtype(first_column):
            # 将第一列设置为索引
            mk_score_df = mk_score_df.set_index(mk_score_df.columns[0])
            print(f"已将第一列 '{mk_score_df.index.name}' 设置为基因名索引")
        else:
            print("警告: 第一列不是字符串类型，无法作为基因名")
            print(f"第一列数据类型: {first_column.dtype}")
            print(f"第一列前5个值: {first_column.head().tolist()}")
            print("继续使用数字索引，但这可能导致后续处理错误")

    # 读取AnnData对象
    print(f"读取AnnData对象: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    # 确保基因名称类型一致（转换为字符串）
    print("确保基因名称类型一致...")
    mk_score_df.index = mk_score_df.index.astype(str)
    adata.var_names = adata.var_names.astype(str)

    # 检查数据一致性
    print(f"标记分数DataFrame中的基因数量: {len(mk_score_df.index)}")
    print(f"AnnData对象中的基因数量: {len(adata.var_names)}")

    # 先比较长度
    if len(mk_score_df.index) != len(adata.var_names):
        print("警告: 标记分数DataFrame和AnnData的基因数量不匹配")

        # 计算共同基因
        common_genes = np.intersect1d(mk_score_df.index, adata.var_names)
        print(f"找到 {len(common_genes)} 个共同基因")

        # 计算各自独有的基因
        only_in_mk_score = np.setdiff1d(mk_score_df.index, adata.var_names)
        only_in_adata = np.setdiff1d(adata.var_names, mk_score_df.index)

        print(f"仅存在于标记分数文件中的基因数量: {len(only_in_mk_score)}")
        print(f"仅存在于AnnData文件中的基因数量: {len(only_in_adata)}")

        # 打印一些示例基因
        if len(only_in_mk_score) > 0:
            print(f"仅存在于标记分数文件中的基因示例: {', '.join(only_in_mk_score[:5])}")
        if len(only_in_adata) > 0:
            print(f"仅存在于AnnData文件中的基因示例: {', '.join(only_in_adata[:5])}")

        # 筛选共同基因
        print("筛选共同基因以继续分析...")
        mk_score_df = mk_score_df.loc[common_genes]
        adata = adata[:, common_genes].copy()
    else:
        # 长度相同但内容可能不同
        if not np.all(mk_score_df.index == adata.var_names):
            print("警告: 标记分数DataFrame和AnnData的基因索引顺序不完全匹配")
            # 找到不匹配的位置
            mismatches = np.where(mk_score_df.index != adata.var_names)[0]
            print(f"找到 {len(mismatches)} 个不匹配的基因位置")
            if len(mismatches) > 0:
                print(f"前5个不匹配的位置: {mismatches[:5]}")
                print(
                    f"对应的基因: {mk_score_df.index[mismatches[0]]} (标记分数) vs {adata.var_names[mismatches[0]]} (AnnData)")

            # 重新排序以匹配
            print("重新排序基因以匹配...")
            adata = adata[:, mk_score_df.index].copy()

    print(f"处理后的数据: 基因数量 = {len(mk_score_df.index)}, 细胞数量 = {adata.shape[0]}")
    return mk_score_df, adata
