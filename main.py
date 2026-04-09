import scanpy as sc

# 读取 h5ad 文件
adata = sc.read_h5ad("/Users/wuyang/Documents/SC-ST data/PRAD/ST/INT25.h5ad")

# 查看基本信息
print(adata)
print(adata.obsm["spatial"])