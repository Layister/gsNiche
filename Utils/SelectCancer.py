
import json
import pandas as pd


# 读取Excel文件
file_path = '/Users/wuyang/Documents/MyPaper/3/dataset/HEST_v1_1_0.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# 筛选癌症样本并处理数据
cancer_samples = df[
    (df['disease_state'] == 'Cancer') & # 排除不是癌症的数据
    (df['oncotree_code'].notna()) &     # 排除没有类型的癌症
    (df['oncotree_code'] != '') &       # 排除没有类型的癌症
    (df['oncotree_code'] != 'UNKNOWN')  # 排除未知类型的癌症
    ].copy()

# 创建结果字典
result_dict = {}

# 按物种和癌症类型分组
grouped = cancer_samples.groupby(['species', 'oncotree_code'])

# 遍历每个分组
for (species, cancer_type), group in grouped:
    # 获取该组的所有样本ID（转为列表）
    sample_ids = group['id'].tolist()

    # 如果该物种尚未在字典中，添加新条目
    if species not in result_dict:
        result_dict[species] = {
            'cancer_types': {},
            'all_cancer_types': []
        }

    # 避免重复添加癌症类型到all_cancer_types列表
    if cancer_type not in result_dict[species]['all_cancer_types']:
        result_dict[species]['all_cancer_types'].append(cancer_type)

    # 添加癌症类型和所有样本ID
    result_dict[species]['cancer_types'][cancer_type] = sample_ids

# 将结果保存为JSON文件
with open('/Users/wuyang/Documents/MyPaper/3/dataset/cancer_samples.json', 'w', encoding='utf-8') as f:
    json.dump(result_dict, f, ensure_ascii=False, indent=4)
print("\n结果已保存到cancer_samples.json文件")

# 打印结果
print("物种癌症类型及所有样本ID:")
for species, data in result_dict.items():
    print(f"\n物种: {species}")
    print(f"所有癌症类型: {data['all_cancer_types']}")
    print("每种癌症的样本ID:")
    for cancer_type, sample_ids in data['cancer_types'].items():
        print(f"  {cancer_type}: {sample_ids}")
        print(f"  样本数量: {len(sample_ids)}")  # 显示每个癌症类型的样本数量
