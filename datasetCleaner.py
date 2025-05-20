import pandas as pd

# 读取文件
# 假设文件名为 'dataset.txt'
# 使用分号作为分隔符，并假设文件没有标题行
df = pd.read_csv('dataset.txt', sep=';', header=None, names=['Input', 'Output'])

# 删除重复行
df = df.drop_duplicates()

# 保存去重后的数据到新文件（可选）
df.to_csv('dataset_cleaned.txt', sep=';', index=False, header=False)

print("重复行已删除，结果已保存到 'dataset_cleaned.txt'")