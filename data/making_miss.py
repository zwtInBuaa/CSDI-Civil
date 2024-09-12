import pandas as pd
import numpy as np

# 构建缺失，为随机连续缺失和随机点缺失的综合

# 读取CSV文件
csv_file = '02.csv'
df = pd.read_csv(csv_file, parse_dates=['Datatime'], index_col='Datatime')


# 构造随机缺失值
def introduce_random_missing_values(df, missing_ratio):
    missing_mask = np.random.rand(*df.shape) < missing_ratio
    df_with_missing = df.mask(missing_mask)
    return df_with_missing


# 删除随机点
def delete_random_points(df, delete_ratio):
    total_points = df.shape[0] * df.shape[1]
    delete_count = int(total_points * delete_ratio)

    indices = np.random.choice(total_points, delete_count, replace=False)
    row_indices = indices // df.shape[1]
    col_indices = indices % df.shape[1]

    for row, col in zip(row_indices, col_indices):
        df.iat[row, col] = np.nan

    return df


# 设置缺失比例和删除比例
missing_ratio = 0.00  # 总体缺失比例为30%
delete_ratio = 0.7  # 删除比例为10%

# 引入随机缺失值
df_with_missing = introduce_random_missing_values(df, missing_ratio)

# 删除随机点
df_final = delete_random_points(df_with_missing, delete_ratio)

# 将数据写入CSV文件
output_csv_file = 'miss' + str(int((missing_ratio + delete_ratio) * 100)) + '.csv'
df_final.to_csv(output_csv_file)

print(f'生成的CSV文件 {output_csv_file} 包含随机缺失值和删除的点。')
