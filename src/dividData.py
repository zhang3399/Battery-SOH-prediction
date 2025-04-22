import pandas as pd

"""混合式数据集划分
参数:
   df (DataFrame): 完整数据集
   div_batt (list): 需要分拆的电池ID列表
   div_name(str): 电池ID属性名
   test_new_batt (str): 完整保留测试的电池ID
   test_ratio (float): 分拆比例
返回:
   train_df, val_df, test_df
"""


def create_hybrid_split(df, div_batt, div_name, test_ratio=0.2, test_new_batt=None):
    final_test_batt = test_new_batt
    final_test_df = df[df[div_name] == final_test_batt]

    # 其他电池内部按时间划分
    train_dfs, val_dfs = [], []
    for batt_id in div_batt:
        batt_data = df[df[div_name] == batt_id]
        split_idx = int(len(batt_data) * (1 - test_ratio))

        train_part = batt_data.iloc[:split_idx]
        val_part = batt_data.iloc[split_idx:]

        train_dfs.append(train_part)
        val_dfs.append(val_part)

    return pd.concat(train_dfs), pd.concat(val_dfs), final_test_df