import torch
from torch.utils.data import Dataset
import numpy as np
# class BatteryDataset(Dataset):
#     def __init__(self, sequences):
#         self.sequences = sequences
#
#     def __len__(self):
#         return len(self.sequences)
#
#     def __getitem__(self, idx):
#         sequence, target = self.sequences[idx]
#         return torch.FloatTensor(sequence), torch.FloatTensor([target])


# def create_sequences(df, seq_length, feature_cols, target_col):
#     sequences = []
#     data = df[feature_cols].values
#     targets = df[target_col].values
#
#     for i in range(len(data) - seq_length):
#         seq = data[i:i + seq_length]
#         target = targets[i + seq_length]
#         sequences.append((seq, target))
#
#     return sequences

class BatteryDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, target, _ = self.sequences[idx]  # 忽略电池ID
        return torch.FloatTensor(sequence), torch.FloatTensor([target])


def create_sequences(df, seq_length, feature_cols, target_col="soh"):
    sequences = []
    data = df[feature_cols].values
    targets = df[target_col].values
    battery_ids = df['battery_id'].values  # 新增电池ID跟踪

    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = targets[i + seq_length]
        battery_id = battery_ids[i + seq_length]  # 记录当前序列对应的电池ID
        sequences.append((seq, target, battery_id))
        if(i==1):
            print("序列sequences：", sequences)

    return sequences




# 在dataset.py中添加专用函数
def create_predict_sequences(df, seq_length, feature_cols):
    """预测专用序列生成（不含target）"""
    sequences = []
    data = df[feature_cols].values

    # 生成连续序列（不跳步）
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i + seq_length]
        sequences.append(seq)

    return np.array(sequences)