import yaml
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from src.data_preprocessing import create_dataset
from src.dataset import BatteryDataset, create_sequences
from src.model import BatteryLSTM
from src.train import train_model
from src.dividData import create_hybrid_split
import joblib
import os

def plot_results(actuals, predictions, filename):
    """可视化预测结果"""
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.6)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    plt.xlabel('Actual SOH')
    plt.ylabel('Predicted SOH')
    plt.title('Actual vs Predicted SOH')
    plt.savefig(filename)
    plt.close()


def main():
    # 加载配置文件
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # 数据预处理（确保create_dataset已修复RUL计算泄露）
    df = create_dataset()

    # 定义特征列（根据最新特征工程调整）
    feature_cols = [
        # 'cycle',
        # 'charge_time',
        # 'temp_diff',
        # 'cv_time_ratio',
        # 'max_temp_charge',
        'cc_time',

        'current_charge_std',
        'Voltage_measured',

        # 'voltage_min',
        # 'current_discharge_std',
        # 'temp_max_discharge',

        'discharge_duration',
        'capacity'
        # 'energy_throughput',

    ]
    target_col = config['data']['target_col']

    # 按电池ID划分数据集（严格隔离）
    train_batt_ids = ['05', '06', '07']  # 划分训练集、测试集数据
    # test_batt_ids = '18'  # 测试完整新电池数据（可选）

    train_df, test_df, final_test_df = create_hybrid_split(df,
                                                           div_batt=train_batt_ids,
                                                           div_name='battery_id',
                                                           test_ratio=config['data']['test_ratio'],
                                                           # test_new_batt=test_batt_ids
                                                           )

    # test_df = final_test_df
    # 特征标准化（仅在训练数据上拟合）
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # 保存scaler对象
    os.makedirs('models', exist_ok=True)  # 确保目录存在
    joblib.dump(scaler, 'models/scaler.pkl')

    # 创建时序序列（确保按电池独立处理）
    def safe_create_sequences(source_df):
        sequences = []
        for batt_id in source_df['battery_id'].unique():
            batt_data = source_df[source_df['battery_id'] == batt_id]
            batt_seq = create_sequences(
                batt_data,
                config['data']['sequence_length'],
                feature_cols,
                target_col
            )
            sequences.extend(batt_seq)
        return sequences

    train_seq = safe_create_sequences(train_df)
    test_seq = safe_create_sequences(test_df)

    # 创建数据加载器
    train_loader = DataLoader(
        BatteryDataset(train_seq),
        batch_size=config['training']['batch_size'],
        shuffle=True  # 只在训练集shuffle
    )
    val_loader = DataLoader(
        BatteryDataset(test_seq),
        batch_size=config['training']['batch_size'],
        shuffle=False  # 验证集不shuffle
    )

    # 初始化模型
    model = BatteryLSTM(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        output_size=config['model']['output_size']
    )

    # 模型训练
    trained_model, best_metrics = train_model(
        model,
        train_loader,
        val_loader,
        config['training']
    )

    # 加载最佳模型进行预测
    trained_model.load_state_dict(torch.load('models/best_model.pth'))
    trained_model.eval()

    # 收集预测结果
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = trained_model(inputs)
            all_preds.extend(outputs.numpy().flatten())
            all_targets.extend(targets.numpy().flatten())

    # 计算评估指标
    eval_metrics = {
        'MSE': mean_squared_error(all_targets, all_preds),
        'RMSE': np.sqrt(mean_squared_error(all_targets, all_preds)),
        'MAE': mean_absolute_error(all_targets, all_preds),
        'R²': r2_score(all_targets, all_preds)
    }

    # 打印并保存指标
    print("\n======== 最终模型评估指标 ========")
    for metric, value in eval_metrics.items():
        print(f"{metric}: {value:.4f}")

    pd.DataFrame({
        'Metric': eval_metrics.keys(),
        'Value': eval_metrics.values()
    }).to_excel('results/metrics.xlsx', index=False)
    # 可视化结果
    plot_results(all_targets, all_preds, 'results/soh_prediction.png')

    # 时间序列趋势对比
    plt.figure(figsize=(12, 6))
    plt.plot(all_targets, label='Actual', alpha=0.7)
    plt.plot(all_preds, label='Predicted', alpha=0.7)
    plt.xlabel('Cycle Index')
    plt.ylabel('SOH')
    plt.title('Capacity Degradation Prediction')
    plt.legend()
    plt.savefig('results/soh_trend_comparison.png')
    plt.close()


if __name__ == '__main__':
    main()