import torch
import pandas as pd
import yaml
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.model import BatteryLSTM
from src.dataset import create_sequences
import joblib  # 用于保存和加载scaler
import matplotlib.pyplot as plt

class BatteryHealthPredictor:
    def __init__(self, config_path='config.yaml'):
        # 加载配置
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # 初始化模型
        self.model = BatteryLSTM(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout'],
            output_size=self.config['model']['output_size']
        )

        # 加载训练好的模型参数
        self.model.load_state_dict(torch.load('models/best_model.pth'))
        self.model.eval()

        # 加载数据预处理工具
        self.scaler = joblib.load('models/scaler.pkl')  # 需要先保存scaler

        # 定义特征列（必须与训练时一致）
        self.feature_cols = [
            # 'cycle', 'charge_time', 'cc_time', 'cv_time_ratio',
            # 'max_temp_charge', 'current_charge_std', 'voltage_min',
            # 'current_discharge_std', 'temp_max_discharge',
            # 'discharge_duration', 'capacity'
            'current_charge_std',
            'Voltage_measured',
            'discharge_duration',
            'capacity'
        ]

    def _create_predict_sequences(self, df, seq_length):
        """预测序列生成器"""
        data = df[self.feature_cols].values
        sequences = []

        # 滑动窗口生成序列
        for i in range(len(data) - seq_length + 1):
            seq = data[i:i + seq_length]
            sequences.append(seq)

        return np.array(sequences)


    def preprocess_data(self, new_df):
        """数据预处理流程"""
        # 保留原始周期值和SOH
        cleaned_df = new_df.copy()
        cleaned_df['original_cycle'] = cleaned_df['cycle']
        cleaned_df['original_soh'] = cleaned_df['soh']

        # 数据清洗
        cleaned_df = cleaned_df.dropna(subset=self.feature_cols).copy()

        # 标准化处理
        cleaned_df[self.feature_cols] = self.scaler.transform(
            cleaned_df[self.feature_cols]
        )
        return cleaned_df

    def predict(self, input_data):
        """
        执行预测的入口方法
        支持输入类型：
        - 单个电池的DataFrame（需包含完整历史数据）
        - 文件路径（csv/excel）
        """
        if isinstance(input_data, str):
            if input_data.endswith('.csv'):
                df = pd.read_csv(input_data)
            elif input_data.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(input_data)
            else:
                raise ValueError("Unsupported file format")
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError("Input must be DataFrame or file path")

        # 数据预处理
        processed_df = self.preprocess_data(df)

        # 创建预测序列
        sequences = self._create_predict_sequences(
            processed_df,
            self.config['data']['sequence_length']
        )
        # 转换为Tensor
        input_tensor = torch.FloatTensor(sequences)

        # 添加维度验证
        if len(input_tensor.shape) == 2:
            input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度

        # 执行预测
        with torch.no_grad():
            predictions = self.model(input_tensor).numpy().flatten()

        # 生成结果报告
        start_idx = self.config['data']['sequence_length'] - 1
        end_idx = start_idx + len(predictions)

        # 生成结果报告
        report = pd.DataFrame({
            'Cycle': processed_df['original_cycle'].iloc[start_idx:end_idx].values,
            'Actual_SOH': processed_df['original_soh'].iloc[start_idx:end_idx].values,
            'Predicted_SOH': predictions
        })

        return report

if __name__ == '__main__':
    # 初始化预测器
    predictor = BatteryHealthPredictor()

    # 示例1：直接使用DataFrame
    new_data = pd.read_excel('test_battery_data.xlsx', engine='openpyxl')
    results = predictor.predict(new_data)

    # 保存结果
    results.to_excel('predictions.xlsx', index=False)

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.plot(results['Cycle'], results['Actual_SOH'], label='Actual')
    plt.plot(results['Cycle'], results['Predicted_SOH'], label='Predicted')
    plt.title('Battery Health Prediction')
    plt.xlabel('Cycle Number')
    plt.ylabel('State of Health (SOH)')
    plt.legend()
    plt.savefig('prediction_visualization.png')
