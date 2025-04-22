import scipy.io
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from collections import defaultdict

def load_battery_data(battery_id):
    """动态构建跨目录路径"""
    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 回溯到父目录（项目根目录）
    project_root = os.path.dirname(current_dir)
    # 构建数据文件路径
    filename = f"B00{battery_id}.mat"
    data_path = os.path.join(project_root, 'data', filename)

    mat = scipy.io.loadmat(
        data_path,
        simplify_cells=True,  # 关键参数：简化数据结构
        verify_compressed_data_integrity=False
    )
    return mat[f'B00{battery_id}']['cycle']  # 直接访问cycle数据

# 改为使用按电池ID组织的缓存
charge_cache = defaultdict(dict)  # 结构: {batt_id: {cycle_num: features}}


def safe_get_mode(voltage_data):
    """安全获取电压众数的函数"""
    try:
        # 转换为numpy数组并移除无效值
        clean_data = voltage_data[~np.isnan(voltage_data)]

        if len(clean_data) == 0:
            print("Warning: 电压数据全为NaN")
            return np.nan

        # 计算众数并验证结果
        mode_result = mode(clean_data)

        # 处理多众数情况：取第一个众数
        if len(mode_result.mode) > 1:
            print(f"发现多个众数: {mode_result.mode}，取第一个值")

        return mode_result.mode[0]

    except Exception as e:
        print(f"计算众数时发生错误: {str(e)}")
        return np.nan


def extract_operation_features(cycle):
    """提取单个操作的特征"""
    features = {}
    data = cycle['data']

    try:
        if cycle['type'] == 'charge':
            # 提取充电特征
            features.update({
                # 'charge_time': data['Time'][-1],
                #
                # 'cv_time_ratio': np.mean(data['Voltage_charge'] >= 4.2),
                # 'max_temp_charge': np.max(data['Temperature_measured']),
                # 'Voltage_charge': np.std(data['Voltage_charge']),
                # 'Current_measured': np.max(data['Current_measured']),

                'cc_time': np.argmax(data['Voltage_measured'] >= 4.2) / len(data['Time']),  # 恒流阶段占比

                'current_charge_std': np.std(data['Current_measured']),
                'Voltage_measured': np.argmax(data['Voltage_measured'] >= 4.2) / len(data['Time']),

            })

        elif cycle['type'] == 'discharge':
            # 提取放电特征
            features.update({
                # 'current_discharge_std': np.std(data['Current_measured']),
                # 'current_discharge_std': np.min(data['Current_measured']),
                # 'Voltage_discharge_std': np.max(data['Voltage_measured']),
                # 'temp_max_discharge': np.max(data['Temperature_measured'])-np.min(data['Temperature_measured']),
                # 'current_discharge_at_load': np.argmax(data['Current_load'] == safe_get_mode(data['Current_load'])) / len(data['Time']),
                # 'Voltage_discharge_at_load':  np.max(data['Voltage_load']),

                'discharge_duration': data['Time'][-1] - data['Time'][0],
                'capacity': data['Capacity']  # 取最终放电容量
            })

    except KeyError as e:
        print(f"特征提取错误: {str(e)}")

    return features


def create_dataset():
    all_data = []
    for batt_id in ['05', '06', '07', '18']:  # 遍历所有电池
        print(f"Processing battery {batt_id}...")
        cycles = load_battery_data(batt_id)  # 假设已实现数据加载

        current_cycle_num = 0  # 独立维护每个电池的周期计数器
        last_charge_features = {}  # 存储最近一次充电特征
        initial_capacity = None
        for cycle in cycles:
            # 跳过阻抗测试
            if cycle['type'] == 'impedance':
                continue

            # 提取基础特征
            features = extract_operation_features(cycle)

            if cycle['type'] == 'charge':
                # 缓存充电特征，等待后续放电配对
                last_charge_features = features

            elif cycle['type'] == 'discharge' and last_charge_features != {}:
                if not last_charge_features:
                    print(f"Warning: 放电周期 {current_cycle_num} 缺少充电数据")
                    continue
                # 合并特征
                full_features = {
                    'battery_id': batt_id,
                    'cycle': current_cycle_num,
                    **last_charge_features,
                    **features
                }
                # 初始化容量记录
                if initial_capacity is None:
                    initial_capacity = full_features['capacity']
                    print(f"电池 {batt_id} 初始容量: {initial_capacity:.2f}Ah")
                # 计算健康指标
                full_features['soh'] = full_features['capacity'] / initial_capacity
                all_data.append(full_features)
                # 周期计数器递增
                current_cycle_num += 1
                last_charge_features = {}  # 重置缓存
    # 转换为DataFrame
    df = pd.DataFrame(all_data)




    # 计算RUL (到容量衰减30%的剩余周期)
    for batt_id, group in df.groupby('battery_id'):
        # eol = group[group['soh'] <= 0.7].index.min()
        eol = group.index.max()
        if pd.notna(eol):
            df.loc[group.index, 'rul'] = eol - group.index

    # 特征工程
    # df['energy_throughput'] = df['charge_time'] * df['current_discharge_std']
    # df['temp_diff'] = df['temp_max_discharge'] - df['max_temp_charge']

    df.to_excel('path.xlsx',
                index=False,  # 不保存索引
                sheet_name='Battery Data',  # 自定义工作表名
                float_format="%.3f")  # 浮点数精度控制

    # 丢弃中间周期数据
    # df = df[df['rul'] > 0].dropna()
    print(df.reset_index(drop=True))

    # 计算特征与目标变量的相关系数
    corr_matrix = df.corr(method="spearman")  # 可选值为{‘pearson’, ‘kendall’, ‘spearman’}
    soh_corr = corr_matrix['soh'].sort_values(ascending=False)

    # 可视化
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    # plt.show()
    plt.savefig('results/特征相关性分析.png')
    # 筛选标准建议
    significant_features = soh_corr[abs(soh_corr) > 0.3].index.tolist()
    print("显著相关特征:", significant_features)

    return df.reset_index(drop=True)

if __name__ == "__main__":
    create_dataset()