import pandas as pd

from Processsing import data_process


def analyze_data(file_path):
    key_metrics = ['Pressure', 'Welding_Time', 'Angle', 'Force', 'Current', 'Thickness_A']
    quality_metrics = ['PullTest', 'NuggetDiameter']
    # 合并为需分析的所有列
    target_columns = key_metrics + quality_metrics

    try:
        df = pd.read_csv(file_path)

        print("原始数据：")
        print(df.shape[0])

        # 合并样本，构成495条数据
        df = df.groupby('Sample_ID').agg({
            'Pressure': 'first',  # 取每个样本的第一组数据
            'Welding_Time': 'first',
            'Angle': 'first',
            'Force': 'first',
            'Current': 'first',
            'Thickness_A': 'first',
            'PullTest': 'first',
            'NuggetDiameter': 'first',
            'Category': 'first'
        }).reset_index()

    except Exception as e:
        print(f"读取数据失败：{e}")
        return None

    # 步骤2：保留前70%的样本
    df_size = int(len(df) * 0.7)  # 计算70%样本量的索引位置
    df_70 = df[:df_size].copy()  # 截取前70%数据

    df_70, outlier_indices = data_process.preprocess_data(df_70)
    print(f"保留70%样本后，样本量：{len(df_70)}")

    # 步骤3：计算统计量（count/mean/std/min/max）
    # 选择需要分析的列（若需全列分析可省略columns参数）
    # 此处假设所有列均需分析，若有特定列可改为 columns=['列名1', '列名2', ...]
    stats = df_70.agg(
        {col: ['count', 'mean', 'std', 'min', 'max'] for col in target_columns}
    )

    # 美化索引名称
    stats = stats.rename(index={
        'count': '样本量（Count）',
        'mean': '均值（Mean）',
        'std': '标准差（Std）',
        'min': '最小值（Min）',
        'max': '最大值（Max）'
    })

    # 保留2位小数（根据需求调整）
    stats = stats.round(2)

    # 步骤4：输出结果
    print("\n70%样本的统计分析结果：")
    print(stats)

    # 可选：将结果保存为Excel（方便后续使用）
    stats.to_excel("data_statistics_70%.xlsx")
    print("\n统计结果已保存至 'data_statistics_70%.xlsx'")

    return stats

# 使用示例
if __name__ == "__main__":
    from Processsing.main_load import main_load_data

    # 加载数据
    file_path = '../Dataset/Data_RSW_1.csv'

    analyze_data(file_path)