import os
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def show_smooth_data(original: np.ndarray, smoothed: np.ndarray, title: str, y_label: str) -> None:
    """可视化原始数据与平滑后数据的对比"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(original, label='Original', alpha=0.6)
    plt.plot(smoothed, label='Smoothed', color='red', linewidth=2)
    plt.title(title)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.show()

def detect_outliers_zscore(data: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> List[int]:
    """Z-score法检测异常值（适用于近似正态分布数据）"""
    outlier_indices = []
    for col in columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            continue
        mean = data[col].mean()
        std = data[col].std()
        if std == 0:
            continue
        z_scores = (data[col] - mean) / std
        col_outliers = data[abs(z_scores) > threshold].index.tolist()
        outlier_indices.extend(col_outliers)
    return list(set(outlier_indices))

def detect_outliers_dbscan(data: pd.DataFrame, columns: List[str], eps: float = 0.5, min_samples: int = 5) -> List[int]:
    """DBSCAN聚类法检测异常值（适用于含空间/时序关联的数据）"""
    numeric_data = data[columns].select_dtypes(include='number')
    if numeric_data.empty:
        return []
    scaled_data = StandardScaler().fit_transform(numeric_data)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_data)
    return data.index[clusters == -1].tolist()

def preprocess_data(
    df: pd.DataFrame,
    feature_columns: List[str] = ['Pressure', 'Welding_Time', 'Angle', 'Force', 'Current', 'Thickness_A'],
    target_columns: List[str] = ['PullTest', 'NuggetDiameter'],
    outlier_method: str = 'dbscan',  # 新增：异常检测方法（'zscore'/'dbscan'/'iqr'）
    zscore_threshold: float = 3.0,   # Z-score阈值
    dbscan_eps: float = 0.5,         # DBSCAN参数0.5
    dbscan_min_samples: int = 5,     # DBSCAN参数
    sigma: float = 5,
    smooth_target: bool = True,
    visualize_smoothing: bool = False,
    handle_feature_outliers: bool = True
) -> Tuple[pd.DataFrame, List[int]]:
    """支持多种异常值检测方法的数据预处理函数"""
    initial_count = len(df)
    outlier_indices = []
    columns_to_check = target_columns.copy()
    if handle_feature_outliers:
        columns_to_check.extend(feature_columns)

    # 选择异常值检测方法
    if outlier_method == 'iqr':
        # 保留原有IQR方法
        for col in columns_to_check:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            col_outliers = df[(df[col] < lower) | (df[col] > upper)].index.tolist()
            outlier_indices.extend(col_outliers)
    elif outlier_method == 'zscore':
        outlier_indices = detect_outliers_zscore(df, columns_to_check, zscore_threshold)
    elif outlier_method == 'dbscan':
        outlier_indices = detect_outliers_dbscan(df, columns_to_check, dbscan_eps, dbscan_min_samples)
    else:
        raise ValueError(f"不支持的异常检测方法：{outlier_method}，可选：'iqr'/'zscore'/'dbscan'")

    # 去重并删除异常值
    outlier_indices = list(set(outlier_indices))
    data_clean = df.drop(outlier_indices).copy()
    cleaned_count = len(data_clean)
    print(f"数据清洗: 移除 {initial_count - cleaned_count} 个异常样本 "
          f"(原始: {initial_count}, 清洗后: {cleaned_count})")

    if cleaned_count == 0:
        raise ValueError("清洗后无剩余样本，请调整异常检测参数")

    # 目标值平滑
    if smooth_target:
        for target in target_columns:
            original_data = data_clean[target].values.astype(float)
            smoothed_data = gaussian_filter1d(original_data, sigma=sigma)
            data_clean[target] = smoothed_data
            if visualize_smoothing:
                show_smooth_data(original_data, smoothed_data, f'{target}: Original vs. Smoothed', target)

    # 归一化
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    data_clean[feature_columns] = scaler_X.fit_transform(data_clean[feature_columns])

    # scaler_y = MinMaxScaler(feature_range=(0, 1))
    # data_clean[target_columns] = scaler_y.fit_transform(data_clean[target_columns])

    # 目标值归一化
    scaler_pull = MinMaxScaler(feature_range=(0, 1))
    data_clean['PullTest'] = scaler_pull.fit_transform(data_clean['PullTest'].to_frame())

    scaler_nugget = MinMaxScaler(feature_range=(0, 1))
    data_clean['NuggetDiameter'] = scaler_nugget.fit_transform(data_clean['NuggetDiameter'].to_frame())

    # 移除无关列
    columns_to_drop = ['Sample_ID', 'Category']
    columns_to_drop = [col for col in columns_to_drop if col in data_clean.columns]
    if columns_to_drop:
        data_clean = data_clean.drop(columns=columns_to_drop)

    folder_path = "../CSV_Result"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"文件夹 {folder_path} 已创建")
    else:
        print(f"文件夹 {folder_path} 已存在")

    df_save_path = "../CSV_Result/data_clean.csv"
    data_clean.to_csv(df_save_path, index=False, encoding="utf-8-sig")

    scaler_folder = "../Data_scaler"
    if not os.path.exists(scaler_folder):
        os.makedirs(scaler_folder, exist_ok=True)
        print(f"文件夹 {scaler_folder} 已创建")
    else:
        print(f"文件夹 {scaler_folder} 已存在")

    # 保存归一化器
    joblib.dump(scaler_X, '../Data_scaler/scaler_X.pkl')
    joblib.dump(scaler_pull, '../Data_scaler/scaler_pull.pkl')
    joblib.dump(scaler_nugget, '../Data_scaler/scaler_nugget.pkl')
    print("已保存特征和目标值归一化器")

    return data_clean, outlier_indices


def split_data(df: pd.DataFrame,
               img: np.ndarray,
               test_size: float = 0.25,
               random_state: int = 42,
               feature_columns: List[str] = ['Pressure', 'Welding_Time', 'Angle', 'Force', 'Current','Thickness_A'],
               target_columns: List[str] = ['PullTest', 'NuggetDiameter']) -> Tuple:
    """
    将数据划分为训练集和测试集
    """
    # 验证输入数据一致性
    if len(df) != len(img):
        raise ValueError(f"DataFrame样本数 ({len(df)}) 与图像样本数 ({len(img)}) 不一致")

    # 提取特征和目标值
    X_data = df[feature_columns].values
    y = df[target_columns].values
    X_img = img

    # 划分数据集（不打乱顺序，适合时序或有序数据）
    X_data_train, X_data_test, \
        X_img_train, X_img_test, \
        y_train, y_test = train_test_split(
        X_data, X_img, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True  # 保持原始顺序
    )

    # 打印划分结果
    print(f"数据集划分完成: 训练集 {len(X_data_train)} 个样本, 测试集 {len(X_data_test)} 个样本")


    return (X_data_train, X_data_test,
            X_img_train, X_img_test,
            y_train, y_test)
