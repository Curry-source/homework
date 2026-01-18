import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import plot

from Processsing import data_process
import cv2

from Processsing.thermal_image_preprocessing import WeldingPreprocessor
from Processsing.thermal_image_preprocessing_2 import WeldingPreprocessor_IR


# 1. 读取表格数据
def read_data(file_path):
    """读取xlsx文件"""
    # df = pd.read_excel(file_path)  # 指定工作表
    df = pd.read_csv(file_path)

    print("原始数据：")
    print(df.shape[0])

    #合并样本，构成495条数据
    df = df.groupby('Sample_ID').agg({
        'Pressure': 'first',  # 取每个样本的第一组数据
        'Welding_Time':'first',
        'Angle':'first',
        'Force':'first',
        'Current':'first',
        'Thickness_A':'first',
        'PullTest':'first',
        'NuggetDiameter':'first',
        'Category':'first'
    }).reset_index()

    print("\n按标签合并后的结果：")
    print(df.shape[0])

    return df

# 主流程
def main_load_data(file_path):
    # -------------------1. 加载数据-------------------
    df = read_data(file_path)

    # 计算80%的索引位置  0.7
    df_size = int(len(df) * 0.7)

    # 只保留各数据集的70%
    df = df[:df_size]

    # -------------------2. 预处理数据-------------------
    df,outlier_indices = data_process.preprocess_data(df)
    print(f"预处理后数据集大小：{df.shape[0]} samples")

    df_save_path = "../CSV_Result/data_normalize.csv"
    df.to_csv(df_save_path, index=False, encoding="utf-8-sig")
    print(f"预测结果已保存到：{df_save_path}")

    # 图像目录路径
    image_dir = "../Dataset/Images_RGB"

    # 初始化预处理工具
    preprocessor = WeldingPreprocessor(
        image_dir="path/to/images",
        img_size=(64, 64),
        grayscale=False,
        augment=True,  # 启用常规增强
        mosaic_prob=0.3  # 30%概率应用Mosaic
    )

    # 预处理图片
    print("开始预处理图片...")
    processed_data= preprocessor.preprocess_images(image_dir )

    processed_data = processed_data[:df_size]

    processed_data = np.delete(processed_data, outlier_indices, axis=0)

    # ！！！！！！三分支
    image_dir_2 = "../Dataset/Images_IR"

    preprocessor_2 = WeldingPreprocessor_IR(
        img_size=(64, 64),  # 目标图像尺寸，根据需求设置
        grayscale=False,  # 热成像图如果是彩色伪热图设为False，纯灰度图设为True
        augment=True,  # 是否启用图像增强
        mosaic_prob=0.3  # Mosaic增强的概率（0-1之间）
    )

    processed_data_2 = preprocessor_2.preprocess_images(
        image_folder=image_dir_2  # 热成像图片所在文件夹路径
    )

    processed_data_2 = processed_data_2[:df_size]

    processed_data_2 = np.delete(processed_data_2, outlier_indices, axis=0)
    # ！！！！！！三分支

    if len(processed_data) == 0:
        print("错误: 未找到有效图片数据，请检查输入路径和文件格式")
    else:
        print(f"预处理完成，共获得 {len(processed_data)} 个样本")
        print(f"数据形状: {processed_data.shape} (样本数, 通道数, 高度, 宽度)")

    return df,processed_data,processed_data_2