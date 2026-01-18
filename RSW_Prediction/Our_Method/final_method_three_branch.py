import os# 操作系统接口（文件操作）
import random# 生成随机数
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.src.saving import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from keras_tuner import HyperModel, RandomSearch, Objective
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from tabulate import tabulate
from Processsing.data_process import split_data
from Processsing.main_load import main_load_data

# 固定随机数种子
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()

class DataLoader:
    """数据加载类，适配三分支输入"""

    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def load_data(self, file_path):

        result = main_load_data(file_path)

        if isinstance(result, tuple) and len(result) >= 3:
            df, img_1, img_2 = result[0], result[1], result[2]  # img_1: 第一图像分支，img_2: 第二图像分支
        else:
            raise ValueError("main()函数返回值格式不正确，期望包含DataFrame和两个图像数据的元组")

        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
                print("警告：main()返回的df不是DataFrame，已尝试自动转换")
            except Exception as e:
                raise TypeError("无法将main()返回的df转换为DataFrame: " + str(e))

        # 转换双图像格式为通道最后（N, C, H, W）->（N, H, W, C）
        def convert_image_format(img):
            if len(img.shape) == 4:
                print(f"原始{img.shape[0]}样本图像形状: {img.shape} (样本数、通道数、高、宽)")
                img = np.transpose(img, (0, 2, 3, 1))
                print(f"转换后图像形状: {img.shape} (样本数、高、宽、通道数)")
            return img

        img_1 = convert_image_format(img_1)
        img_2 = convert_image_format(img_2)

        # 划分数据（固定随机种子确保双图像与特征样本对齐）
        X_data_train, X_data_test, \
            X_img1_train, X_img1_test, \
            y_train, y_test = split_data(df, img_1)

        _, _, \
            X_img2_train, X_img2_test, \
            _, _ = split_data(df, img_2)

        # 校验样本数量一致性
        assert len(X_data_train) == len(X_img1_train) == len(X_img2_train) == len(y_train), \
            "训练集各分支样本数量不匹配"
        assert len(X_data_test) == len(X_img1_test) == len(X_img2_test) == len(y_test), \
            "测试集各分支样本数量不匹配"

        print("\n数据处理结果:")
        print(f"训练集样本大小: {X_data_train.shape[0]} samples")
        print(f"测试集样本大小: {X_data_test.shape[0]} samples")
        print(f"第一图像分支（训练集）: {X_img1_train.shape} (N, H, W, C)")
        print(f"第二图像分支（训练集）: {X_img2_train.shape} (N, H, W, C)")
        print(f"特征分支（训练集）: {X_data_train.shape} (N, 特征数)")

        return (X_data_train, X_data_test,
                X_img1_train, X_img1_test,
                X_img2_train, X_img2_test,
                y_train, y_test)

    def create_datasets(self, X_data, X_img1, X_img2, y=None, shuffle=True):
        """创建tf.data.Dataset对象（确保键名与模型输入层匹配）"""
        # 确保标签为numpy数组，分离双输出
        if y is not None:
            y_pulltest = y[:, 0].copy()
            y_nugget = y[:, 1].copy()
            output_dict = {
                'output_pulltest': tf.convert_to_tensor(y_pulltest, dtype=tf.float32),
                'output_nugget': tf.convert_to_tensor(y_nugget, dtype=tf.float32)
            }
        else:
            output_dict = None

        # 关键修复：输入字典的键名必须与模型输入层名称完全一致
        input_dict = {
            'features': tf.convert_to_tensor(X_data, dtype=tf.float32),
            'cnn1_input': tf.convert_to_tensor(X_img1, dtype=tf.float32),  # 匹配模型第一图像输入层
            'cnn2_input': tf.convert_to_tensor(X_img2, dtype=tf.float32)  # 匹配模型第二图像输入层
        }

        # 构建数据集
        if output_dict is not None:
            dataset = tf.data.Dataset.from_tensor_slices((input_dict, output_dict))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(input_dict)

        if shuffle and output_dict is not None:
            dataset = dataset.shuffle(buffer_size=len(X_data))
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset


class ThreeBranchCrossModalAttention(layers.Layer):
    """三分支跨模态注意力层"""

    def __init__(self, hidden_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # 动态权重分配层
        self.dynamic_weight_layers = [
            layers.Dense(hidden_dim // 4, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(hidden_dim, activation='sigmoid')
        ]

        # 注意力参数
        # 图像分支1
        self.cnn1_query = layers.Dense(hidden_dim)
        self.cnn2_key = layers.Dense(hidden_dim)
        self.lstm_key = layers.Dense(hidden_dim)
        self.cnn2_value = layers.Dense(hidden_dim)
        self.lstm_value = layers.Dense(hidden_dim)

        # 图像分支2
        self.cnn2_query = layers.Dense(hidden_dim)
        self.cnn1_key = layers.Dense(hidden_dim)
        self.cnn1_value = layers.Dense(hidden_dim)

        # 特征分支
        self.lstm_query = layers.Dense(hidden_dim)

        # 融合层
        self.cnn12_query = layers.Dense(hidden_dim)
        self.cnn12_key = layers.Dense(hidden_dim)
        self.cnn12_value = layers.Dense(hidden_dim)
        self.fusion_dense = layers.Dense(hidden_dim)
        self.batch_norm = layers.BatchNormalization()
        self.final_fusion = layers.Dense(hidden_dim, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)

    def _dynamic_feature_weight(self, x):
        """改进的动态特征权重分配"""
        # 添加残差连接，避免信息丢失 -----------这个 dynamic_weight_layers 本质是一个轻量级的特征增强网络：通过降维提取核心关联、Dropout 防止过拟合、Sigmoid 约束增强范围，最终学习出与输入特征维度匹配的动态缩放因子，结合残差连接实现 “安全且自适应” 的特征加权。这种设计在保持计算效率的同时，能让模型更灵活地关注对任务重要的特征。相比直接用 Dense(hidden_dim) 学习权重，先降维再升维的设计能减少参数数量（例如 hidden_dim=1024 时，参数从 1024×1024 减少为 1024×256 + 256×1024，约节省一半），同时强迫网络聚焦于更关键的特征交互，避免学习冗余的权重模式。不同于可能输出负值的激活函数（如 ReLU 可能输出 0，但不会负；Tanh 输出 [-1,1] 可能导致抑制），Sigmoid 的 [0,1] 输出配合 x + 1 确保了权重始终 ≥1，既实现了动态增强，又避免了 “过度抑制有用特征” 的风险（尤其适合特征重要性不确定的场景）。
        residual = x
        for layer in self.dynamic_weight_layers:
            x = layer(x)
        # 残差连接 + 加权：通过x学习 "缩放因子"，对原始特征进行动态加权（x + 1确保权重至少为 1，避免特征被过度抑制）
        x_weighted = residual * x + residual # 原始特征 + 加权特征（类似ResNet的残差结构）
        return x_weighted

    def _attention_block(self, query, key, value, name=None):
        """通用的注意力计算块"""
        # 计算注意力分数
        scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.cast(self.hidden_dim, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # 应用dropout到注意力权重
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        context = tf.matmul(attention_weights, value)
        return context

    def call(self, cnn1_features, cnn2_features, lstm_features, training=None):
        """改进的融合逻辑"""

        # 准备注意力输入
        cnn1_expanded = tf.expand_dims(cnn1_features, axis=1)  # (batch, 1, hidden_dim)
        cnn2_expanded = tf.expand_dims(cnn2_features, axis=1)
        lstm_expanded = tf.expand_dims(lstm_features, axis=1)

        # --------------------自注意力机制---------------
        query_cnn1 = self.cnn1_query(cnn1_expanded)
        key_cnn1 = self.cnn1_key(cnn1_expanded)
        value_cnn1 = self.cnn1_value(cnn1_expanded)
        cnn1_expanded = self._attention_block(query_cnn1, key_cnn1, value_cnn1)

        # 动态加权特征
        cnn1_expanded = self._dynamic_feature_weight(cnn1_expanded)
        cnn2_expanded = self._dynamic_feature_weight(cnn2_expanded)
        lstm_expanded = self._dynamic_feature_weight(lstm_expanded)

        # 1. CNN1与CNN2双向注意力
        query_cnn1 = self.cnn1_query(cnn1_expanded)
        key_cnn2 = self.cnn2_key(cnn2_expanded)
        value_cnn2 = self.cnn2_value(cnn2_expanded)
        context_cnn12 = self._attention_block(query_cnn1, key_cnn2, value_cnn2)

        query_cnn2 = self.cnn2_query(cnn2_expanded)
        key_cnn1 = self.cnn1_key(cnn1_expanded)
        value_cnn1 = self.cnn1_value(cnn1_expanded)
        context_cnn21 = self._attention_block(query_cnn2, key_cnn1, value_cnn1)

        # 融合CNN1和CNN2的结果
        cnn12_concat = tf.concat([context_cnn12, context_cnn21], axis=-1)
        cnn12_fused = tf.squeeze(cnn12_concat, axis=1)
        cnn12_fused = self.batch_norm(self.fusion_dense(cnn12_fused))
        cnn12_fused = tf.expand_dims(cnn12_fused, axis=1)

        # 2. CNN12融合结果与LSTM交互
        query_cnn12 = self.cnn12_query(cnn12_fused)
        key_lstm = self.lstm_key(lstm_expanded)
        value_lstm = self.lstm_value(lstm_expanded)
        context_cnn12_lstm = self._attention_block(query_cnn12, key_lstm, value_lstm)

        query_lstm = self.lstm_query(lstm_expanded)
        key_cnn12 = self.cnn12_key(cnn12_fused)
        value_cnn12 = self.cnn12_value(cnn12_fused)
        context_lstm_cnn12 = self._attention_block(query_lstm, key_cnn12, value_cnn12)

        # 3. 全局融合
        global_concat = tf.concat([context_cnn12_lstm, context_lstm_cnn12], axis=-1)
        global_fused = tf.squeeze(global_concat, axis=1)
        global_fused = self.final_fusion(global_fused)

        return global_fused

    def get_config(self):
        """支持序列化"""
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate
        })
        return config


class MetricsLogger(Callback):
    """记录三分支模型双输出的训练指标"""

    def __init__(self):
        super().__init__()
        self.best_val_r2_mean = -float('inf')
        self.best_val_mae_mean = float('inf')
        self.best_epoch = 0
        # 双输出单独指标
        self.best_val_pulltest_r2 = -float('inf')
        self.best_val_pulltest_mae = float('inf')
        self.best_val_nugget_r2 = -float('inf')
        self.best_val_nugget_mae = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # 提取双输出验证指标
        val_pulltest_r2 = logs.get('val_output_pulltest_r2_score', -float('inf'))
        val_nugget_r2 = logs.get('val_output_nugget_r2_score', -float('inf'))
        val_pulltest_mae = logs.get('val_output_pulltest_mae', float('inf'))
        val_nugget_mae = logs.get('val_output_nugget_mae', float('inf'))

        # 计算均值指标
        val_r2_mean = (val_pulltest_r2 + val_nugget_r2) / 2
        val_mae_mean = (val_pulltest_mae + val_nugget_mae) / 2

        # 更新最佳指标
        if val_r2_mean > self.best_val_r2_mean:
            self.best_val_r2_mean = val_r2_mean
            self.best_val_mae_mean = val_mae_mean
            self.best_epoch = epoch + 1
            self.best_val_pulltest_r2 = val_pulltest_r2
            self.best_val_pulltest_mae = val_pulltest_mae
            self.best_val_nugget_r2 = val_nugget_r2
            self.best_val_nugget_mae = val_nugget_mae


def build_lenet5_branch(input_shape, hp, branch_name="cnn"):
    """构建可复用的LeNet-5图像分支"""
    inputs = layers.Input(shape=input_shape, name=f'{branch_name}_input')
    x = inputs

    # 第一卷积块（超参可调）
    x = layers.Conv2D(
        filters=hp.Int(f'{branch_name}_conv1_filters', 8, 32, step=8, default=16),
        kernel_size=hp.Choice(f'{branch_name}_conv1_kernel', [5, 7, 8], default=7),
        strides=1,
        padding='valid',
        name=f'{branch_name}_conv1'
    )(x)
    x = layers.BatchNormalization(name=f'{branch_name}_bn1')(x)
    x = layers.Activation('relu', name=f'{branch_name}_relu1')(x)
    x = layers.MaxPool2D(pool_size=2, strides=2, name=f'{branch_name}_pool1')(x)

    # 第二卷积块
    x = layers.Conv2D(
        filters=hp.Int(f'{branch_name}_conv2_filters', 32, 64, step=16, default=32),
        kernel_size=hp.Choice(f'{branch_name}_conv2_kernel', [5, 7], default=5),
        strides=1,
        padding='valid',
        name=f'{branch_name}_conv2'
    )(x)
    x = layers.BatchNormalization(name=f'{branch_name}_bn2')(x)
    x = layers.Activation('relu', name=f'{branch_name}_relu2')(x)
    x = layers.MaxPool2D(pool_size=2, strides=2, name=f'{branch_name}_pool2')(x)

    # 第三卷积块
    x = layers.Conv2D(
        filters=hp.Int(f'{branch_name}_conv3_filters', 64, 128, step=32, default=64),
        kernel_size=hp.Choice(f'{branch_name}_conv3_kernel', [3, 5], default=3),
        strides=1,
        padding='valid',
        name=f'{branch_name}_conv3'
    )(x)
    x = layers.BatchNormalization(name=f'{branch_name}_bn3')(x)
    x = layers.Activation('relu', name=f'{branch_name}_relu3')(x)
    x = layers.MaxPool2D(pool_size=2, strides=2, name=f'{branch_name}_pool3')(x)

    # 全连接层
    x = layers.Flatten(name=f'{branch_name}_flatten')(x)
    x = layers.Dense(
        hp.Int(f'{branch_name}_dense_units', 128, 256, step=64, default=128),
        activation='relu',
        name=f'{branch_name}_dense'
    )(x)
    x = layers.LayerNormalization(name=f'{branch_name}_ln')(x)

    return inputs, x


class ThreeBranchHyperModel(HyperModel):
    """三分支模型（双LeNet-5图像分支 + BiLSTM特征分支）"""

    def __init__(self, image1_channels, image2_channels,
                 feature_dim, image_height, image_width, num_outputs=2):
        self.image1_channels = image1_channels  # 第一图像分支通道数
        self.image2_channels = image2_channels  # 第二图像分支通道数
        self.feature_dim = feature_dim  # 特征分支维度
        self.image_height = image_height  # 图像高度
        self.image_width = image_width  # 图像宽度
        self.num_outputs = num_outputs  # 双输出

    def build(self, hp):
        # -------------------------- 1. 定义输入层 --------------------------
        # 特征输入（用于BiLSTM分支）
        features_input = layers.Input(shape=(self.feature_dim,), name='features')

        # 图像输入（两个独立的LeNet-5分支）
        image1_shape = (self.image_height, self.image_width, self.image1_channels)
        image2_shape = (self.image_height, self.image_width, self.image2_channels)

        # -------------------------- 2. 构建三分支 --------------------------
        # 分支1: 第一图像分支（LeNet-5）
        img1_input, cnn1_features = build_lenet5_branch(
            image1_shape, hp, branch_name="cnn1")

        # 分支2: 第二图像分支（LeNet-5）
        img2_input, cnn2_features = build_lenet5_branch(
            image2_shape, hp, branch_name="cnn2")

        # 分支3: 特征分支（BiLSTM）
        #(1, self.feature_dim) 相当于timesteps=1，input_dim=self.feature_dim（1 个时间步，每个时间步有self.feature_dim个特征），这样的好处是会聚焦于单个时间步内的不同特征的关联性，且不会额外引入不必要的“时序顺序关系”导致形成错误的上下文依赖。当输入是无天然顺序的静态特征（如图像特征、用户属性特征等），且需要用 LSTM 处理但不关注时序关系时（例如利用 LSTM 的门控机制增强特征筛选能力）。此时核心是保留原始特征的高维度（input_dim），而非构建时序。双向性可能存在两点微弱价值：一是模型容量的被动提升：双向 LSTM 相当于并行运行两个独立的单向 LSTM（前向 + 后向），参数数量翻倍，理论上能拟合更复杂的特征交互（尤其当工艺参数间关联极强时）；二是特征融合的冗余性保障：两个方向的 LSTM 输出会被拼接或叠加，相当于对同一特征进行两次不同角度的门控筛选，可能在一定程度上降低单一 LSTM 单元的决策偏差。
        x = layers.Reshape((1, self.feature_dim), name='lstm_reshaper')(features_input)
        lstm_units = hp.Int('lstm_units', 64, 256, step=64, default=128)

        x = layers.Bidirectional(
            layers.LSTM(
                units=lstm_units,
                return_sequences=False,
                activation='tanh',
                recurrent_activation='sigmoid',
                name='bilstm'
            ),
            name='bidirectional_lstm'
        )(x)

        lstm_features = layers.Dense(
            hp.Int('lstm_dense_units', 128, 256, step=64, default=128),
            activation='relu',
            name='lstm_dense'
        )(x)
        lstm_features = layers.LayerNormalization(name='lstm_ln')(lstm_features)

        # -------------------------- 3. 三分支特征对齐 --------------------------
        # 确保所有分支特征维度一致，以便注意力融合
        hidden_dim = hp.Int('attention_hidden_dim', 128, 256, step=64, default=128)
        cnn1_features = layers.Dense(hidden_dim, name='cnn1_projection')(cnn1_features)
        cnn2_features = layers.Dense(hidden_dim, name='cnn2_projection')(cnn2_features)
        lstm_features = layers.Dense(hidden_dim, name='lstm_projection')(lstm_features)

        # -------------------------- 4. 跨模态注意力融合 --------------------------
        attention = ThreeBranchCrossModalAttention(hidden_dim=hidden_dim, name='cross_attention')
        fused_features = attention(cnn1_features, cnn2_features, lstm_features)

        # -------------------------- 5. 输出层（双输出） --------------------------
        x = layers.Flatten(name='final_flatten')(fused_features)

        # 全连接层
        x = layers.Dense(
            hp.Int('fc_units1', 128, 512, step=128, default=256),
            activation='relu',
            name='fc1'
        )(x)
        x = layers.Dropout(
            hp.Float('fc_dropout1', 0.1, 0.5, step=0.1, default=0.3),
            name='dropout1'
        )(x)

        x = layers.Dense(
            hp.Int('fc_units2', 64, 256, step=64, default=128),
            activation='relu',
            name='fc2'
        )(x)
        x = layers.Dropout(
            hp.Float('fc_dropout2', 0.1, 0.5, step=0.1, default=0.2),
            name='dropout2'
        )(x)

        # 双输出层
        output_pulltest = layers.Dense(
            1, name='output_pulltest',
            kernel_regularizer=tf.keras.regularizers.L2(1e-5)
        )(x)
        output_nugget = layers.Dense(
            1, name='output_nugget',
            kernel_regularizer=tf.keras.regularizers.L2(1e-5)
        )(x)

        # -------------------------- 6. 构建模型 --------------------------
        model = models.Model(
            inputs=[features_input, img1_input, img2_input],
            outputs=[output_pulltest, output_nugget]
        )

        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Float('learning_rate', 1e-5, 1e-3, sampling='log', default=1e-4)
            ),
            loss={'output_pulltest': 'mae', 'output_nugget': 'mae'},
            loss_weights={'output_pulltest': 0.4, 'output_nugget': 0.6},  # 平衡双输出损失
            metrics={
                'output_pulltest': ['mae', tf.keras.metrics.R2Score(name='r2_score')],
                'output_nugget': ['mae', tf.keras.metrics.R2Score(name='r2_score')]
            }
        )

        return model


def extended_calculate_metrics(y_pred, y_true):
    """
    计算双输出的评估指标
    参数:
        y_pred: 列表，包含两个numpy数组 [pull_pred, nugget_pred]
        y_true: 列表，包含两个numpy数组 [pull_true, nugget_true]

    返回:
        metrics: 字典，包含两个输出的各项指标
    """
    # 从列表中提取两个输出的预测值和真实值
    y_pred_pulltest = y_pred[0].flatten()  # 拉剪力预测值
    y_pred_nugget = y_pred[1].flatten()  # 熔核直径预测值
    y_true_pulltest = y_true[0].flatten()  # 拉剪力真实值
    y_true_nugget = y_true[1].flatten()  # 熔核直径真实值

    # 计算拉剪力指标
    pull_mae = mean_absolute_error(y_true_pulltest, y_pred_pulltest)
    pull_rmse = np.sqrt(mean_squared_error(y_true_pulltest, y_pred_pulltest))
    pull_r2 = r2_score(y_true_pulltest, y_pred_pulltest)
    pull_pearson, _ = pearsonr(y_true_pulltest, y_pred_pulltest)

    # 计算熔核直径指标
    nugget_mae = mean_absolute_error(y_true_nugget, y_pred_nugget)
    nugget_rmse = np.sqrt(mean_squared_error(y_true_nugget, y_pred_nugget))
    nugget_r2 = r2_score(y_true_nugget, y_pred_nugget)
    nugget_pearson, _ = pearsonr(y_true_nugget, y_pred_nugget)

    # 计算均值指标
    mean_mae = (pull_mae + nugget_mae) / 2
    mean_rmse = (pull_rmse + nugget_rmse) / 2
    mean_r2 = (pull_r2 + nugget_r2) / 2
    mean_pearson = (pull_pearson + nugget_pearson) / 2

    # 组织成字典返回
    return {
        "output_pulltest": {
            "MAE": pull_mae,
            "RMSE": pull_rmse,
            "R2": pull_r2,
            "Pearson": pull_pearson
        },
        "output_nugget": {
            "MAE": nugget_mae,
            "RMSE": nugget_rmse,
            "R2": nugget_r2,
            "Pearson": nugget_pearson
        },
        "mean": {
            "MAE": mean_mae,
            "RMSE": mean_rmse,
            "R2": mean_r2,
            "Pearson": mean_pearson
        }
    }


def plot_dual_output_predictions(model_name, model, data_loader, scaler_pull, scaler_nugget, save_path=None):
    """绘制双输出的预测结果对比"""
    # 初始化数据列表
    X_data, X_img1, X_img2, y_true = [], [], [], []

    for batch in data_loader:
        # 提取特征数据（工艺参数）
        X_data.append(batch[0]['features'].numpy())

        # 提取两个图像分支数据（使用实际图像键名）
        X_img1.append(batch[0]['cnn1_input'].numpy())  # 替换为模型实际的第一个图像输入键名
        X_img2.append(batch[0]['cnn2_input'].numpy())  # 替换为模型实际的第二个图像输入键名

        # 提取双输出真实值
        y_true_batch = np.column_stack([
            batch[1]['output_pulltest'].numpy(),
            batch[1]['output_nugget'].numpy()
        ])
        y_true.append(y_true_batch)

    # 拼接所有批次数据
    X_data = np.concatenate(X_data, axis=0)
    X_img1 = np.concatenate(X_img1, axis=0)
    X_img2 = np.concatenate(X_img2, axis=0)
    y_true = np.concatenate(y_true, axis=0)  # 形状：(n_samples, 2)

    # 模型预测（匹配三分支模型的输入键名）
    y_pred = model.predict({
        'features': X_data,
        'cnn1_input': X_img1,
        'cnn2_input': X_img2
    })  # y_pred是列表：[pull_pred, nugget_pred]

    # 反归一化（如果训练时使用了标准化）
    # y_pred_pull = scaler_pull.inverse_transform(y_pred[0].reshape(-1, 1)).flatten()
    # y_pred_nugget = scaler_nugget.inverse_transform(y_pred[1].reshape(-1, 1)).flatten()
    # y_true_pull = scaler_pull.inverse_transform(y_true[:, 0].reshape(-1, 1)).flatten()
    # y_true_nugget = scaler_nugget.inverse_transform(y_true[:, 1].reshape(-1, 1)).flatten()
    y_pred_pull = y_pred[0].flatten()
    y_pred_nugget = y_pred[1].flatten()
    y_true_pull = y_true[:, 0].flatten()
    y_true_nugget = y_true[:, 1].flatten()

    # 计算评估指标（传入列表格式，与修改后的函数适配）
    metrics = extended_calculate_metrics(
        [y_pred_pull, y_pred_nugget],  # 预测值：[拉剪力, 熔核直径]
        [y_true_pull, y_true_nugget]  # 真实值：[拉剪力, 熔核直径]
    )

    # -------------------------- 2. 构造DataFrame（结构化数据） --------------------------
    process_param_names = ['Pressure', 'Welding_Time', 'Angle', 'Force', 'Current', 'Thickness_A']
    # 将所有数据整合为字典（键=列名，值=数据）
    data_dict = {}

    # 添加预测值列
    data_dict["Pred_Pull_Strength"] = y_pred_pull  # 预测拉力强度
    data_dict["Pred_Nugget_Diameter"] = y_pred_nugget  # 预测熔核直径
    # 添加真实值列
    data_dict["True_Pull_Strength"] = y_true_pull
    data_dict["True_Nugget_Diameter"] = y_true_nugget

    # 转换为DataFrame
    df_pred = pd.DataFrame(data_dict)

    # 保存到CSV
    df_save_path = "../CSV_Result/rsw_model_predictions.csv"
    df_pred.to_csv(df_save_path, index=False, encoding="utf-8-sig")
    print(f"预测结果已保存到：{df_save_path}")
    print("\nCSV文件前5行预览：")
    print(df_pred.head())

    # 绘制对比图
    plt.figure(figsize=(16, 8))

    # 拉剪力预测对比
    plt.subplot(1, 2, 1)
    plt.plot(y_true_pull, label='True Value', color='blue', linewidth=1.5)
    plt.plot(y_pred_pull, label='Pred Value', color='red', linestyle='--', alpha=0.8)
    plt.title(f'Pulltest (R²: {metrics["output_pulltest"]["R2"]:.4f})')
    plt.xlabel('Index')
    plt.ylabel('Pulltest')
    plt.legend()
    plt.grid(alpha=0.3)
    # 添加指标文本框
    plt.text(0.05, 0.95,
             f'MAE: {metrics["output_pulltest"]["MAE"]:.4f}\n'
             f'RMSE: {metrics["output_pulltest"]["RMSE"]:.4f}\n'
             f'R²: {metrics["output_pulltest"]["R2"]:.4f}\n'
             f'Pearson: {metrics["output_pulltest"]["Pearson"]:.4f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 熔核直径预测对比
    plt.subplot(1, 2, 2)
    plt.plot(y_true_nugget, label='True Value', color='blue', linewidth=1.5)
    plt.plot(y_pred_nugget, label='Pred Value', color='red', linestyle='--', alpha=0.8)
    plt.title(f'Nugget (R²: {metrics["output_nugget"]["R2"]:.4f})')
    plt.xlabel('Index')
    plt.ylabel('Nugget')
    plt.legend()
    plt.grid(alpha=0.3)
    # 添加指标文本框
    plt.text(0.05, 0.95,
             f'MAE: {metrics["output_nugget"]["MAE"]:.4f}\n'
             f'RMSE: {metrics["output_nugget"]["RMSE"]:.4f}\n'
             f'R²: {metrics["output_nugget"]["R2"]:.4f}\n'
             f'Pearson: {metrics["output_nugget"]["Pearson"]:.4f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测对比图已保存至: {save_path}")
    plt.show()

    return [y_pred_pull, y_pred_nugget], [y_true_pull, y_true_nugget], metrics


def hyperparameter_tuning(hypermodel, X_data_train, X_img1_train, X_img2_train, y_train,
                          data_loader, epochs=50, max_trials=10, weight_pulltest=0.5,
                          batch_size=32, retrain_times=5):
    """三分支模型超参数调优"""
    # 划分训练集和验证集
    X_data_train, X_data_val, X_img1_train, X_img1_val, X_img2_train, X_img2_val, y_train, y_val = train_test_split(
        X_data_train, X_img1_train, X_img2_train, y_train, test_size=0.25, random_state=42)

    # 创建数据集
    train_dataset = data_loader.create_datasets(
        X_data_train, X_img1_train, X_img2_train, y_train, shuffle=True)
    val_dataset = data_loader.create_datasets(
        X_data_val, X_img1_val, X_img2_val, y_val, shuffle=False)

    # 多目标优化回调
    class MultiR2Callback(Callback):
        def __init__(self, weight_pulltest=0.5):
            super().__init__()
            self.weight_pulltest = weight_pulltest
            self.weight_nugget = 1 - weight_pulltest

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            # 计算综合R²
            train_pull_r2 = logs.get('output_pulltest_r2_score', -float('inf'))
            train_nug_r2 = logs.get('output_nugget_r2_score', -float('inf'))
            val_pull_r2 = logs.get('val_output_pulltest_r2_score', -float('inf'))
            val_nug_r2 = logs.get('val_output_nugget_r2_score', -float('inf'))

            logs['combined_r2'] = (train_pull_r2 * self.weight_pulltest +
                                   train_nug_r2 * self.weight_nugget)

            logs['val_combined_r2'] = (val_pull_r2 * self.weight_pulltest +
                                       val_nug_r2 * self.weight_nugget)

    # 基础回调函数
    def get_base_callbacks():
        return [
            MultiR2Callback(weight_pulltest=weight_pulltest),
            EarlyStopping(
                patience=30,
                monitor='val_combined_r2',
                mode='max',
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_combined_r2',
                mode='max',
                factor=0.5,
                patience=5,
                verbose=1
            )
        ]

    # 超参数搜索器配置
    tuner = RandomSearch(
        hypermodel,
        objective=Objective("val_combined_r2", direction="max"),
        max_trials=max_trials,
        executions_per_trial=2,
        project_name='../Search_Optimization/three_branch_optimization',
        overwrite=True
    )

    # 执行搜索
    tuner.search(
        [X_data_train, X_img1_train, X_img2_train],  # 作为列表传递输入
        [y_train[:, 0], y_train[:, 1]],  # 作为列表传递两个输出
        validation_data=(
            [X_data_val, X_img1_val, X_img2_val],  # 验证输入
            [y_val[:, 0], y_val[:, 1]]  # 验证输出
        ),
        epochs=epochs,
        callbacks=get_base_callbacks(),
        batch_size=batch_size,
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(1)[0]
    print("\n=== 最佳超参数确定 ===")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")

    # 多次重训练并选择最优模型
    best_combined_r2 = -float('inf')
    best_model = None
    best_history = None
    best_metrics_logger = None

    print(f"\n=== 开始{retrain_times}次重训练以减少随机性 ===")
    for i in range(retrain_times):
        print(f"\n--- 第{i + 1}/{retrain_times}次训练 ---")

        current_model = tuner.hypermodel.build(best_hps)
        current_metrics = MetricsLogger()
        current_callbacks = get_base_callbacks() + [current_metrics]

        # 修改训练数据的传递方式
        current_history = current_model.fit(
            [X_data_train, X_img1_train, X_img2_train],
            [y_train[:, 0], y_train[:, 1]],
            validation_data=(
                [X_data_val, X_img1_val, X_img2_val],
                [y_val[:, 0], y_val[:, 1]]
            ),
            epochs=epochs,
            callbacks=current_callbacks,
            batch_size=batch_size,
            verbose=1
        )

        # 评估当前模型性能
        current_val_combined = current_history.history['val_combined_r2']
        current_best_r2 = max(current_val_combined)
        current_best_epoch = np.argmax(current_val_combined) + 1

        print(f"第{i + 1}次训练 - 最佳epoch: {current_best_epoch}")
        print(f"第{i + 1}次训练 - 最佳综合验证R²: {current_best_r2:.4f}")

        # 更新最佳模型
        if current_best_r2 > best_combined_r2:
            best_combined_r2 = current_best_r2
            best_model = current_model
            best_history = current_history
            best_metrics_logger = current_metrics

    # 分析最终最佳模型
    print(f"\n=== 多次训练结果总结 ===")
    print(f"最佳综合验证R²: {best_combined_r2:.4f}")

    # 提取最佳模型的详细指标
    val_pull_r2 = best_history.history['val_output_pulltest_r2_score']
    val_nug_r2 = best_history.history['val_output_nugget_r2_score']
    val_combined = best_history.history['val_combined_r2']

    best_epoch = np.argmax(val_combined) + 1
    print(f"最佳epoch: {best_epoch}")
    print(f"pulltest验证集R²: {val_pull_r2[best_epoch - 1]:.4f}")
    print(f"nugget验证集R²: {val_nug_r2[best_epoch - 1]:.4f}")
    print(f"综合验证集R²: {val_combined[best_epoch - 1]:.4f}")

    return best_model, best_history, best_metrics_logger


def print_result_table(train_metrics, test_metrics):
    """生成训练和测试指标对比表格"""
    # 组织表格数据
    table_data = [
        ["指标", "训练集(pulltest)", "训练集(nugget)", "训练集(均值)",
         "测试集(pulltest)", "测试集(nugget)", "测试集(均值)"]
    ]
    # MAE行
    table_data.append([
        "MAE",
        f"{train_metrics['best_val_pulltest_mae']:.4f}",
        f"{train_metrics['best_val_nugget_mae']:.4f}",
        f"{train_metrics['best_val_mae_mean']:.4f}",
        f"{test_metrics['output_pulltest']['MAE']:.4f}",
        f"{test_metrics['output_nugget']['MAE']:.4f}",
        f"{test_metrics['mean']['MAE']:.4f}"
    ])
    # R²行
    table_data.append([
        "R²",
        f"{train_metrics['best_val_pulltest_r2']:.4f}",
        f"{train_metrics['best_val_nugget_r2']:.4f}",
        f"{train_metrics['best_val_r2_mean']:.4f}",
        f"{test_metrics['output_pulltest']['R2']:.4f}",
        f"{test_metrics['output_nugget']['R2']:.4f}",
        f"{test_metrics['mean']['R2']:.4f}"
    ])

    # 打印表格
    print("\n训练与测试指标对比:")
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

    # 保存为CSV
    os.makedirs('../results', exist_ok=True)
    pd.DataFrame(table_data[1:], columns=table_data[0]).to_csv('../results/metrics_summary.csv', index=False)
    print("指标表格已保存至 results/metrics_summary.csv")


def main_function():
    batch_size = 5# 批量大小（每次训练5个样本）
    epochs = 500  # 总训练轮数
    model_name = 'three_branch_lenet5_bilstm_93'# 模型命名
    num_trials = 15# 超参数搜索次数

    file_path = '../Dataset/Data_RSW_1.csv'#简单说：CSV就是用逗号分隔的表格文件(这里的csv只有参数分支)
    data_loader = DataLoader(batch_size=batch_size)#创建一个数据加载器实例，设置批量大小为5

    # 加载数据（三分支输入）
    result = data_loader.load_data(file_path)
    X_data_train, X_data_test, \
        X_img1_train, X_img1_test, \
        X_img2_train, X_img2_test, \
        y_train, y_test = result

    # 获取数据参数
    image1_channels = X_img1_train.shape[-1]
    image2_channels = X_img2_train.shape[-1]
    image_height = X_img1_train.shape[1]
    image_width = X_img1_train.shape[2]
    feature_dim = X_data_train.shape[1]

    # 创建三分支超模型
    hypermodel = ThreeBranchHyperModel(
        image1_channels=image1_channels,
        image2_channels=image2_channels,
        feature_dim=feature_dim,
        image_height=image_height,
        image_width=image_width,
        num_outputs=2
    )

    # 超参数调优
    print("\n=== 阶段1: 超参数调优 ===")
    best_model, best_history, train_metrics_logger = hyperparameter_tuning(
        hypermodel,
        X_data_train,
        X_img1_train,
        X_img2_train,
        y_train,
        data_loader=data_loader,
        batch_size=batch_size,
        epochs=epochs,
        max_trials=num_trials
    )

    # 保存最佳模型
    os.makedirs('../models', exist_ok=True)
    model_path = os.path.join('../models', f'best_{model_name}.h5')
    best_model.save(model_path)
    print(f"最佳模型已保存至 {model_path}")


    model_path = os.path.join('../models', f'best_{model_name}.h5')
    best_model = load_model(
        model_path,
        custom_objects={'ThreeBranchCrossModalAttention': ThreeBranchCrossModalAttention,
                        'mae': tf.keras.losses.MeanAbsoluteError()},

    )
    # best_model.summary()

    # 测试集评估
    print("\n=== 阶段2: 测试集评估 ===")
    test_dataset = data_loader.create_datasets(
        X_data_test, X_img1_test, X_img2_test, y_test, shuffle=False)

    # 加载scaler
    scaler_pull = joblib.load('../Data_scaler/scaler_pull.pkl') if os.path.exists('../Data_scaler/scaler_pull.pkl') else None
    scaler_nugget = joblib.load('../Data_scaler/scaler_nugget.pkl') if os.path.exists('../Data_scaler/scaler_nugget.pkl') else None

    # 绘制预测结果并获取测试指标
    _, _, test_metrics = plot_dual_output_predictions(
        model_name,
        best_model,
        test_dataset,
        scaler_pull,
        scaler_nugget,
        save_path=f'../Image_Result/three_branch_946.png'
    )

    # 生成并打印指标表格
    train_metrics = {
        'best_val_pulltest_mae': train_metrics_logger.best_val_pulltest_mae,
        'best_val_nugget_mae': train_metrics_logger.best_val_nugget_mae,
        'best_val_mae_mean': train_metrics_logger.best_val_mae_mean,
        'best_val_pulltest_r2': train_metrics_logger.best_val_pulltest_r2,
        'best_val_nugget_r2': train_metrics_logger.best_val_nugget_r2,
        'best_val_r2_mean': train_metrics_logger.best_val_r2_mean
    }
    print_result_table(train_metrics, test_metrics)


if __name__ == "__main__":
    main_function()
