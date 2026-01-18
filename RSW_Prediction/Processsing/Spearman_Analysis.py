import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import matplotlib.patches as patches
import os

# 导入数据预处理模块（仅保留一次有效导入）
from Processsing import data_process


class SpearmanCorrelationAnalyzer:
    """Spearman相关矩阵分析器（表格全称+图表简写）"""

    def __init__(self):
        # 保持全称不变（确保表格输出为全称）
        self.key_metrics = ['Pressure', 'Welding_Time', 'Angle', 'Force', 'Current', 'Thickness_A']
        self.quality_metrics = ['PullTest', 'NuggetDiameter']
        self.all_metrics = self.key_metrics + self.quality_metrics

        # 全称到简写的映射字典（与指标一一对应）
        self.metric_short_name = {
            'Pressure': 'PR',          # 压力
            'Welding_Time': 'WT',      # 焊接时间
            'Angle': 'EA',             # 电极角度
            'Force': 'EF',             # 电极力
            'Current': 'WC',           # 焊接电流
            'Thickness_A': 'MT',       # 材料厚度
            'PullTest': 'TS',          # 拉力测试强度
            'NuggetDiameter': 'ND'     # 熔核直径
        }

    def create_advanced_spearman_correlation(self, df, save_path="../Image_Result/spearman_correlation_matrix.png"):
        """创建高级Spearman相关矩阵可视化（图表用简写）"""
        if not self._validate_data(df):
            return None

        # 基于全称筛选数据（不修改原始列名）
        data = df[self.all_metrics].copy()
        global_limits = self._calculate_global_limits(data)
        corr_matrix, p_values = self._calculate_spearman_matrix(data)

        fig, axes, cbar_ax, left_label_ax, bottom_label_ax = self._create_figure_layout()

        self._plot_all_subplots(axes, data, corr_matrix, p_values, global_limits)
        self._add_decorations(fig, left_label_ax, bottom_label_ax, cbar_ax)
        self._save_and_display(fig, save_path)

        return corr_matrix

    def _validate_data(self, df):
        """验证数据完整性（基于全称验证）"""
        missing_metrics = [metric for metric in self.all_metrics if metric not in df.columns]
        if missing_metrics:
            print(f"警告: 以下指标在数据中不存在: {missing_metrics}")
            return False
        return True

    def _calculate_global_limits(self, data):
        """计算统一的全局坐标范围（基于全称）"""
        global_limits = {}
        for metric in self.all_metrics:
            data_min = data[metric].min()
            data_max = data[metric].max()
            margin = (data_max - data_min) * 0.15  # 增加15%边距避免数据超出可视范围
            global_limits[metric] = (data_min - margin, data_max + margin)
        return global_limits

    def _calculate_spearman_matrix(self, data):
        """计算Spearman相关系数矩阵和p值（矩阵索引为全称）"""
        n_vars = len(data.columns)
        corr_matrix = pd.DataFrame(np.eye(n_vars), index=data.columns, columns=data.columns)
        p_values = pd.DataFrame(np.eye(n_vars), index=data.columns, columns=data.columns)

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                corr, p_val = spearmanr(data.iloc[:, i], data.iloc[:, j])
                corr_matrix.iloc[i, j] = corr_matrix.iloc[j, i] = corr
                p_values.iloc[i, j] = p_values.iloc[j, i] = p_val

        return corr_matrix, p_values

    def _create_figure_layout(self):
        """创建图形布局（适配指标数量）"""
        n_metrics = len(self.all_metrics)

        fig = plt.figure(figsize=(18, 18))

        # 调整网格比例，为标签预留空间
        gs = fig.add_gridspec(
            n_metrics + 2, n_metrics + 2,
            width_ratios=[0.15] + [1] * n_metrics + [0.1],
            height_ratios=[0.08] + [1] * n_metrics + [0.2],
            wspace=0.2, hspace=0.2
        )

        # 初始化子图数组
        axes = np.zeros((n_metrics, n_metrics), dtype=object)
        for i in range(n_metrics):
            for j in range(n_metrics):
                axes[i, j] = fig.add_subplot(gs[i + 1, j + 1])

        # 辅助轴（颜色条、标签）
        cbar_ax = fig.add_subplot(gs[1:-1, -1])
        left_label_ax = fig.add_subplot(gs[1:-1, 0])
        bottom_label_ax = fig.add_subplot(gs[-1, 1:-1])

        return fig, axes, cbar_ax, left_label_ax, bottom_label_ax

    def _plot_all_subplots(self, axes, data, corr_matrix, p_values, global_limits):
        """绘制所有子图（直方图/散点图/相关系数热图）"""
        n_metrics = len(self.all_metrics)
        cmap = plt.cm.coolwarm  # 相关系数颜色映射

        for i in range(n_metrics):
            for j in range(n_metrics):
                ax = axes[i, j]
                metric_i = self.all_metrics[i]
                metric_j = self.all_metrics[j]
                short_i = self.metric_short_name[metric_i]  # 映射为简写
                short_j = self.metric_short_name[metric_j]

                if i == j:
                    # 对角线：直方图（显示简写标签）
                    self._plot_histogram(ax, data[metric_i], short_i, global_limits, i, j, n_metrics)
                elif i < j:
                    # 上三角：相关系数热图（显示简写标签）
                    self._plot_correlation_heatmap(ax, corr_matrix.iloc[i, j], p_values.iloc[i, j],
                                                   short_i, short_j, cmap)
                else:
                    # 下三角：散点图（显示简写标签）
                    self._plot_scatter(ax, data[metric_j], data[metric_i],
                                       short_j, short_i, i, j, n_metrics, global_limits)

    def _plot_histogram(self, ax, data, label, global_limits, row_idx, col_idx, total_vars):
        """绘制单指标直方图（显示简写标签）"""
        if global_limits is not None:
            # 通过简写反向映射全称，获取坐标范围
            full_name = [k for k, v in self.metric_short_name.items() if v == label][0]
            data_range = global_limits[full_name]
            bins = np.linspace(data_range[0], data_range[1], 12)  # 均匀分箱
        else:
            bins = 12

        # 绘制直方图
        ax.hist(data, bins=bins, alpha=0.7, color='steelblue', edgecolor='black', density=True)

        # 固定坐标范围
        if global_limits is not None:
            ax.set_xlim(data_range)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.2)  # 预留统计文本空间

        # 设置刻度和统计信息
        self._setup_histogram_ticks(ax, row_idx, col_idx, total_vars)
        self._add_statistics_text(ax, data)

    def _setup_histogram_ticks(self, ax, row_idx, col_idx, total_vars):
        """优化直方图刻度显示"""
        ax.tick_params(axis='both', which='major', labelsize=8)

        # 限制刻度数量，避免拥挤
        if len(ax.get_xticks()) > 4:
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        if len(ax.get_yticks()) > 4:
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))

        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        self._rotate_yticklabels(ax)  # 旋转y轴标签避免重叠

    def _plot_scatter(self, ax, x_data, y_data, x_label, y_label, row_idx, col_idx, total_vars, global_limits):
        """绘制双指标散点图（显示简写标签）"""
        # 绘制散点
        ax.scatter(x_data, y_data, alpha=0.6, s=15, color='steelblue',
                   edgecolors='white', linewidth=0.3)

        # 固定坐标范围（基于全称映射）
        if global_limits is not None:
            x_full = [k for k, v in self.metric_short_name.items() if v == x_label][0]
            y_full = [k for k, v in self.metric_short_name.items() if v == y_label][0]
            ax.set_xlim(global_limits[x_full])
            ax.set_ylim(global_limits[y_full])

        # 设置刻度
        self._setup_scatter_ticks(ax, row_idx, col_idx, total_vars)
        ax.grid(False)

    def _setup_scatter_ticks(self, ax, row_idx, col_idx, total_vars):
        """优化散点图刻度显示"""
        ax.tick_params(axis='both', which='major', labelsize=10)

        # 限制刻度数量
        if len(ax.get_xticks()) > 4:
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        if len(ax.get_yticks()) > 4:
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))

        self._rotate_yticklabels(ax)
        self._rotate_xticklabels(ax)

    def _rotate_yticklabels(self, ax):
        """旋转y轴标签（垂直显示）"""
        labels = ax.get_yticklabels()
        if labels:
            for label in labels:
                label.set_rotation(90)
                label.set_fontsize(10)
                label.set_ha('center')
                label.set_va('center')

    def _rotate_xticklabels(self, ax):
        """旋转x轴标签（水平显示）"""
        labels = ax.get_xticklabels()
        if labels:
            for label in labels:
                label.set_rotation(0)
                label.set_fontsize(10)
                label.set_ha('center')
                label.set_va('center')

    def _plot_correlation_heatmap(self, ax, corr, p_val, var1, var2, cmap):
        """绘制相关系数热图（显示简写标签+显著性）"""
        # 归一化相关系数到[0,1]用于颜色映射
        norm_corr = (corr + 1) / 2
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor=cmap(norm_corr), transform=ax.transAxes))

        # 标注相关系数和p值（带显著性*）
        significance = '*' if p_val < 0.05 else ''
        corr_text = f'{corr:.3f}{significance}'
        p_text = f'p={p_val:.3f}' if p_val >= 0.001 else 'p<0.001'

        ax.text(0.5, 0.3, p_text, transform=ax.transAxes,
                ha='center', va='center', fontsize=14,
                color='white' if abs(corr) > 0.5 else 'black')

        ax.text(0.5, 0.6, corr_text, transform=ax.transAxes,
                ha='center', va='center', fontsize=14,
                color='white' if abs(corr) > 0.5 else 'black')

        ax.axis('off')  # 关闭坐标轴

    def _add_statistics_text(self, ax, data):
        """在直方图上添加均值和标准差"""
        stats_text = f'Mean: {data.mean():.2f}\nStd: {data.std():.2f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                va='top', ha='right', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    def _add_decorations(self, fig, left_label_ax, bottom_label_ax, cbar_ax):
        """添加外围标签和颜色条"""
        self._add_outer_labels(left_label_ax, bottom_label_ax)
        self._add_colorbar(fig, cbar_ax)
        plt.tight_layout(pad=4.0)  # 调整布局避免重叠

    def _add_outer_labels(self, left_ax, bottom_ax):
        """添加外围简写标签（左侧垂直，底部水平）"""
        left_ax.axis('off')
        bottom_ax.axis('off')
        n_metrics = len(self.all_metrics)

        # 左侧标签（垂直显示简写）
        for i, metric in enumerate(self.all_metrics):
            short_label = self.metric_short_name[metric]
            y_pos = (i + 0.5) / n_metrics
            left_ax.text(0.3, 1 - y_pos, short_label, rotation=90,
                         ha='center', va='center', fontsize=16,
                         transform=left_ax.transAxes)

        # 底部标签（水平显示简写）
        for j, metric in enumerate(self.all_metrics):
            short_label = self.metric_short_name[metric]
            x_pos = (j + 0.5) / n_metrics
            bottom_ax.text(x_pos, 0.3, short_label,
                           ha='center', va='center', fontsize=16,
                           transform=bottom_ax.transAxes)

    def _add_colorbar(self, fig, cbar_ax):
        """添加相关系数颜色条"""
        norm = plt.Normalize(-1, 1)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])  # 固定颜色条刻度

    def _save_and_display(self, fig, save_path):
        """保存图像并显示"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"相关矩阵（图表简写版）已保存至: {save_path}")

    def export_correlation_table(self, corr_matrix, save_path="../CSV_Result/spearman_correlation_table.csv"):
        """导出相关系数表格（保留全称）"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        corr_matrix.to_csv(save_path)
        print(f"相关系数表格（全称版）已导出至: {save_path}")


# 主程序运行
if __name__ == "__main__":
    # 加载数据
    file_path = '../Dataset/Data_RSW_1.csv'  # 替换为实际数据路径
    df = pd.read_csv(file_path)
    print("原始数据样本量：", len(df))

    # 按Sample_ID合并样本（保留每组第一组数据）
    df = df.groupby('Sample_ID').agg({
        'Pressure': 'first',
        'Welding_Time': 'first',
        'Angle': 'first',
        'Force': 'first',
        'Current': 'first',
        'Thickness_A': 'first',
        'PullTest': 'first',
        'NuggetDiameter': 'first',
        'Category': 'first'
    }).reset_index()

    # 保留70%样本
    df_size = int(len(df) * 0.7)
    df = df[:df_size]

    # 数据预处理（去异常值等）
    df, outlier_indices = data_process.preprocess_data(df)
    print(f"预处理后样本量（已去除异常值）：{len(df)}")

    # 生成并导出相关矩阵
    analyzer = SpearmanCorrelationAnalyzer()
    corr_matrix = analyzer.create_advanced_spearman_correlation(df)
    if corr_matrix is not None:
        analyzer.export_correlation_table(corr_matrix)