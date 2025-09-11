import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from scipy import stats
import seaborn as sns
from typing import List, Tuple, Optional

def plot_matrix_heatmap(matrix: np.ndarray,
                        title: str = "矩阵热力图",
                        x_labels: list = None,
                        y_labels: list = None,
                        cmap: str = "viridis",
                        annotate: bool = False,
                        fmt: str = ".2f",
                        figsize: tuple = (10, 8),
                        show_colorbar: bool = True,
                        vmin: float = None,
                        vmax: float = None,
                        save_path: str = None) -> None:
    """
    将矩阵用热力图可视化
    
    参数:
        matrix: 待可视化的矩阵（numpy数组）
        title: 热力图标题
        x_labels: x轴标签列表（长度需与矩阵列数一致）
        y_labels: y轴标签列表（长度需与矩阵行数一致）
        cmap: 颜色映射（如"viridis"、"coolwarm"、"binary"等）
        annotate: 是否在热力图上标注数值
        fmt: 数值标注的格式（如".2f"表示保留2位小数）
        figsize: 图像尺寸（宽, 高）
        show_colorbar: 是否显示颜色条
        vmin: 颜色映射的最小值（None则自动计算）
        vmax: 颜色映射的最大值（None则自动计算）
        save_path: 图像保存路径（如"heatmap.png"，None则不保存）
    """
    # 验证输入矩阵
    if not isinstance(matrix, np.ndarray):
        raise TypeError("输入必须是numpy数组")
    if matrix.ndim != 2:
        raise ValueError("仅支持二维矩阵可视化")
    
    # 创建画布
    plt.figure(figsize=figsize)
    
    # 绘制热力图
    im = plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    
    # 添加标题
    plt.title(title, fontsize=15)
    
    # 设置坐标轴标签
    if x_labels is not None:
        if len(x_labels) != matrix.shape[1]:
            raise ValueError(f"x_labels长度({len(x_labels)})与矩阵列数({matrix.shape[1]})不匹配")
        plt.xticks(range(matrix.shape[1]), x_labels, rotation=45, ha="right")
    else:
        plt.xticks(range(matrix.shape[1]))
    
    if y_labels is not None:
        if len(y_labels) != matrix.shape[0]:
            raise ValueError(f"y_labels长度({len(y_labels)})与矩阵行数({matrix.shape[0]})不匹配")
        plt.yticks(range(matrix.shape[0]), y_labels)
    else:
        plt.yticks(range(matrix.shape[0]))
    
    # 添加数值标注
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text_color = "white" if im.norm(matrix[i, j]) > 0.5 else "black"  # 根据背景色选择文字颜色
                plt.text(j, i, format(matrix[i, j], fmt),
                         ha="center", va="center", color=text_color, fontsize=8)
    
    # 显示颜色条
    if show_colorbar:
        cbar = plt.colorbar(im)
        cbar.set_label("数值", fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    # 显示图像
    plt.show()

# 设置绘图风格
plt.style.use('seaborn-v0_8-colorblind')
sns.set_palette('colorblind')

def plot_discrete_distribution_curve(
    values,
    probabilities=None,
    title="离散概率分布曲线",
    x_label="取值",
    y_label="概率",
    color="green",
    fill_alpha=0.3,
    show_points=True,
    figsize=(10, 6)
):
    """
    用阶梯曲线展示离散概率分布，并填充曲线下区域
    
    参数:
        values: 离散分布的取值（1D数组）
        probabilities: 对应取值的概率，若为None则从values计算频率
        title: 图表标题
        x_label: x轴标签
        y_label: y轴标签
        color: 曲线和填充区域的颜色
        fill_alpha: 填充区域的透明度（0-1）
        show_points: 是否显示离散点
        figsize: 图表尺寸
    """
    # 处理输入数据
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("values必须是一维数组")
    
    # 计算概率（如果未提供）
    if probabilities is None:
        unique_vals, counts = np.unique(values, return_counts=True)
        probabilities = counts / np.sum(counts)
    else:
        unique_vals = np.asarray(values)
        probabilities = np.asarray(probabilities)
        if len(unique_vals) != len(probabilities):
            raise ValueError("values和probabilities长度必须一致")
    
    # 确保值是排序的
    sorted_indices = np.argsort(unique_vals)
    x = unique_vals[sorted_indices]
    y = probabilities[sorted_indices]
    
    # 为阶梯曲线准备数据（在离散点之间保持水平）
    # 添加前后两个点使曲线完整
    x_extended = np.concatenate([[x[0] - (x[1]-x[0])/2], x, [x[-1] + (x[-1]-x[-2])/2]])
    y_extended = np.concatenate([[0], y, [0]])
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 绘制阶梯曲线（mid参数使阶梯在x位置居中）
    plt.step(x_extended, y_extended, where='mid', color=color, linewidth=2.5, label="概率质量函数")
    
    # 填充曲线下的区域（透明颜色）
    plt.fill_between(x_extended, y_extended, color=color, alpha=fill_alpha)
    
    # 显示离散点
    if show_points:
        plt.scatter(x, y, color=color, s=50, zorder=3, label="离散点")
    
    # 设置图表属性
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xlim(x_extended[0], x_extended[-1])
    plt.ylim(0, max(y) * 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


class DistributionVisualizer:
    """概率分布可视化工具"""
    
    @staticmethod
    def plot_discrete_distribution(
        dist, 
        title: str, 
        x_label: str = "值", 
        y_label: str = "概率",
        x_range: Tuple[int, int] = None,
        sample_size: int = 10000,
        show_samples: bool = True
    ) -> None:
        """
        可视化离散概率分布
        
        参数:
            dist: scipy的离散分布对象（如stats.binom, stats.poisson）
            title: 图表标题
            x_label: x轴标签
            y_label: y轴标签
            x_range: x轴范围 (min, max)，None则自动计算
            sample_size: 用于绘制直方图的样本量
            show_samples: 是否显示样本直方图
        """
        # 确定x轴范围
        if x_range is None:
            # 生成样本并根据样本确定范围
            samples = dist.rvs(size=sample_size)
            x_min, x_max = min(samples), max(samples)
            # 扩展一点范围使图表更美观
            x_min = max(0, x_min - 2) if 'binom' in str(dist) else x_min - 2
            x_max += 2
        else:
            x_min, x_max = x_range
        
        x = np.arange(x_min, x_max + 1)
        pmf = dist.pmf(x)  # 概率质量函数
        
        # 创建画布
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制概率质量函数
        ax.plot(x, pmf, 'bo', ms=8, label='PMF')
        ax.vlines(x, 0, pmf, colors='b', lw=2)
        
        # 绘制样本直方图
        if show_samples:
            samples = dist.rvs(size=sample_size)
            ax.hist(
                samples, 
                bins=np.arange(x_min, x_max + 2) - 0.5, 
                density=True, 
                alpha=0.3, 
                color='b', 
                label=f'{sample_size}个样本'
            )
        
        # 设置图表属性
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(0, max(pmf) * 1.1)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_continuous_distribution(
        dist, 
        title: str, 
        x_label: str = "值", 
        y_label: str = "概率密度",
        x_range: Tuple[float, float] = None,
        sample_size: int = 10000,
        show_samples: bool = True,
        show_cdf: bool = True
    ) -> None:
        """
        可视化连续概率分布
        
        参数:
            dist: scipy的连续分布对象（如stats.norm, stats.expon）
            title: 图表标题
            x_label: x轴标签
            y_label: y轴标签
            x_range: x轴范围 (min, max)，None则自动计算
            sample_size: 用于绘制直方图的样本量
            show_samples: 是否显示样本直方图
            show_cdf: 是否显示累积分布函数
        """
        # 确定x轴范围
        if x_range is None:
            # 使用分布的均值±3倍标准差作为范围
            mean, std = dist.mean(), dist.std()
            x_min, x_max = mean - 3 * std, mean + 3 * std
        else:
            x_min, x_max = x_range
        
        x = np.linspace(x_min, x_max, 1000)
        pdf = dist.pdf(x)  # 概率密度函数
        
        # 创建画布
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 绘制概率密度函数
        ax1.plot(x, pdf, 'b-', lw=2, label='PDF')
        
        # 绘制样本直方图
        if show_samples:
            samples = dist.rvs(size=sample_size)
            ax1.hist(
                samples, 
                bins=50, 
                density=True, 
                alpha=0.3, 
                color='b', 
                label=f'{sample_size}个样本'
            )
        
        # 设置主坐标轴属性
        ax1.set_title(title, fontsize=14)
        ax1.set_xlabel(x_label, fontsize=12)
        ax1.set_ylabel(y_label, fontsize=12, color='b')
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(0, max(pdf) * 1.1)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.3)
        
        # 绘制累积分布函数（次要坐标轴）
        if show_cdf:
            ax2 = ax1.twinx()
            cdf = dist.cdf(x)
            ax2.plot(x, cdf, 'r--', lw=2, label='CDF')
            ax2.set_ylabel('累积概率', fontsize=12, color='r')
            ax2.set_ylim(0, 1.05)
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()

def demonstrate_distributions():
    """展示各种概率分布的可视化效果"""
    # 1. 正态分布
    norm_dist = stats.norm(loc=0, scale=1)  # 均值0，标准差1的正态分布
    DistributionVisualizer.plot_continuous_distribution(
        norm_dist,
        title="标准正态分布 N(0, 1)",
        x_label="x",
        y_label="概率密度"
    )
    
    # 2. 指数分布
    expon_dist = stats.expon(scale=2)  # 尺度参数=2（均值=2）
    DistributionVisualizer.plot_continuous_distribution(
        expon_dist,
        title="指数分布 Exp(λ=0.5)",
        x_label="x",
        x_range=(0, 10)
    )
    
    # 3. 二项分布
    binom_dist = stats.binom(n=10, p=0.5)  # n=10次试验，成功概率p=0.5
    DistributionVisualizer.plot_discrete_distribution(
        binom_dist,
        title="二项分布 Binomial(n=10, p=0.5)",
        x_label="成功次数"
    )
    
    # 4. 泊松分布
    poisson_dist = stats.poisson(mu=3)  # 均值=3
    DistributionVisualizer.plot_discrete_distribution(
        poisson_dist,
        title="泊松分布 Poisson(λ=3)",
        x_label="事件数"
    )
    
    # 5. 均匀分布
    uniform_dist = stats.uniform(loc=0, scale=10)  # [0, 10)区间的均匀分布
    DistributionVisualizer.plot_continuous_distribution(
        uniform_dist,
        title="均匀分布 Uniform(0, 10)",
        x_label="x"
    )

if __name__ == "__main__":
    demonstrate_distributions()
    