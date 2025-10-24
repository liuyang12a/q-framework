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
    title="distribution",
    x_label="node_id",
    y_label="probability mass",
    color="green",
    color_group = None,
    fill_alpha=0.3,
    show_points=True,
    figsize=(10, 6),
    fixed_lim=False,
    y_lim=1.0,
    legend_ncol=1,
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
    plt.step(x_extended, y_extended, where='mid', color=color, linewidth=2.5, label="PMF")
    
    # 填充曲线下的区域（透明颜色）
    plt.fill_between(x_extended, y_extended, color=color, alpha=fill_alpha)
    
    # 显示离散点
    if show_points:
        if color_group is not None:
            group, colors = color_group
            for label, idx_group in group.items():
                xg = [x[i] for i in idx_group]
                yg = [y[i] for i in idx_group]
                plt.scatter(xg, yg, color=colors[label], s=50, zorder=3, label=label)
        else:
            plt.scatter(x, y, color=color, s=50, zorder=3, label="scatter")

    
    
    # 设置图表属性
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=30)
    plt.ylabel(y_label, fontsize=30)
    plt.xlim(x_extended[0], x_extended[-1])
    if fixed_lim:
        plt.ylim(0,y_lim)
    else:
        plt.ylim(0, max(y) * 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(fontsize=20,loc='best',ncol=legend_ncol)
    plt.tight_layout()
    plt.show()

def plot_line_chart(
    x_data: List[np.ndarray],
    y_data: List[np.ndarray],
    labels: List[str],
    title: str = "line",
    x_label: str = "X",
    y_label: str = "Y",
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    linewidth=4,
    grid: bool = True,
    legend: bool = True,
    highlight_points: Optional[List[Tuple[int, str]]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    legend_ncol=1,
    log_x=False,
    log_y=False
) -> None:
    """
    绘制折线图，支持多组数据、自定义样式和关键点标注
    
    参数:
        x_data: X轴数据列表，每组数据对应一条折线
        y_data: Y轴数据列表，与x_data一一对应
        labels: 每条折线的标签列表
        title: 图表标题
        x_label: X轴标签
        y_label: Y轴标签
        colors: 每条折线的颜色列表，None则使用默认颜色
        markers: 每条折线的标记样式列表
        linestyles: 每条折线的线条样式列表
        grid: 是否显示网格
        legend: 是否显示图例
        highlight_points: 要标注的关键点，格式为[(索引, 标注文本), ...]
        figsize: 图表尺寸
        save_path: 保存图表的路径，None则不保存
    """
    # 验证输入数据
    if len(x_data) != len(y_data) or len(x_data) != len(labels):
        raise ValueError("x_data, y_data和labels的长度必须一致")
    
    # 创建画布
    plt.figure(figsize=figsize)
    
    # 绘制每条折线
    for i in range(len(x_data)):
        # 设置样式参数
        color = colors[i] if colors and i < len(colors) else None
        marker = markers[i] if markers and i < len(markers) else None
        linestyle = linestyles[i] if linestyles and i < len(linestyles) else '-'
        
        # 绘制折线
        plt.plot(
            x_data[i], y_data[i],
            label=labels[i],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            markersize=6
        )
    
    # 标注关键点
    if highlight_points:
        for point in highlight_points:
            idx, text = point
            if 0 <= idx < len(x_data[0]) and len(x_data) == 1:  # 仅支持单组数据的关键点标注
                plt.scatter(
                    x_data[0][idx], y_data[0][idx],
                    color='red',
                    s=100,
                    zorder=5
                )
                plt.annotate(
                    text,
                    xy=(x_data[0][idx], y_data[0][idx]),
                    xytext=(10, 10),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='black')
                )
    
    # 设置图表属性
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(x_label, fontsize=30, labelpad=10)
    plt.ylabel(y_label, fontsize=25, labelpad=10)

    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')
    
    # 设置网格
    if grid:
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置图例
    if legend:
        plt.legend(fontsize=20, loc='upper right',ncol=legend_ncol)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()
