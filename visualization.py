import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

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

# 示例使用
if __name__ == "__main__":
    # 1. 生成示例矩阵（随机矩阵）
    np.random.seed(42)
    random_matrix = np.random.rand(5, 5)  # 5x5的随机矩阵（0-1之间）
    
    # 2. 生成示例矩阵（相关系数矩阵）
    data = np.random.multivariate_normal(mean=[0, 0, 0, 0], cov=np.eye(4), size=100)
    corr_matrix = np.corrcoef(data.T)  # 4x4的相关系数矩阵（-1到1之间）
    
    # 3. 可视化随机矩阵
    plot_matrix_heatmap(
        matrix=random_matrix,
        title="随机矩阵热力图",
        x_labels=[f"特征{i}" for i in range(1, 6)],
        y_labels=[f"样本{i}" for i in range(1, 6)],
        cmap="coolwarm",
        annotate=True,
        fmt=".3f"
    )
    
    # 4. 可视化相关系数矩阵
    plot_matrix_heatmap(
        matrix=corr_matrix,
        title="特征相关系数矩阵",
        x_labels=["变量A", "变量B", "变量C", "变量D"],
        y_labels=["变量A", "变量B", "变量C", "变量D"],
        cmap="viridis",
        vmin=-1,  # 相关系数范围固定为-1到1
        vmax=1,
        annotate=True,
        fmt=".2f",
        figsize=(8, 6)
    )