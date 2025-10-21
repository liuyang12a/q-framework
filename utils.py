import os
import numpy as np

from datetime import datetime
from typing import Dict, Any, List, Callable, Optional
from scipy.stats import invgamma


def get_today():
    timestamp = datetime.timestamp(datetime.now())
    dt = datetime.fromtimestamp(timestamp) 
    return dt.strftime("%Y-%m-%d")

def trans_list2str(l, seg):
    s=''
    for i in range(len(l)-1):
        s += (l[i]+seg)
    
    return s + l[-1]
    
def sample_inverse_gamma(shape: float, scale: float, size: int = 1) -> np.ndarray:
    """
    从逆Gamma分布中采样
    
    参数:
        shape: 形状参数（α > 0）
        scale: 尺度参数（β > 0）
        size: 采样数量
    
    返回:
        采样结果的numpy数组
    """
    # 检查参数有效性
    if shape <= 0:
        raise ValueError("形状参数必须大于0")
    if scale <= 0:
        raise ValueError("尺度参数必须大于0")
    if size <= 0:
        raise ValueError("采样数量必须为正整数")
    
    # 从逆Gamma分布采样
    samples = invgamma.rvs(a=shape, scale=scale, size=size)
    return samples

def dict_to_single_line_text(
    d: Dict[Any, Any],
    format_type: str = "key_value",
    key_value_sep: str = "=",
    pair_sep: str = ";",
    prefix: str = "",
    suffix: str = ""
) -> str:
    """
    将字典的键值对转换为单行文本
    
    参数:
        d: 要转换的字典
        format_type: 输出格式类型
            - "key_value": 键值对格式 (key=value, key2=value2)
            - "json": JSON格式 {"key": value, "key2": value2}
            - "python": Python字典格式 {key: value, key2: value2}
        key_value_sep: 键和值之间的分隔符（仅用于key_value格式）
        pair_sep: 键值对之间的分隔符
        prefix: 文本前缀
        suffix: 文本后缀
    
    返回:
        单行文本字符串
    """
    # 处理不同格式
    if format_type == "json":
        items = []
        for k, v in d.items():
            # 字符串值添加双引号
            key_str = f'"{k}"' if isinstance(k, str) else str(k)
            if isinstance(v, str):
                val_str = f'"{v}"'
            elif isinstance(v, (list, dict, tuple)):
                # 简单处理嵌套结构
                val_str = str(v).replace("'", '"')
            else:
                val_str = str(v).lower() if isinstance(v, bool) else str(v)
            items.append(f"{key_str}: {val_str}")
        content = "{" + pair_sep.join(items) + "}"
    
    elif format_type == "python":
        items = [f"{k}: {v}" for k, v in d.items()]
        content = "{" + pair_sep.join(items) + "}"
    
    else:  # key_value格式
        items = []
        for k, v in d.items():
            # 处理值为复杂类型的情况
            if isinstance(v, (list, dict, tuple)):
                val_str = str(v).replace(" ", "").replace("\n", "")
            else:
                val_str = str(v)
            items.append(f"{k}{key_value_sep}{val_str}")
        content = pair_sep.join(items)
    
    # 添加前缀和后缀
    return f"{prefix}{content}{suffix}"


def traverse_files(
    root_dir: str,
    recursive: bool = True,
    filter_func: Optional[Callable[[str], bool]] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> List[str]:
    """
    遍历文件夹下的所有文件
    
    参数:
        root_dir: 要遍历的根目录
        recursive: 是否递归遍历子文件夹
        filter_func: 用于过滤文件的函数，返回True保留文件
        progress_callback: 处理每个文件的回调函数
    
    返回:
        所有符合条件的文件路径列表
    """
    file_paths = []
    
    # 检查根目录是否存在
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"目录不存在: {root_dir}")
    
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"{root_dir} 不是一个目录")
    
    # 遍历目录
    for entry in os.scandir(root_dir):
        if entry.is_file(follow_symlinks=False):
            # 处理文件
            file_path = entry.path
            # 应用过滤函数
            if filter_func is None or filter_func(file_path):
                file_paths.append(file_path)
                # 调用进度回调
                if progress_callback:
                    progress_callback(file_path)
        elif entry.is_dir(follow_symlinks=False) and recursive:
            # 递归处理子目录
            sub_files = traverse_files(
                entry.path,
                recursive,
                filter_func,
                progress_callback
            )
            file_paths.extend(sub_files)
    
    return file_paths