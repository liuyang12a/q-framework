import json

from datetime import datetime
from typing import Dict, Any, Optional


def get_today():
    timestamp = datetime.timestamp(datetime.now())
    dt = datetime.fromtimestamp(timestamp) 
    return dt.strftime("%Y-%m-%d")



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
