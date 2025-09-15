import csv
import networkx as nx
import json
import numpy as np
import os
from typing import Any, Dict, List, Optional, Union

def load_network_from_csv(file_path, directed=False):

    network = nx.DiGraph()
    edge_list = []
    try:
        with open(file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:

                if not row:
                    continue
                    
                if len(row) < 3:
                    print(f"warning: line {row} has wrong format, skipped!")
                    continue
                
                try:
                    source = int(row[0])
                    target = int(row[1])
                    edge_type = int(row[2])

                    if directed and edge_type == 1:
                            edge_list.append((source, target))
                    else:
                        edge_list.append((source, target))
                        edge_list.append((target, source))
                except ValueError as e:
                    print(f"warning: can not transform {row} to Integer: {e}")
    except Exception as e:
        print(e)


    network.add_edges_from(edge_list) 
    return network

def digraph_to_adjacency_matrix(digraph, dtype=int):

    nodes = range(len(list(digraph.nodes())))

    adj_sparse = nx.adjacency_matrix(digraph, nodelist=nodes, weight=None)
    
    adj_matrix = adj_sparse.toarray().astype(dtype)
    
    return adj_matrix, nodes

def load_json_file(
    file_path: str,
    default: Optional[Union[Dict, List]] = None,
    encoding: str = 'utf-8'
) -> Union[Dict, List, Any]:
    """
    加载JSON文件并返回解析后的数据
    
    参数:
        file_path: JSON文件路径
        default: 当文件不存在或加载失败时返回的默认值
        encoding: 文件编码格式
    
    返回:
        解析后的JSON数据，或默认值（如果加载失败）
    """
    # 检查文件是否存在
    def _deserialize(obj: Any) -> Any:
            if not isinstance(obj, dict):
                return obj
            
            # 处理序列化的ndarray
            if '_type' in obj and obj['_type'] == 'ndarray':
                return np.array(
                    obj['data'],
                    dtype=obj['dtype']
                ).reshape(obj['shape'])
            
            # 递归处理其他字典
            return {k: _deserialize(v) for k, v in obj.items()}

    if not os.path.exists(file_path):
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON文件不存在: {file_path}")
    
    # 检查是否为文件
    if not os.path.isfile(file_path):
        if default is not None:
            return default
        raise IsADirectoryError(f"{file_path}是一个目录，不是文件")
    
    try:
        # 尝试加载并解析JSON文件
        with open(file_path, 'r', encoding=encoding) as f:
            return json.load(f,object_hook=_deserialize)
    
    except json.JSONDecodeError as e:
        print(f"JSON解析错误 in {file_path}: {str(e)}")
    except UnicodeDecodeError as e:
        print(f"编码错误 in {file_path}: {str(e)}，尝试使用其他编码？")
    except IOError as e:
        print(f"文件读写错误 in {file_path}: {str(e)}")
    
    # 处理加载失败的情况
    if default is not None:
        return default
    raise RuntimeError(f"无法加载JSON文件: {file_path}")

def save_to_json(
    data: Any,
    file_path: str,
    indent: int = 2,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
    create_dir: bool = True
) -> bool:
    """
    将数据保存为JSON文件的便捷函数
    
    参数:
        data: 要保存的数据（必须是JSON可序列化的）
        file_path: 目标文件路径
        indent: 格式化缩进空格数
        sort_keys: 是否按键名排序
        ensure_ascii: 是否确保ASCII编码（False保留中文等字符）
        create_dir: 如果目录不存在，是否自动创建
    
    返回:
        保存成功返回True，否则返回False
    """
    def _serialize(obj: Any) -> Any:
            # 处理ndarray类型
            if isinstance(obj, np.ndarray):
                return {
                    '_type': 'ndarray',
                    'dtype': str(obj.dtype),
                    'shape': obj.shape,
                    'data': obj.tolist()
                }
            # 处理其他NumPy标量类型
            elif isinstance(obj, np.generic):
                return obj.item()
            # 处理列表和元组
            elif isinstance(obj, (list, tuple)):
                return [_serialize(item) for item in obj]
            # 处理字典
            elif isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            # 其他类型直接返回
            return obj
    try:
        # 确保目录存在
        if create_dir:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        
        # 写入JSON文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(
                _serialize(data),
                f,
                indent=indent,
                sort_keys=sort_keys,
                ensure_ascii=ensure_ascii
            )
        return True
    except (TypeError, ValueError) as e:
        print(f"数据无法序列化为JSON: {str(e)}")
    except IOError as e:
        print(f"文件写入错误: {str(e)}")
    return False



class JSONDataSaver:
    """JSON数据存储类，提供更灵活的JSON文件存储功能"""
    
    def __init__(
        self,
        default_path: str = "data.json",
        default_indent: int = 2,
        default_sort: bool = False,
        ensure_ascii: bool = False
    ):
        """
        初始化JSON数据存储器
        
        参数:
            default_path: 默认文件路径
            default_indent: 默认缩进空格数
            default_sort: 默认是否排序键
            ensure_ascii: 是否确保ASCII编码
        """
        self.default_path = default_path
        self.default_indent = default_indent
        self.default_sort = default_sort
        self.ensure_ascii = ensure_ascii
    
    def save(
        self,
        data: Any,
        file_path: Optional[str] = None,
        indent: Optional[int] = None,
        sort_keys: Optional[bool] = None,
        create_dir: bool = True
    ) -> bool:
        """
        保存数据到JSON文件
        
        参数:
            data: 要保存的数据
            file_path: 目标文件路径，为None则使用默认路径
            indent: 缩进空格数，为None则使用默认值
            sort_keys: 是否排序键，为None则使用默认值
            create_dir: 是否自动创建目录
        
        返回:
            保存成功返回True
        """
        # 使用默认参数（如果未指定）
        actual_path = file_path or self.default_path
        actual_indent = indent if indent is not None else self.default_indent
        actual_sort = sort_keys if sort_keys is not None else self.default_sort
        
        return save_to_json(
            data=data,
            file_path=actual_path,
            indent=actual_indent,
            sort_keys=actual_sort,
            ensure_ascii=self.ensure_ascii,
            create_dir=create_dir
        )
    
    def append_to_array(
        self,
        item: Any,
        file_path: Optional[str] = None,
        indent: Optional[int] = None,
        sort_keys: Optional[bool] = None
    ) -> bool:
        """
        向JSON文件中的数组追加元素（如果文件不存在则创建新数组）
        
        参数:
            item: 要追加的元素
            file_path: 目标文件路径
            indent: 缩进空格数
            sort_keys: 是否排序键
        
        返回:
            操作成功返回True
        """
        actual_path = file_path or self.default_path
        
        # 尝试加载现有数据
        data = []
        if os.path.exists(actual_path):
            try:
                with open(actual_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    print(f"文件内容不是数组，无法追加元素")
                    return False
            except (json.JSONDecodeError, IOError) as e:
                print(f"读取文件失败: {str(e)}")
                return False
        
        # 追加新元素
        data.append(item)
        
        # 保存更新后的数据
        return self.save(
            data=data,
            file_path=actual_path,
            indent=indent,
            sort_keys=sort_keys
        )