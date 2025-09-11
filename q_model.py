from ctmc import kolmogorov_forward, ctmc_stationary_distribution, power_iteration_method

import numpy as np

def cosine_distance_discrete(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    计算两个离散概率分布之间的余弦距离
    
    参数:
        p: 分布P的概率质量函数 (一维数组)
        q: 分布Q的概率质量函数 (一维数组)
        eps: 微小值，用于避免除以0的情况
    
    返回:
        cosine_dist: 余弦距离 (0到2之间)
    """
    # 验证输入
    if p.shape != q.shape:
        raise ValueError(f"p和q的形状必须相同，实际分别为{p.shape}和{q.shape}")
    
    # 计算点积和模长
    dot_product = np.dot(p, q)
    norm_p = np.linalg.norm(p)
    norm_q = np.linalg.norm(q)
    
    # 计算余弦相似度
    denominator = norm_p * norm_q
    if denominator < eps:
        return 1.0  # 当两个向量都接近零时，余弦距离为1
    
    cosine_similarity = dot_product / denominator
    
    # 余弦距离 = 1 - 余弦相似度
    return 1 - cosine_similarity

def kl_divergence_discrete(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    计算两个离散概率分布之间的KL散度 D_KL(P || Q)
    
    参数:
        p: 分布P的概率质量函数 (形状相同的一维数组)
        q: 分布Q的概率质量函数
        eps: 微小值，用于避免log(0)和除以0的情况
    
    返回:
        kl: KL散度值 (非负)
    """
    # 验证输入
    if p.shape != q.shape:
        raise ValueError(f"p和q的形状必须相同，实际分别为{p.shape}和{q.shape}")
    if not np.isclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError(f"p必须是概率分布，其和为{np.sum(p)}而非1")
    if not np.isclose(np.sum(q), 1.0, atol=1e-6):
        raise ValueError(f"q必须是概率分布，其和为{np.sum(q)}而非1")
    
    # 替换为微小值以避免数值问题
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    
    # 计算KL散度: sum(p * log(p / q))
    return np.sum(p * np.log(p / q))

class QModel:
    def __init__(self, states, Q):
        self.states = states
        self.num_states = len(states)
        self.state_index = {state: i for i, state in enumerate(states)}
        
        # 验证生成器矩阵
        self.Q = np.array(Q)
        if self.Q.shape != (self.num_states, self.num_states):
            raise ValueError(f"Wrong shape of Q, ({self.num_states}, {self.num_states}) is expected.")
        
        for i in range(self.num_states):
            self.Q[i][i] = self.Q[i][i] - sum(Q[i])

        try:
            self.stationary = ctmc_stationary_distribution(self.Q)
        except ValueError:
            self.stationary = power_iteration_method(self.Q)
        except RuntimeError:
            self.stationary = None

    def set_cycle_stride(self, t):
        self.transition_matrix = kolmogorov_forward(self.Q, t).T

    def forward(self, pre_spreading_state):
        next_spreading_state = np.matmul(self.transition_matrix,pre_spreading_state) 
        return next_spreading_state
    
    def run(self, init_spreading_state, max_time, cycle_time):
        steps = max_time/cycle_time
        self.set_cycle_stride(cycle_time)
        spreading_state = init_spreading_state

        spreading_dynamics = []
        for _ in range(int(steps)):
            spreading_state = self.forward(spreading_state)
            spreading_dynamics.append(spreading_state)
            
        if self.stationary is not None:
            distances = {
                "kl":[],
                "cosine":[]
            }
            for state in spreading_dynamics:
                distances['kl'].append(kl_divergence_discrete(self.stationary, state))
                distances['cosine'].append(cosine_distance_discrete(self.stationary, state))
        else:
            distances = None

        return spreading_dynamics, distances

class SI(QModel):
    def __init__(self, states, graph, lam, alpha):
        super().__init__(states, graph*lam)
        self.threshold = alpha/len(states)

class SIR(QModel):
    def __init__(self, states, graph, lam, alpha, delta1):
        if min(states)<=0:
            x = 1-min(states)
            new_states = [0]
            for i in range(len(states)):
                new_states.append(states[i]+x)
            states = new_states
        else:
            states = [0] + states
        
        graph = graph*lam
        graph = np.concat( (np.array([[delta1]]*(len(states)-1)), graph), axis=1)
        graph = np.concat( (np.array([[0.0]*len(states)]), graph), axis=0)

        super().__init__(states, graph)
        self.threshold = alpha/len(states)

class SIRS(QModel):
    def __init__(self, states, graph, lam, alpha, delta1, delta2):
        if min(states)<=0:
            x = 1-min(states)
            new_states = [0]
            for i in range(len(states)):
                new_states.append(states[i]+x)
            states = new_states
        else:
            states = [0] + states
        
        graph = graph*lam
        graph = np.concat( (np.array([[delta1]]*(len(states)-1)), graph), axis=1)
        graph = np.concat( (np.array([[delta2]*len(states)]), graph), axis=0)

        super().__init__(states, graph)
        self.threshold = alpha/len(states)
    