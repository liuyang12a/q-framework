import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.integrate import solve_ivp
from scipy.linalg import null_space


def kolmogorov_forward(Q: np.ndarray, t: float, method: str = 'RK45') -> np.ndarray:
    """
    用Kolmogorov前向方程求解状态转移矩阵P(t)
    
    参数:
        Q: 生成器矩阵（n×n）
        t: 目标时刻
        method: 微分方程求解方法（如'RK45'、'BDF'，刚性系统建议用'BDF'）
    
    返回:
        P: 时刻t的状态转移矩阵（n×n），P[i][j]表示t时刻从i到j的转移概率
    """
    # 验证生成器矩阵
    check_generator_matrix(Q)
    n = Q.shape[0]
    
    # 定义微分方程：dP/dt = P @ Q（将矩阵展平为向量处理）
    def dP_dt(t: float, P_flat: np.ndarray) -> np.ndarray:
        P = P_flat.reshape(n, n)  # 向量转矩阵
        dP = P @ Q  # 前向方程
        return dP.flatten()  # 矩阵转向量
    
    # 初始条件：P(0) = 单位矩阵
    P0 = np.eye(n).flatten()
    
    # 求解微分方程（从0到t）
    sol = solve_ivp(
        fun=dP_dt,
        t_span=(0.0, t),
        y0=P0,
        method=method,
        t_eval=[t]  # 仅需t时刻的解
    )
    
    # 提取t时刻的解并转换为矩阵
    if not sol.success:
        raise RuntimeError(f"求解失败：{sol.message}")
    P_t = sol.y[:, 0].reshape(n, n)
    
    # 确保每行和为1（概率归一化，数值误差修正）
    row_sums = P_t.sum(axis=1, keepdims=True)
    P_t = P_t / row_sums
    
    return P_t

def ctmc_stationary_distribution(Q: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    求解连续时间马尔可夫链(CTMC)的平稳分布
    
    参数:
        Q: 生成器矩阵(shape=(n, n))
        tol: 数值计算的容差
    
    返回:
        pi: 平稳分布向量(shape=(n,)), 满足pi @ Q = 0且sum(pi) = 1
    """
    # 验证输入矩阵
    n = Q.shape[0]
    if Q.shape != (n, n):
        raise ValueError("生成器矩阵必须是方阵")
    if not np.allclose(np.sum(Q, axis=1), 0.0, atol=tol):
        raise ValueError("生成器矩阵每行之和必须为0")
    
    # 方法1: 求解零空间 (pi @ Q = 0)
    # 计算Q的左零空间（行向量）
    null = null_space(Q.T)  # 转置后求右零空间等价于原矩阵的左零空间
    
    # 零空间应该是一维的（对于不可约CTMC）
    if null.shape[1] != 1:
        raise ValueError("生成器矩阵的零空间维度不为1，可能不满足不可约性条件")
    
    # 提取零空间向量并归一化
    pi = null.flatten()
    pi = pi / np.sum(pi)
    
    # 确保所有元素为正（平稳分布应为概率分布）
    if np.any(pi < -tol):
        raise ValueError("计算得到的平稳分布包含负值，可能矩阵不满足正常返条件")
    pi = np.maximum(pi, 0.0)  # 处理数值误差导致的微小负值
    pi = pi / np.sum(pi)  # 重新归一化
    
    return pi

def power_iteration_method(Q: np.ndarray, tol: float = 1e-10, max_iter: int = 10000) -> np.ndarray:
    """
    用幂迭代法求解平稳分布（通过矩阵指数近似）
    
    参数:
        Q: 生成器矩阵
        tol: 收敛容差
        max_iter: 最大迭代次数
    
    返回:
        pi: 平稳分布向量
    """
    n = Q.shape[0]
    # 初始化概率分布（均匀分布）
    pi = np.ones(n) / n
    # 使用一个较大的时间步长近似极限分布
    # 这里使用矩阵指数的近似迭代: P(t) = exp(Qt)
    # 迭代公式: pi_{k+1} = pi_k @ (I + Q*dt)
    
    dt = 0.1  # 时间步长
    P = np.eye(n) + Q * dt  # 转移矩阵近似
    
    for _ in range(max_iter):
        pi_new = pi @ P
        pi_new = pi_new / np.sum(pi_new)  # 归一化
        if np.linalg.norm(pi_new - pi) < tol:
            return pi_new
        pi = pi_new
    
    raise RuntimeError(f"幂迭代法在{max_iter}次迭代后未收敛")

class ContinuousTimeMarkovChain:
    """连续时间马尔可夫链(CTMC)实现"""
    
    def __init__(self, states, generator_matrix):
        """
        初始化连续时间马尔可夫链
        
        参数:
            states: 状态列表
            generator_matrix: 生成器矩阵(Q矩阵)，形状为(n, n)，其中n是状态数量
                             Q[i][j]表示从状态i到状态j的转移率(i≠j)
                             Q[i][i] = -sum(Q[i][j] for j≠i)
        """
        self.states = states
        self.num_states = len(states)
        self.state_index = {state: i for i, state in enumerate(states)}
        
        # 验证生成器矩阵
        self.generator = np.array(generator_matrix)
        if self.generator.shape != (self.num_states, self.num_states):
            raise ValueError(f"生成器矩阵形状应为({self.num_states}, {self.num_states})")
        
        # 检查对角元素是否正确
        for i in range(self.num_states):
            row_sum = np.sum(self.generator[i, :])
            if not np.isclose(row_sum, 0.0, atol=1e-9):
                raise ValueError(f"生成器矩阵第{i}行的和应为0，实际为{row_sum}")
    
    def _get_transition_rates(self, current_state):
        """获取当前状态的转移率"""
        idx = self.state_index[current_state]
        return self.generator[idx, :]
    
    def _holding_time(self, current_state):
        """计算在当前状态的停留时间（服从指数分布）"""
        idx = self.state_index[current_state]
        # 总转移率 = -Q[i][i]
        total_rate = -self.generator[idx, idx]
        
        if total_rate <= 0:
            return np.inf  # 吸收态，停留时间无穷大
        
        # 指数分布的参数为λ=total_rate
        return expon.rvs(scale=1/total_rate)
    
    def _next_state(self, current_state):
        """根据转移率选择下一个状态"""
        idx = self.state_index[current_state]
        rates = self.generator[idx, :].copy()
        
        # 排除当前状态
        rates[idx] = 0.0
        
        # 计算转移概率
        total_rate = np.sum(rates)
        if total_rate <= 0:
            return current_state  # 吸收态，无法转移
        
        probabilities = rates / total_rate
        
        # 选择下一个状态
        next_idx = np.random.choice(self.num_states, p=probabilities)
        return self.states[next_idx]
    
    def simulate_continuous(self, initial_state, max_time=np.inf, max_steps=np.inf):
        """
        模拟CTMC过程
        
        参数:
            initial_state: 初始状态
            max_time: 最大模拟时间
            max_steps: 最大转移步数
            
        返回:
            包含时间点和对应状态的列表
        """
        if initial_state not in self.states:
            raise ValueError(f"初始状态{initial_state}不在状态列表中")
        
        # 初始化模拟结果
        time_points = [0.0]
        states = [initial_state]
        
        current_time = 0.0
        current_state = initial_state
        steps = 0
        
        while True:
            # 检查终止条件
            if current_time >= max_time or steps >= max_steps:
                break
            
            # 计算停留时间
            holding_time = self._holding_time(current_state)
            
            # 检查是否到达最大时间
            if current_time + holding_time > max_time:
                break
            
            # 更新时间
            current_time += holding_time
            time_points.append(current_time)
            
            # 转移到下一个状态
            current_state = self._next_state(current_state)
            states.append(current_state)
            
            steps += 1
            
            # 检查是否进入吸收态
            idx = self.state_index[current_state]
            if -self.generator[idx, idx] <= 1e-9:  # 总转移率接近0
                break
        
        return time_points, states
    

    def stationary_distribution(self):
        """计算平稳分布（如果存在）"""
        # 解方程组：πQ = 0 且 Σπ = 1
        A = np.vstack([self.generator.T, np.ones(self.num_states)])
        b = np.zeros(self.num_states + 1)
        b[-1] = 1.0
        
        # 使用最小二乘法求解
        pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return {self.states[i]: pi[i] for i in range(self.num_states)}
    
    def transition_probability(self, t):
        """计算t时刻的转移概率矩阵P(t) = exp(Qt)"""
        return np.matrix_exponentiate(self.generator * t)
    
    def visualize_simulation(self, time_points, states):
        """可视化模拟结果"""
        plt.figure(figsize=(12, 6))
        
        # 绘制状态随时间的变化
        state_indices = [self.state_index[s] for s in states]
        plt.step(time_points, state_indices, where='post', linewidth=2)
        
        plt.yticks(range(self.num_states), self.states)
        plt.xlabel("时间")
        plt.ylabel("状态")
        plt.title("连续时间马尔可夫链模拟")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 计算状态停留时间分布
        holding_times = np.diff(time_points)
        state_holding = {state: [] for state in self.states}
        
        for i in range(len(holding_times)):
            state_holding[states[i]].append(holding_times[i])
        
        # 绘制状态停留时间分布
        plt.figure(figsize=(12, 6))
        for i, state in enumerate(self.states):
            if state_holding[state]:
                plt.hist(
                    state_holding[state], 
                    bins=20, 
                    alpha=0.5, 
                    label=state,
                    density=True
                )
        
        plt.xlabel("停留时间")
        plt.ylabel("密度")
        plt.title("状态停留时间分布")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

