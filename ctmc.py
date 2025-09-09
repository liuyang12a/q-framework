import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from scipy.stats import expon

from scipy.integrate import solve_ivp
from typing import Tuple

def check_generator_matrix(Q: np.ndarray) -> bool:
    """验证生成器矩阵的合法性（每行和为0）"""
    n = Q.shape[0]
    if Q.shape != (n, n):
        raise ValueError("生成器矩阵必须是方阵")
    row_sums = np.sum(Q, axis=1)
    if not np.allclose(row_sums, 0.0, atol=1e-9):
        raise ValueError(f"生成器矩阵每行之和必须为0，实际行和：{row_sums}")
    return True

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

def example_usage():
    """示例：求解简单CTMC的转移矩阵"""
    # 1. 定义生成器矩阵Q（两状态模型：健康->患病->健康）
    # Q = [
    #   [-λ, λ],  # 状态0（健康）：以λ速率转移到状态1
    #   [μ, -μ]   # 状态1（患病）：以μ速率转移到状态0
    # ]
    λ = 0.2  # 健康→患病的转移率
    μ = 0.1  # 患病→健康的转移率
    Q = np.array([
        [-λ, λ],
        [μ, -μ]
    ])
    
    # 2. 求解t=5时刻的转移矩阵
    t = 5.0
    P = kolmogorov_forward(Q, t, method='BDF')
    
    # 3. 输出结果
    print(f"生成器矩阵Q:\n{Q}\n")
    print(f"t={t}时刻的转移矩阵P(t):\n{P}\n")
    print(f"验证每行和为1（概率归一化）：{P.sum(axis=1)}")
    
    # 4. 解析解对比（两状态模型有解析解）
    def analytical_P(t):
        a = (λ + μ) * t
        p00 = (μ + λ * np.exp(-a)) / (λ + μ)
        p01 = 1 - p00
        p10 = (λ - λ * np.exp(-a)) / (λ + μ)
        p11 = 1 - p10
        return np.array([[p00, p01], [p10, p11]])
    
    P_analytical = analytical_P(t)
    print(f"\n解析解P(t):\n{P_analytical}")
    print(f"数值解与解析解误差：{np.linalg.norm(P - P_analytical)}")


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

# 示例使用
if __name__ == "__main__":
    # 定义状态
    states = ["健康", "患病", "康复", "死亡"]
    
    # 定义生成器矩阵(Q矩阵)
    # Q[i][j]表示从状态i到状态j的转移率(i≠j)
    generator = [
        # 健康 -> 健康(负和), 患病, 康复, 死亡
        [-0.05, 0.04, 0.0, 0.01],
        # 患病 -> 健康, 患病(负和), 康复, 死亡
        [0.02, -0.1, 0.07, 0.01],
        # 康复 -> 健康, 患病, 康复(负和), 死亡
        [0.03, 0.02, -0.06, 0.01],
        # 死亡 -> 所有转移率为0(吸收态)
        [0.0, 0.0, 0.0, 0.0]
    ]
    
    # 创建CTMC模型
    ctmc = ContinuousTimeMarkovChain(states, generator)
    
    # 可视化转移率矩阵
    ctmc.visualize_transition_rates()
    
    # 模拟过程
    time_points, states_sequence = ctmc.simulate(
        initial_state="健康", 
        max_time=100
    )
    
    # 可视化模拟结果
    ctmc.visualize_simulation(time_points, states_sequence)
    
    # 计算平稳分布
    stationary_dist = ctmc.stationary_distribution()
    print("平稳分布:")
    for state, prob in stationary_dist.items():
        print(f"  {state}: {prob:.4f}")