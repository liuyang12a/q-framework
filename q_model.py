from ctmc import kolmogorov_forward

import numpy as np

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

    def set_cycle_stride(self, t):
        self.transition_matrix = kolmogorov_forward(self.Q, t)

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

        return spreading_dynamics


class SI(QModel):
    def __init__(self, states, graph, lam, alpha):
        super().__init__(states, graph*lam)
        self.threshold = alpha/len(states)


    