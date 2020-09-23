import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve

class Network:
    EXPLICIT = 0
    dt = 1e-3

    def __init__(self, N, p, A_p, A_m, g, gamma, F, tao_p=50, tao_m=100, tao=5, tao_0=2 * 1e5, noise=None,
                 stdp_kernel=None, max_time=10000, seed=None):
        # initializing parameters
        self.N = N
        self.p = p
        self.A_p = A_p
        self.A_m = A_m
        self.g = g
        self.F = F
        self.gamma = gamma
        self.tao_p = tao_p
        self.tao_m = tao_m
        self.tao = tao
        self.tao_0 = tao_0
        self.noise = noise
        # either use a given function as STDP kernel, or the one in the paper
        self.stdp_kernel = lambda delta_t: (self.A_p * (np.exp(delta_t / self.tao_p))) if delta_t < 0 else (
                self.A_m * (np.exp(-delta_t / self.tao_m))) if not stdp_kernel else stdp_kernel

        # set the length of the second stage simulation
        self.max_time = max_time
        # randomize patterns
        if seed:
            np.random.seed(seed)
        self.memory_patterns = 2 * np.eye(self.N)[:p,
                                   :] - 1  # TODO:2 * np.random.binomial(1, .5, (self.p, self.N)) - 1
        # do they need to be orthogonal? it

        # is computationally inefficient to randomize orthogonal sign vectors
        self.coef = np.zeros(self.p)  # the strength of every memory pattern
        self.coef_history = None
        self.coef[Network.EXPLICIT] = np.random.rand(1)
        self.P = (1.0 / self.N) * np.dstack(
            [x * y for x, y in
             [np.meshgrid(self.memory_patterns[i], self.memory_patterns[i]) for i in range(self.p)]]).T

        self.W = np.sum(self.coef[:, np.newaxis, np.newaxis] * self.P, axis=0)
        self.__overall_int_of_k=0.1
        self.delta_k_long=-self.A_m*self.tao_m-self.A_p*self.tao_p+self.__overall_int_of_k
        self.find_f()
    def find_f(self):
        self.b=fsolve(lambda b:self.F(self.gamma*(b**3)*self.__overall_int_of_k)-b,np.ndarray([1]))[0]
        self.f =self.b * self.memory_patterns[Network.EXPLICIT]
    def run_first_stage(self):
        explicit_pattern = self.memory_patterns[Network.EXPLICIT]

    def run_second_stage(self):
        self.coef[Network.EXPLICIT + 1:] = np.random.uniform(9, 10, self.p - 1)

    def delta_u_dynamics(self, value,with_noise=False,t_greater_then_zero=False):
        return (-value + self.g * (self.W @ value)+(np.random.normal(0,self.noise,self.N) if with_noise else 0)) / self.tao if t_greater_then_zero else 0.


    def w_dynamics(self, f_t, t):
        pass

    def euler_iterator(self, initial_value, func,t0=0, noise_func=None):
        if noise_func:
            def iterator():
                cur_val = initial_value
                sqrt_dt = np.sqrt(Network.dt)
                t = t0
                while True:
                    cur_val += Network.dt * func(cur_val, t) + sqrt_dt * noise_func()
                    t += Network.dt
                    yield cur_val
        else:
            def iterator():
                cur_val = initial_value
                t = t0
                while True:
                    cur_val += Network.dt * func(cur_val, t)
                    t += Network.dt
                    yield cur_val
        return iterator

    def noise_dynamics(self):
        pass
