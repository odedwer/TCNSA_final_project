import numpy as np


class Network:
    EXPLICIT = 0
    dt = 1e-3

    def __init__(self, N, p, A_p, A_m, g, tao_p=50, tao_m=100, tao=5, tao_0=2 * 1e5, noise=None,
                 stdp_kernel=None, max_time=10000, seed=None):
        # initializing parameters
        self.N = N
        self.A_p = A_p
        self.A_m = A_m
        self.tao_p = tao_p
        self.tao_m = tao_m
        self.tao = tao
        self.tao_0 = tao_0
        self.g = g
        self.p = p
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
        # the strength of every memory pattern
        self.coef = np.zeros(self.p)
        self.coef_history = None
        self.coef[Network.EXPLICIT] = np.random.rand(1)
        self.P = (1.0 / self.N) * np.dstack(
            [x * y for x, y in
             [np.meshgrid(self.memory_patterns[i], self.memory_patterns[i]) for i in range(self.p)]]).T
        self.W = np.sum(self.coef[:, np.newaxis, np.newaxis] * self.P, axis=0)

    def run_first_stage(self):
        explicit_pattern = self.memory_patterns[Network.EXPLICIT]

    def run_second_stage(self):
        self.coef[Network.EXPLICIT + 1:] = np.random.uniform(9, 10, self.p - 1)

    def u_dynamics(self):
        pass

    def firing_rate_dynamics(self):
        pass

    def w_dynamics(self):
        pass

    def euler_iterator(self,initial_value,func,dt):
        def itterator():
            t=0
            while(True):

                yield initial_value + dt*func(t)
        return itterator

    def noise_dynamics(self):
        pass
