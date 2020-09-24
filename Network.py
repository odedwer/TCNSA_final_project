import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve


class Network:
    EXPLICIT = 0

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
        self.dt = min(tao_p, tao_m, tao, tao_0) / 1000.
        # either use a given function as STDP kernel, or the one in the paper
        self.stdp_kernel = self.default_stdp_kernel if stdp_kernel is None else stdp_kernel

        # set the length of the second stage simulation
        self.max_time = max_time
        # randomize patterns
        if seed:
            np.random.seed(seed)
        self.memory_patterns = np.zeros((self.p, self.N))
        self.init_memory_patterns()
        # do they need to be orthogonal? it

        # is computationally inefficient to randomize orthogonal sign vectors
        self.coef = np.zeros(self.p)  # the strength of every memory pattern
        self.coef[Network.EXPLICIT] = 9.5  # np.random.rand(1)  # init the explicit pattern strength
        self.coef_history = None
        self.P = (1.0 / self.N) * np.dstack(
            [x * y for x, y in
             [np.meshgrid(self.memory_patterns[i], self.memory_patterns[i]) for i in range(self.p)]]).T

        self.W = np.sum(self.coef[:, np.newaxis, np.newaxis] * self.P, axis=0)
        self.__overall_int_of_k = 1.2
        self.delta_k_long = -self.A_m * self.tao_m - self.A_p * self.tao_p + self.__overall_int_of_k
        self.find_f()

    def init_memory_patterns(self):
        self.memory_patterns[0] = 2 * np.random.binomial(1, .5, self.N) - 1
        for i in range(1, self.p):
            cur_vec = np.random.binomial(1, .5, self.N)
            while ((1. / self.N) * (self.memory_patterns @ cur_vec) != 0).any():
                cur_vec = 2 * np.random.binomial(1, .5, self.N) - 1
            self.memory_patterns[i] = cur_vec.copy()

    def default_stdp_kernel(self, delta_t):
        A = np.full_like(delta_t, self.A_m)
        tao_arr = np.full_like(delta_t, self.tao_m)
        negative = delta_t < 0
        A[negative] = self.A_p
        tao_arr[negative] = self.tao_p
        return A * np.exp(-np.abs(delta_t) / tao_arr)

    def find_f(self):
        self.b = fsolve(lambda b: self.F(self.gamma * (b ** 3) * self.__overall_int_of_k) - b, 100.)[0]
        self.f = self.memory_patterns[Network.EXPLICIT] * self.b

    def run_second_stage(self):
        self.coef[Network.EXPLICIT + 1:] = np.random.uniform(9, 10, self.p - 1)

    def delta_u_dynamics(self, value, t, with_noise=False):
        return (-value + self.g * (self.W @ value) + (
            np.random.normal(0, self.noise / np.sqrt(self.dt),
                             self.N) if with_noise else 0)) / self.tao
        # scaling by 1/sqrt(dt) so that when performing the euler method for ODE (multiplying by dt),
        # the noise will be scaled by sqrt(dt) - to get the euler-maruyama method

    def w_dynamics(self, delta_u, t):
        """

        :param delta_u: array of delta_u per timestep, timestep rows and N columns
        :return:
        """
        Ks = np.vstack([self.stdp_kernel(np.linspace(0, t, delta_u.shape[0]) - t), self.stdp_kernel(
            t - np.linspace(0, t, delta_u.shape[0]))])
        delta_u_int = ((Ks @ (self.f+self.g*delta_u)) * self.dt)
        outer = self.gamma * np.outer(delta_u_int, self.f + self.g * delta_u[-1])
        return (-self.W + outer[0].T + outer[1])/self.tao_0

    def euler_iterator(self, initial_value, func, t0=0):
        def iterator(args=None) -> np.array:
            cur_val = initial_value
            t = t0
            while True:
                cur_val += self.dt * func(cur_val, t, args) if args is not None else func(cur_val, t)
                t += self.dt
                yield cur_val

        return iterator

    def run_first_phase(self):
        delta_u = np.full((1, self.N), 0)
        dWdt = np.array([np.inf])
        coefs = np.zeros((5000, self.P.shape[0]))
        t = self.dt
        i = 0
        while (dWdt > self.dt * 1e-20).any() and i < 5000:
            print(i)
            delta_u = np.vstack(
                [delta_u, delta_u[-1] + self.dt * self.delta_u_dynamics(delta_u[-1], t, with_noise=True)])
            dWdt = self.dt * self.w_dynamics(delta_u, t)
            self.W += dWdt
            t += self.dt
            coefs[i, :] = self.P[:, 0, :] @ self.W[:, 0] / self.P[:, 0, 0]
            i += 1
        print(i)
        print(dWdt)
        return np.vstack([self.coef,coefs])
