import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from tqdm import tqdm


class Network:
    EXPLICIT = 0
    WIDTH = 1000

    def __init__(self, N, p, A_p, A_m, g, gamma, tao_p=50, tao_m=100, tao=5, tao_0=2 * 1e5, noise=None,
                 stdp_kernel=None, seed=None):
        # initializing parameters
        self.N = N
        self.p = p
        self.A_p = A_p
        self.A_m = A_m
        self.g = g
        self.gamma = gamma
        self.tao_p = tao_p
        self.tao_m = tao_m
        self.tao = tao
        self.tao_0 = tao_0
        self.noise = noise * (np.sqrt(self.N) / np.sqrt(1024))
        self.dt = 0.5
        self.second_phase_flag = False
        # either use a given function as STDP kernel, or the one in the paper
        self.stdp_kernel = self.default_stdp_kernel if stdp_kernel is None else stdp_kernel
        self.current_time = 0.
        # randomize patterns
        if seed:
            np.random.seed(seed)
        self.memory_patterns = np.zeros((self.p, self.N))
        self.init_memory_patterns()
        # do they need to be orthogonal? it

        # is computationally inefficient to randomize orthogonal sign vectors
        self.coef = np.zeros(self.p)  # the strength of every memory pattern
        self.coef_history = None
        self.P = (1.0 / self.N) * np.dstack(
            [x * y for x, y in
             [np.meshgrid(self.memory_patterns[i], self.memory_patterns[i]) for i in range(self.p)]]).T

        self.__overall_int_of_k = 10
        # self.delta_k_long = -self.A_p * self.tao_p - self.A_m * self.tao_m + self.__overall_int_of_k
        self.delta_k_long_calc = quad(self.kl_kernel_for_calc, -np.inf, np.inf)[0]
        self.delta_k_long = quad(self.kl_kernel, -np.inf, np.inf)[0]
        self.find_f()
        self.coef[Network.EXPLICIT] = 10  # self.gamma * (
        # self.b ** 2) * self.N * (self.A_p * self.tao_p + self.A_m * self.tao_m + self.delta_k_long)
        self.W = np.sum(self.coef[:, np.newaxis, np.newaxis] * self.P, axis=0)

    # def get_noise_level(self):
    #     upper_noise_bound = (self.tao * (self.b ** 2)) / (self.g ** 2)
    #     lower_noise_bound = max((1. / (self.gamma * (self.g ** 3) * self.A_p)) * (self.tao / (self.tao_p ** 2)),
    #                             (1. / (self.gamma * (self.g ** 3) * self.A_m)) * (self.tao / (self.tao_m ** 2)))
    #     self.noise = (upper_noise_bound + lower_noise_bound) / 2.

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
        return A * np.exp(-np.abs(delta_t) / tao_arr) + self.kl_kernel(delta_t)

    def kl_kernel(self, delta_t):
        return (20+1e-11) * (np.exp(
            -np.abs(delta_t) / np.abs(self.A_m * self.tao_m + self.A_p * self.tao_p)) / self.delta_k_long_calc)

    def kl_kernel_for_calc(self, delta_t):
        return np.exp(-np.abs(delta_t) / np.abs(self.A_m * self.tao_m + self.A_p * self.tao_p))

    def find_f(self):
        self.b = fsolve(
            lambda b: self.gamma * (b ** 3) * (self.A_m * self.tao_m + self.A_p * self.tao_p + self.delta_k_long) - 1,
            200.)[0]  # TODO: make sure this is the correct linearization
        self.f = self.memory_patterns[Network.EXPLICIT] * self.b

    def run_second_stage(self):
        self.coef[Network.EXPLICIT + 1:] = np.random.uniform(9, 10, self.p - 1)

    def delta_u_dynamics(self, value, t, with_noise=False):
        a = (-value + self.g * (self.W @ value) + (
            np.random.normal(0, self.noise / np.sqrt(self.dt),
                             self.N) if with_noise else 0)) / self.tao
        return a
        # return (-value + self.g * (self.W @ value) + (
        #     np.random.normal(0, self.noise, self.N) if with_noise else 0)) / self.tao
        # scaling by 1/sqrt(dt) so that when performing the euler method for ODE (multiplying by dt),
        # the noise will be scaled by sqrt(dt) - to get the euler-maruyama method

    def w_dynamics(self, delta_u, t):
        """

        :param delta_u: array of delta_u per timestep, timestep rows and N columns
        :return:
        """
        firing_rates = self.f + self.g * delta_u
        Ks = np.vstack(
            [self.stdp_kernel(np.linspace(0, t, delta_u.shape[0]) - t), self.stdp_kernel(
                t - np.linspace(0, t, delta_u.shape[0]))])
        delta_u_int = ((Ks @ firing_rates) * self.dt)
        outer0 = self.gamma * np.outer(firing_rates[-1, :], delta_u_int[0, :]).T
        outer1 = self.gamma * np.outer(firing_rates[-1, :], delta_u_int[1, :])
        return (-self.W + outer0 + outer1) / self.tao_0

    def run_first_phase(self, LIMIT=None, with_noise=False):
        if LIMIT is None:
            LIMIT = int(self.tao_0 * 5) / self.dt
        print(f"LIMIT={LIMIT}")
        if self.coef_history is None:
            self.delta_u = np.full((1, self.N), 0)
            self.coef_history = np.zeros((1, self.P.shape[0]))
            self.coef_history[0, :] = self.coef
        for _ in tqdm(range(int(LIMIT))):
            self.delta_u = np.vstack(
                [self.delta_u,
                 self.delta_u[-1, :] + self.dt * self.delta_u_dynamics(self.delta_u[-1], self.current_time,
                                                                       with_noise=with_noise)])
            self.W += self.dt * self.w_dynamics(self.delta_u, self.current_time)
            self.current_time += self.dt
            self.coef_history = np.vstack([self.coef_history, self.P[:, 0, :] @ self.W[:, 0] / self.P[:, 0, 0]])
        self.coef[:] = self.coef_history[-1, :]
        return self.coef_history.copy(), self.delta_u.copy()

    def run_second_phase(self, min_pattern_strength, max_pattern_strength, with_noise=False, max_time=None,
                         explicit_pattern=None):
        if not self.second_phase_flag:
            self.second_phase_flag = True
            self.coef_history = None
            self.delta_u = None
            self.current_time = 0
        if max_time is None:
            max_time = 10000
        max_iteration = int(max_time / self.dt)
        if explicit_pattern is None:
            explicit_pattern = Network.EXPLICIT
        if self.coef_history is None:
            self.coef[np.arange(self.p) != explicit_pattern] = np.random.uniform(min_pattern_strength,
                                                                                 max_pattern_strength,
                                                                                 self.p - 1)
            self.coef_history = np.zeros((max_iteration, self.P.shape[0]))
            self.coef_history[0, :] = self.coef
            iteration_range = range(1, max_iteration)
            self.W = np.sum(self.coef[:, np.newaxis, np.newaxis] * self.P, axis=0)
            self.delta_u = np.random.normal(0, self.noise, (1, self.N)) if with_noise else np.zeros((1, self.N))
        else:
            first_row_index = self.coef_history.shape[0]
            self.coef_history = np.vstack([self.coef_history, np.zeros((max_iteration, self.P.shape[0]))])
            self.coef_history[first_row_index, :] = self.coef
            iteration_range = range(first_row_index, first_row_index + max_iteration)
        self.f = self.b * self.memory_patterns[explicit_pattern]
        self.current_time += self.dt
        self.W = np.sum(self.coef[:, np.newaxis, np.newaxis] * self.P, axis=0)
        for i in tqdm(iteration_range):
            self.delta_u = np.vstack([self.delta_u,
                                      self.delta_u[-1] + self.dt * self.delta_u_dynamics(self.delta_u[-1],
                                                                                         self.current_time,
                                                                                         with_noise=with_noise)])
            self.W += self.dt * self.w_dynamics(self.delta_u, self.current_time)
            self.current_time += self.dt
            self.coef_history[i, :] = self.P[:, 0, :] @ self.W[:, 0] / self.P[:, 0, 0]
        self.coef = self.coef_history[-1, :]
        return self.coef_history.copy(), self.delta_u.copy()
