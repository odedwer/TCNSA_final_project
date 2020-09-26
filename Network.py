import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve


class Network:
    EXPLICIT = 0
    WIDTH = 1000

    def __init__(self, N, p, A_p, A_m, g, gamma, F, tao_p=50, tao_m=100, tao=5, tao_0=2 * 1e5, noise=None,
                 stdp_kernel=None, seed=None):
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
        self.dt = min(tao_p, tao_m, tao, tao_0) / 10.
        # either use a given function as STDP kernel, or the one in the paper
        self.stdp_kernel = self.default_stdp_kernel if stdp_kernel is None else stdp_kernel
        self.current_time=0.
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

        self.__overall_int_of_k = 0.1
        self.delta_k_long = -self.A_p * self.tao_p - self.A_m * self.tao_m + self.__overall_int_of_k
        self.find_f()
        self.coef[Network.EXPLICIT] = self.gamma * (
                self.b ** 2) * self.N * self.__overall_int_of_k  # np.random.rand(1)  # init the explicit pattern strength
        self.W = np.sum(self.coef[:, np.newaxis, np.newaxis] * self.P, axis=0)

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

    def pre_first_stdp_kernel(self, delta_t):
        return self.A_p * np.exp(delta_t / self.tao_p)

    def post_first_stdp_kernel(self, delta_t):
        return self.A_m * np.exp(-delta_t / self.tao_m)

    def find_f(self):
        self.b = fsolve(lambda b: self.F(self.gamma * (b ** 3) * self.__overall_int_of_k) - 1, 10.)[0]#todo change beack to b
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
        # Ks_m_infty = np.vstack([self.stdp_kernel(np.arange(-10000, 0, self.dt) - t), self.stdp_kernel(
        #     t - np.arange(-10000, 0, self.dt))])
        # inf_int0 = (np.sum(np.outer(Ks_m_infty[0],self.f),axis=0)*self.dt)[:,np.newaxis]
        # inf_outer0 = inf_int0 @ self.f[np.newaxis,:] * self.gamma
        # inf_int1 = (np.sum(np.outer(Ks_m_infty[1], self.f), axis=0) * self.dt)[:, np.newaxis]
        # inf_outer1 = (inf_int1 @ self.f[np.newaxis, :] * self.gamma).T

        firing_rates = self.f + self.g * delta_u
        Ks = np.vstack(
            [self.stdp_kernel(np.linspace(0, t, delta_u.shape[0]) - t), self.stdp_kernel(
                t - np.linspace(0, t, delta_u.shape[0]))])
        delta_u_int = ((Ks @ firing_rates) * self.dt)
        outer0 = self.gamma * np.outer(firing_rates[-1, :], delta_u_int[0, :]).T
        outer1 = self.gamma * np.outer(firing_rates[-1, :], delta_u_int[1, :])
        return (-self.W + outer0 + outer1) / self.tao_0

    def run_first_phase(self,,LIMIT=2000):
        delta_u = np.full((1, self.N), 0)
        coefs = np.zeros((LIMIT + 1, self.P.shape[0])) #todo change to final time of running.
        coefs[0, :] = self.coef
        i = 0
        coef_diff = np.inf
        while i<LIMIT:
            print(i)
            delta_u = np.vstack(
                [delta_u, delta_u[-1] + self.dt * self.delta_u_dynamics(delta_u[-1], self.current_time, with_noise=with_noise)])
            dw = self.w_dynamics(delta_u, self.current_time)
            self.W += self.dt * dw
            self.current_time += self.dt
            i += 1
            coefs[i, :] = self.P[:, 0, :] @ self.W[:, 0] / self.P[:, 0, 0] #todo here is the problem?
            coef_diff = np.abs(coefs[i, Network.EXPLICIT] - coefs[i - 1, Network.EXPLICIT])
        print(i)
        return np.vstack([self.coef, coefs]), delta_u


    def run_second_phase(self, pattern,delta_u, implicit_introduction_time=1, with_noise=False,LIMIT=2000):
        LIMIT = 20000
        coefs = np.zeros((LIMIT + 1, self.P.shape[0]))
        coefs[0, :] = self.coef
        self.current_time = self.dt
        i = 0

        if isinstance(pattern,list):
            pattern = sum(pattern)

        self.f+=(pattern*self.b)
        for j in range(implicit_introduction_time):
            delta_u = np.vstack(
                [delta_u, delta_u[-1] + self.dt * self.delta_u_dynamics(delta_u[-1], self.current_time, with_noise=with_noise)])
            self.W += self.dt * self.w_dynamics(delta_u, self.current_time)
            self.current_time += self.dt
            i += 1
            coefs[i, :] = self.P[:, 0, :] @ self.W[:, 0] / self.P[:, 0, 0]
            coef_diff = np.abs(coefs[i, Network.EXPLICIT] - coefs[i - 1, Network.EXPLICIT])
        self.f-=(pattern*self.b)

        coef_diff = np.inf
        while coef_diff >1e-10 and i<LIMIT:
            print(i)
            delta_u = np.vstack(
                [delta_u, delta_u[-1] + self.dt * self.delta_u_dynamics(delta_u[-1], self.current_time, with_noise=with_noise)])
            self.W += self.dt * self.w_dynamics(delta_u, self.current_time)
            self.current_time += self.dt
            i += 1
            coefs[i, :] = self.P[:, 0, :] @ self.W[:, 0] / self.P[:, 0, 0]
            coef_diff = np.abs(coefs[i, Network.EXPLICIT] - coefs[i - 1, Network.EXPLICIT])
        return np.vstack([self.coef, coefs]), delta_u
