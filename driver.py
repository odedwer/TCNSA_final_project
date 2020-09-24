from Network import *
from time import time
import matplotlib.pylab as plt
import scipy.linalg as linalg

# %%
network = Network(512, 4, 2, -1.2, 0.1, 9000e-1, lambda x: np.power(x, 3), 50, 100, 5, 2e5, 0.1118, seed=97)
# %%
first_W = network.W.copy()

start = time()
coefs, delta_u = network.run_first_phase()
overall = (time() - start) / 60.
# %%
plt.figure()
plt.plot(coefs)
# %%
network.W - first_W
# %%
W = network.W
P = network.P
# %%
c1 = ((network.P[0, 0, :] @ network.W[:, 0]) / network.P[0, 0, 0])
c2 = ((network.P[1, 0, :] @ network.W[:, 0]) / network.P[1, 0, 0])
c3 = ((network.P[2, 0, :] @ network.W[:, 0]) / network.P[2, 0, 0])
c4 = ((network.P[3, 0, :] @ network.W[:, 0]) / network.P[3, 0, 0])
# %%
c = ((P[:, 0, :] @ W[:, 0]) / P[:, 0, 0])
# %%
t = 2
N = 5
steps = 10
delta_u = (np.ones((steps, N)) * np.arange(1, steps + 1, 1)[:, np.newaxis]).ravel()
np.random.shuffle(delta_u)
delta_u.shape = (steps, N)

# %%
Ksp = np.arange(steps)
Ksm = np.arange(steps) + 10
Ksp_sum = Ksp @ delta_u
Ksm_sum = Ksm @ delta_u
Ksp_outer = np.outer(delta_u[-1], Ksp_sum).T
Ksm_outer = np.outer(delta_u[-1], Ksm_sum)
# %%
W = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        W[i, j] = delta_u[-1][j] * Ksp_sum[i] + delta_u[-1][i] * Ksm_sum[j]
# %%
c = np.vstack([Ksp, Ksm])
c_sum = c @ delta_u
c_outer = np.outer(c_sum, delta_u[-1])
c_outer.shape = (2, c_outer.shape[0] // 2, c_outer.shape[1])
# %%
c_outer[1].T == Ksm_outer
# %%
x = np.linspace(-100, 100, 10000)
kernel = network.stdp_kernel(x)
# %%
plt.figure()
plt.plot(x, kernel)
# %%
from scipy.integrate import quad

integ = quad(lambda t: network.stdp_kernel(t), -2500, 2500)[0]
