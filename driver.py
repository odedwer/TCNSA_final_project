from Network import *
from time import time
import matplotlib.pylab as plt
import scipy.linalg as linalg

# %%
# networkS = NetworkSlow(32, 4, 2, -1.2, 0.1, 9000e-1, np.tanh, 50, 100, 5, 2e5, 0.1118, seed=97)
networkF = Network(4, 2, 2, -1.2, 0.1, 1/9000.,-400, 50, 100, 5, 2e5, 0.1118, seed=97)
# %%
# first_W_S = networkS.W.copy()

# coefs_S, delta_u_S = networkS.run_first_phase()
#%%
first_W_F = networkF.W.copy()

#%%
coefs_F = networkF.run_first_phase(LIMIT=10000)

#%%0

coefs_F_2= networkF.run_second_phase(networkF.memory_patterns[1],delta_u_F,implicit_introduction_time=200,)

coefs_F= np.vstack([coefs_F,coefs_F_2])
# %%
plt.figure()
plt.plot(coefs_F, label='F')
plt.legend()
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
c_outer0 = np.outer(delta_u[-1],c_sum[0,:]).T
c_outer1 = np.outer(delta_u[-1],c_sum[1,:])
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
