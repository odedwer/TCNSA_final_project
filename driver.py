from Network import *
from time import time
import matplotlib.pylab as plt
import scipy.linalg as linalg

# %%
network = Network(256, 4, 2, -1.2, 0.1, 9000e-1, lambda x: np.power(x, 3), 50, 100, 5, 2e5, 0.1118, seed=97)
# %%
first_W = network.W.copy()

start = time()
coefs = network.run_first_phase()
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
c = ((network.P[:, 0, :] @ first_W[:, 0]) / network.P[:, 0, 0])
# %%
t = 2
N = 5
steps = 10
delta_u = (np.ones((steps, N)) * np.arange(1, steps + 1, 1)[:, np.newaxis]).ravel()
np.random.shuffle(delta_u)
delta_u.shape = (steps, N)
# %%
a = np.arange(steps)
b = np.arange(steps) + 10
a_mult = a[:, np.newaxis] * delta_u
b_mult = b[:, np.newaxis] * delta_u
a_sum = np.sum(a_mult, axis=0)
b_sum = np.sum(b_mult, axis=0)
a_outer = np.outer(1 + 3 * delta_u[-1], a_sum)
b_outer = np.outer(1 + 3 * delta_u[-1], b_sum)
# %%
c = np.vstack([a, b])
c_sum = c @ delta_u
c_outer = np.outer(c_sum, 1 + 3 * delta_u[-1])
c_outer.shape = (2, c_outer.shape[0] // 2, c_outer.shape[1])
# %%
c_outer[1].T == b_outer
