from Network import *
from time import time

network = Network(1024, 2, 2, -1.2, 0.1, 9000e-1, np.tanh, 50, 100, 5, 2e5, 0.1118)
# %%
first_W = network.W.copy()

start = time()
network.run_first_phase()
overall = time() - start
# %%
network.W - first_W
# %%
W = network.W
P = network.P
#%%
np.linalg.tensorsolve()
