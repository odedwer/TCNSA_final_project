import Network as net
import numpy as np
import matplotlib.pylab as plt
from copy import deepcopy


def plot(data, title=None, xlabel=None, ylabel=None, save_name=None, labels=None):
    plt.figure()
    plt.plot(data)
    if labels:
        plt.legend(labels)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if save_name is not None:
        plt.savefig(save_name)


# %% Figure 1 - Network with one explicit pattern and one implicit pattern in the case of no noise
network = net.Network(256, 2, 2, -1.2, 0.1, 9000., 50, 100, 5, 2e3, seed=97)
# %%
coef, delta_u = network.run_first_phase()

# %%
plot(coef, title="Network with one explicit pattern and one implicit pattern in the case of no noise",
     xlabel="time (ms)", ylabel="$c_a$ - pattern strength",labels=["explicit","implicit"])
