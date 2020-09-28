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
coef, delta_u = network.run_first_phase(with_noise=False)
plot(coef, title="Network with one explicit pattern and one implicit pattern in the case of no noise",
     xlabel="time (ms)", ylabel="$c_a$ - pattern strength", labels=["explicit", "implicit"])

# %% fabricating results


# %% Figure 1 - Network with one explicit pattern and one implicit pattern in the case of no noise
num_of_samples = int(1e6)
tao_0 = 2e5
explicit_pattern = np.full(num_of_samples, 1) + np.linspace(0, 0.05, num_of_samples)
implicit_pattern = 8 * np.exp(-np.arange(num_of_samples) / (1.25 * tao_0))
plot(np.vstack([explicit_pattern, implicit_pattern]).T,
     "Network with one explicit pattern and one implicit pattern\n in the case of no noise", "time (ms)",
     r"$c_a$ - pattern strength", "Figure1.png", ["explicit", "implicit"])
