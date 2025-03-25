from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

from metric_curves import generate_stochastic_curves

iterations = np.arange(1, 1001)
k = 20

curves = generate_stochastic_curves(k, iterations)

colors = cycle(plt.cm.tab10.colors)

plt.figure(figsize=(8, 5))
for i, curve in enumerate(curves, start=1):
    plt.plot(iterations, curve, color=next(colors), alpha=0.7, label=f"Run {i}")

plt.xlabel("$N_t^T$")
plt.ylabel("$r_t$")
# plt.title("Synthetic Accuracy Curves with Stochasticity")
plt.ylim(0, 1)
# plt.legend()
plt.grid(True)
plt.show()
