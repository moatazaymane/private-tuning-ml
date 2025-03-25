import numpy as np


def accuracy_curve(t, A_max=0.9, gr=0.1, t0=50, noise_std=0.02):
    """
    Generate a synthetic accuracy curve using a logistic function with noise.

    Parameters:
    - t: Epochs (numpy array)
    - A_max: Maximum accuracy
    - k: Growth rate of the curve
    - t0: Inflection point (epoch where accuracy grows fastest)
    - noise_std: Standard deviation of Gaussian noise

    Returns:
    - Numpy array of simulated accuracy values.
    """
    base_curve = A_max / (1 + np.exp(-gr * (t - t0)))
    noise = np.random.normal(0, noise_std, size=t.shape)
    return np.clip(base_curve + noise, 0, 1)


def generate_stochastic_curves(k, iterations):
    curves = []
    for _ in range(k):
        A_max = np.clip(np.random.normal(0.6, 0.4), 0.1, 0.9)
        noise_std = np.abs(np.random.normal(0.05, 0.02))
        gr_val = np.abs(np.random.normal(0.001, 0.002))

        curve = accuracy_curve(iterations, A_max=A_max, gr=gr_val, noise_std=noise_std)
        curves.append(curve)

    return curves
