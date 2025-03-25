import numpy as np


def generate_means(k, rare_prob=0.1):
    """
    Generates k different means in the range [0, 1] with values above 0.9 being rare.

    Parameters:
    - k: Number of means to generate.
    - rare_prob: Probability of selecting a value close to or above 0.9. Default is 0.1.
    - seed: Random seed for reproducibility. Default is None.

    Returns:
    - A numpy array of k generated means.
    """
    num_large_means = int(k * rare_prob)
    means = []

    for i in range(k):
        if i + 1 <= num_large_means:
            mean = np.random.uniform(0.9, 1)
        else:
            mean = np.random.uniform(0, 0.6)

        means.append(mean)

    return np.array(means)
