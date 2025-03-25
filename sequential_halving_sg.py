import numpy as np


def sequential_halving_subgaussian(
    k,
    I,
    epsilon_alpha,
    delta_tune,
    C,
    means,
    seed=None,
    var_proxy=1 / 4,
    distribution="gaussian",
    UCB=False,
    beta=None,
):
    """
    Implements the Sequential Halving Algorithm for Hyperparameter Selection.

    Parameters:
    - k: Number of configurations.
    - I: training iterations
    - epsilon_alpha: RDP budget per iteration.
    - delta_tune: Privacy cost parameter.
    - C: Privacy cost growth constant.
    - means: Pre-generated array of means for the configurations.
    - seed: Random seed for reproducibility. Default is None.
    - var_proxy: variance proxy
    - distribution: name of the sub-Gaussian dist
    - UCB: whether to use upper confidence bound
    - beta: confidence parameter (replaces "confidence")

    Returns:
    - Best configuration after sequential halving.
    """

    if distribution == "gaussian":

        np.random.seed(seed)
        L = int(np.ceil(np.log2(k)))
        A = list(range(k))
        stl = 0
        Iprev = 0
        for l in range(1, L + 1):

            I_l = (I / np.sqrt(k) * np.log(k)) * 2 ** ((l - 1) / 2)
            T_l = (C - 1 - (1 - k ** (-1 / 2)) / (np.sqrt(2) * np.log2(k))) * 2 ** (
                (l - 1) / 2
            )
            stl += T_l
            rewards = []

            for i in A:
                R_i = np.random.normal(
                    means[i], np.sqrt(var_proxy), min(int(T_l), I_l - Iprev)
                )
                print((int(T_l), I_l - Iprev))
                mu_hat = np.mean(R_i)
                sigma_sq = (6**2 / T_l**2) * np.log(
                    4 * np.exp(1 / 6) * (C - 1) * I * epsilon_alpha / delta_tune
                )
                N = np.random.normal(0, np.sqrt(sigma_sq), 1)
                mu_tilde = mu_hat + N

                # UCB calculation with modified t
                if UCB and beta is not None:
                    t = np.sqrt(np.log(2 / beta)) * T_l / (sigma_sq + 1 / 4)
                    ucb_mu_tilde = mu_tilde + t
                    rewards.append((ucb_mu_tilde, i))
                else:
                    rewards.append((mu_tilde, i))

            rewards.sort(key=lambda x: x[0], reverse=True)
            A = [rewards[i][1] for i in range(len(A) // 2)]

        assert len(A) == 1

        add_I = I - stl
        if I - stl > stl:
            print("REFINING MEAN ESTIMATE")
            R_i = np.random.normal(means[rewards[0][1]], np.sqrt(var_proxy), int(add_I))
            mu_hat_final = np.mean(R_i)
            best_config, true_mean, empirical_mean = (
                rewards[0][1],
                means[rewards[0][1]],
                np.clip(mu_hat_final, 0, 1),
            )
        else:
            best_config, true_mean, empirical_mean = (
                rewards[0][1],
                means[rewards[0][1]],
                np.clip(rewards[0][0], 0, 1),
            )

    else:
        raise
    return best_config, true_mean, empirical_mean
