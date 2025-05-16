import numpy as np
import numpy as np
from generate_means import generate_means, generate_multivariate_means
from random_stopping_sg import get_privacy_spent, privacy_analysis
from multiprocessing import Pool, cpu_count
from functools import partial



def sequential_halving_subgaussian(
    k,
    I,
    epsilon_alpha,
    delta_tune,
    C,
    means,
    tau,
    seed=None,
    var_proxy=1 / 4,
    distribution="gaussian",
    algorithm="1",
    multivariate=True
):
    """
    Implements the Sequential Halving Algorithm for Hyperparameter Selection.

    Parameters:
    - k: Number of configurations.
    - I: training iterations
    - epsilon_alpha: RDP budget of the first iteration iteration (alpha is the optimal order after I iterations).
    - delta_tune: Privacy cost parameter.
    - C: Privacy cost growth constant (the final epsilon = C* epsilond(dpsgd)).
    - means: Pre-generated array of means for the configurations.
    - seed: Random seed for reproducibility. Default is None.
    - var_proxy: variance proxy
    - distribution: name of the sub-Gaussian dist

    Returns:
    - Best configuration after sequential halving.
    """

    if distribution == "gaussian":

        np.random.seed(seed)
        L = int(np.ceil(np.log2(k)))
        A = list(range(k))
        stl = 0
        prev = 0
        for l in range(1, L + 1):
            
            if algorithm == '1':
                I_l = 2 * np.sqrt(2 * epsilon_alpha) * I
                I_l /= tau * (np.log2(k) + 1)

                T_l =  (C - 1) * tau 
                T_l /= k * np.log2(k)
                T_l *= 2 ** (l - 1)
                stl = np.log2(k)*I_l
                total = np.log2(k) * I_l
            else:
                 I_l = (I / (np.sqrt(k) * np.log2(k))) * 2 ** ((l - 1) / 2)
                 T_l = (C - 1 - (1 - k ** (-1 / 2)) / (np.sqrt(2) * np.log2(k))) * 2 ** (
                    (l - 1) / 2
                )
                 stl += T_l
                 total = I_l
            
            mn = min(int(T_l), int(I_l))


            stl += mn
            rewards = []

            for i in A:
                assert mn >= 1, f'{(int(T_l), I_l)}'

                if not multivariate:
                    R_i = np.random.normal(
                        means[i], np.sqrt(var_proxy), int(mn)
                    )
                else:
                    R_i = np.random.multivariate_normal(
                        mean=means[i][prev:T_l+1],
                        cov=var_proxy * np.eye(mn)
                    )

                mu_hat = np.mean(R_i)
                sigma_sq = (6**2 / mn**2) * np.log(
                    4 * np.exp(1 / 6) * (C - 1) * I * epsilon_alpha / delta_tune
                )
                N = np.random.normal(0, np.sqrt(sigma_sq), 1)
                mu_tilde = mu_hat + N

                rewards.append((mu_tilde, i))
            prev = T_l
            rewards.sort(key=lambda x: x[0], reverse=True)
            A = [rewards[i][1] for i in range(len(A) // 2)]

        assert len(A) == 1


   
        best_config, true_mean, empirical_mean = (
            rewards[0][1],
            means[rewards[0][1]],
            np.clip(rewards[0][0], 0, 1),
        )


    else:
        raise
        
    return best_config, true_mean, empirical_mean


def run_experiment(r, shared_params):
    k, C, tau, I, total_delta, eps_alpha, seed, multi = shared_params

    if not multi:
        means = generate_means(k, rare_prob=r)
    else:
        means = generate_multivariate_means(k, rare_prob=r, num = int(I/100))


    _, true_mean, _ = sequential_halving_subgaussian(
        k=k,
        I=I,
        epsilon_alpha=eps_alpha,
        delta_tune=total_delta,
        C=C,
        means=means,
        tau=tau,
        seed=seed,
        var_proxy=1 / 4,
        distribution="gaussian",
        multivariate = multi
    )

    diff = np.max(means) - true_mean
    assert diff >=0
    return diff


def run_experiment2(C, shared_params):
    k, r, tau, I, total_delta, eps_alpha, seed, multi = shared_params

    if not multi:
        means = generate_means(k, rare_prob=r)
    else:
        means = generate_multivariate_means(k, rare_prob=r)

    _, true_mean, _ = sequential_halving_subgaussian(
        k=k,
        I=I,
        epsilon_alpha=eps_alpha,
        delta_tune=total_delta,
        C=C,
        means=means,
        tau=tau,
        seed=seed,
        var_proxy=1 / 4,
        distribution="gaussian",
        multivariate = multi
    )

    diff = np.max(means) - true_mean
    assert diff >=0
    return diff


if __name__ == '__main__':
    # Shared setup
    exp = 2
    multi = False
    k = 2**6
    C = 5
    I_b = 9e8
    tau = 8*k*(np.log2(k) + 1)
    I =  k*(np.log2(k) + 1)*I_b

    print(f'Ib - I {(I, I_b)}')
    num_seeds = 100

    n, bs = 800000, 1 
    q = bs / n
    sigma =3.5
    total_delta = 1 / n
    orders = np.arange(1.5, 100, 0.05)

    _epsilon_dpsgd = privacy_analysis.compute_rdp(
        q=q, noise_multiplier=sigma, steps=I, orders=orders
    )

    _one_iter_epsilon_dpsgd = privacy_analysis.compute_rdp(
        q=q, noise_multiplier=sigma, steps=1, orders=orders
    )

    epsilon_dpsgd = {orders[i]: eps for i, eps in enumerate(_epsilon_dpsgd)}
    one_iter_epsilon_dpsgd = {orders[i]: eps for i, eps in enumerate(_one_iter_epsilon_dpsgd)}


    epsilon_adp, best_alpha = get_privacy_spent(
        rdp=_epsilon_dpsgd, orders=orders, delta=total_delta
    )

    print(f"EPSILON ADP {(epsilon_adp, C*epsilon_adp, epsilon_dpsgd[best_alpha])}")
    eps_alpha = one_iter_epsilon_dpsgd[best_alpha]

    if exp == 1:
        rares = np.linspace(0.02, 0.1, 10)

        # Use partial to pass shared params to each call
        per_seed_diffs = np.zeros((num_seeds, len(rares)))

        for seed in range(num_seeds):
            print(f'SEEED {seed}')
            shared_params = (k, C, tau, I, total_delta, eps_alpha, seed, multi)

            np.random.seed(seed)  # Ensure different randomness per seed
            with Pool(processes=cpu_count()) as pool:
                diffs = pool.map(partial(run_experiment, shared_params=shared_params), rares)
            per_seed_diffs[seed, :] = diffs
            print(diffs)
        # Compute mean and std across seeds
        mean_diffs = np.mean(per_seed_diffs, axis=0)
        std_diffs = np.std(per_seed_diffs, axis=0)

        import os 

        os.makedirs("results", exist_ok=True)
        np.save("results/_5_sg_sh_mean_diffs.npy", mean_diffs)
        np.save("results/_5_sg_sh_std_diffs.npy", std_diffs)
    
    else:
        C_values = np.arange(3, 8, 1)
        r = .05
        per_seed_diffs = np.zeros((num_seeds, len(C_values)))

        for seed in range(num_seeds):
            shared_params = (k, r, tau, I, total_delta, eps_alpha, seed, multi)

            np.random.seed(seed)  # Ensure different randomness per seed
            with Pool(processes=cpu_count()) as pool:
                diffs = pool.map(partial(run_experiment2, shared_params=shared_params), C_values)
            per_seed_diffs[seed, :] = diffs
            print(diffs)
        # Compute mean and std across seeds
        mean_diffs = np.mean(per_seed_diffs, axis=0)
        std_diffs = np.std(per_seed_diffs, axis=0)

        import os 

        os.makedirs("results", exist_ok=True)
        np.save("results/_C_sg_sh_mean_diffs.npy", mean_diffs)
        np.save("results/_C_sg_sh_std_diffs.npy", std_diffs)

