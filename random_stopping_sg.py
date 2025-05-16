from concurrent.futures import ThreadPoolExecutor
from generate_means import generate_means
import numpy as np
import opacus
from opacus.accountants.analysis import rdp as privacy_analysis
from scipy.stats import poisson

get_privacy_spent = opacus.accountants.analysis.rdp.get_privacy_spent


def random_stopping_subgaussian(
    I,
    total_epsilon,
    total_delta,
    mu,
    means,
    seed=None,
    var_proxy=1 / 4,
    distribution="gaussian",
    stopping_time_dist="poisson",
    epsilon_dpsgd=None,
    eps_prime=None,
    alpha_range=(1.1, 100.0),
    step_size=1,
    multivariate = False
):
    """
    Parameters:
    - I: Integer, number of samples per run
    - total_epsilon: Maximum epsilon allowed for the differential privacy guarantee
    - total_delta: Maximum delta allowed for the differential privacy guarantee
    - mu: Mean of the stopping time distribution (used if stopping_time_dist='poisson')
    - means: Array of means for the subgaussian distribution
    - seed: Random seed for reproducibility (default is None)
    - var_proxy: Variance proxy for the distribution (default is 1/4)
    - distribution: Distribution type, default is 'gaussian'
    - stopping_time_dist: The distribution used for stopping time ('poisson' or other custom distribution)
    - epsilon_dpsgd: Epsilon for differential privacy with the DP-SGD mechanism (used for ADP conversion)
    - alpha_range: Range of alpha values for iterating over and adjusting epsilon_hat
    - step_size: Step size for iterating over alpha values

    Returns:
    - true_mean: The true mean corresponding to the best empirical mean
    - empirical_mean: The empirical mean of the best run
    - epsilon_prime: The final epsilon value after adjustment
    - delta_hat: The final delta value after adjustment
    """

    if seed is not None:
        np.random.seed(seed)

    if stopping_time_dist == "poisson":
        tau = poisson.rvs(mu)

    else:
        raise ValueError(
            f"Unsupported stopping time distribution: {stopping_time_dist}"
        )

    if eps_prime != None:
        best_epsilon_prime = eps_prime

    best_epsilon_prime = find_best_epsilon_prime(
        epsilon_dpsgd=epsilon_dpsgd,
        total_epsilon=total_epsilon,
        total_delta=total_delta,
        mu=mu,
        alpha_range=alpha_range,
        step_size=step_size,
    )

    if best_epsilon_prime == float("inf"):
        return None, None, 0

    empirical_means = []
    random_indexes = []

    for _ in range(tau):

        random_idx = np.random.randint(0, len(means))

        if not multivariate:
            mean = means[random_idx]
            if distribution == "gaussian":
                result = np.random.normal(mean, np.sqrt(var_proxy), size=I)

        elif distribution == "gaussian":
            result = np.random.multivariate_normal(
                mean=means[random_idx][:],
                cov=var_proxy * np.eye(len(means[random_idx][:]))
            )

        empirical_mean = np.mean(result)
        empirical_means.append(empirical_mean)
        random_indexes.append(random_idx)

    if tau == 0:
        return None, np.max(means), 0

    best_empirical_mean = max(empirical_means)
    best_run_idx = random_indexes[empirical_means.index(best_empirical_mean)]
    true_mean = means[best_run_idx]

    return best_run_idx, true_mean, np.clip(best_empirical_mean, 0, 1)


def find_valid_mu(
    total_epsilon, total_delta, epsilon_dpsgd=1, alpha_range=(1.1, 100.0), step_size=1
):
    """Perform binary search to find the largest valid integer mu."""
    low, high = 0, 1000
    best_mu = 0
    best_eps_prime = None

    while low <= high:
        mu = (low + high) // 2
        eps_prime = find_best_epsilon_prime(
            epsilon_dpsgd=epsilon_dpsgd,
            total_epsilon=total_epsilon,
            total_delta=total_delta,
            mu=mu,
            alpha_range=alpha_range,
            step_size=step_size,
        )

        if eps_prime <= total_epsilon:
            best_mu = mu
            best_eps_prime = eps_prime
            low = mu + 1
        else:
            high = mu - 1
    
    print(f"BEST EXP TAU {best_mu}")
    return best_mu, best_eps_prime


def find_best_epsilon_prime(
    epsilon_dpsgd,
    total_epsilon,
    total_delta,
    mu,
    alpha_range=(1.1, 100.0),
    step_size=1,
    tol=1e-12,
    multithreaded=True,
):
    """
    Finds the best epsilon_prime that satisfies privacy constraints using binary search on delta_hat.

    Parameters:
    - epsilon_dpsgd: The privacy budget for each iteration of DPSGD.
    - total_epsilon: The total epsilon budget for the experiment.
    - total_delta: The total delta budget for the experiment.
    - mu: A scaling parameter involved in the computation of epsilon_prime.
    - alpha_range: The range of alpha values to search over.
    - step_size: The step size for iterating over alpha values.
    - tol: The tolerance for binary search on delta_hat.
    - multithreaded: If True, run the loop on alpha in parallel.

    Returns:
    - best_epsilon_prime: The best epsilon_prime value found.
    """

    best_epsilon_prime = float("inf")
    delta_min, delta_max = 1e-12, 1

    while delta_max - delta_min > tol:
        delta_hat = (delta_min + delta_max) / 2  # Middle value for binary search

        alpha_values = np.arange(alpha_range[0], alpha_range[1], step_size)

        def process_alpha(alpha):
            epsilon_hat, _ = get_privacy_spent(
                orders=[alpha], rdp=epsilon_dpsgd[alpha], delta=delta_hat
            )
            if epsilon_hat < 0 or epsilon_hat > np.log(1 + (1 / (alpha - 1))):
                return None

            epsilon_prime_candidate = (
                epsilon_dpsgd[alpha] + mu * delta_hat + np.log(mu) / (alpha - 1)
            )
            epsilon_prime_candidate, _ = get_privacy_spent(
                rdp=epsilon_prime_candidate, orders=[alpha], delta=total_delta
            )

            return (
                epsilon_prime_candidate
                if epsilon_prime_candidate <= total_epsilon
                else None
            )

        if multithreaded:
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(process_alpha, alpha_values))
        else:
            results = [process_alpha(alpha) for alpha in alpha_values]

        for epsilon_prime_candidate in filter(None, results):
            if epsilon_prime_candidate < best_epsilon_prime:
                best_epsilon_prime = epsilon_prime_candidate
                delta_max = delta_hat
                break
        else:
            delta_min = delta_hat

    return best_epsilon_prime


def compute_diff(rare_prob,k,  params):
    means = generate_means(k, rare_prob=rare_prob)
    params['means'] = means
    _, true_mean, _ = random_stopping_subgaussian(**params)
    return np.max(means) - true_mean


def compute_diff_exp2(C, k, rare_prob, params, epsilon_adp):


    exp_tau, eps_prime = find_valid_mu(
            total_epsilon=C * epsilon_adp,
            total_delta=params["total_delta"],
            epsilon_dpsgd=params["epsilon_dpsgd"],
            alpha_range=params["alpha_range"],
            step_size=params["step_size"],
        )

    params["total_epsilon"] = C * epsilon_adp
    params["mu"] = exp_tau
    params["eps_prime"] = eps_prime
    means = generate_means(k, rare_prob=rare_prob)
    params['means'] = means
    _, true_mean, _ = random_stopping_subgaussian(**params)
    return np.max(means) - true_mean


if __name__ == "__main__":

    from generate_means import generate_means
    import numpy as np
    from multiprocessing import Pool
    from functools import partial
    exp = 1

    num_seeds = 100
    k = 2**6
    C = 5
    rares = np.linspace(0.02, 0.1, 10)
    n, bs = 50000, 25
    q = bs / n
    I =  64*80000
    sigma = 4
    total_delta = 1 / n
    order_start, order_end, step_size = 1.1, 100, 0.05
    orders = np.arange(1.1, 100.0, step_size)

    _epsilon_dpsgd = privacy_analysis.compute_rdp(
        q=q, noise_multiplier=sigma, steps=I, orders=orders
    )

    epsilon_dpsgd = {orders[i]: eps for i, eps in enumerate(_epsilon_dpsgd)}
    epsilon_adp, _ = get_privacy_spent(
        rdp=_epsilon_dpsgd, orders=orders, delta=total_delta
    )

    if exp == 1:

        exp_tau, eps_prime = find_valid_mu(
            total_epsilon=C * epsilon_adp,
            total_delta=total_delta,
            epsilon_dpsgd=epsilon_dpsgd,
            alpha_range=(order_start, order_end),
            step_size=step_size,
        )

        # Partial function that binds 'k' and 'params' to 'compute_diff'
        per_seed_diffs = np.zeros((num_seeds, len(rares)))

        for seed in range(num_seeds):
            params = {
                "I": I,
                "total_epsilon": C * epsilon_adp,
                "total_delta": total_delta,
                "mu": exp_tau,
                "eps_prime": eps_prime,
                "seed": seed,
                "var_proxy": 1 / 4,
                "distribution": "gaussian",
                "stopping_time_dist": "poisson",
                "epsilon_dpsgd": epsilon_dpsgd,
                "alpha_range": (order_start, order_end),
                "step_size": step_size
            }

            compute_diff_with_params = partial(compute_diff, k=k, params=params)
            with Pool() as pool:
                diffs = pool.map(compute_diff_with_params, rares)

            per_seed_diffs[seed, :] = diffs

        mean_diffs = np.mean(per_seed_diffs, axis=0)
        std_diffs = np.std(per_seed_diffs, axis=0)

        import os 
        os.makedirs("results", exist_ok=True)
        np.save("results/5_multi_sg_rs_mean_diffs.npy", mean_diffs)
        np.save("results/5_multi_sg_rs_std_diffs.npy", std_diffs)
    
    else:
        C_values = np.arange(3, 8, 1)
        r = .05
        # Partial function that binds 'k' and 'params' to 'compute_diff'
        per_seed_diffs = np.zeros((num_seeds, len(C_values)))



        for seed in range(num_seeds):

            params = {
                "I": I,
                "total_delta": total_delta,
                "seed": seed,
                "var_proxy": 1 / 4,
                "distribution": "gaussian",
                "stopping_time_dist": "poisson",
                "epsilon_dpsgd": epsilon_dpsgd,
                "alpha_range": (order_start, order_end),
                "step_size": step_size
                }
            compute_diff_with_params = partial(compute_diff_exp2, k=k, params=params, epsilon_adp = epsilon_adp, rare_prob = r)
            with Pool() as pool:
                diffs = pool.map(compute_diff_with_params, C_values)

            per_seed_diffs[seed, :] = diffs

        mean_diffs = np.mean(per_seed_diffs, axis=0)
        std_diffs = np.std(per_seed_diffs, axis=0)

        import os 
        os.makedirs("results", exist_ok=True)
        np.save("results/C_sg_rs_mean_diffs.npy", mean_diffs)
        np.save("results/C_sg_rs_std_diffs.npy", std_diffs)
