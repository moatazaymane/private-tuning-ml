from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

from generate_means import generate_means
from random_stopping_sg import (find_valid_mu, get_privacy_spent,
                                random_stopping_subgaussian)
from sequential_halving_sg import sequential_halving_subgaussian


def run_experiment_subgaussian(
    k,
    I,
    epsilon_alpha,
    C,
    means,
    seed,
    epsilon_dpsgd,
    delta_dpsgd,
    epsilon_adp,
    var_proxy=1 / 4,
    distribution="gaussian",
    total_delta=None,
    alpha_range=(1.1, 100.0),
    step_size=1,
    found=False,
    UCB=False,
    beta=None,
):
    max_mean = np.max(means)
    idxs = np.where(means == max_mean)[0]

    best_config_sh, _, empirical_mean_sh = sequential_halving_subgaussian(
        k=k,
        I=I,
        epsilon_alpha=epsilon_alpha,
        delta_tune=(C - 1) * delta_dpsgd,
        C=C,
        means=means,
        seed=seed,
        var_proxy=var_proxy,
        distribution=distribution,
        UCB=UCB,
        beta=beta,
    )

    exp_tau, eps_prime = find_valid_mu(
        total_epsilon=C * epsilon_adp,
        total_delta=total_delta,
        epsilon_dpsgd=epsilon_dpsgd,
        alpha_range=alpha_range,
        step_size=step_size,
    )

    best_config_rs, _, empirical_mean_rs = random_stopping_subgaussian(
        I=I,
        total_epsilon=C * epsilon_adp,
        total_delta=total_delta,
        mu=exp_tau,
        eps_prime=eps_prime,
        means=means,
        seed=seed,
        var_proxy=1 / 4,
        distribution="gaussian",
        stopping_time_dist="poisson",
        epsilon_dpsgd=epsilon_dpsgd,
        alpha_range=alpha_range,
        step_size=step_size,
    )

    if found:
        found_rs = best_config_rs in idxs
        found_sh = best_config_sh in idxs
        return found_rs, found_sh

    return empirical_mean_sh, empirical_mean_rs, max_mean


def run_single_seed(
    seed,
    k,
    I,
    epsilon_alpha,
    C_values,
    epsilon_dpsgd,
    epsilon_adp,
    total_delta,
    alpha_range,
    step_size,
    found,
):
    np.random.seed(seed=seed)
    rare_prob = np.random.uniform(0.01, 0.05)  # high means optimal arms percentage
    means = generate_means(k, rare_prob=rare_prob)

    if found:
        found_count_sh, found_count_rs = {}, {}

        for C in C_values:
            delta_dpsgd = total_delta / C
            found_rs, found_sh = run_experiment_subgaussian(
                k=k,
                I=I,
                epsilon_alpha=epsilon_alpha,
                delta_dpsgd=delta_dpsgd,
                epsilon_adp=epsilon_adp,
                C=C,
                means=means,
                seed=seed,
                total_delta=total_delta,
                epsilon_dpsgd=epsilon_dpsgd,
                alpha_range=alpha_range,
                step_size=step_size,
                found=found,
            )

            found_count_sh[C] = found_count_sh.get(C, 0) + found_sh
            found_count_rs[C] = found_count_rs.get(C, 0) + found_rs

        return found_count_sh, found_count_rs

    else:

        avg_sh, avg_rs, max_mean_sh, max_mean_rs = {}, {}, {}, {}

        for C in C_values:
            delta_dpsgd = total_delta / C
            empirical_mean_sh, empirical_mean_rs, max_mean = run_experiment_subgaussian(
                k=k,
                I=I,
                epsilon_alpha=epsilon_alpha,
                delta_dpsgd=delta_dpsgd,
                epsilon_adp=epsilon_adp,
                C=C,
                means=means,
                seed=seed,
                total_delta=total_delta,
                epsilon_dpsgd=epsilon_dpsgd,
                alpha_range=alpha_range,
                step_size=step_size,
                found=found,
            )

            avg_sh[C] = empirical_mean_sh
            avg_rs[C] = empirical_mean_rs
            max_mean_sh[C] = max_mean
            max_mean_rs[C] = max_mean

        return avg_sh, avg_rs, max_mean_sh, max_mean_rs


def run_single_seed_rare(
    seed,
    k,
    I,
    epsilon_alpha,
    C,
    rare_probas,
    epsilon_dpsgd,
    epsilon_adp,
    total_delta,
    alpha_range,
    step_size,
    found,
):

    np.random.seed(seed=seed)
    avg_sh, avg_rs, max_mean = {}, {}, {}

    for rare_prob in rare_probas:
        means = generate_means(k, rare_prob=rare_prob / 100)
        delta_dpsgd = total_delta / C
        empirical_mean_sh, empirical_mean_rs, _max_mean = run_experiment_subgaussian(
            k=k,
            I=I,
            epsilon_alpha=epsilon_alpha,
            delta_dpsgd=delta_dpsgd,
            epsilon_adp=epsilon_adp,
            C=C,
            means=means,
            seed=seed,
            total_delta=total_delta,
            epsilon_dpsgd=epsilon_dpsgd,
            alpha_range=alpha_range,
            step_size=step_size,
            found=found,
        )

        avg_sh[rare_prob] = empirical_mean_sh
        avg_rs[rare_prob] = empirical_mean_rs
        max_mean[rare_prob] = _max_mean

    return avg_sh, avg_rs, max_mean


if __name__ == "__main__":
    from matplotlib.ticker import MaxNLocator
    import seaborn as sns
    from random_stopping_sg import privacy_analysis

    k = 2**10
    num_seeds = 1
    n, bs, MAX_PHYSICAL_BATCH_SIZE = 50000, 25, 25
    max_grad_norm = 1
    q = bs / n
    I = 10
    sigma = 1
    total_delta = 1 / n
    C_values = np.arange(4, 20, 6)
    order_start, order_end, step_size = 1.5, 100, 0.05
    orders = np.arange(order_start, order_end, step_size)
    UCB, beta = True, 0.001
    plot_rare, C, rares = True, 10, [.5, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    found = False
    _epsilon_dpsgd = privacy_analysis.compute_rdp(
        q=q, noise_multiplier=sigma, steps=I, orders=orders
    )

    epsilon_dpsgd = {orders[i]: eps for i, eps in enumerate(_epsilon_dpsgd)}
    epsilon_adp, _ = get_privacy_spent(
        rdp=_epsilon_dpsgd, orders=orders, delta=total_delta
    )
    epsilon_alpha = _epsilon_dpsgd[0]

    seeds = range(num_seeds)
    total_eps = {C: [] for C in C_values}

    with ThreadPoolExecutor(max_workers=num_seeds) as executor:

        if plot_rare:
            results = list(
                executor.map(
                    lambda seed: run_single_seed_rare(
                        seed,
                        k,
                        I,
                        epsilon_alpha,
                        C,
                        rares,
                        epsilon_dpsgd,
                        epsilon_adp,
                        total_delta,
                        (order_start, order_end),
                        step_size,
                        found,
                    ),
                    seeds,
                )
            )
        else:
            results = list(
                executor.map(
                    lambda seed: run_single_seed(
                        seed,
                        k,
                        I,
                        epsilon_alpha,
                        C_values,
                        epsilon_dpsgd,
                        epsilon_adp,
                        total_delta,
                        (order_start, order_end),
                        step_size,
                        found,
                    ),
                    seeds,
                )
            )

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    if plot_rare:

        max_mean_sh = {rare: [] for rare in rares}
        max_mean_rs = {rare: [] for rare in rares}
        max_mean_general = {rare: [] for rare in rares}

        for idx, seed in enumerate(seeds):
            avg_sh, avg_rs, max_mean = results[idx]

            for rare_prob in rares:
                max_mean_sh[rare_prob].append(avg_sh[rare_prob])
                max_mean_rs[rare_prob].append(avg_rs[rare_prob])
                max_mean_general[rare_prob].append(max_mean[rare_prob])

        rare_probas_percent = rares
        mean_sh_values = [np.mean(max_mean_sh[rare]) for rare in rares]
        std_sh_values = [np.std(max_mean_sh[rare]) for rare in rares]

        mean_rs_values = [np.mean(max_mean_rs[rare]) for rare in rares]
        std_rs_values = [np.std(max_mean_rs[rare]) for rare in rares]

        mean_general_values = [np.mean(max_mean_general[rare]) for rare in rares]
        std_general_values = [np.std(max_mean_general[rare]) for rare in rares]


        color_2 = "#D6AEDD"   # Muted pink (for μ1)
        color_1 = "#B22222"  # Muted Orange (for RUE)
        color_3 = "#0073e6"  # Dark Blue (for SH) #1f3a5f


        plt.plot(
            rare_probas_percent,
            mean_rs_values,
            label=f"RUE - $\epsilon = {int(C*epsilon_adp)}$",
            color=color_1,  # Darker blue for the RUE line
        )

        plt.fill_between(
            rare_probas_percent,
            np.array(mean_rs_values) - np.array(std_rs_values),
            np.array(mean_rs_values) + np.array(std_rs_values),
            color=color_1,
            alpha=0.2,
        )

        plt.plot(
            rare_probas_percent,
            mean_general_values,
            label="$\mu_1$",
            linestyle="--",  # Dashed line style
            color=color_2,  # Muted orange for μ1
        )

        # Adding the SH plot with dark blue
        plt.plot(
            rare_probas_percent,
            mean_sh_values,
            label=f"SH - $\epsilon = {int(C*epsilon_adp)}$",
            color=color_3,  # Dark blue for SH
        )

        plt.fill_between(
            rare_probas_percent,
            np.array(mean_sh_values) - np.array(std_sh_values),
            np.array(mean_sh_values) + np.array(std_sh_values),
            color=color_3,
            alpha=0.2,
        )

        plt.xlabel("$R$ (%)")
        plt.ylabel("$\mu$")
        plt.legend(loc="lower right")
        plt.savefig(f'figures/SH-RS-rate-k-{k}-C-{C}-sig-{sigma}-I-{I}.png', dpi=300)
        plt.show()

    elif found:
        found_count_sh, found_count_rs = {}, {}
        for found_rs, found_sh in results:
            for C in C_values:
                found_count_sh[C] = found_count_sh.get(C, 0) + found_sh
                found_count_rs[C] = found_count_rs.get(C, 0) + found_rs

        eps_values = [total_eps[C][0] for C in C_values]

        plt.plot(eps_values, found_count_sh, label=f"SH", color="blue", marker="o")
        plt.plot(eps_values, found_count_rs, label="RUE", color="red", marker="x")

        plt.xlabel(r"$\epsilon$")
        plt.ylabel("Fraction Found")
        plt.legend(loc="lower right")
        # plt.savefig('figures/SH-RS-count.png', dpi=300)
        plt.show()

    else:
        avg_sh = {seed: {C: [] for C in C_values} for seed in seeds}
        avg_rs = {seed: {C: [] for C in C_values} for seed in seeds}
        max_means_sh = {
            seed: {C: [] for C in C_values} for seed in seeds
        }  # To track max mean for SH
        max_means_rs = {
            seed: {C: [] for C in C_values} for seed in seeds
        }  # To track max mean for RS

        for idx, seed in enumerate(seeds):
            avg_diff_sh, avg_diff_rs, max_mean_sh_seed, max_mean_rs_seed = results[idx]
            for C in C_values:
                avg_sh[seed][C].append(avg_diff_sh[C])
                avg_rs[seed][C].append(avg_diff_rs[C])
                max_means_sh[seed][C].append(max_mean_sh_seed[C])
                max_means_rs[seed][C].append(max_mean_rs_seed[C])

                if len(total_eps[C]) == 0:
                    total_eps[C].append(C * epsilon_adp)

        mean_sh = [np.mean([avg_sh[seed][C][0] for seed in seeds]) for C in C_values]
        mean_rs = [np.mean([avg_rs[seed][C][0] for seed in seeds]) for C in C_values]
        std_sh = [np.std([avg_sh[seed][C][0] for seed in seeds]) for C in C_values]
        std_rs = [np.std([avg_rs[seed][C][0] for seed in seeds]) for C in C_values]

        max_mean_sh = [
            np.mean([max_means_sh[seed][C][0] for seed in seeds]) for C in C_values
        ]
        max_mean_rs = [
            np.mean([max_means_rs[seed][C][0] for seed in seeds]) for C in C_values
        ]

        eps_values = [total_eps[C][0] for C in C_values]

        plt.plot(eps_values, mean_sh, label="Sequential Halving", color="blue")
        plt.fill_between(
            eps_values,
            np.array(mean_sh) - np.array(std_sh),
            np.array(mean_sh) + np.array(std_sh),
            color="blue",
            alpha=0.2,
        )
        plt.plot(eps_values, max_mean_sh, label="SH", linestyle="--", color="blue")

        plt.plot(eps_values, mean_rs, label="RUE", color="red")
        plt.fill_between(
            eps_values,
            np.array(mean_rs) - np.array(std_rs),
            np.array(mean_rs) + np.array(std_rs),
            color="red",
            alpha=0.2,
        )
        plt.plot(
            eps_values, max_mean_rs, label="$\mu_1$", linestyle="--", color="orange"
        )

        plt.xlabel(r"$\epsilon$")
        plt.ylabel("$\hat{\mu}_1$")
        plt.legend(loc="lower right")
        # plt.savefig('figures/SH-RS-mean.png', dpi=300)
        plt.show()
