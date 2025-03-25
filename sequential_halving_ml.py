import json
from pathlib import Path

import numpy as np

from generate_ml_models import generate_ml_models
from train_model import train_model


def SH_ML(
    prev_l,
    stop_l,
    train_ds,
    test_ds,
    k,
    seed,
    batch_size,
    I,
    epsilon_alpha,
    delta_tune,
    C,
    sigma,
    max_grad_norm,
    UCB,
    beta,
    device,
    MAX_PHYSICAL_BATCH_SIZE,
    sampling="poisson",
    var_proxy=1 / 4,
    continue_training=False,
):

    if prev_l == 0:
        A = generate_ml_models(
            seed=seed,
            k=k,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            sigma=sigma,
            I=I,
            sampling=sampling,
        )

    else:
        load_path = Path(f"experiments/sh_ml/{seed}/{l-1}/")
        with open(load_path / "A.json", "r") as f:
            A = json.load(f)

    log2_k = np.log2(k)
    training_iters = 0

    for l in range(prev_l + 1, stop_l + 1):
        Il = I * np.sqrt(2 * epsilon_alpha) / (k * log2_k) * 2 ** ((l - 1) / 2)
        A, _ = sh(
            l,
            A,
            train_ds,
            test_ds,
            device="cpu",
            k=k,
            I=I,
            epsilon_alpha=epsilon_alpha,
            delta_tune=delta_tune,
            C=C,
            seed=seed,
            var_proxy=var_proxy,
            UCB=UCB,
            beta=beta,
            MAX_PHYSICAL_BATCH_SIZE=MAX_PHYSICAL_BATCH_SIZE,
        )
        training_iters += Il

    assert I >= training_iters

    if stop_l == log2_k:
        A["remaining_iters"] = training_iters - I
        if continue_training:
            assert len(A) == 1

            for id in A:
                idx = id

            if I > training_iters:
                accuracies, _ = train_model(
                    train_ds=train_ds,
                    l=l,
                    seed=seed,
                    test_ds=test_ds,
                    model_info=A[idx]["info"],
                    model_id=idx,
                    iterations=int(training_iters),
                    eval_points=[],
                    device=device,
                    MAX_PHYSICAL_BATCH_SIZE=MAX_PHYSICAL_BATCH_SIZE,
                )
                save_a(A, seed, log2_k)

            A[idx]["accuracies"] = accuracies

    save_a(A, seed, stop_l)

    return A


def sh(
    l,
    A,
    train_ds,
    test_ds,
    k,
    I,
    epsilon_alpha,
    delta_tune,
    C,
    seed,
    var_proxy=1 / 4,
    UCB=False,
    beta=None,
    MAX_PHYSICAL_BATCH_SIZE=256,
    device="cuda",
):

    save_a(A, seed, l - 1)
    np.random.seed(seed)
    log2_k = np.log2(k)

    Iprev = (
        0
        if l == 1
        else (I * np.sqrt(2 * epsilon_alpha) / (k * log2_k) * 2 ** ((l - 2) / 2))
    )
    Il = I * np.sqrt(2 * epsilon_alpha) / (k * log2_k) * 2 ** ((l - 1) / 2)
    Tl = int((C - 1 - (1 - k**-0.5) / (np.sqrt(2) * log2_k)) * 2 ** ((l - 1) / 2))
    s_values = np.arange(1, Tl + 1, 1)
    Ts = int(Iprev) + s_values * (int(Il) - int(Iprev)) / Tl if Tl > 0 else []

    _Ts = np.round(Ts).astype(int)

    Ts = np.unique(_Ts)
    print(Ts)
    print((len(Ts), len(_Ts)))

    assert len(_Ts) == len(Ts), "Increase I"

    rewards = []
    for id in A:
        accuracies, new_acc = train_model(
            train_ds=train_ds,
            l=l,
            test_ds=test_ds,
            seed=seed,
            model_info=A[id]["info"],
            model_id=id,
            iterations=int(Il),
            eval_points=Ts,
            device=device,
            MAX_PHYSICAL_BATCH_SIZE=MAX_PHYSICAL_BATCH_SIZE,
        )
        A[id]["accuracies"] = accuracies
        mu_hat = np.mean(accuracies)
        sigma_sq = (
            (6**2 / Tl**2)
            * np.log(4 * np.exp(1 / 6) * (C - 1) * I * epsilon_alpha / delta_tune)
            if Tl > 0
            else 0
        )
        N = np.random.normal(0, np.sqrt(sigma_sq), 1) if sigma_sq > 0 else 0
        mu_tilde = mu_hat + N

        if UCB and beta is not None:
            t = (
                np.sqrt(np.log(2 / beta)) * Tl / (sigma_sq + var_proxy)
                if sigma_sq > 0
                else 0
            )
            ucb_mu_tilde = mu_tilde + t
            rewards.append((ucb_mu_tilde, id))
        else:
            rewards.append((mu_tilde, id))

    rewards.sort(key=lambda x: x[0], reverse=True)
    kept_ids = [rewards[i][1] for i in range(len(A) // 2)]
    new_A = {id: A[id] for id in kept_ids}

    return new_A, A


def save_a(A, seed=None, l=None, path=None):

    if path:

        with open(path, "w") as f:
            json.dump(A, f)
    else:
        save_dir = Path(f"experiments/sh_ml/{seed}/{l-1}/")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "A.json"
        with open(save_path, "w") as f:
            json.dump(A, f)
