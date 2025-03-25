import os
from pathlib import Path

import numpy as np
from scipy.stats import poisson

from generate_ml_models import generate_ml_models
from nets.utils import fix_seed
from random_stopping_sg import find_valid_mu
from sequential_halving_ml import save_a
from train_model import train_model


def RS_ML(
    seed,
    train_ds,
    test_ds,
    k,
    batch_size,
    I,
    epsilon_dpsgd: dict,
    epsilon,
    delta,
    sigma,
    max_grad_norm,
    device,
    MAX_PHYSICAL_BATCH_SIZE,
    sampling="poisson",
    alpha_range=(1.1, 100.0),
    step_size=1,
    tau=None,
    rem=None,
    restart=True,
):

    fix_seed(experiment=seed)
    A = generate_ml_models(
        seed=seed,
        k=k,
        batch_size=batch_size,
        max_grad_norm=max_grad_norm,
        sigma=sigma,
        I=I,
        sampling=sampling,
    )

    if tau == None:
        exp_tau, _ = find_valid_mu(
            total_epsilon=epsilon,
            total_delta=delta,
            epsilon_dpsgd=epsilon_dpsgd,
            alpha_range=alpha_range,
            step_size=step_size,
        )
        tau = poisson.rvs(exp_tau)

    if rem == None:
        rem = int(tau)

    print(f"TAU {tau}")

    for _ in range(rem):
        idx = np.random.choice(list(A.keys()), size=1)[0]
        # print((idx, list(A.keys())))
        if restart == False and os.path.exists(f"experiments/rs_ml/{seed}/{idx}.json"):
            continue

        eval_points = np.round(np.linspace(1, I, 100)).astype(int)
        accuracies, _ = train_model(
            train_ds=train_ds,
            test_ds=test_ds,
            seed=seed,
            model_info=A[idx]["info"],
            model_id=idx,
            iterations=I,
            eval_points=eval_points,
            device=device,
            MAX_PHYSICAL_BATCH_SIZE=MAX_PHYSICAL_BATCH_SIZE,
        )

        A[idx]["accuracies"] = accuracies
        A[idx]["tau"] = tau

        save_a(A=A[idx], path=f"experiments/rs_ml/{seed}/{idx}.json")

    mx = 0
    for idx in A:
        mx = max(mx, max(A[idx].get("accuracies", [0])))

    return mx


if __name__ == "__main__":
    import warnings

    import numpy as np
    from torch.utils.tensorboard import SummaryWriter
    from torchvision.datasets import FashionMNIST

    from datasets.cv_datasets import CV_Dataset
    from random_stopping_sg import get_privacy_spent, privacy_analysis
    from sequential_halving_ml import SH_ML

    warnings.simplefilter("ignore")

    writer = SummaryWriter()
    dts_train = FashionMNIST(root=".", train=True, download=True)
    dts_test = FashionMNIST(root=".", train=False, download=True)
    train_ds = CV_Dataset(data=dts_train.data, labels=dts_train.targets)
    test_ds = CV_Dataset(data=dts_test.data, labels=dts_test.targets)

    k = 2**5
    n, bs, MAX_PHYSICAL_BATCH_SIZE = 50000, 25, 25
    max_grad_norm = 1
    q = bs / n
    I = 1
    sigma = 0.7
    total_delta = 1 / n
    C = 3
    order_start, order_end, step_size = 1.5, 100, 0.05
    orders = np.arange(order_start, order_end, step_size)
    sampling = "poisson"
    device = "cpu"

    _epsilon_dpsgd = privacy_analysis.compute_rdp(
        q=q, noise_multiplier=sigma, steps=I, orders=orders
    )
    epsilon_dpsgd = {orders[i]: eps for i, eps in enumerate(_epsilon_dpsgd)}
    epsilon_adp, _ = get_privacy_spent(
        rdp=_epsilon_dpsgd, orders=orders, delta=total_delta
    )
    epsilon_alpha = _epsilon_dpsgd[0]

    print(f"KL div {epsilon_alpha}")
    delta_tune = (C - 1) * (total_delta) / C
    print(
        f"Total Epsilon One training run {epsilon_adp} - Total Epsilon {C*epsilon_adp}"
    )
    seed = 0
    save_dir = Path(f"experiments/rs_ml/{seed}/")
    save_dir.mkdir(parents=True, exist_ok=True)

    accuracies = []
    epsilons = []
    start_seed, num_seeds = 0, 1
    for seed in range(start_seed, start_seed + num_seeds):
        mx_acc = RS_ML(
            sigma=sigma,
            train_ds=train_ds,
            test_ds=test_ds,
            k=k,
            seed=0,
            batch_size=bs,
            I=I,
            epsilon_dpsgd=epsilon_dpsgd,
            epsilon=C * epsilon_adp,
            delta=total_delta,
            max_grad_norm=max_grad_norm,
            device=device,
            MAX_PHYSICAL_BATCH_SIZE=MAX_PHYSICAL_BATCH_SIZE,
            sampling=sampling,
            alpha_range=(order_start, order_end),
            step_size=step_size,
        )
        accuracies.append(mx_acc)

    _accs = np.array(accuracies)
    print((C * epsilon_adp, np.mean(_accs), np.std(_accs)))
