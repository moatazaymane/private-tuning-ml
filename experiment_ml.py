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
    I = 15000
    sigma = 0.7
    total_delta = 1 / n
    C = 3
    order_start, order_end, step_size = 1.5, 100, 0.5
    orders = np.arange(order_start, order_end, step_size)
    sampling = "poisson"
    device = "cpu"
    UCB, beta = False, 0.001

    _epsilon_dpsgd = privacy_analysis.compute_rdp(
        q=q, noise_multiplier=sigma, steps=I, orders=orders
    )
    epsilon_dpsgd = {orders[i]: eps for i, eps in enumerate(_epsilon_dpsgd)}
    epsilon_adp, _ = get_privacy_spent(
        rdp=_epsilon_dpsgd, orders=orders, delta=total_delta
    )
    epsilon_alpha = _epsilon_dpsgd[0]

    print(epsilon_alpha)
    delta_tune = (C - 1) * (total_delta) / C
    print(f"Epsilon One training run {epsilon_adp} - Total Epsilon {C*epsilon_adp}")

    log2_k = int(np.log2(k))
    A1 = SH_ML(
        prev_l=0,
        stop_l=log2_k,
        sigma=sigma,
        train_ds=train_ds,
        test_ds=test_ds,
        k=k,
        seed=0,
        batch_size=bs,
        I=I,
        epsilon_alpha=epsilon_alpha,
        delta_tune=delta_tune,
        C=C,
        max_grad_norm=max_grad_norm,
        UCB=UCB,
        beta=beta,
        device=device,
        MAX_PHYSICAL_BATCH_SIZE=MAX_PHYSICAL_BATCH_SIZE,
        sampling=sampling,
        var_proxy=1 / 4,
    )
