import json
import math
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from numpy import random as nrandom
from sortedcontainers import SortedSet


def log_message(message, seed=None, step=None, back=False):
    linew = 10
    start = "\n" if back else ""
    if step and seed:
        print(
            f"{start}{'-'*linew} Seed {seed} {'-'*linew} Step {step} {'-'*linew} {message} {'-'*linew}"
        )

    elif seed:
        print(f"{start}{'-'*linew} Seed {seed} {'-'*linew} {message} {'-'*linew}")

    elif step:
        print(f"{start}{'-'*linew} Step {step} {'-'*linew} {message} {'-'*linew}")

    else:
        print(f"{start}{'-'*linew} {message} {'-'*linew} ")


def save_dict(base_path, path, data):
    with open(f"{base_path}/{path}", "w") as f:
        json.dump(data, f)


def fix_seed(experiment: int):
    np.random.seed(experiment)
    torch.random.manual_seed(experiment)
    random.seed(experiment)


def init_hyperparameters(
    experiment: int,
    private_seeds: dict | None,
    name="cnn-small",
    private_params: dict | None = None,
) -> dict:
    """
    initializes hyperparameters
    Args:
        experiment: seed
        fixed: seed for fixed hyperparameters (compare to ps22)
        name: architecture name to guide the NAS
        private_seeds: contains the seeds to vary of possibly fix hyperparameters that impact the privacy accounting

    Returns:
        hyperparameter dictionary
    """

    fix_seed(experiment=experiment)
    if name == "cnn-small":
        hyperparameters = {"cnn": {}, "head": {}}

        selections = [
            [(32, 2), (16, 2)],
            [(16, 2), (32, 2)],
            [(16, 1), (32, 2)],
            [(16, 2), (32, 1)],
            [(16, 1), (32, 1)],
            [(32, 1), (32, 1), (64, 1)],
            [(32, 2), (32, 2), (64, 2)],
            [(32, 2), (32, 1), (64, 1)],
            [(32, 2), (32, 2), (64, 1)],
            [(32, 2), (64, 2), (128, 2)],
            [(32, 2), (64, 1), (128, 1)],
            [(32, 2), (64, 2), (128, 1)],
            [(32, 1), (64, 1), (128, 1)],
        ]

        dropouts_cnn = [
            [[0.3], [0.2], [0.1], [0.0]],
            [
                [0.0, 0.0],
                [0.3, 0.0],
                [0.0, 0.1],
                [0.1, 0.0],
                [0.1, 0.1],
                [0.2, 0.1],
                [0.2, 0.2],
                [0.2, 0.3],
            ],
            [
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.2, 0.2, 0.2],
                [0.2, 0.1, 0.1],
                [0.3, 0.2, 0.2],
                [0.2, 0.2, 0.1],
                [0.0, 0.2, 0.0],
                [0.0, 0, 0.3],
            ],
        ]

        maps = random.Random(x=experiment).choice(selections)

        selected = dropouts_cnn[len(maps) - 1]
        random.Random(x=experiment).shuffle(selected)

        dropout_cnn = random.Random(x=experiment).choice(selected)

        hyperparameters["cnn"]["num_layers"] = len(maps)
        hyperparameters["cnn"]["maps"] = maps
        hyperparameters["cnn"]["groups"] = create_groups(maps=maps, seed=experiment)

        residual_types = ["None", "Type1"]
        hyperparameters["residual"] = random.Random(x=experiment).choice(residual_types)
        # kernels
        kernels = [
            [(8,), (7,), (5,), (4,)],
            [(8, 4), (7, 5), (7, 3), (5, 5), (5, 3), (3, 3)],
            [
                (8, 4, 2),
                (8, 4, 3),
                (7, 5, 3),
                (7, 3, 3),
                (5, 5, 5),
                (5, 5, 3),
                (5, 3, 3),
                (3, 3, 3),
            ],
        ]

        kernels = random.Random(x=experiment).choice(
            kernels[hyperparameters["cnn"]["num_layers"] - 1]
        )
        hyperparameters["cnn"]["kernels"] = kernels
        linear_sizes = [
            (32,),
            (16,),
            (80, 40),
            (100, 60),
            (128, 84),
            (200, 160),
            (200,),
            (80,),
            (128,),
            (100,),
            (40, 10),
            (80, 16),
            (32, 16),
            (16, 8),
        ]
        dropouts = [
            (0.3, 0.1),
            (0.0, 0.0),
            (0.2, 0.1),
            (0.1, 0.1),
            (0.2, 0.2),
            (0.3, 0.2),
        ]

        random.Random(x=experiment).shuffle(linear_sizes)
        random.Random(x=experiment).shuffle(dropouts)

        # backbone dropout
        hyperparameters["cnn"]["dropout"] = dropout_cnn

        hyperparameters["head"]["linear"] = random.Random(x=experiment).choice(
            linear_sizes
        )
        hyperparameters["head"]["dropout"] = random.Random(x=experiment).choice(
            dropouts
        )[: len(hyperparameters["head"]["linear"])]
        hyperparameters["head"]["bias"] = True

        activations = ["relu", "tanh"]

        hyperparameters["cnn"]["activation"] = random.Random(x=experiment).choice(
            activations
        )
        random.Random(x=experiment).shuffle(activations)
        hyperparameters["head"]["activation"] = random.Random(x=experiment).choice(
            activations
        )

        # choose training hyperparameters
        learning_rates = [2e-4, 2e-3, 2e-2, 2e-1, 1, 1.25, 1.5, 2]
        random.Random(x=experiment).shuffle(learning_rates)
        lr = random.Random(x=experiment).choice(learning_rates)
        hyperparameters["lr"] = lr

        if private_params is not None:
            hyperparameters["batch_size"] = private_params["batch_size"]
            hyperparameters["sigma"] = private_params["sigma"]
            hyperparameters["max_grad_norm"] = private_params["max_grad_norm"]
            hyperparameters["steps"] = private_params["steps"]
            hyperparameters["sampling"] = private_params["sampling"]

        else:
            batch_sizes = [512, 768, 1024, 1280, 1536, 1792, 2048]
            sigmas = [1, 1.5, 2, 3]
            max_grad_norms = [1, 1.5, 2]
            steps_l = [500, 600, 700, 800, 900, 1000, 1500, 2000]
            batch_size = random.Random(x=private_seeds["batch_size"]).choice(
                batch_sizes
            )
            sigma = random.Random(x=private_seeds["sigma"]).choice(sigmas)
            max_grad_norm = random.Random(x=private_seeds["max_grad_norm"]).choice(
                max_grad_norms
            )
            steps = random.Random(x=private_seeds["steps"]).choice(steps_l)
            hyperparameters["batch_size"] = batch_size
            hyperparameters["sigma"] = sigma
            hyperparameters["max_grad_norm"] = max_grad_norm
            hyperparameters["steps"] = steps
            hyperparameters["sampling"] = "poisson"
        return hyperparameters
    else:
        raise NotImplementedError


def create_groups(maps: List[int], seed: int):
    """
    Number of groups (create groups by normalizing cnn feature maps across the channel dimension)
    Args:
        maps: List of channels used in the cnn backbone
        seed

    Returns:
        List of groups
    """

    groups = []

    for num, _ in maps:
        divs = get_divisors(num)
        group = random.Random(x=seed).choice(divs)
        groups.append(group)
    return groups


def get_divisors(num):
    """
    Get divisors of number (sorted) - does not include 1

    Args:
        num: number


    Returns:
        sorted list of divisors
    """

    lst = SortedSet()
    for div in range(2, int(math.sqrt(num)) + 1):
        if num % div == 0:
            div2 = num // div

            lst.add(div)
            lst.add(div2)
    lst.add(num)
    return list(lst)


def make_cnn_layer(
    layer_num: int,
    in_features: int,
    out_features: int,
    kernel_size,
    proba: float,
    GN: int | None = None,
    bias=True,
    stride=1,
    max_pool=True,
    activation="relu",
    pad=False,
    pad_mx=False,
):
    layers = []

    if pad:
        layers.append(
            [
                f"Conv_{layer_num}",
                nn.Conv2d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=kernel_size,
                    bias=bias,
                    stride=stride,
                    padding="same",
                ),
            ]
        )
    else:
        layers.append(
            [
                f"Conv_{layer_num}",
                nn.Conv2d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=kernel_size,
                    bias=bias,
                    stride=stride,
                ),
            ]
        )

    if proba > 0:
        layers.append([f"Dropout_{layer_num}", nn.Dropout(p=proba)])

    if activation == "relu":
        layers.append([f"Relu_{layer_num}", nn.ReLU()])
    elif activation == "tanh":
        layers.append([f"Tanh_{layer_num}", nn.Tanh()])
    else:
        raise NotImplementedError

    # Suggestion by brock et al (2021) signal propagation improves with post activation normalization
    if GN is not None:
        layers.append(
            [
                f"Group_Norm{layer_num}",
                nn.GroupNorm(num_groups=GN, num_channels=out_features),
            ]
        )

    if max_pool and pad_mx:
        layers.append(
            [f"Max_Pool_{layer_num}", nn.MaxPool2d(kernel_size=2, padding="same")]
        )
    elif max_pool:
        layers.append([f"Max_Pool_{layer_num}", nn.MaxPool2d(kernel_size=2)])

    return layers
