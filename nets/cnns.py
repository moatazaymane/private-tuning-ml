import math

import torch.nn as nn

from nets.utils import init_hyperparameters, make_cnn_layer


class CNN(nn.Module):
    def __init__(
        self, hyperparameters: dict, in_channels: int, num_classes=10, img_size=28
    ):
        super().__init__()
        self._hyperparameters = hyperparameters
        maps, linear = (
            hyperparameters["cnn"]["maps"],
            hyperparameters["head"]["linear"],
        )
        num_layers, kernels = (
            hyperparameters["cnn"]["num_layers"],
            hyperparameters["cnn"]["kernels"],
        )
        head_bias = hyperparameters["head"]["bias"]
        backbone_dropout = hyperparameters["cnn"]["dropout"]
        head_dropout = hyperparameters["head"]["dropout"]
        groups = hyperparameters["cnn"]["groups"]
        assert len(groups) == num_layers
        layers = []
        max_pool = 2
        out_features = in_channels
        sz = img_size
        pad = False
        mx = True
        st = 2
        for i in range(num_layers):
            st = maps[i][1]
            sz1 = sz - 1
            pad = False
            mx = True
            k = kernels[i]
            if kernels[i] >= sz:
                k = sz
            sz1 = math.ceil((sz - k) / (2 * st)) // 2
            if sz1 <= 2:
                sz1 = math.ceil((sz - k) / (2 * st))
                mx = False

            if sz1 <= 2:
                st = 1
                sz1 = sz
                mx = False
                pad = True

            if sz <= 2:
                mx = False
                pad = True

            layers += [
                *make_cnn_layer(
                    layer_num=i + 1,
                    in_features=out_features,
                    out_features=maps[i][0],
                    kernel_size=k,
                    GN=groups[i],
                    proba=backbone_dropout[i],
                    stride=min(maps[i][1], st),
                    max_pool=mx,
                    activation=hyperparameters["cnn"]["activation"],
                    pad=pad,
                )
            ]

            out_features = maps[i][0]
            sz = sz1
        layers.append(["Flatten", nn.Flatten()])

        for i, size in enumerate(list(linear) + [num_classes]):
            layers.append(
                [
                    f"Linear_{i+1}",
                    nn.LazyLinear(
                        size,
                        bias=head_bias,
                    ),
                ]
            )
            if i != len(linear) and head_dropout[i] > 0:
                layers.append([f"Dropout_{i + 1}", nn.Dropout(p=head_dropout[i])])

            if i != len(linear) and hyperparameters["head"]["activation"] == "relu":
                layers.append([f"Head_Activation_{i + 1}", nn.ReLU()])

            elif i != len(linear) and hyperparameters["head"]["activation"] == "tanh":
                layers.append([f"Head_Activation_{i + 1}", nn.Tanh()])

            if i == len(linear):
                layers.append(["Softmax", nn.Softmax(dim=1)])

        self._layers = layers
        self._net = nn.ModuleDict(self._layers)

    def forward(self, z):
        out = z
        for _, (_, layer) in enumerate(self._net.items(), 1):
            out = layer(out)
        return out

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @property
    def accuracy(self):
        return self._accuracy


def init_model(
    seeds: dict,
    in_features: int,
    num_classes: int,
    model_name="cnn-small",
    private_params: dict | None = None,
    activation="relu",
    optimizer_name="sgd",
):
    """
    Itializes a model with a predefined architecture, experiment seed determines all hyperparameters of the model

    Args:
        in_features: number of features (cnn: channels)
        num_classes: number of classes in classification
        seed: experiment seed

    Returns:
        model
    """

    if model_name == "cnn-small":
        hyperparameters = init_hyperparameters(
            experiment=seeds["seed"],
            private_seeds=seeds["private"],
            name=model_name,
            private_params=private_params,
        )
        hyperparameters["activation"] = activation
        hyperparameters["optimizer_name"] = optimizer_name
        model = CNN(
            hyperparameters=hyperparameters,
            in_channels=in_features,
            num_classes=num_classes,
        )

        return model, hyperparameters

    else:
        raise NotImplementedError
