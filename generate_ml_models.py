def generate_ml_models(
    seed, k, batch_size, sigma, max_grad_norm, I, sampling="poisson"
):

    ids = [i for i in range(k * seed, k * seed + k)]
    Models = {}
    private_params = {}
    private_params["batch_size"] = batch_size
    private_params["sampling"] = sampling
    private_params["steps"] = I
    private_params["max_grad_norm"] = max_grad_norm
    private_params["sigma"] = sigma

    for id in ids:

        seeds = {"seed": id, "private": id}
        model_info = {
            "seeds": seeds,
            "in_features": 1,
            "num_classes": 10,
            "model_name": "cnn-small",
            "activation": "relu",
            "optimizer_name": "sgd",
            "private_params": private_params,
        }

        Models[id] = {"info": model_info}

    return Models
