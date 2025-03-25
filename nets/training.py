import glob
from code.validation import val_loop
from collections import defaultdict
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm

from nets.cnns import init_model
from nets.utils import fix_seed, log_message, save_dict


def train_one_seed(
    writer,
    train_ds: Dataset,
    test_ds: Dataset,
    exp: int,
    selection_steps: int,
    base_path: int,
    self_comparaison=[5, 1e-2],
    accuracy_gaps: List[float] | None = None,
    prev_step=-1,
    device="cpu",
    save_models=False,
    eval_after=1,
    MAX_PHYSICAL_BATCH_SIZE=256,
    use_tqdm=True,
):
    """
    This makes it possible to train an architecture for a certain number of steps (one, for example) and store a model checkpoint and certain metrics in a specified folder indexed by seed and step.
    For tuning hyperparameters, several seeds can run a gradient step in parallel before joining the threads, and deciding which seeds need to be trained for longer.
    We calculate the privacy budget based on the number of architectures we try and also the number of steps
    """

    fix_seed(experiment=exp)
    # other models were trained in depth one before
    accuracies = defaultdict(int)  # max accuracies reached after every grad step
    step_accuracies = defaultdict(list)  # list of all accuracies reached at step
    model_accuracies = defaultdict(int)
    best_models = (
        []
    )  # list containing model path and accuracies | Only used if the save limit is lower than the seeds tried
    pref_max_step_acc = defaultdict(int)
    seeds = {
        "seed": exp,
        "private": {
            "batch_size": exp,
            "sigma": exp,
            "max_grad_norm": exp,
            "steps": exp,
        },
    }

    step = 0

    if prev_step != 0:
        assert f"{base_path}/seed-{exp}/models/step-{prev_step}" in glob.glob(
            f"{base_path}/seed-{exp}/models/*"
        )
        state = torch.load(
            f"{base_path}/seed-{exp}/models/step-{prev_step}/model_{exp}"
        )
        model_info = state["model_info"]
        model, hyperparameters = init_model(**model_info)
        model.load_state_dict(state["model_state_dict"])
        optimizer = SGD(params=model.parameters(), lr=hyperparameters["lr"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        step = state["step"]
        accuracies = state["max_accuracies"]
        model_accuracies = state["model_accuracies"]
        step_accuracies = state["step_accuracies"]
        pref_max_step_acc = state["pref_max_accuracy_steps"]

        _, _hyperparameters = init_model(**model_info)

        assert hyperparameters == _hyperparameters
        log_message(
            message=f"Checked if the hyperparameters are the same as previous runs",
            seed=exp,
        )

    else:
        model_info = {
            "seeds": seeds,
            "in_features": 1,
            "num_classes": 10,
            "model_name": "cnn-small",
            "activation": "relu",
            "optimizer_name": "sgd",
        }

        model, hyperparameters = init_model(**model_info)
        optimizer = SGD(params=model.parameters(), lr=hyperparameters["lr"])

    hyperparameters["steps"] = selection_steps
    dataloader = DataLoader(
        train_ds, shuffle=True, batch_size=hyperparameters["batch_size"]
    )
    test_dl = DataLoader(
        dataset=test_ds, batch_size=MAX_PHYSICAL_BATCH_SIZE, shuffle=False
    )
    prve = PrivacyEngine(accountant="rdp")

    model, optimizer, train_dl = prve.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=hyperparameters["sigma"],
        max_grad_norm=hyperparameters["max_grad_norm"],
        poisson_sampling=hyperparameters["sampling"] == "poisson",
    )

    errors = ModuleValidator.validate(model, strict=False)
    # log_message(message=f"Using Opacus Module validator, existing Errors in model architecture {len(errors)}", seed=exp,)

    assert hyperparameters["batch_size"] % MAX_PHYSICAL_BATCH_SIZE == 0

    k = hyperparameters["batch_size"] // MAX_PHYSICAL_BATCH_SIZE

    criterion = nn.CrossEntropyLoss()
    stop = False

    assert step < selection_steps

    with BatchMemoryManager(
        data_loader=train_dl,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        optimizer=optimizer,
    ) as memory_safe_data_loader:
        while True:
            if use_tqdm:
                it = tqdm(enumerate(memory_safe_data_loader))
            else:
                it = enumerate(memory_safe_data_loader)

            for mini_batch_step, batch in it:
                for ke in batch:
                    batch[ke].to(device)
                model.to(device)
                model.train()
                out = model(batch["data"].to(device))
                loss = criterion(out, batch["label"].to(device))
                # backward pass
                loss.backward()
                # updates
                optimizer.step()
                # zero gradients
                optimizer.zero_grad()

                if (mini_batch_step + 1) % k == 0:
                    step += 1
                    if step % eval_after == 0:
                        acc_test, _ = val_loop(
                            criterion=criterion,
                            dl_test=test_dl,
                            model=model,
                            device=device,
                            verbose=False,
                        )

                        if writer is not None:
                            writer.add_scalar(f"test accuracy {exp}", acc_test, step)
                        log_message(
                            message=f"Accuracy reached by model {acc_test:.3f}",
                            step=step,
                            seed=exp,
                        )

                        accuracies[step] = max(accuracies[step], acc_test)
                        pref_max_step_acc[step] = acc_test
                        pref_max_step_acc[step] = max(
                            pref_max_step_acc[step - 1], pref_max_step_acc[step]
                        )
                        step_accuracies[step].append(acc_test)
                        model_accuracies[step] = acc_test
                        best_models.sort(reverse=True)

                        if (
                            accuracy_gaps is not None
                            and acc_test
                            <= pref_max_step_acc[step] - accuracy_gaps[step - 1]
                        ) or (
                            len(model_accuracies) > self_comparaison[0]
                            and abs(
                                model_accuracies[step]
                                - model_accuracies[step - self_comparaison[0]]
                            )
                            < self_comparaison[1]
                        ):
                            # we stop training the model
                            stop = True
                            log_message(
                                message=f"Model was eliminated during training step -- acc in step Acc(step = {step - self_comparaison[0]}) = {model_accuracies[step - self_comparaison[0]]} | Acc(step = {step}) = {acc_test}",
                                seed=exp,
                                step=step + eval_after,
                                back=True,
                            )

                        if save_models == True:
                            log_message(
                                message="Saving model state dict", step=step, seed=exp
                            )
                            Path(f"{base_path}/seed-{exp}/models/step-{step}").mkdir(
                                parents=True, exist_ok=True
                            )
                            _path = (
                                f"{base_path}/seed-{exp}/models/step-{step}/model_{exp}"
                            )

                            torch.save(
                                {
                                    "model_state_dict": model._module.state_dict(),
                                    "optimizer_state_dict": optimizer.original_optimizer.state_dict(),
                                    "step": step,
                                    "hyperparameters": hyperparameters,
                                    "model_info": model_info,
                                    "best_models": best_models,
                                    "model_accuracies": model_accuracies,
                                    "max_accuracies": accuracies,
                                    "step_accuracies": step_accuracies,
                                    "pref_max_accuracy_steps": pref_max_step_acc,
                                },
                                _path,
                            )

                        hyperparameters["steps"] = step

                        if step < selection_steps and stop:
                            model_info["eliminated"] = 1

                        log_message(message="Saving info", step=step, seed=exp)
                        Path(f"{base_path}/seed-{exp}/info/step-{step}").mkdir(
                            parents=True, exist_ok=True
                        )
                        save_dict(
                            base_path=f"{base_path}/seed-{exp}/info/step-{step}",
                            path="hyperparameters.json",
                            data=hyperparameters,
                        )
                        save_dict(
                            base_path=f"{base_path}/seed-{exp}/info/step-{step}",
                            path="models_info.json",
                            data={
                                "pref_max_accuracy_steps": pref_max_step_acc,
                                "step_accuracies": step_accuracies,
                                "model_accuracies": model_accuracies,
                                "max_accuracies": accuracies,
                                "best_models": best_models,
                            },
                        )
                        save_dict(
                            base_path=f"{base_path}/seed-{exp}/info/step-{step}",
                            path="model_info.json",
                            data=model_info,
                        )

                        if stop:
                            stop = True
                            break

                    if step == selection_steps:
                        stop = True
                        break
            if stop:
                break
