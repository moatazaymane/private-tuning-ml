import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import numpy as np

from nets.training import train_one_seed
from nets.utils import log_message


def train_models(
    seed: int,
    train_ds,
    test_ds,
    selection_steps: int,
    limit: int,
    accuracy_gaps: List[float],
    base_path: str,
    eval_after=1,
    self_comparaison=[5, 1e-2],
    MAX_PHYSICAL_BATCH_SIZE=256,
    device="cuda",
    prev_step=0,
    save_models=False,
    parallel=False,
    writer=None,
    workers=8,
    remove_prev=False,
    train=True,
    use_tqdm=True,
):
    """
    This makes it possible to train an architecture for a certain number of steps (one, for example) and store a model checkpoint and certain metrics in a specified folder indexed by seed and step.
    For tuning hyperparameters, several seeds can run a gradient step in parallel before joining the threads, and deciding which seeds need to be trained for longer.
    We calculate the privacy budget based on the number of architectures we try and also the number of steps

    Args:
        seed: starting seed, usually set to one
        limit: the breadth of the search, how many architectures to try
        eval_after: how many steps before evaluation, saving models, metrics ...
        accuracy_gaps: list of accuracy gaps, at a given step we compare the accuracy of the model to max_{step <= cur_step, seeds} accuracy to accuracy(seed, step) and decide to carry on training the model or not
        self_comparaison: we used the metrics saved during training for early stopping, if there is no improvement in accuracy
        prev_step: will load metrics and checkpoints to carry on training the models from prev_step
        parallel: boolean value, if set to true the models will be trained using multiple threads
        save_models: boolean value, if set to true checkpoints will be saved after eval_after steps
        writer: writer for tensorboard logging, can be None

    Returns:
        None
    """

    if remove_prev:
        os.remove(path=base_path)

    if prev_step != 0:
        arr = np.load(f"{base_path}/step-{prev_step}/remaining_models.npy")
        models = set(list(arr))

    else:
        models = set(seed for seed in range(seed, seed + limit))
    for step in range(prev_step, selection_steps, eval_after):
        for exp in range(seed, seed + limit):
            if exp not in models:
                continue

            if parallel == False and train:
                log_message(
                    message=f"Training for {eval_after} steps --- starting step {step}",
                    seed=exp,
                    back=True,
                )

                train_one_seed(
                    writer=writer,
                    train_ds=train_ds,
                    test_ds=test_ds,
                    exp=exp,
                    selection_steps=step + eval_after,
                    base_path=base_path,
                    self_comparaison=self_comparaison,
                    accuracy_gaps=accuracy_gaps,
                    prev_step=step,
                    device=device,
                    save_models=save_models,
                    eval_after=eval_after,
                    MAX_PHYSICAL_BATCH_SIZE=MAX_PHYSICAL_BATCH_SIZE,
                    use_tqdm=use_tqdm,
                )

            elif train:
                log_message(
                    message=f"Training for {eval_after} steps --- starting step {step} --- Parallel",
                    seed=exp,
                    back=True,
                )

                with ThreadPoolExecutor(max_workers=min(os.cpu_count(), workers)) as e:
                    e.submit(
                        train_one_seed,
                        writer,
                        train_ds,
                        test_ds,
                        exp,
                        step + eval_after,
                        base_path,
                        self_comparaison,
                        accuracy_gaps,
                        step,
                        device,
                        save_models,
                        eval_after,
                        MAX_PHYSICAL_BATCH_SIZE,
                        use_tqdm,
                    )

        # postprocessing | calculating max_{step <= cur_step | all_seeds} Accuracy to compare it to Accuracy(seed, step) and decide whether to eliminate the model or not | The model might also be already eliminated based on other criteria in the iner training loop
        pref_seed_max = 0
        models = set(list(models))
        for exp in range(seed, seed + limit):
            info_path = f"{base_path}/seed-{exp}/info"
            if f"{info_path}/step-{step + eval_after}" in glob.glob(f"{info_path}/*"):
                with open(
                    f"{info_path}/step-{step + eval_after}/models_info.json"
                ) as f:
                    models_info = json.load(f)

                if "eliminated" in models_info.keys() and exp in models:
                    models.remove(exp)

                # it contains the maximim for the same seed over the steps |  models_info['pref_max_accuracy_steps'][step] = max_{step <= cur_step} Accuracy(seed)
                pref_seed_max = max(
                    models_info["pref_max_accuracy_steps"][str(step + eval_after)],
                    pref_seed_max,
                )

            elif exp in models:
                models.remove(exp)

        back = True

        for _, exp in enumerate(range(seed, seed + limit)):
            if exp not in models:
                continue

            info_path = f"{base_path}/seed-{exp}/info"

            if f"{info_path}/step-{step + eval_after}" in glob.glob(f"{info_path}/*"):
                with open(
                    f"{info_path}/step-{step + eval_after}/models_info.json"
                ) as f:
                    models_info = json.load(f)

            else:
                models.remove(exp)

            max_reached = models_info["pref_max_accuracy_steps"][str(step + eval_after)]

            if (
                pref_seed_max - accuracy_gaps[step + eval_after - 1] > max_reached
            ) and exp in models:
                models.remove(exp)
                log_message(
                    message=f"Model was eliminated in postprocessing step -- Max(seed, step<=cur) acc = {pref_seed_max} -- Max(step <=cur) acc(seed) = {max_reached}",
                    seed=exp,
                    step=step + eval_after,
                    back=back,
                )
                back = False

        log_message(
            message=f"Number of remaining models after training all seeds for {eval_after} steps {len(models)}",
            step=step + eval_after,
        )

        models = np.array(list(models))
        Path(f"{base_path}/step-{step+eval_after}").mkdir(parents=True, exist_ok=True)
        np.save(f"{base_path}/step-{step + eval_after}/remaining_models.npy", models)
    log_message(
        message=f"Remaining models after training all seeds for {eval_after} steps {models}",
        step=prev_step + selection_steps,
        back=True,
    )
