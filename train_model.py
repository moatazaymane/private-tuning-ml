import os
from pathlib import Path
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from nets.cnns import init_model
from nets.utils import fix_seed, log_message
from validation import val_loop


def train_model(
    seed,
    train_ds,
    test_ds,
    model_info,
    model_id,
    iterations,
    device,
    MAX_PHYSICAL_BATCH_SIZE,
    base_path=None,
    model_index = None,
    eval_points=None,
    writer=None,
    use_tqdm=False,
    save_model=True,
    l=None,
    eval_after=1
):

    fix_seed(experiment=seed)
    new_acc = []

    if l is None and os.path.isdir(Path(base_path) / f"model-{model_index}"):
        # Only executes in random stopping
        
        model_dir = Path(base_path) / f"model-{model_index}"

        saved_steps = sorted(
            [int(f.name.split('-')[-1]) for f in model_dir.glob("step-*") if f.is_file()]
        )

        print(saved_steps)

        latest_step = max([s for s in saved_steps if s <= iterations], default=0)
        iterations = iterations - latest_step

        if latest_step:
            print(f'Loading checkpoint from latest step {latest_step} - remaining iterations {iterations}')
            checkpoint_path = model_dir / f"step-{latest_step}"
            state = torch.load(checkpoint_path)
            model, hyperparameters = init_model(**state["model_info"])
            step = state["step"]
            accuracies = state['accuracies']

    elif base_path and l != 1 and l != None :

        state = torch.load(f"models/seed-{seed}/models/{model_id}")
        model, hyperparameters = init_model(**state["model_info"])
        step = state["step"]

    else:
        model, hyperparameters = init_model(**model_info)
        accuracies = []
        step = 0
    current_iters = 0
    optimizer = SGD(params=model.parameters(), lr=hyperparameters["lr"])
    if l != None and l != 1 and os.path.exists(f"models/seed-{seed}/models/{model_id}"):
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        accuracies = state["accuracies"]

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

    _ = ModuleValidator.validate(model, strict=False)
    # log_message(message=f"Using Opacus Module validator, existing Errors in model architecture {len(errors)}", seed=exp,)
    assert hyperparameters["batch_size"] % MAX_PHYSICAL_BATCH_SIZE == 0

    k = hyperparameters["batch_size"] // MAX_PHYSICAL_BATCH_SIZE
    criterion = nn.CrossEntropyLoss()

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
                if current_iters == iterations:
                    break
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
                    current_iters += 1

                    # print((current_iters, iterations))
                    if (eval_points and current_iters in eval_points) or (eval_points == None and (current_iters + 1)%eval_after == 0):
                        acc_test, _ = val_loop(
                            criterion=criterion,
                            dl_test=test_dl,
                            model=model,
                            device=device,
                            verbose=False,
                        )

                        if writer is not None:
                            writer.add_scalar(
                                f"test accuracy {seed} - {model_id}", acc_test, step
                            )
                        log_message(
                            message=f"test accuracy {seed} - {model_id} {acc_test:.3f}",
                            step=step + 1,
                            seed=seed,
                        )

                        accuracies.append(acc_test)
                        new_acc.append(acc_test)

                        path = Path(base_path) / f"model-{model_index}"
                        path.mkdir(parents=True, exist_ok=True)
                        torch.save(
                                {
                                    "model_state_dict": model._module.state_dict(),
                                    "optimizer_state_dict": optimizer.original_optimizer.state_dict(),
                                    "step": step,
                                    "hyperparameters": hyperparameters,
                                    "model_info": model_info,
                                    "accuracies": accuracies,
                                },
                                Path(base_path) / f"model-{model_index}/step-{step+1}",
                            )
                    
            break
    return accuracies, new_acc
