import torch
from torch.utils.data import DataLoader


def val_loop(
    model,
    criterion,
    dl_test: DataLoader | None = None,
    batch=None,
    device="cuda",
    verbose=False,
):
    dps, correct, loss = 0, 0, 0
    model.eval()
    with torch.no_grad():
        if batch is not None:
            y_predicted = model(batch["data"].to(device))
            y_predicted_cls = y_predicted.argmax(dim=1, keepdim=True)
            correct += (
                y_predicted_cls.eq(batch["label"].view_as(y_predicted_cls).to(device))
                .sum()
                .item()
            )
            loss += criterion(y_predicted, batch["label"].to(device))
            dps += len(batch["data"])

            if verbose:
                print((correct, dps))
            assert correct <= dps

        else:
            for batch in dl_test:
                y_predicted = model(batch["data"].to(device))
                y_predicted_cls = y_predicted.argmax(dim=1, keepdim=True)
                correct += (
                    y_predicted_cls.eq(
                        batch["label"].view_as(y_predicted_cls).to(device)
                    )
                    .sum()
                    .item()
                )
                loss += criterion(y_predicted, batch["label"].to(device))

                dps += len(batch["data"])
                if verbose:
                    print((correct, dps))
                assert correct <= dps

    return correct / dps, loss / dps
