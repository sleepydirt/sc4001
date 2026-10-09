import copy

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_shape=(1, 1, 64, 64)):
    """Helper to count FLOPs"""
    was_training = model.training
    model.eval()
    with FlopCounterMode(display=False) as counter:
        model(torch.zeros(*input_shape))
    model.train(was_training)
    return counter.get_total_flops()


@torch.no_grad()
def evaluate(model, loader, criterion=None):
    """Loss and confusion-based metrics, pneumonia = 1"""
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    model.eval()
    total_loss, count = 0.0, 0
    tp = tn = fp = fn = 0
    for images, labels in loader:
        logits = model(images)
        total_loss += criterion(logits, labels).item() * len(labels)
        count += len(labels)

        predicted = (logits > 0).float()
        tp += int(((predicted == 1) & (labels == 1)).sum())
        tn += int(((predicted == 0) & (labels == 0)).sum())
        fp += int(((predicted == 1) & (labels == 0)).sum())
        fn += int(((predicted == 0) & (labels == 1)).sum())

    accuracy = (tp + tn) / count
    return {
        "loss": total_loss / count,
        "accuracy": accuracy,
        "error": 1.0 - accuracy,
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def train_model(model, train_loader, val_loader, config, learning_rate, verbose=True):
    """Training loop with Adam optimiser.
    
    Early stopping is based on val loss"""
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_error": []}
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    diverged = False

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        running_loss, count = 0.0, 0
        for images, labels in train_loader:
            optimiser.zero_grad()
            loss = criterion(model(images), labels)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * len(labels)
            count += len(labels)

        if count == 0:
            break

        metrics = evaluate(model, val_loader, criterion)
        history["train_loss"].append(running_loss / count)
        history["val_loss"].append(metrics["loss"])
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_error"].append(metrics["error"])

        improved = metrics[config.monitor] < best_score - config.early_stopping_min_delta
        if verbose:
            print(f"  epoch {epoch:2d}/{config.max_epochs}  "
                  f"train_loss={running_loss / count:.4f}  val_loss={metrics['loss']:.4f}  "
                  f"val_acc={metrics['accuracy']:.4f}  val_error={metrics['error']:.4f}"
                  + ("  *" if improved else ""), flush=True)

        if diverged:
            if verbose:
                print("  stopped: non-finite loss", flush=True)
            break

        if improved:
            best_score = metrics[config.monitor]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                if verbose:
                    print(f"  early stopping after epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "diverged": diverged,
        "epochs_run": len(history["train_loss"]),
    }
