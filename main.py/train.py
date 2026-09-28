import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    BEST_MODEL_PATH,
    LAST_MODEL_PATH,
    LOG_FILE,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    GRAPHS_DIR
)

from dataset import (
    get_train_loader,
    get_val_loader
)

from model import get_model

from utils import (
    train_one_epoch,
    validate
)


def main():

    # Create output directories
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    train_loader = get_train_loader()
    val_loader = get_val_loader()

    print(f"Training Images   : {len(train_loader.dataset)}")
    print(f"Validation Images : {len(val_loader.dataset)}")

    print("\nLoading ConvNeXt-Base...\n")

    # Load Model
    model = get_model()

    # Triplet Loss
    criterion = nn.TripletMarginLoss(
        margin=1.0,
        p=2
    )

    # Optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    best_loss = float("inf")

    # Store metrics
    epochs_list = []
    train_losses = []
    val_losses = []

    print("=" * 60)
    print("Training Started")
    print("=" * 60)

    # CSV Log
    with open(LOG_FILE, "w", newline="") as log_file:

        log_writer = csv.writer(log_file)
        log_writer.writerow([
            "Epoch",
            "Train Loss",
            "Validation Loss"
        ])

        for epoch in range(EPOCHS):

            print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

            train_loss, train_acc = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=DEVICE
            )

            val_loss, val_acc = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=DEVICE
            )

            print(f"Train Loss      : {train_loss:.4f}")
            print(f"Validation Loss : {val_loss:.4f}")

            epochs_list.append(epoch + 1)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Save logs
            log_writer.writerow([
                epoch + 1,
                f"{train_loss:.4f}",
                f"{val_loss:.4f}"
            ])

            # Save Best Model
            if val_loss < best_loss:

                best_loss = val_loss

                torch.save(
                    model.state_dict(),
                    BEST_MODEL_PATH
                )

                print("-> Best Model Saved")

        # Save Last Model
        torch.save(
            model.state_dict(),
            LAST_MODEL_PATH
        )

    print("\n" + "=" * 60)
    print("Training Finished Successfully")
    print("=" * 60)

    print(f"Best Validation Loss : {best_loss:.4f}")

    print(f"\nBest Model : {BEST_MODEL_PATH}")
    print(f"Last Model : {LAST_MODEL_PATH}")
    print(f"Log File   : {LOG_FILE}")

    print("\nAll outputs saved successfully.")

    print(f"\nCheckpoints : {CHECKPOINTS_DIR}")
    print(f"Logs        : {LOGS_DIR}")
    print(f"Graphs      : {GRAPHS_DIR}")

    print("=" * 60)


if __name__ == "__main__":
    main()