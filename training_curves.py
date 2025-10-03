import csv
import wandb

wandb.init(name="baseline_curves")

with open("training_log.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epoch = int(row["epoch"])
        train_loss = float(row["train_loss"])
        train_acc = float(row["train_acc"])
        val_loss = float(row["val_loss"])
        val_acc = float(row["val_acc"])

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
            },
            step=epoch,
        )

wandb.finish()
