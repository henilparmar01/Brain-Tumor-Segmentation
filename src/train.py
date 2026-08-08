import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import UNet
from dataloader import create_dataloader
from loss import BCEDiceLoss


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):

    model.train()

    running_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        unit="batch"
    )

    for images, masks in progress_bar:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        masks = masks.unsqueeze(1)

        loss = loss_fn(outputs, masks)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}"
        })

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss


   


def validate_one_epoch(model, dataloader, loss_fn, device):

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for images, masks in dataloader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            masks = masks.unsqueeze(1)

            loss = loss_fn(outputs, masks)

            running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss

def main():

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader
    print("Creating DataLoader...")
    train_loader, val_loader = create_dataloader()

    # Model
    print("Creating Model...")
    model = UNet().to(device)

    # Loss Function
    print("Creating Loss Function...")
    loss_fn = nn.BCEDiceLoss()

    # Optimizer
    print("Creating Optimizer...")
    optimizer = Adam(model.parameters(), lr=0.001)

    #Learning Rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )

    # Number of Epochs
    epochs = 10

    #best model tracking
    best_val_loss = float("inf")

    #Early stopping
    patience = 5
    counter = 0

    # Training history
    train_losses = []
    val_losses = []

    print("\n===================================")
    print("      Training Started 🚀")
    print("===================================\n")



    for epoch in range(epochs):

        print(f"\nEpoch [{epoch+1}/{epochs}]")

        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device
        )

        val_loss = validate_one_epoch(
            model,
            val_loader,
            loss_fn,
            device
        )

        # save loss history
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        #scheduler
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            counter = 0

            torch.save(model.state_dict(), "best_model.pth")

            print("Best model saved!")

        else:

            counter += 1
            print(
                f"  No improvement for "
                f"{counter} / {patience} epochs."
            )


        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
         )

        if counter >= patience:

            print("\n   Earlt Stopping Triggered!")
            break  

    print("\n===================================")
    print("     Training Completed ")
    print("===================================")

    # Save training history
    history = {
        "train_loss":train_losses,
        "val_loss":val_losses
    }

    return history

if __name__ == "__main__":
    main()