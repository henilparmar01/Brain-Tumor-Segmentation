import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

from model import UNet
from dataloader import create_dataloader
from loss import TverskyFocalLoss
from config import LEARNING_RATE, EPOCHS

def dice_score(predictions, targets, smooth=1.0):
    probabilities = torch.sigmoid(predictions)
    predictions_binary = (probabilities > 0.5).float()

    predictions_binary = predictions_binary.view(-1)
    targets = targets.view(-1)

    intersection = (predictions_binary * targets).sum()

    dice = (2.0 * intersection + smooth) / (
        predictions_binary.sum() + targets.sum() + smooth
    )

    return dice.item()


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, scaler):

    model.train()

    running_loss = 0.0
    running_dice = 0.0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        unit="batch"
    )

    for images, masks in progress_bar:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            device_type=device.type,
            enabled=(device.type == "cuda")
        ):
            outputs = model(images)
            
            masks = masks.unsqueeze(1)
            
            loss = loss_fn(outputs, masks)

        

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()
        with torch.no_grad():
            batch_dice = dice_score(outputs, masks)
            running_dice += batch_dice

        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}"
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)

    return epoch_loss, epoch_dice


   


def validate_one_epoch(model, dataloader, loss_fn, device):

    model.eval()

    running_loss = 0.0
    running_dice = 0.0

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        unit="batch"
    )

    with torch.no_grad():

        for images, masks in progress_bar:

            images = images.to(device)
            masks = masks.to(device)

            with autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):
                outputs = model(images)
                
                masks = masks.unsqueeze(1)
                
                loss = loss_fn(outputs, masks)

            

            running_loss += loss.item()
            with torch.no_grad():
                batch_dice = dice_score(outputs, masks)
                running_dice += batch_dice
                            
                        

    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)

    return epoch_loss, epoch_dice

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
    loss_fn = TverskyFocalLoss()

    # Optimizer
    print("Creating Optimizer...")
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    #Learning Rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )

    # Number of Epochs
    epochs = EPOCHS

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

        train_loss, train_dice = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scaler
        )

        val_loss, val_dice = validate_one_epoch(
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
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
         )

        if counter >= patience:

            print("\n   Early Stopping Triggered!")
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