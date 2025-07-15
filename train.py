import os
import time
import argparse
import numpy as np
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import Fire
from LossFunction import FocalLoss, DiceLoss, CombinedLoss

# Constants
LIMIT_PATIENCE = 30  # Maximum number of epochs with no improvement before early stopping
NET_NAME = "Net"  # Default network name

def calc_confusion_matrix(output, target, threshold=0.5):
    """
    Calculate confusion matrix metrics (TP, TN, FP, FN) for binary classification.
    - TP: True Positives (correctly predicted positive samples).
    - TN: True Negatives (correctly predicted negative samples).
    - FP: False Positives (negative samples incorrectly predicted as positive).
    - FN: False Negatives (positive samples incorrectly predicted as negative).
    """
    output = (output >= threshold).float()
    TP = len(torch.where(torch.isclose(output, torch.tensor(1.0)) & torch.isclose(target, torch.tensor(1.0)))[0])
    TN = len(torch.where(torch.isclose(output, torch.tensor(0.0)) & torch.isclose(target, torch.tensor(0.0)))[0])
    FP = len(torch.where(torch.isclose(output, torch.tensor(1.0)) & torch.isclose(target, torch.tensor(0.0)))[0])
    FN = len(torch.where(torch.isclose(output, torch.tensor(0.0)) & torch.isclose(target, torch.tensor(1.0)))[0])
    return np.array([TP, TN, FP, FN])

def calc_precision(TP, TN, FP, FN):
    """Calculate precision: TP / (TP + FP) with zero division handling."""
    return TP / (TP + FP) if TP + FP > 0 else 0

def calc_recall(TP, TN, FP, FN):
    """Calculate recall: TP / (TP + FN) with zero division handling."""
    return TP / (TP + FN) if TP + FN > 0 else 0

def calc_f1_score(precision, recall):
    """Calculate F1-score: 2 * (precision * recall) / (precision + recall) with zero division handling."""
    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

def print_and_record_result(epoch, writer, loss, index, type='train'):
    """
    Print and record training/testing metrics to TensorBoard.
    Args:
        epoch (int): Current epoch number.
        writer (SummaryWriter): TensorBoard writer instance.
        loss (float): Average loss for the epoch.
        index (np.array): Confusion matrix values [TP, TN, FP, FN].
        type (str): 'train' or 'test' to specify the dataset type.
    Returns:
        float: F1-score for the epoch.
    """
    TP, TN, FP, FN = index
    precision = calc_precision(TP, TN, FP, FN)
    recall = calc_recall(TP, TN, FP, FN)
    f1 = calc_f1_score(precision, recall)
    print(f"Epoch {epoch} {type} set average Loss: {loss:.8f}; precision: {precision:.6f}, "
          f"recall: {recall:.6f}, F1: {f1:.6f}")
    writer.add_scalar(f"{type}_loss", loss, global_step=epoch)
    writer.add_scalar(f"{type}_precision", precision, global_step=epoch)
    writer.add_scalar(f"{type}_recall", recall, global_step=epoch)
    writer.add_scalar(f"{type}_f1", f1, global_step=epoch)
    return f1

def test_model(model):
    """Test the model architecture with random input to verify shapes and output range."""
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", total_params)

    batch_size = 4  # Batch size of 4 samples
    input_channels = 12  # 12 input channels (terrain, wind, moisture, etc.)
    height, width = 256, 256  # 256x256 pixels as per dataset specification

    x = torch.randn(batch_size, input_channels, height, width)  # Random input tensor
    output = model(x)

    print(f"Input shape: {x.shape}")  # Expected: [4, 12, 256, 256]
    print(f"Output shape: {output.shape}")  # Expected: [4, 1, 256, 256]
    print(f"Output min: {output.min()}, max: {output.max()}, mean: {output.mean()}")

def Init_net(model_name):
    if model_name == "rcda":
        from Models.RCDA import RCDA
        net = RCDA()
    elif model_name == "rcda-5":
        from Models.RCDA import RCDA
        net = RCDA(depth=5)
    elif model_name == "cda":
        from Models.CDA import CDA
        net = CDA()
    elif model_name == "ragca":
        from Models.RAGCA import RAGCA
        net = RAGCA()
    elif model_name == "rca":
        from Models.RCA import RCA
        net = RCA()
    elif model_name == "unet":
        from Models.UNet import UNet
        net = UNet(img_ch=12)
    elif model_name == "attunet":
        from Models.WPN import AttU_Net
        net = AttU_Net(img_ch=12)
    elif model_name == "resunet":
        net = smp.Unet(encoder_name="resnet34", in_channels=12, classes=1, activation="sigmoid")
    elif model_name == "wpn":
        from Models.WPN import WPN
        net = WPN(img_ch=12)
    elif model_name == "fire_simulator":
        from Models.FireSimulator import FireSpreadEmulator
        net = FireSpreadEmulator()
    elif model_name == "funetcast":
        from Models.FUNetCast import FUNetCast
        net = FUNetCast()
    elif model_name == "r2u_net":
        from Models.WPN import R2U_Net
        net = R2U_Net(img_ch=12)
    elif model_name == "r2att_u_net":
        from Models.WPN import R2AttU_Net
        net = R2AttU_Net(img_ch=12)
    elif model_name == "asufm":
        from Models.asufm.asufm import ASUFM
        net = ASUFM(num_classes=1)
    else:
        raise ValueError(f"Unsupported model parameters: {model_name}.")
    return net

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a segmentation model for wildfire spread prediction.")
    parser.add_argument('--limit_patience', type=int, default=LIMIT_PATIENCE,
                        help='Maximum number of epochs with no improvement before early stopping.')
    parser.add_argument('--net_name', type=str, default=NET_NAME, help='Name of the network model.')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model architecture details (e.g., rcda, unet, etc.).')
    parser.add_argument('--weight_save_name', type=str, default=NET_NAME,
                        help='Name for saving the model weights.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimization.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading.')
    parser.add_argument('--epochs', type=int, default=200, help='Maximum number of training epochs.')
    parser.add_argument('--loss_function', type=str, default='FocalLoss',
                        help='Loss function type: FocalLoss, DiceLoss, CombinedLoss, or BCELoss.')
    parser.add_argument('--focal_alpha', type=float, default=0.75, help='Alpha parameter for Focal Loss.')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss.')
    parser.add_argument('--dice_weight', type=float, default=0.5,
                        help='Weight of Dice Loss in Combined Loss.')

    args = parser.parse_args()
    if not os.path.exists("weights"):
        os.mkdir("weights")
    WEIGHT_SAVE_PATH = f"weights/{args.weight_save_name}.pth"

    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # 1. Prepare dataset
    fire_train_data = Fire('train', augmentation=True)
    fire_eval_data = Fire('test', augmentation=False)
    train_dataloader = DataLoader(fire_train_data, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(fire_eval_data, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

    # 2. Build neural network
    net = Init_net(args.model_name)

    # Count total parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params}")
    net.to(device)

    # 3. Define loss function
    if args.loss_function == 'FocalLoss':
        loss_function = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.loss_function == 'DiceLoss':
        loss_function = DiceLoss()
    elif args.loss_function == 'CombinedLoss':
        loss_function = CombinedLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, dice_weight=args.dice_weight)
    elif args.loss_function == 'BCELoss':
        loss_function = nn.BCELoss()
    else:
        raise ValueError(f"Unsupported loss function: {args.loss_function}. Choose from 'FocalLoss', "
                         "'DiceLoss', 'CombinedLoss', or 'BCELoss'.")

    loss_function.to(device)

    # 4. Define optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate)

    # 5. (Optional) Set up TensorBoard for visualization
    log_path = os.path.join("Logs", args.net_name)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    # 6. Start training
    total_train_step = 0
    best_epoch = -1
    best_f1 = -1
    patience = 0

    for epoch in range(1, args.epochs + 1):
        print(f"-------------Epoch {epoch} training started--------------")
        start_time = time.time()

        net.train()
        train_fx = np.zeros(4)  # [TP, TN, FP, FN]
        train_loss = 0.0

        for data in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_step += 1
            train_loss += loss.item()
            train_fx += calc_confusion_matrix(outputs, labels)

        # Record training results
        train_loss /= len(train_dataloader)
        train_f1 = print_and_record_result(epoch, writer, train_loss, train_fx)

        # Evaluate on test set
        test_fx = np.zeros(4)  # [TP, TN, FP, FN]
        test_loss = 0.0
        net.eval()
        with torch.no_grad():
            for data in tqdm(test_dataloader, desc=f"Testing Epoch {epoch}"):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = loss_function(outputs, labels)
                test_loss += loss.item()
                test_fx += calc_confusion_matrix(outputs, labels)

        end_time = time.time()
        test_loss /= len(test_dataloader)
        test_f1 = print_and_record_result(epoch, writer, test_loss, test_fx, type='test')

        if test_f1 >= best_f1:
            patience = 0
            best_f1 = test_f1
            best_epoch = epoch
            torch.save(net, WEIGHT_SAVE_PATH)
            print(f"Model saved at epoch {epoch} with best F1-score: {best_f1:.6f}")
        else:
            patience += 1

        print(f"Epoch {epoch} training completed, time taken: {round(end_time - start_time, 2)} seconds")
        if patience > args.limit_patience:
            print(f"Epoch {epoch} training ended, exceeded {args.limit_patience} epochs with no improvement, "
                  "training stopped early")
            break

    # Final summary
    print(f"The best model is from epoch {best_epoch} with the highest test F1-score: {best_f1:.6f}")
    print(f"Training completed, model weights saved to: {WEIGHT_SAVE_PATH}")
    writer.close()