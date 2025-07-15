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

LIMIT_PATIENCE = 30
NET_NAME = "Net"


def calc_confusion_matrix(output, target, threshold=0.5):
    # 计算混淆矩阵，包括: (以下正类为 1 ， 负类为 0)
    # TP（True Positives）表示模型将正类样本正确预测为正类的数量。
    # TN（True Negatives）表示模型将负类样本正确预测为负类的数量。
    # FP（False Positives）表示模型将负类样本错误地预测为正类的数量。
    # FN（False Negatives）表示模型将正类样本错误地预测为负类的数量。
    output = (output >= threshold).float()
    TP = len(torch.where(torch.isclose(output, torch.tensor(1.0)) & (torch.isclose(target, torch.tensor(1.0))))[0])
    TN = len(torch.where(torch.isclose(output, torch.tensor(0.0)) & (torch.isclose(target, torch.tensor(0.0))))[0])
    FP = len(torch.where(torch.isclose(output, torch.tensor(1.0)) & (torch.isclose(target, torch.tensor(0.0))))[0])
    FN = len(torch.where(torch.isclose(output, torch.tensor(0.0)) & (torch.isclose(target, torch.tensor(1.0))))[0])
    # confusion_matrix = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
    return np.array([TP, TN, FP, FN])


def calc_precision(TP, TN, FP, FN):
    # 计算精确度
    return TP / (TP + FP) if TP + FP > 0 else 0


def calc_recall(TP, TN, FP, FN):
    # 计算召回率
    return TP / (TP + FN) if TP + FN > 0 else 0


def calc_f1_score(precision, recall):
    # 计算F1 score
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0


# def ResultAnaysis(preds, labels):
#     predictions = (preds > 0.5).float().cpu().numpy().flatten()
#     true_labels = labels.cpu().numpy().flatten()
#     TN, FP, FN, TP = confusion_matrix(true_labels, predictions).revel() # from sklearn.metrics import confusion_matrix
#     precision = precision_score(true_labels, predictions, zero_division=0)
#     recall = recall_score(true_labels, predictions, zero_division=0)
#     f1 = f1_score(true_labels, predictions, zero_division=0)
#     return precision, recall, f1

def PrintAndRecordResult(epoch, writer, loss, index, TYPE='train'):
    TP, TN, FP, FN = index
    precision = calc_precision(TP, TN, FP, FN)
    recall = calc_recall(TP, TN, FP, FN)
    f1 = calc_f1_score(precision, recall)
    print(
        f"第 {epoch} 轮 {TYPE} 集平均Loss: {(loss):.8f}; precision:{(precision):.6f}、recall:{(recall):.6f}、F1:{(f1):.6f}")
    # 写入
    writer.add_scalar(f"{TYPE}_loss", loss, global_step=epoch)
    writer.add_scalar(f"{TYPE}_precsion", precision, global_step=epoch)
    writer.add_scalar(f"{TYPE}_recall", recall, global_step=epoch)
    writer.add_scalar(f"{TYPE}_f1", f1, global_step=epoch)
    return f1


def TestModel(model):
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", total_params)

    # Generate random inputs data (matching paper's specs)
    batch_size = 4  # Batch of 4 samples
    input_channels = 12  # 12 inputs channels (terrain, wind, moisture, etc.)
    height, width = 256, 256  # 256x256 pixels (as per paper)

    # Random inputs tensor (simulating 12-channel inputs image)
    x = torch.randn(batch_size, input_channels, height, width)
    # Forward pass
    output = model(x)

    # Check shapes
    print(f"Input shape: {x.shape}")  # Expected: [4, 12, 256, 256]
    print(f"Output shape: {output.shape}")  # Expected: [4, 1, 256, 256]
    print(output.min(), output.max(), output.mean())  # Check output values


if __name__ == '__main__':
    # 加载配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit_patience', type=int, default=LIMIT_PATIENCE, help='最大容忍无优化的轮数')
    parser.add_argument('--net_name', type=str, default=NET_NAME, help='网络模型名称')
    parser.add_argument('--model_params', type=str, default=None, help='网络模型细节')
    parser.add_argument('--weight_save_name', type=str, default="NET_NAME", help='模型权重保存名')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch_size', type=int, default=8, help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载的工作线程数')
    parser.add_argument('--epochs', type=int, default=400, help='最大训练轮数')
    parser.add_argument('--loss_function', type=str, default='FocalLoss',
                        help='损失函数类型: FocalLoss, DiceLoss, CombinedLoss, BCELoss')
    parser.add_argument('--focal_alpha', type=float, default=0.75, help='Focal Loss 的 alpha 参数')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss 的 gamma 参数')
    parser.add_argument('--dice_weight', type=float, default=0.5, help='Combined Loss 中 Dice Loss 的权重')

    args = parser.parse_args()
    WEIGHT_SAVE_PATH = f"Experiment/weights/{args.weight_save_name}.pth"
    # 通过这种方式定义使用什么设备进行训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1、准备数据集
    # 划分数据集
    fire_train_data = Fire('train', augmentation=True)
    fire_eval_data = Fire('test', augmentation=False)
    train_dataloader = DataLoader(fire_train_data, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)
    test_dataloader = DataLoader(fire_eval_data, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
    # 2、搭建神经网络
    if args.model_params == "rcda":
        from Models.RCDA import RCDA
        net = RCDA()
    elif args.model_params == "rcda-5":
        from Models.RCDA import RCDA
        net = RCDA(depth=5)
    elif args.model_params == "cda":
        from Models.CDA import CDA
        net = CDA()
    elif args.model_params == "ragca":
        from Models.RAGCA import RAGCA
        net = RAGCA()
    elif args.model_params == "rca":
        from Models.RCA import RCA
        net = RCA()
    elif args.model_params == "unet":
        from Models.UNet import UNet
        net = UNet(img_ch=12)
    elif args.model_params == "attunet":
        from Models.WPN import AttU_Net
        net = AttU_Net(img_ch=12)
    elif args.model_params == "resunet":
        net = smp.Unet(encoder_name="resnet34", in_channels=12, classes=1, activation="sigmoid")
    elif args.model_params == "wpn":
        from Models.WPN import WPN
        net = WPN(img_ch=12)
    elif args.model_params == "fire_simulator":
        from Models.FireSimulator import FireSpreadEmulator
        net = FireSpreadEmulator()
    elif args.model_params == "funetcast":
        from Models.FUNetCast import FUNetCast
        net = FUNetCast()
    elif args.model_params == "r2u_net":
        from Models.WPN import R2U_Net
        net = R2U_Net(img_ch=12)
    elif args.model_params == "r2att_u_net":
        from Models.WPN import R2AttU_Net
        net = R2AttU_Net(img_ch=12)
    elif args.model_params == "asufm":
        from Models.asufm.asufm import ASUFM
        net = ASUFM(num_classes=1)
    else:
        raise ValueError(f"Unsupported model parameters: {args.model_params}.")
    # Count the number of parameters
    total_params = sum(p.numel() for p in net.parameters())
    print("Total parameters: ", total_params)
    net.to(device)
    # 3、确定损失函数
    if args.loss_function == 'FocalLoss':
        loss_function = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.loss_function == 'DiceLoss':
        loss_function = DiceLoss()
    elif args.loss_function == 'CombinedLoss':
        loss_function = CombinedLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, dice_weight=args.dice_weight)
    elif args.loss_function == 'BCELoss':
        loss_function = nn.BCELoss()
    else:
        raise ValueError(
            f"Unsupported loss function: {args.loss_function}. Choose from 'FocalLoss', 'DiceLoss', 'CombinedLoss', or 'BCELoss'.")

    loss_function.to(device)
    # 4、优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate)

    # 5、(可选) 将结果保存到tensorboard 进行可视化
    log_path = os.path.join("Logs", args.net_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    print("训练采用" + "GPU" if torch.cuda.is_available() else "CPU")

    # 6、开始训练
    total_train_step = 0
    best_epoch = -1
    best_f1 = -1
    patience = 0
    for epoch in range(1, args.epochs + 1):
        print("-------------第 {} 轮训练开始--------------".format(epoch))
        start_time = time.time()
        train_start_time = time.time()
        net.train()
        # print("此轮使用的learning rate = {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        train_fx = np.zeros(4)  # TP, TN, FP, FN
        train_loss = 0
        for data in tqdm(train_dataloader):
            # 计算
            inputs, labels = data
            inputs = inputs.to(device)  # gpu 加速
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_step += 1
            # 结果统计
            train_loss += loss.item()
            # outputs = (F.sigmoid(outputs) > 0.5).float()
            train_fx += calc_confusion_matrix(outputs, labels)
            # 输出 tensorboard (每隔 x 次输出一次)
            # if total_train_step % 200 == 0:
            # print(f"训练batch次数: {total_train_step} , 此时 Loss: {loss:.8f}; precision:{precision:.6f}、recall:{recall:.6f}、F1:{f1:.6f}、OA:{OA:.6f}、Jaccard:{jaccard:.6f}; 累计耗时: {time.time() - train_start_time:.2f} 秒")
        # 结果保存
        PrintAndRecordResult(epoch, writer, train_loss / len(train_dataloader), train_fx)
        # scheduler.step()
        # 测试数据集，不需要优化参数
        test_fx = np.zeros(4)  # TP, TN, FP, FN
        test_loss = 0
        net.eval()
        with torch.no_grad():
            for data in tqdm(test_dataloader):
                # 计算
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                loss = loss_function(outputs, labels)
                # 结果统计
                test_loss += loss.item()
                # outputs = (F.sigmoid(outputs) > 0.5).float()
                test_fx += calc_confusion_matrix(outputs, labels)

        end_time = time.time()
        # 结果保存
        test_f1 = PrintAndRecordResult(epoch, writer, test_loss / len(test_dataloader), test_fx, TYPE='test')
        if test_f1 >= best_f1:
            patience = 0
            best_f1 = test_f1
            best_epoch = epoch
            torch.save(net, WEIGHT_SAVE_PATH)
        else:
            patience += 1
        print(f"第 {epoch} 轮训练结束，耗时: {round(time.time() - start_time, 2)} 秒")
        if patience > args.limit_patience:
            print(f"第 {epoch} 轮训练结束, 已超过 {args.limit_patience} 轮 测试集没有任何优化, 提前停止训练")
            break
        epoch += 1

    # 保存最好的结果 和 训练过程中的 相关指数
    print(f"最后的网络模型是第 {best_epoch} 轮 训练的结果; 最好的测试集f1-score达到: {best_f1}")
    print(f"训练完成，模型 {args.net_name} 权重已保存到: {WEIGHT_SAVE_PATH}")
    writer.close()