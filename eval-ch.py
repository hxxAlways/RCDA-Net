import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Fire
from train import calc_confusion_matrix, calc_precision, calc_recall, calc_f1_score, Init_net

def calc_IoU(TP, TN, FP, FN):
    return TP / (TP + FP + FN) if TP + FP + FN > 0 else 0

def evaluate(model, dataloader, device, thresholds=[0.2,0.3,0.4,0.5,0.6], use_increment=True, ignore=None):
    model.eval()
    best_f1, best_precision, best_recall, best_IoU, best_threshold = 0, 0, 0, 0, 0
    n = len(thresholds)
    Index = np.zeros((n, 4))  # n * [TP, TN, FP, FN]
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            if ignore is not None:
                for ig in ignore:
                    inputs[0, ig] *= 0
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            if not use_increment:
                outputs -= inputs.squeeze()[0]
                labels -= inputs.squeeze()[0]
            for idx in range(n):
                Index[idx] += calc_confusion_matrix(outputs, labels, threshold=thresholds[idx])
    for i in range(n):
        TP, TN, FP, FN = Index[i]
        precision = calc_precision(TP, TN, FP, FN)
        recall = calc_recall(TP, TN, FP, FN)
        f1 = calc_f1_score(precision, recall)
        IoU = calc_IoU(TP, TN, FP, FN)
        print(f"Evaluating with threshold: {thresholds[i]}; F1-score: {f1:.6f}")
        if f1 > best_f1:
            best_f1, best_precision, best_recall, best_IoU, best_threshold = f1, precision, recall, IoU, thresholds[i]
    return best_precision, best_recall, best_f1, best_IoU, best_threshold
def add_text(ax, text, fontweight='normal', fontsize=18, loc_x=0.11, loc_y=0.9, width=0.04, height=0.12, pad=0.14):
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx
    ax.text(s=text,
            x=minx + xlen * loc_x,
            y=miny + ylen * (loc_y - pad + height),
            fontweight=fontweight,
            fontsize=fontsize,
            horizontalalignment='center',
            verticalalignment='bottom')
def add_north(ax, labelsize=24, loc_x=0.07, loc_y=0.9, width=0.04, height=0.12, pad=0.14):
    """
    画一个指北针带'N'文字注释
    主要参数如下
    :param ax: 要画的坐标区域 Axes实例 plt.gca()获取即可
    :param labelsize: 显示'N'文字的大小
    :param loc_x: 以文字下部为中心的占整个ax横向比例
    :param loc_y: 以文字下部为中心的占整个ax纵向比例
    :param width: 指南针占ax比例宽度
    :param height: 指南针占ax比例高度
    :param pad: 文字符号占ax比例间隙
    :return: None
    """
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx
    left = [minx + xlen * (loc_x - width * .5), miny + ylen * (loc_y - pad)]
    right = [minx + xlen * (loc_x + width * .5), miny + ylen * (loc_y - pad)]
    top = [minx + xlen * loc_x, miny + ylen * (loc_y - pad + height)]
    center = [minx + xlen * loc_x, left[1] + (top[1] - left[1]) * .4]
    triangle = mpatches.Polygon([left, top, right, center], color='k')
    add_text(ax, "N", 'normal', labelsize, loc_x, loc_y, width, height, pad)
    ax.add_patch(triangle)
def add_annotation(ax, Index, labelsize=24, loc_x=0.78, loc_y=0.92, width=0.04, height=0.12, pad=0.14):
    IndexTitle = ["precision: %.2f%%" % (Index[0]*100), "recall: %.2f%%" % (Index[1]*100), "f1: %.3f" % Index[2], "IoU: %.3f" % (Index[3])]
    for i in range(4):
        add_text(ax, IndexTitle[i], 'bold', labelsize, loc_x, loc_y, width, height, (1 + 0.5 * i) * pad)
def add_scalebar(ax):
    # 比例尺设置：
    fontprops = fm.FontProperties(size=24)
    scalebar = AnchoredSizeBar(ax.transData,
                               size=60,  # 代表x轴上10单位
                               label='3 km',  # 显示的实际距离
                               loc='lower right',  # 位置
                               pad=0.5,
                               color='black',
                               frameon=False,
                               size_vertical=1,
                               fontproperties=fontprops,
                               bbox_to_anchor=(1.0, 0.05))  # 调整y坐标（0.05）使偏下
    ax.add_artist(scalebar)  # 添加比例尺

def Cases_Analysis(modelname, models, thresholdlist, UID:list[int], date:list[datetime]):
    labels = ['Fire Mask at Time T', 'Incremental Fire Mask at Time T+1']
    colors = ['#67000C', '#FA6949']
    fire_data = Fire('eval', UID=UID, start=date, end=date)
    dataloader = DataLoader(fire_data, shuffle=False, batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models.append(None) # 用于绘制标签 (一定要排在最后)
    plt.rcParams['font.family'] = 'Times New Roman, SimSun'  # 设置字体族，中文为SimSun，英文为Times New Roman
    fig = plt.figure(figsize=(24, 14))
    for dataid, (input, label) in enumerate(dataloader):
        input = input.to(device)
        label = label.to(device)
        for modelid, model in enumerate(models):
            figid = dataid * len(models) + modelid + 1
            ax = plt.subplot(len(dataloader), len(models), figid)
            data = input.squeeze()[0].cpu().detach().numpy()
            Index = None
            if model:
                model.eval()
                output = model(input)
                TP, TN, FP, FN = calc_confusion_matrix(output, label, threshold=thresholdlist[modelid])
                precision = calc_precision(TP, TN, FP, FN)
                recall = calc_recall(TP, TN, FP, FN)
                f1 = calc_f1_score(precision, recall)
                IoU = calc_IoU(TP, TN, FP, FN)
                Index = [precision, recall, f1, IoU]
                # print(f"precision:{precision:.6f}、recall:{recall:.6f}、F1:{f1:.6f}、IoU:{IoU:.6f}")
                # 画图
                pred = (output > thresholdlist[modelid]).float().cpu().squeeze().detach().numpy()
                data[pred == 1] = 0.5
                plt.title(f"({chr(figid + 96)}) {modelname[modelid]} Prediction on case-{UID[dataid]}", y=-0.1, font={'size': 26})
            else:
                label = label.cpu().squeeze().detach().numpy()
                data[label == 1] = 0.5
                plt.title(f"({chr(figid + 96)}) Label on case-{UID[dataid]}", y=-0.1, font={'size': 26})
            Title_list = ['Input', 'Prediction', 'Label']
            plt.imshow(data, cmap="Reds", interpolation='nearest')
            plt.tight_layout()
            plt.xticks([])
            plt.yticks([])
            add_scalebar(ax)
            add_north(ax) # 添加指北针
            if Index:
                add_annotation(ax, Index) # 添加指标
            # add_text(ax, f"UID:{UID[dataid]} Year:{date[dataid].year}", 'normal', 24, 0.26, 0.05)

    plt.subplots_adjust(wspace=0.05, hspace=0.12)  # Reduce space between subplots
    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=color, label=label)
                      for color, label in zip(colors, labels)]
    # 在figure底部添加全局图例
    fig.legend(handles=legend_handles,
               loc='lower center',
               ncol=2,  # 分4列显示
               handlelength=4,  # 增加长度（默认1.0）
               handleheight=2,  # 增加高度（默认0.7）
               fontsize=26,
               bbox_to_anchor=(0.5, 0.02),  # 调整位置
               frameon=True)
    plt.subplots_adjust(bottom=0.2)  # 增大底部边距，以显示图例
    plt.show()
    plt.savefig("Comparison_Fig.png") # 这个保存出的图有点问题...

if __name__ == '__main__':
    # 加载配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_name', type=str, default=[], nargs='+', help='...')
    parser.add_argument('--model_name', type=str, default=[], nargs='+',
                        help='Model architecture details (e.g., rcda, unet, etc.).')
    parser.add_argument('--UID_list', type=str, default=[], nargs='+', help='List of...')
    parser.add_argument('--date_list', type=str, default=[], nargs='+', help='List of...')
    parser.add_argument('--threshold_list', type=str, default=[], nargs='+', help='List of...')

    args = parser.parse_args()
    fire_train_data = Fire('train', augmentation=False)
    fire_eval_data = Fire('eval', augmentation=False)
    train_dataloader = DataLoader(fire_train_data, batch_size=1)
    eval_dataloader = DataLoader(fire_eval_data, shuffle=False, batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    assert len(args.weight_name) == len(args.model_name), 'The length of weight_name and model_name must match'
    for weight_name, model_name in zip(args.weight_name, args.model_name):
        weight_path = f"weights/{weight_name}.pth"
        # 加载模型
        model = Init_net(model_name)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        models.append(model)
        print("Weight loaded from: ", weight_name)
        # 总体分析
        # print(evaluate(model, eval_dataloader, device))
    # 案例分析
    UID = [int(uid) for uid in args.UID_list]
    threshold_list = [float(threshold) for threshold in args.threshold_list]
    date = []
    for firedate in args.date_list:
        year, month, day = firedate.split('-')
        date.append(datetime(int(year), int(month), int(day)))
    if len(UID) == len(date) and len(models) == len(threshold_list) and len(UID) > 0 and len(threshold_list) > 0:
        Cases_Analysis(args.model_name, models, threshold_list, UID, date)