import torch
import argparse
import numpy as np
from tqdm import tqdm
from dataset import Fire
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from train import calc_confusion_matrix, calc_precision, calc_recall, calc_f1_score, Init_net

def calc_IoU(TP, TN, FP, FN):
    """Calculate Intersection over Union (IoU): TP / (TP + FP + FN) with zero division handling."""
    return TP / (TP + FP + FN) if TP + FP + FN > 0 else 0

def evaluate(model, dataloader, device, thresholds=[0.2, 0.3, 0.4, 0.5, 0.6], use_increment=True, ignore=None):
    """
    Evaluate model performance across multiple thresholds.
    Args:
        model: PyTorch model instance.
        dataloader: DataLoader for evaluation dataset.
        device: Device to run the evaluation (CPU or GPU).
        thresholds: List of threshold values to test.
        use_increment: Boolean to determine if incremental prediction is used.
        ignore: List of channels to zero out (if any).
    Returns:
        Tuple of best precision, recall, F1-score, IoU, and the corresponding threshold.
    """
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
    """Add text annotation to the plot at a specified relative position."""
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
    """Add a north arrow ('N') annotation to the plot."""
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx
    left = [minx + xlen * (loc_x - width * 0.5), miny + ylen * (loc_y - pad)]
    right = [minx + xlen * (loc_x + width * 0.5), miny + ylen * (loc_y - pad)]
    top = [minx + xlen * loc_x, miny + ylen * (loc_y - pad + height)]
    center = [minx + xlen * loc_x, left[1] + (top[1] - left[1]) * 0.4]
    triangle = mpatches.Polygon([left, top, right, center], color='k')
    add_text(ax, "N", 'normal', labelsize, loc_x, loc_y, width, height, pad)
    ax.add_patch(triangle)

def add_annotation(ax, Index, labelsize=24, loc_x=0.78, loc_y=0.92, width=0.04, height=0.12, pad=0.14):
    """Add performance metrics (precision, recall, F1, IoU) as annotations."""
    IndexTitle = [f"precision: {Index[0]*100:.2f}%", f"recall: {Index[1]*100:.2f}%", f"f1: {Index[2]:.3f}", f"IoU: {Index[3]:.3f}"]
    for i in range(4):
        add_text(ax, IndexTitle[i], 'bold', labelsize, loc_x, loc_y, width, height, (1 + 0.5 * i) * pad)

def add_scalebar(ax):
    """Add a scale bar to the plot representing a 3 km distance."""
    fontprops = fm.FontProperties(size=24)
    scalebar = AnchoredSizeBar(ax.transData,
                               size=60,  # Represents 60 units (e.g., pixels) on x-axis
                               label='3 km',  # Actual distance represented
                               loc='lower right',
                               pad=0.5,
                               color='black',
                               frameon=False,
                               size_vertical=1,
                               fontproperties=fontprops,
                               bbox_to_anchor=(1.0, 0.05))  # Adjusted y-position
    ax.add_artist(scalebar)

def cases_analysis(model_names, models, threshold_list, UID_list, date_list, datatype='eval'):
    """
    Perform case-by-case analysis and visualization of model predictions.
    Args:
        model_names: List of model names for labeling.
        models: List of model instances (including None for ground truth).
        threshold_list: List of thresholds for each model.
        UID_list: List of unique identifiers for cases.
        date_list: List of datetime objects for cases.
        datatype: Type of dataset ('train' or 'eval')
    """
    labels = ['Fire Mask at Time T', 'Incremental Fire Mask at Time T+1']
    colors = ['#67000C', '#FA6949']
    fire_data = Fire(datatype, UID=UID_list, start=date_list, end=date_list)
    dataloader = DataLoader(fire_data, shuffle=False, batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models.append(None)  # Add None for ground truth label plotting

    plt.rcParams['font.family'] = 'Times New Roman, SimSun'  # Set font family
    fig = plt.figure(figsize=(24, 14))
    for data_id, (input, label) in enumerate(dataloader):
        input = input.to(device)
        label = label.to(device)
        for model_id, model in enumerate(models):
            fig_id = data_id * len(models) + model_id + 1
            ax = plt.subplot(len(dataloader), len(models), fig_id)
            data = input.squeeze()[0].cpu().detach().numpy()
            Index = None
            if model:
                model.eval()
                output = model(input)
                TP, TN, FP, FN = calc_confusion_matrix(output, label, threshold=threshold_list[model_id])
                precision = calc_precision(TP, TN, FP, FN)
                recall = calc_recall(TP, TN, FP, FN)
                f1 = calc_f1_score(precision, recall)
                IoU = calc_IoU(TP, TN, FP, FN)
                Index = [precision, recall, f1, IoU]
                pred = (output > threshold_list[model_id]).float().cpu().squeeze().detach().numpy()
                data[pred == 1] = 0.5  # Overlay prediction
                plt.title(f"({chr(fig_id + 96)}) {model_names[model_id]} Prediction on case-{UID_list[data_id]}",
                          y=-0.1, fontdict={'size': 26})
            else:
                label = label.cpu().squeeze().detach().numpy()
                data[label == 1] = 0.5  # Overlay ground truth
                plt.title(f"({chr(fig_id + 96)}) Label on case-{UID_list[data_id]}",
                          y=-0.1, fontdict={'size': 26})
            plt.imshow(data, cmap="Reds", interpolation='nearest')
            plt.tight_layout()
            plt.xticks([])
            plt.yticks([])
            add_scalebar(ax)
            add_north(ax)  # Add north arrow
            if Index:
                add_annotation(ax, Index)  # Add metrics

    plt.subplots_adjust(wspace=0.05, hspace=0.12)  # Minimize subplot spacing
    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=color, label=label)
                      for color, label in zip(colors, labels)]
    fig.legend(handles=legend_handles,
               loc='lower center',
               ncol=2,
               handlelength=4,
               handleheight=2,
               fontsize=26,
               bbox_to_anchor=(0.5, 0.02),
               frameon=True)
    plt.subplots_adjust(bottom=0.2)  # Increase bottom margin for legend

    # Save figure with high resolution
    output_path = "Comparison_Fig.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Figure saved to: {output_path}")
    plt.show()

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate and visualize model performance on wildfire cases.")
    parser.add_argument('--weight_name', type=str, default=[], nargs='+', help='List of weight file names.')
    parser.add_argument('--model_name', type=str, default=[], nargs='+',
                        help='List of model architecture names (e.g., rcda, wpn, funetcast).')
    parser.add_argument('--UID_list', type=str, default=[], nargs='+', help='List of unique case IDs.')
    parser.add_argument('--date_list', type=str, default=[], nargs='+', help='List of fire occurrence dates.')
    parser.add_argument('--data_type', type=str, default='eval', help='Dataset types used for case analysis. Choices are "train", "eval".')
    parser.add_argument('--threshold_list', type=str, default=[], nargs='+', help='List of initial prediction thresholds (as strings, e.g., 0.2 0.5 0.6) to evaluate. '
                         'If provided, these values are used as starting thresholds. The list will be dynamically '
                         'updated with the best threshold identified during evaluation. If not specified, '
                         'the best threshold will be determined automatically and used.')

    args = parser.parse_args()
    fire_train_data = Fire('train', augmentation=False)
    fire_eval_data = Fire('eval', augmentation=False)
    train_dataloader = DataLoader(fire_train_data, batch_size=1)
    eval_dataloader = DataLoader(fire_eval_data, shuffle=False, batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    threshold_list = []
    assert len(args.weight_name) == len(args.model_name), 'The length of weight_name must match model_name'
    assert args.data_type == 'train' or args.data_type == 'eval', 'The data type must be "train" or "eval". Got {}'.format(args.data_type)
    for weight_name, model_name in zip(args.weight_name, args.model_name):
        weight_path = f"weights/{weight_name}.pth"
        # Load model
        try:
            model = Init_net(model_name)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.to(device)
            models.append(model)
            print(f"Weight loaded from: {weight_name}")
            # Overall analysis
            precision, recall, f1_score, iou, best_threshold = evaluate(model, eval_dataloader, device)
            threshold_list.append(best_threshold)
            print(f"Overall Model Performance - Best Threshold: {best_threshold:.3f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1_score:.4f}")
            print(f"  IoU: {iou:.4f}")
        except Exception as e:
            print(f"Error loading model {weight_name}: {e}")
            continue

    # Case analysis
    UID_list = [int(uid) for uid in args.UID_list]
    if args.threshold_list:
        threshold_list = [float(threshold) for threshold in args.threshold_list]
    date_list = []
    for fire_date in args.date_list:
        year, month, day = fire_date.split('-')
        date_list.append(datetime(int(year), int(month), int(day)))

    if len(UID_list) == len(date_list) and len(models) == len(threshold_list) and len(UID_list) > 0 and len(threshold_list) > 0:
        cases_analysis(args.model_name, models, threshold_list, UID_list, date_list, datatype=args.data_type)