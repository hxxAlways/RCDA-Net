import os
import math
import torch
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

FIG_SIZE = 256
PI = math.pi
TRAIN_INPUTS = 'dataset/train/inputs'
TRAIN_LABELS = 'dataset/train/labels'
TEST_INPUTS = 'dataset/test/inputs'
TEST_LABELS = 'dataset/test/labels'
# TRAIN_INPUTS = 'C:/work/FirePredict/Code/OutputData/dataset/train/inputs'
# TRAIN_LABELS = 'C:/work/FirePredict/Code/OutputData/dataset/train/labels'
# TEST_INPUTS = 'C:/work/FirePredict/Code/OutputData/dataset/test/inputs'
# TEST_LABELS = 'C:/work/FirePredict/Code/OutputData/dataset/test/labels'

band_info = [
    ("Fire Mask (t)", (0.0, 1.0)),
    ("DEM", (1.0, 3413.0)),
    ("B Reflectance", (0.0, 1.0)),
    ("G Reflectance", (0.0, 1.0)),
    ("R Reflectance", (0.0, 1.0)),
    ("NDVI", (-1.0, 1.0)),
    ("Wind Speed", (0.0163726806640625, 13.046875)),
    ("Wind Direction", (-PI, PI)),
    ("Temperature", (270.5065612792969, 300.23919677734375)),
    ("Precipitation", (0.0, 0.0012839797418564558)),
    ("Humidity", (0.002109996974468231, 0.015439476817846298)),
    ("Air Density", (0.9423993229866028, 1.270668625831604))
]


class Fire(Dataset):
    """
    datatype: train dataset or test dataset
    use_increment: Whether to enable incremental masking (default enabled)
    augmentation: Whether to enable data augmentation (not enabled by default)
    UID:list[int]: Designated forest fire event UID list (default is empty, recommended not to specify this value during training)
    start/end:list[datetime]: The specified start and end time of the forest fire event (default is empty, recommended not to specify this value during training)
    """
    def __init__(self, datatype, use_increment=True, augmentation=False, UID:list[int]=None, start:list[datetime]=None, end:list[datetime]=None):
        super().__init__()
        self.DARA_INPUTS = TRAIN_INPUTS if datatype == 'train' else TEST_INPUTS
        self.DARA_LABELS = TRAIN_LABELS if datatype == 'train' else TEST_LABELS
        self.datapath = [f for f in os.listdir(self.DARA_INPUTS) if f.endswith('.npy')]
        self.use_increment = use_increment
        self.augmentation = augmentation
        if UID and start and end:
            assert len(UID) == len(start) == len(end), "UID and start and end must have the same length."
            newdata = []
            for uid, st, ed in zip(UID, start, end):
                for path in self.datapath:
                    idate = datetime.strptime(path[-14:-4], "%Y-%m-%d")
                    if (str(uid) in path) and idate <= ed and idate >= st:
                        newdata.append(path)
            self.datapath = newdata

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, idx):
        input = np.load(os.path.join(self.DARA_INPUTS, self.datapath[idx]))
        label = np.load(os.path.join(self.DARA_LABELS, self.datapath[idx]))
        # Incremental mask
        if self.use_increment:
            label = label - input[0]

        # Data augmentation
        if self.augmentation:
            input, label = self.transform_data(input, label)

        # Mean normalization processing
        for idx, item in enumerate(input):
            band_name, (min_value, max_value) = band_info[idx]
            input[idx] = (item - min_value) / (max_value - min_value)

        input = torch.from_numpy(input.astype(np.float32))
        label = torch.from_numpy(label.reshape((1, FIG_SIZE, FIG_SIZE)).astype(np.float32))
        return input, label

    def transform_data(self, data, label, wd_ch=7):
        # flip horizontal θ→−θ
        if random.random() < 0.5:
            data = np.flip(data, axis=2)
            label = np.flip(label, axis=1)
            data[wd_ch] = -data[wd_ch]  # Horizontal Mirror: Wind Direction Reversing Left and Right

        # flip vertical θ→π−θ
        if random.random() < 0.5:
            data = np.flip(data, axis=1)
            label = np.flip(label, axis=0)
            data[wd_ch] = PI - data[wd_ch]
            data[wd_ch][data[wd_ch] > PI] -= 2 * PI  # Maintain at [- π, π]

        # Random counterclockwise rotation (0, 90, 180, 270) θ → θ − (π/2)
        angle = random.choice([0, 90, 180, 270])
        k = angle // 90
        if k > 0:
            data = np.rot90(data, k=k, axes=(1, 2)).copy()
            label = np.rot90(label, k=k, axes=(0, 1)).copy()
            # Rotating wind direction
            data[wd_ch] -= k * (PI / 2)
            data[wd_ch][data[wd_ch] <= -PI] += 2 * PI
            data[wd_ch][data[wd_ch] > PI] -= 2 * PI

        return data, label


if __name__ == '__main__':
    train = Fire('train', augmentation=True)
    test = Fire('test', augmentation=False)
    print(train.__len__())
    print(test.__len__())
    train_dataloader = DataLoader(train, batch_size=1, shuffle=True, num_workers=0)
    eval_dataloader = DataLoader(test, batch_size=1, shuffle=True, num_workers=0)
    for data in tqdm(train_dataloader):
        inputs, labels = data
        # print(inputs.shape)
        # print(labels.shape, labels.min(), labels.max())
