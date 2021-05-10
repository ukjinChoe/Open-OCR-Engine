import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pathlib import Path
from torch.utils.data import Dataset
from utils.data_manipulation import resize, normalize_mean_variance, generate_affinity, generate_target

class DatasetSYNTH(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataPath = Path(cfg.SynthDataPath)
        self.basePath = self.dataPath.parent

        with self.dataPath.open('rb') as f:
            dsets = pickle.load(f)

        self.imnames, self.charBB, self.txt = [], [], []
        for d in tqdm(dsets, total=len(dsets), desc="loading dataset"):
            self.imnames.append(d['fn'])
            self.charBB.append(d['charBB'])
            self.txt.append(d['txt'])

    def __len__(self):
        return len(self.imnames)

    def __getitem__(self, item):
        item = item % len(self.imnames)
        image = plt.imread(self.basePath / self.imnames[item], 'PNG')  # Read the image

        if len(image.shape) == 2:
            image = np.repeat(image[:, :, None], repeats=3, axis=2)
        elif image.shape[2] == 1:
            image = np.repeat(image, repeats=3, axis=2)
        else:
            image = image[:, :, 0: 3]

        image, character = resize(image, self.charBB[item].copy())  # Resize the image to (768, 768)
        
        normal_image = image.astype(np.uint8).copy()
        image = normalize_mean_variance(image).transpose(2, 0, 1)

        # Generate character heatmap
        weight_character = generate_target(image.shape, character.copy())

        # Generate affinity heatmap
        weight_affinity, affinity_bbox = generate_affinity(image.shape, character.copy(), self.txt[item])

        cv2.drawContours(
            normal_image,
            np.array(affinity_bbox).reshape([len(affinity_bbox), 4, 1, 2]).astype(np.int64), -1, (0, 255, 0), 2)

        enlarged_affinity_bbox = []

        for i in affinity_bbox:
            center = np.mean(i, axis=0)
            i = i - center[None, :]
            i = i*60/25
            i = i + center[None, :]
            enlarged_affinity_bbox.append(i)

        cv2.drawContours(
            normal_image,
            np.array(enlarged_affinity_bbox).reshape([len(affinity_bbox), 4, 1, 2]).astype(np.int64),
            -1, (0, 0, 255), 2
        )

        return     image.astype(np.float32), \
                weight_character.astype(np.float32), \
                weight_affinity.astype(np.float32), \
                normal_image

                