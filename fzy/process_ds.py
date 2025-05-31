import json
import random
random.seed(42)
import jsonlines
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import torchvision.transforms as transforms
import cv2
from pathlib import Path
from fzy.ds import *
OUTPUT_DIR = DATASET_BASE / 'processed'

def process_ds():
    # ActivityNet, Breakfast, Charades, QVHighlights, VALOR32K, YouCook2
    # 实例化
    ds_list = [
        ActivityNetCaps(),
        Charades(),
        QVHighlights(),
        # YouCook2()
    ]
    for ds in ds_list:
        ds.load_train()
    



if __name__ == "__main__":
    process_ds()