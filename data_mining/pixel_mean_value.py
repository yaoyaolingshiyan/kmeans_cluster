# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import sys

sys.path.append("../../")
import libs.configs.cfgs as cfgs
from help_utils.tools import mkdir
from tqdm import tqdm


def count_mean_pixel(set_image_path, set_name):
    count = 0
    all_mean = np.zeros(3, np.float64)
    set_files = os.listdir(set_image_path)
    for file in tqdm(set_files, desc="Processing:%s" % set_name):
        img = cv2.imread(os.path.join(set_image_path, file))
        count += 1
        mean = np.mean(img, axis=(0, 1)).astype(np.float64)  # 适用于每张图片的大小都相等的情况
        all_mean += mean
    return all_mean / count


if __name__ == "__main__":
    split_train_image = cfgs.split_train_image
    split_val_image = cfgs.split_val_image
    name_to_paths = {
        "train": split_train_image,
        "val": split_val_image,
    }

    name_to_mean = dict()
    for set_name, set_image_path in name_to_paths.items():
        mean = count_mean_pixel(set_image_path, set_name)
        name_to_mean[set_name] = mean.tolist()
    save_path = os.path.join(cfgs.DATA_MINING_PATH, cfgs.VERSION)
    mkdir(save_path)
    with open("mining_result/pixel_mean.txt", "w") as f:
        f.write("set_name: [Blue,Green,Red]\n")
        for set_name, set_mean in name_to_mean.items():
            f.write(set_name + " : " + str(set_mean) + "\n")
    print("statistics completed(Blue,green,Red):")
    print("--" * 20)
    print(name_to_mean)
