import os
import numpy as np
import cv2
import sys

sys.path.append("../../")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import libs.configs.cfgs as cfgs
from help_utils.tools import mkdir
from libs.label_name_dict.label_dict import NAME_LABEL_MAP
from tqdm import tqdm


def object_num_per_image(gt_dir, set_name):
    label_to_number_list = {}
    for label_name in NAME_LABEL_MAP:
        label_to_number_list[label_name] = []
    label_to_number_list["total"] = []
    file_list = os.listdir(gt_dir)
    for file in tqdm(file_list, desc="加载data"):
        with open(os.path.join(gt_dir, file), encoding="utf-8") as gt_f:
            lines = gt_f.readlines()
            if len(lines) == 0:  # 跳过没有目标的图片
                continue
            lines = [line.strip() for line in lines]
            label_to_number = {}
            for label_name in NAME_LABEL_MAP:
                label_to_number[label_name] = 0
            label_to_number["total"] = 0
            for line in lines:
                line_split = line.split(" ")
                if len(line_split) < 10:
                    continue
                label_name = line_split[8]
                assert label_name in NAME_LABEL_MAP, "label %s in  file:%s illegal"
                label_to_number[label_name] += 1
                label_to_number["total"] += 1

            for label_name, number in label_to_number.items():
                label_to_number_list[label_name].append(number)

    label_list = [label_name for label_name in NAME_LABEL_MAP]
    label_list.append("total")  # total 放在最后方便观看
    for label_name in label_list:
        label_to_number_list[label_name] = np.array(label_to_number_list[label_name])
    x = np.arange(len(label_list))
    y_max = [np.max(label_to_number_list[label_name]) for label_name in label_list]
    y_mean = [np.mean(label_to_number_list[label_name]) for label_name in label_list]
    y_sum = [np.sum(label_to_number_list[label_name]) for label_name in label_list]
    bar_width = 0.3
    plt.bar(x, y_mean, bar_width, color='salmon', label='mean')
    # plt.bar(x + bar_width, y_max, bar_width, color='orchid', label='max')
    plt.legend()  # 显示图例
    plt.xticks(x + bar_width / 2, label_list, rotation=45)  # 显示x坐标轴的标签
    save_path = os.path.join(cfgs.DATA_MINING_PATH, cfgs.VERSION)
    mkdir(save_path)
    plt.savefig(os.path.join(save_path, set_name + "_object_number_per_image.png"), dpi=200)
    plt.close()
    with open(os.path.join(save_path, set_name + "_object_number_per_image.txt"), "w") as f:
        f.write("%s\t\t\t\t%s\t%s\t%s\n" % ("label", "max", "mean", 'sum'))
        for i in range(len(label_list)):
            f.write("%s\t\t\t\t%d\t%d\t%d\n" % (label_list[i], y_max[i], y_mean[i], y_sum[i]))


if __name__ == "__main__":
    split_train_label = cfgs.split_train_label
    split_val_label = cfgs.split_val_label
    name_to_paths = {
        "train": cfgs.split_train_label,
        "val": cfgs.split_val_label,
    }
    index = 0
    for set_name, set_gt_path in name_to_paths.items():
        object_num_per_image(set_gt_path, set_name)
