import numpy as np
import os
from tqdm import tqdm
import sys
import matplotlib

matplotlib.use("Agg")
sys.path.append("../../../")
from data.data_mining.kmeans_anchor_boxes.kmeans import kmeans, avg_iou
import libs.configs.cfgs as cfgs
from help_utils.tools import mkdir
import matplotlib.colors as colors
import matplotlib.pyplot as plt


def load_dataset(gt_dir):
    dataset = []
    file_list = os.listdir(gt_dir)
    height = 800
    width = 800
    for file in tqdm(file_list, desc="加载data"):
        with open(os.path.join(gt_dir, file), encoding="utf-8") as gt_f:
            lines = gt_f.readlines()
            if len(lines) == 0:  # 跳过没有目标的图片
                continue
            lines = [line.strip() for line in lines]
            for line in lines:
                line_split = line.split(" ")
                if len(line_split) < 10:
                    continue
                origin = [int(float(split)) for split in line_split[:8]]
                xmin = min(origin[0::2]) / width
                xmax = max(origin[0::2]) / width
                ymin = min(origin[1::2]) / height
                ymax = max(origin[1::2]) / height
                if xmax - xmin <= 0 or ymax - ymin <= 0:
                    continue
                dataset.append([xmax - xmin, ymax - ymin])
    return np.array(dataset)


def kmeans_1_to_20(set_gt_path, set_name):
    data = load_dataset(set_gt_path)
    cluster_num_list = []
    accuracy_list = []
    out_list = []
    ratios_list = []
    for cluster_num in range(1, 21, 1):
        out = kmeans(data, k=cluster_num)
        accuracy = avg_iou(data, out) * 100
        ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()

        cluster_num_list.append(cluster_num)
        accuracy_list.append(accuracy)
        out_list.append(out)
        ratios_list.append(ratios)
        print("--" * 10 + 'set name: ' + set_name + ',cluster num : ' + str(cluster_num) + "--" * 10)
        print("Accuracy: {:.2f}%".format(accuracy))
        print("Boxes:\n {}".format(out))
        print("Ratios:\n {}".format(sorted(ratios)))

    return cluster_num_list, accuracy_list, out_list, ratios_list


if __name__ == "__main__":
    split_train_label = cfgs.split_train_label
    split_val_label = cfgs.split_val_label
    name_to_paths = {
        "train": cfgs.split_train_label,
        "val": cfgs.split_val_label,
    }
    index = 0
    save_path = os.path.join(cfgs.DATA_MINING_PATH, cfgs.VERSION)
    mkdir(save_path)
    for set_name, set_gt_path in name_to_paths.items():
        print("--" * 10 + set_name + "--" * 10)
        cluster_num_list, accuracy_list, out_list, ratios_list = kmeans_1_to_20(set_gt_path, set_name)

        c = colors.cnames.keys()
        c_dark = list(filter(lambda x: x.startswith('dark'), c))
        plt.plot(cluster_num_list, accuracy_list, color=c_dark[index], label=set_name)
        with open(os.path.join(save_path, set_name + "_kmeans_box_result.txt"), "w") as f:
            f.write("%s\t\t\t%s\n" % ("k", "box_list([x,y])"))
            for i in range(0, len(out_list)):
                out = out_list[i]
                k = i + 1
                f.write("%s\t\t\t%s\n" % (str(k), str(out.tolist())))
        index += 1
    plt.legend(loc='upper left')  # 显示图例
    plt.xlabel('cluster_num')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join(save_path, 'kmeans_result.png'))
