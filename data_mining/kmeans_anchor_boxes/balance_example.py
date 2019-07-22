import numpy as np
import os
from tqdm import tqdm
import sys
import matplotlib
import random

matplotlib.use("Agg")
sys.path.append("../../../")
from kmeans_anchor_boxes.kmeans import kmeans, avg_iou
import matplotlib.colors as colors
import matplotlib.pyplot as plt

NAME_LABEL_MAP = {
        'back_ground': 0,
        'large-vehicle': 1,
        'swimming-pool': 2,
        'helicopter': 3,
        'bridge': 4,
        'plane': 5,
        'ship': 6,
        'soccer-ball-field': 7,
        'basketball-court': 8,
        'airport': 9,
        'container-crane': 10,
        'ground-track-field': 11,
        'small-vehicle': 12,
        'harbor': 13,
        'baseball-diamond': 14,
        'tennis-court': 15,
        'roundabout': 16,
        'storage-tank': 17,
        'helipad': 18}


def get_number_balanced_class_dataset(class_dataset, number_per_class):
    res = []
    class_dataset_len = len(class_dataset)
    quotient = int(number_per_class / class_dataset_len)
    remainder = number_per_class % class_dataset_len
    for i in range(quotient):
        res += class_dataset
    if remainder:
        slice = random.sample(class_dataset, remainder)
        res += slice
    # print(len(res))
    return res


def load_dataset(gt_dir, number_per_class=10000):
    """
    和 example不同的地方在于针对类别不均衡，我们先进行类别均衡操作，然后再聚类
    :param gt_dir:
    :param number_per_class:
    :return:
    """
    dataset = []
    file_list = os.listdir(gt_dir)
    height = 800
    width = 800
    label_dataset = {}
    for label in NAME_LABEL_MAP:
        if label == "back_ground":
            continue
        label_dataset[label] = []
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
                label = line_split[8]
                if xmax - xmin <= 0 or ymax - ymin <= 0:
                    continue
                label_dataset[label].append([xmax - xmin, ymax - ymin])
    for label in label_dataset:
        print(label, ': ', len(label_dataset[label]))
        dataset += get_number_balanced_class_dataset(label_dataset[label], number_per_class)
    print(len(dataset))
    return np.array(dataset)


def kmeans_1_to_20(set_gt_path, set_name):
    data = load_dataset(set_gt_path)
    print(data.shape)
    cluster_num_list = []
    accuracy_list = []
    out_list = []
    ratios_list = []
    for cluster_num in range(1, 21, 1):
        print('start:', cluster_num)
        # out 是k次聚类中心点,
        # print(cluster_num)
        out = kmeans(data, k=cluster_num)
        # print(cluster_num, ': ', out)
        accuracy = avg_iou(data, out) * 100
        # 计算纵横比，保留两位小数
        ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()

        cluster_num_list.append(cluster_num)
        accuracy_list.append(accuracy)
        out_list.append(out)
        ratios_list.append(ratios)
        print("--" * 10 + 'set name: ' + set_name + ',cluster num : ' + str(cluster_num) + "--" * 10)
        print("Accuracy: {:.2f}%".format(accuracy))
        print("Boxes:\n {}".format(out))
        print("Ratios:\n {}".format(sorted(ratios)))
        print('*'*30)

    return cluster_num_list, accuracy_list, out_list, ratios_list


if __name__ == "__main__":
    root_path = '/home/liubo/data/space-tech-remote-sensing-ob-detection-dataset_split_size_800_800_gap_100_ratio_1'
    split_train_label = os.path.join(root_path, 'train', 'labelTxt')
    split_val_label = os.path.join(root_path, 'val', 'labelTxt')
    name_to_paths = {
        "train": split_train_label,
        "val": split_val_label,
    }
    index = 0
    save_path = os.path.join('/home/zmy/work_space/data_mining/mining_result', 'R2CNN_2019_competition_v1')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for set_name, set_gt_path in name_to_paths.items():
        print("--" * 10 + set_name + "--" * 10)
        cluster_num_list, accuracy_list, out_list, ratios_list = kmeans_1_to_20(set_gt_path, set_name)

        # 从颜色字典中取出所有颜色值
        c = colors.cnames.keys()
        # 过滤出dark开头的颜色
        c_dark = list(filter(lambda x: x.startswith('dark'), c))
        plt.plot(cluster_num_list, accuracy_list, color=c_dark[index], label=set_name)
        with open(os.path.join(save_path, set_name + "_kmeans_box_result_balanced.txt"), "w") as f:
            f.write("%s\t\t\t%s\n" % ("k", "box_list([x,y])"))
            for i in range(0, len(out_list)):
                out = out_list[i]
                k = i + 1
                f.write("%s\t\t\t%s\n" % (str(k), str(out.tolist())))
        index += 1
    plt.legend(loc='upper left')  # 显示图例
    plt.xlabel('cluster_num')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join(save_path, 'kmeans_result_balanced.png'))
