import numpy as np
import os
from tqdm import tqdm
from shapely.geometry import Polygon


def intersection(g_quad, p_quad):
    # 创建带比较的两个点的多边形
    g_poly = Polygon(g_quad).convex_hull
    p_ploy = Polygon(p_quad).convex_hull
    inter = g_poly.intersection(p_ploy).area
    union = g_poly.area + g_poly.area - inter
    if union == 0:
        return 0
    else:
        return inter / union  # 交并比

def iou(data1, data2):
    # [Xmin, Ymin, Xmax, Ymax]
    data1 = [int(i) for i in data1]
    data2 = [int(j) for j in data2]
    quad1 = [data1[0], data1[1], data1[2], data1[1], data1[2], data1[3], data1[0], data1[3]]
    quad2 = [data2[0], data2[1], data2[2], data2[1], data2[2], data2[3], data2[0], data2[3]]
    quad1 = np.array(quad1).reshape(4, 2)
    quad2 = np.array(quad2).reshape(4, 2)
    poly1 = Polygon(quad1).convex_hull
    poly2 = Polygon(quad2).convex_hull
    inter = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - inter
    if union == 0:
        return 0
    else:
        return inter / union  # 交并比


def calcu_iou(set_gt_path):
    iou_list = []
    data = os.listdir(set_gt_path)
    # print(data)
    for i in tqdm(data, desc='计算iou'):
        dataset = []
        # print(data[i])
        with open(os.path.join(set_gt_path, i), encoding="utf-8") as gt_f:
            lines = gt_f.readlines()
            if len(lines) == 0:  # 跳过没有目标的图片
                # print(data[i], 'without objects! ')
                continue
            lines = [line.strip() for line in lines]
            for line in lines:
                line_split = line.split(" ")
                if len(line_split) < 10:
                    continue
                origin = [int(float(split)) for split in line_split[:8]]
                xmin = min(origin[0::2])
                xmax = max(origin[0::2])
                ymin = min(origin[1::2])
                ymax = max(origin[1::2])
                if xmax - xmin <= 0 or ymax - ymin <= 0:
                    continue
                dataset.append([xmin, ymin, xmax, ymax])
        # print(dataset)
        for m in range(len(dataset)-1):
            for j in range(m+1, len(dataset)):
                iou_list.append(iou(dataset[m], dataset[j]))
        # print(data[i], 'calculated!')
    return iou_list





if __name__ == "__main__":
    split_train_label = '/home/zmy/work_space/labelxt/train/labelTxt/'
    split_val_label = '/home/zmy/work_space/labelxt/val/labelTxt/'

    # split_train_label = 'D:/competition/kmeans_cluster/labelTxt/train/'
    # split_val_label = 'D:/competition/kmeans_cluster/labelTxt/val/'
    name_to_paths = {
        "train": split_train_label,
        "val": split_val_label,
    }
    save_path = '/home/zmy/work_space/data_mining/mining_result/'
    # save_path = 'D:/competition/kmeans_cluster/labelTxt/save_iou/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for set_name, set_gt_path in name_to_paths.items():
        print("--" * 10 + set_name + "--" * 10)
        iou_list = calcu_iou(set_gt_path)
        iou_array = np.asarray(iou_list, dtype=float)
        mean_iou = iou_array.mean()
        max_iou = max(iou_array)
        num_iou = len(iou_list)
        print('\n')
        print(set_name, 'iou 的总数量为：', num_iou)
        print(set_name, 'iou 的平均值为：', mean_iou)
        print(set_name, 'iou 的最大值为：', max_iou)
        with open(os.path.join(save_path, set_name + "_iou.txt"), "w") as f:
            f.write('mean:'+str(mean_iou)+':max:'+str(max_iou)+':num:'+str(num_iou))
        with open(os.path.join(save_path, set_name + "_every_iou.txt"), "w") as f2:
            for nu in range(len(iou_list)):
                f2.write(str(iou_list[nu])+'\n')

