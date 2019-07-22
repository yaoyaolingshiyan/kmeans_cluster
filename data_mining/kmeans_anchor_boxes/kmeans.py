import numpy as np
import random

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    # 返回维度为(18000, k),元素随机生成的矩阵
    distances = np.empty((rows, k))
    # 长度为18000的一维数组
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows （k, 2）
    clusters = boxes[np.random.choice(rows, k, replace=False)]  # 选出K个中心
    # print('clusters:', clusters)
    # print('type(boxes):', type(boxes))
    # print(clusters)
    # print(type(clusters))
    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
            # print(distances.shape)
        nearest_clusters = np.argmin(distances, axis=1)
        print('*****again*****')
        # print('nearcluster: ', nearest_clusters)

        if (last_clusters == nearest_clusters).all():
            break

        num_list = []
        clusters_list = []
        for cluster in range(k):
            # clusters[cluster] = boxes[nearest_clusters == cluster]
            clusters_list.append(boxes[nearest_clusters == cluster])
            num_list.append(len(clusters[cluster]))
        for j in range(k):
            if num_list[j] == 0:
                cu_older = num_list.index(max(num_list))
                clusters[j] = random.sample(clusters_list[cu_older], 1)
                while clusters[j] == dist(clusters_list[cu_older], axis=0):
                    clusters[j] = random.sample(clusters_list[cu_older], 1)
            else:
                clusters[j] = dist(clusters_list[j], axis=0)

        last_clusters = nearest_clusters

    return clusters
