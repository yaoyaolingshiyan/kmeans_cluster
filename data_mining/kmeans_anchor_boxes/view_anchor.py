import numpy as np
import cv2
import random


def color_generator():
    res = []
    for i in range(50):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        res.append([b, g, r])
    return res


def draw_anchors(img, anchors):
    center = [400, 400]
    colors = color_generator()
    for i in range(len(anchors)):
        width_height = [anchors[i][0] * 800, anchors[i][1] * 800]
        # 输入中心点， 长宽，角度
        rect = (center, width_height, 0)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        cv2.drawContours(img, [rect], -1, colors[i], 2)
    return img


if __name__ == "__main__":
    train_img = np.zeros((800, 800, 3), np.uint8)
    val_img = np.zeros((800, 800, 3), np.uint8)
    # fill the image with white
    train_img.fill(255)
    val_img.fill(255)
    # anchors = [[0.04375, 0.035],
    #            [0.015, 0.015],
    #            [0.3775, 0.46],
    #            [0.00875, 0.01],
    #            [0.17625, 0.13125],
    #            [0.09875, 0.1425],
    #            [0.0325, 0.0575],
    #            [0.12875, 0.2325],
    #            [0.05625, 0.1125],
    #            [0.13625, 0.0375],
    #            [0.02125, 0.02875],
    #            [0.4275, 0.14625],
    #            [0.1125, 0.09875],
    #            [0.7725, 0.62875],
    #            [0.23125, 0.2575],
    #            [0.06125, 0.055],
    #            [0.08125, 0.075],
    #            [0.03, 0.01875]]

    train_anchors = [[0.015000000000000013, 0.016249999999999987],
               [0.25125, 0.2675],
               [0.13249999999999995, 0.24625000000000002],
               [0.08125000000000004, 0.0825],
               [0.058750000000000024, 0.125],
               [0.022500000000000006, 0.020000000000000004],
               [0.026249999999999996, 0.027500000000000024],
               [0.03375000000000006, 0.0325],
               [0.11875000000000002, 0.07625000000000001],
               [0.0625, 0.03749999999999998],
               [0.11625000000000002, 0.125],
               [0.049999999999999996, 0.020000000000000018],
               [0.06000000000000005, 0.060000000000000005],
               [0.195, 0.15125],
               [0.17124999999999996, 0.04375],
               [0.011249999999999982, 0.009999999999999995],
               [0.6025, 0.5137499999999999],
               [0.03125, 0.08250000000000002],
               [0.022500000000000075, 0.04625000000000001],
               [0.04249999999999998, 0.0475]]
    val_anchors = [[0.010000000000000009, 0.01],
                   [0.32875, 0.55875],
                   [0.018750000000000044, 0.043749999999999956],
                   [0.13624999999999993, 0.03750000000000009],
                   [0.11249999999999993, 0.09500000000000008],
                   [0.10750000000000004, 0.49625],
                   [0.12875, 0.22625],
                   [0.06125000000000001, 0.05499999999999999],
                   [0.04500000000000004, 0.03499999999999992],
                   [0.10999999999999999, 0.1375],
                   [0.17625000000000002, 0.15375],
                   [0.515, 0.3425],
                   [0.75875, 0.6637500000000001],
                   [0.37124999999999997, 0.12875000000000003],
                   [0.05625000000000002, 0.115],
                   [0.015000000000000013, 0.0175],
                   [0.23375, 0.26375000000000004],
                   [0.026250000000000002, 0.02375],
                   [0.07250000000000001, 0.075],
                   [0.03375000000000006, 0.05750000000000005]]

    train_img = draw_anchors(train_img, train_anchors)
    val_img = draw_anchors(val_img, val_anchors)
    cv2.imshow('train_image', train_img)
    cv2.imshow('val_image', val_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
