import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt

if __name__ == "__main__":

    name_to_paths = {
        "train": "1",
        "val": "123",
        # TODO
    }
    index = 0
    save_path = "./"

    for set_name, set_gt_path in name_to_paths.items():
        print("--" * 10 + set_name + "--" * 10)
        cluster_num_list, accuracy_list = [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]

        c = colors.cnames.keys()
        c_dark = list(filter(lambda x: x.startswith('dark'), c))
        plt.plot(cluster_num_list, accuracy_list, color=c_dark[index], label=set_name)
        index += 1
    plt.legend(loc='upper left')  # 显示图例
    plt.xlabel('cluster_num')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join(save_path, 'kmeans_result.png'))
