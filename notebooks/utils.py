import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import skimage.measure
import matplotlib.pyplot as plt
import matplotlib


def preprocessing(mask_size=8):
    """
    Perform image preprocessing on a directory of images.

    Args:
        mask_size (int, optional): Size of the mask for pooling. Defaults to 8.

    Returns:
        pd.DataFrame: A DataFrame containing preprocessed images and labels.
    """
    # Construct the path to the data directory based on your project structure
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(base_path, "data", "input", "JF30_subset")

    files = np.array(
        [
            f
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and f != ".DS_Store"
        ]
    )
    files = sorted(files, reverse=True)

    # File name to labels
    file_data = pd.read_csv(
        path[: len(path) - 11] + "/classlabels.txt", sep=",", header=None
    )
    file_data.columns = ["file_name", "label"]

    file_data_dict = {}
    for index, row in file_data.iterrows():
        file_data_dict[row["file_name"]] = row["label"]

    flowers_list = list()
    sizes = list()
    for file in tqdm(files[:]):
        image = Image.open(path + "/" + file)
        data = np.asarray(image)
        # Trimming down to the first smaller integer mod 0 with the mask size
        w = data.shape[0] - (data.shape[0] % mask_size)
        h = data.shape[1] - (data.shape[1] % mask_size)
        data_trimmed = np.stack(
            [data[:w, :h, 0], data[:w, :h, 1], data[:w, :h, 2]], axis=2
        )
        data_pooled = skimage.measure.block_reduce(
            data_trimmed, (mask_size, mask_size, 1), np.mean
        )
        file_name = file.split("-")[0]
        label = file_data_dict[file_name]
        flowers_list.append([data_pooled, label])

    flowers_df = pd.DataFrame(flowers_list, columns=["image", "label"])

    return flowers_df


def visualize_dataset(flowers_dataframe):
    """
    Visualize a dataset of images and their labels.

    Args:
        flowers_dataframe (pd.DataFrame): DataFrame containing images and labels.

    Returns:
        None
    """
    fig, ax_all = plt.subplots(1, 1, figsize=(12, 6))

    gs = matplotlib.gridspec.GridSpec(
        2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1], height_ratios=[1, 1]
    )

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    ax4 = plt.subplot(gs[0, 3])
    ax5 = plt.subplot(gs[0, 4])
    ax6 = plt.subplot(gs[1, 0])
    ax7 = plt.subplot(gs[1, 1])
    ax8 = plt.subplot(gs[1, 2])
    ax9 = plt.subplot(gs[1, 3])
    ax10 = plt.subplot(gs[1, 4])

    ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]

    for i, pair in enumerate(
        list(zip(flowers_dataframe["image"], flowers_dataframe["label"]))
    ):
        img, label = pair
        ax[i].imshow(np.array(img, dtype=int))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title("label = " + str(label), fontsize=14)

    plt.show()


def visualize_resuts(img1, img_list, lab1, lab_list, j_list):
    """
    Visualize results and images for a given set of data.

    Args:
        img1: The test sample image.
        img_list (list): List of images.
        lab1: Label of the test sample.
        lab_list (list): List of labels.
        j_list (list): List of J values.

    Returns:
        None
    """
    fig, ax_all = plt.subplots(1, 1, figsize=(16, 6))

    gs = matplotlib.gridspec.GridSpec(
        2,
        11,
        figure=fig,
        width_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        height_ratios=[1.6, 0.4],
        hspace=0.1,
        wspace=0,
    )

    ax1 = plt.subplot(gs[0, 2:])

    plt.xticks([])
    plt.yticks(fontsize=12)
    plt.ylabel(r"$J_\Gamma^\star$", fontsize=14)

    ax_test = plt.subplot(gs[0, 0])
    ax_test.set_title("Test sample")
    ax2 = plt.subplot(gs[1, 2])
    ax3 = plt.subplot(gs[1, 3])
    ax4 = plt.subplot(gs[1, 4])
    ax5 = plt.subplot(gs[1, 5])
    ax6 = plt.subplot(gs[1, 6])
    ax7 = plt.subplot(gs[1, 7])
    ax8 = plt.subplot(gs[1, 8])
    ax9 = plt.subplot(gs[1, 9])
    ax10 = plt.subplot(gs[1, 10])

    ax1.scatter(np.arange(len(j_list)), j_list, s=100, marker="d")
    ax_test.imshow(np.array(img1, dtype=int))
    ax2.imshow(np.array(img_list[0], dtype=int))
    ax3.imshow(np.array(img_list[1], dtype=int))
    ax4.imshow(np.array(img_list[2], dtype=int))
    ax5.imshow(np.array(img_list[3], dtype=int))
    ax6.imshow(np.array(img_list[4], dtype=int))
    ax7.imshow(np.array(img_list[5], dtype=int))
    ax8.imshow(np.array(img_list[6], dtype=int))
    ax9.imshow(np.array(img_list[7], dtype=int))
    ax10.imshow(np.array(img_list[8], dtype=int))

    ax_list_all = [ax_test, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
    for i, lab in enumerate([lab1] + lab_list):
        ax_list_all[i].set_xticks([])
        ax_list_all[i].set_yticks([])
        ax_list_all[i].set_xlabel("label = " + str(lab), fontsize=11)
