from math import sqrt, floor, ceil
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from tester import Tester
from trainer import Trainer


def denormalize(img):
    channel_means = (0.4914, 0.4822, 0.4465)
    channel_stdevs = (0.2470, 0.2435, 0.2616)
    img = img.astype(dtype=np.float32)

    for i in range(img.shape[0]):
        img[i] = (img[i] * channel_stdevs[i]) + channel_means[i]

    return np.transpose(img, (1, 2, 0))


def get_rows_cols(num: int) -> Tuple[int, int]:
    cols = floor(sqrt(num))
    rows = ceil(num / cols)

    return rows, cols


def visualize_data(
    loader,
    num_figures: int = 12,
    label: str = "",
    classes: List[str] = [],
):
    batch_data, batch_label = next(iter(loader))

    fig = plt.figure()
    fig.suptitle(label)

    rows, cols = get_rows_cols(num_figures)

    for i in range(num_figures):
        plt.subplot(rows, cols, i + 1)
        plt.tight_layout()
        npimg = denormalize(batch_data[i].cpu().numpy().squeeze())
        label = (
            classes[batch_label[i]] if batch_label[i] < len(classes) else batch_label[i]
        )
        plt.imshow(npimg, cmap="gray")
        plt.title(label)
        plt.xticks([])
        plt.yticks([])


def show_misclassified_images(
    images: List[Tensor],
    predictions: List[int],
    labels: List[int],
    classes: List[str],
    label: str = "",
):
    assert len(images) == len(predictions) == len(labels)

    fig = plt.figure(figsize=(5, 12))
    fig.suptitle(label)

    for i in range(len(images)):
        plt.subplot(len(images) // 2, 2, i + 1)
        plt.tight_layout()
        image = images[i]
        npimg = denormalize(image.cpu().numpy().squeeze())
        predicted = classes[predictions[i]]
        correct = classes[labels[i]]
        plt.imshow(npimg, cmap="gray")
        plt.title("Correct: {}\nPredicted: {}".format(correct, predicted))
        plt.xticks([])
        plt.yticks([])


def plot_data(data_list: List[List[int | float]], titles: List[str]):
    assert len(data_list) == len(
        titles
    ), "length of datalist should be equal to length of title list"

    rows, cols = get_rows_cols(len(data_list))

    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j

            if idx >= len(data_list):
                break

            ax = axs[i, j] if len(axs.shape) > 1 else axs[max(i, j)]

            ax.plot(data_list[idx])  # type: ignore
            ax.set_title(titles[idx])


def collect_results(trainer: Trainer, tester: Tester):
    return {
        "train_losses": trainer.losses,
        "train_accuracies": trainer.accuracies,
        "test_losses": tester.losses,
        "test_accuracies": tester.accuracies,
    }
