import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    npimg = denormalize(img)
    plt.imsave("matplotlib.png", npimg, cmap="gray")
    # plt.imshow(npimg, cmap="gray") # Not working on my WSL2 on Windows 11


def denormalize(img):
    channel_means = (0.4914, 0.4822, 0.4465)
    channel_stdevs = (0.2470, 0.2435, 0.2616)
    img = img.numpy().astype(dtype=np.float32)

    for i in range(img.shape[0]):
        img[i] = (img[i] * channel_stdevs[i]) + channel_means[i]

    return np.transpose(img, (1, 2, 0))
