import matplotlib.pyplot as plt
import numpy as np

def plot_some(imgs, labels=None, count = 4):
    fig = plt.figure()
    for i in range(count):
        ax = plt.subplot(1, count, i+1)
        if labels is not None:
            ax.set_title("Label: %s" % int(labels[i]))
        ax.imshow(np.reshape(imgs[i], (28,28)))
    plt.show()
