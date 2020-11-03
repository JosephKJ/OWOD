import os
import torch
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time
from collections import deque
import seaborn as sns

import numpy as np

from detectron2.utils.store import Store


def plot_tsne(X, label, total_num_classes):
    n_components = 2
    (fig, subplots) = plt.subplots(1, 5, figsize=(15, 8))

    perplexities = [5, 30, 50, 100, 150]

    for i, perplexity in enumerate(perplexities):
        ax = subplots[i]

        t0 = time()
        tsne = manifold.TSNE(n_components=n_components, init='random',
                             random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(X)
        t1 = time()
        print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
        ax.set_title("Perplexity=%d" % perplexity)
        #
        # sc = ax.scatter(Y[:, 0], Y[:, 1], c=label, cmap="plasma")
        sns.scatterplot(x=Y[:, 0], y=Y[:, 1], hue=label, ax=ax, legend='full', palette='colorblind')
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        # plt.legend(handles=sc.legend_elements()[0], labels=range(total_num_classes))

        # plt.legend(handles=sc.legend_elements()[0], labels=['0', '1'])

    # plt.show()

    plt.savefig('tsne.png')

maxlen_queue = 100
total_num_classes = 22

queues = [deque(maxlen=maxlen_queue) for _ in range(total_num_classes)]

source_dir = '/home/fk1/workspace/OWOD/output/features'

files = os.listdir(source_dir)
for i, file in enumerate(files):
    path = os.path.join(source_dir, file)
    features, classes = torch.load(path)
    for f, c in zip(features, classes):
        queues[c.detach().cpu().numpy()].append(f.detach().cpu().numpy())

    # if i == 2:
    #     break
x = []
y = []
for i, queue in enumerate(queues):
    for item in queue:
        x.append(item)
        y.append(i)

print('Going to plot')
plot_tsne(x, y, total_num_classes)
