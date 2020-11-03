# Author: Narine Kokhlikyan <narine@slice.com>
# License: BSD

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time
import numpy as np


def plot_tsne(X, label):
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
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=label, cmap="Set1")
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        plt.legend(handles=sc.legend_elements()[0], labels=['0', '1'])

    # plt.show()

    plt.savefig('tsne.png')


n_samples = 300
# X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
# plot_tsne(X, y)

num_samples_from_prior = 10
num_tasks = 10

X = []
color = []
label = []
for i in range(num_tasks):
    for p in range(num_samples_from_prior):
        prior = np.random.rand(1000)
        X.append(prior)
        color.append('C'+str(i))
        label.append(i%2)

plot_tsne(np.array(X), np.array(color), label)
