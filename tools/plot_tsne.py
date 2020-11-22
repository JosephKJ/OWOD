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

def plot_tsne_indiv(X, label, total_num_classes):
    n_components = 2

    # perplexities = [5, 30, 50, 100, 150]
    perplexities = list(range(10, 150, 10))

    for i, perplexity in enumerate(perplexities):
        __, ax = plt.subplots()
        t0 = time()
        tsne = manifold.TSNE(n_components=n_components, init='random',
                             random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(X)
        t1 = time()
        print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
        # ax.set_title("Perplexity=%d" % perplexity)
        #
        # sc = ax.scatter(Y[:, 0], Y[:, 1], c=label, cmap="plasma")

        # palette = sns.color_palette(None, total_num_classes)
        # palette = sns.color_palette("flare", as_cmap=True)

        flatui = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', \
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', \
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', \
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', \
        '#3498db']
        # sns.set_palette(flatui)
        sns.scatterplot(x=Y[:, 0], y=Y[:, 1], hue=label, ax=ax, legend='full', palette=flatui)

        # sns.scatterplot(x=Y[:, 0], y=Y[:, 1], hue=label, ax=ax, legend='full', palette='colorblind')
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        # plt.legend(handles=sc.legend_elements()[0], labels=range(total_num_classes))

        # plt.legend(handles=sc.legend_elements()[0], labels=['0', '1'])

    # plt.show()
    #     plt.legend(fontsize='xx-large', ncol=2, handleheight=2.4, labelspacing=0.05)
        plt.legend(ncol=6)

        plt.savefig('tsne_' + str(perplexity) + '.png')
        plt.pause(0.0001)
        plt.clf()

maxlen_queue = 100
total_num_classes = 22

queues = [deque(maxlen=maxlen_queue) for _ in range(total_num_classes)]

source_dir = '/home/fk1/workspace/OWOD/output/features'

# files = os.listdir(source_dir)
# for i, file in enumerate(files):
#     path = os.path.join(source_dir, file)
#     features, classes = torch.load(path)
#     for f, c in zip(features, classes):
#         if c == 80:
#             c = 20
#             queues[c].append(f.detach().cpu().numpy())
#         elif c == 81:
#             c = 21
#             queues[c].append(f.detach().cpu().numpy())
#         elif c <= total_num_classes:
#             queues[c.detach().cpu().numpy()].append(f.detach().cpu().numpy())
#     if i%100 == 0:
#         print('Processing ' + str(i))
#     # if i == 2:
#     #     break
#
# torch.save(queues, os.path.join(source_dir,'queues_tsne.pkl'))

queues = torch.load(os.path.join(source_dir,'queues_tsne.pkl'))

x = []
y = []
for i, queue in enumerate(queues):
    if i == 20:
        continue
    if i == 21:
        i = 20
    for item in queue:
        x.append(item)
        y.append(i)

print('Going to plot')
plot_tsne_indiv(x, y, total_num_classes)
