from __future__ import print_function
import os
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def loss_plot(hist, path='.', y_max=None, use_subplot=False, keys_to_show=[], net_num = 0, comment=''):
    try:
        x = range(len(hist['D_loss']))
    except:
        keys = hist.keys()
        lens = [len(hist[k]) for k in keys if 'loss' in k]
        maxlen = max(lens)
        x = range(maxlen)

    if use_subplot:
        f, axarr = plt.subplots(2, sharex=True)

    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.tight_layout()

    if len(keys_to_show) == 0:
        keys_to_show = hist.keys()
    for key, value in hist.items():  # hist.iteritems():
        if 'time' in key or key not in keys_to_show:
            continue
        y = value
        if len(x) != len(y):
            print('[warning] loss_plot() found mismatching dimensions: {}'.format(key))
            continue
        if use_subplot and 'acc' in key:
            axarr[1].plot(x, y, label=key)
        elif use_subplot:
            axarr[0].plot(x, y, label=key)
        else:
            plt.plot(x, y, label=key)

    if use_subplot:
        axarr[0].legend(loc=1)
        axarr[0].grid(True)
        axarr[1].legend(loc=1)
        axarr[1].grid(True)
    else:
        plt.legend(loc=1)
        plt.grid(True)

    if y_max is not None:
        if use_subplot:
            x_min, x_max, y_min, _ = axarr[0].axis()
            axarr[0].axis((x_min, x_max, -y_max / 20, y_max))
        else:
            x_min, x_max, y_min, _ = plt.axis()
            plt.axis((x_min, x_max, -y_max / 20, y_max))

    path = os.path.join(path, 'loss_%d_' % net_num + comment + '.png')

    plt.savefig(path)

    plt.close()
