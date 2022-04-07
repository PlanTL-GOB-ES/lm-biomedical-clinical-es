import os
import json
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from scipy.interpolate import interp1d

plt.rcParams.update({
    "font.family": "CMU Serif",
    "font.size": 12
})


def get_stats(data):
    # URLs by domain: min, max, average, std
    num_urls = np.array([d['num_urls'] for d in data])
    print("URLs by domain statistics:")
    print(
        f'min: {np.min(num_urls)}, max: {np.max(num_urls)}, avg: {np.mean(num_urls)}, std: {np.std(num_urls)}, total: {np.sum(num_urls)}')

    # Depths
    depths = [d['depths'] for d in data]
    depths = np.array(list(chain.from_iterable(depths)))
    print("All depths statistics:")
    print(f'min: {np.min(depths)}, max: {np.max(depths)}, avg: {np.mean(depths)}, std: {np.std(depths)}')
    return num_urls, depths


def plot_descendant(data, title, output, interpolate=False):
    y = sorted(data, reverse=True)
    x = range(len(data))

    if interpolate:
        f_interp = interp1d(x, y, kind='linear')
        x_interp = np.linspace(0, len(x), num=50_000, endpoint=False)
        x = x_interp
        y = f_interp(x_interp)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle(title)
    ax.set_yscale('log')
    ax.bar(x, y)
    if interpolate:
        ax.set_xticks((0, max(x)))

    fig.tight_layout()
    fig.savefig(output)


if __name__ == '__main__':
    with open('data/stats.json', 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    data = list(filter(lambda x: x['num_urls'] != 0, data))
    num_urls, depths = get_stats(data)

    plot_descendant(num_urls, 'URL count by domain', 'output/url_count_domain.pdf')
    plot_descendant(depths, 'URL depths', 'output/url_depths.pdf', interpolate=True)
