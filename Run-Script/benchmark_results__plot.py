import numpy as np
from matplotlib import pyplot as plt, rcParams

from distilllib.engine.utils import save_figure

# 要解释的样本数据集
# dataset = 'CamCAN'  # CamCAN DecMeg2014
dataset = 'DecMeg2014'  # CamCAN DecMeg2014

species = ("Vanilla", "KD", "GDK", "MSEKD", "DKD", r"$\mathcal{L}_{FAKD}$", "FAKD")
penguin_means = {
    "Vanilla": ((94.39), (0.13)),
    "Teacher: LFCNN": ((94.83, 94.80, 94.79, 94.80, 95.10, 95.17), (0.17, 0.16, 0.29, 0.20, 0.15, 0.10)),
    "Teacher: VARCNN": ((94.94, 94.91, 94.92, 94.87, 95.34, 95.43), (0.18, 0.14, 0.12, 0.10, 0.06, 0.09)),
    "Teacher: HGRN": ((94.53, 94.47, 94.16, 94.53, 94.59, 94.72), (0.12, 0.10, 0.38, 0.10, 0.22, 0.07)),
}
if dataset == "DecMeg2014":
    penguin_means = {
        "Vanilla": ((74.75), (1.55)),
        "Teacher: LFCNN": ((79.36, 79.23, 78.92, 77.54, 81.72, 81.95), (0.80, 1.16, 0.76, 0.84, 0.36, 0.58)),
        "Teacher: VARCNN": ((79.09, 79.16, 79.26, 77.58, 79.12, 79.53), (0.42, 0.39, 0.25, 0.36, 0.99, 0.57)),
        "Teacher: HGRN": ((79.60, 78.92, 79.12, 79.53, 82.05, 83.03), (0.20, 0.13, 1.19, 0.74, 0.59, 0.20)),
    }

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained', figsize=(7.5, 5))

for attribute, (mean, std) in penguin_means.items():
    offset = width * multiplier
    # ax.bar_label(rects, padding=3)
    # 一个一个添加误差棒
    if attribute == "Vanilla":
        rects = ax.bar(x + offset + width, mean, width, label=attribute)
        ax.errorbar(x[0] + offset + width, mean, yerr=std, ecolor='black', elinewidth=1, capsize=5, capthick=1)
    else:
        rects = ax.bar(x[1:] + offset, mean, width, label=attribute)
        for i in range(len(species)-1):
            ax.errorbar(x[i+1] + offset, mean[i], yerr=std[i], ecolor='black', elinewidth=1, capsize=5, capthick=1)
        multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Methods of Knowledge Distillation')
ax.set_title('Experimental results in the {} dataset'.format(dataset))
ax.set_xticks(x + width, species)
ax.legend(loc='upper center', ncols=len(penguin_means))
ax.set_ylim(93.5, 96)
if dataset == "DecMeg2014":
    ax.set_ylim(72, 85)

plt.show()

save_figure(fig, '../plot/heatmap/', 'benchmark_results_{}'.format(dataset))
