import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MultipleLocator
plt.rcParams['figure.dpi'] = 300  # 图片像素
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.rc('font', family='Times New Roman')


def heat_map(data, filepath, x_label='', y_label='', x_ticklabels=None, y_ticklabels=None, min=None, max=None, center=None):
    if y_ticklabels is None:
        y_ticklabels = []
    if x_ticklabels is None:
        x_ticklabels = []

    h = sns.heatmap(data, vmin=min, vmax=max, center=center, cbar=False)
    cb = h.figure.colorbar(h.collections[0])
    cb.ax.tick_params(labelsize=14)  # 设置colorbar刻度字体大小。

    # 设置坐标位置及内容
    pos_x = np.array(range(len(x_ticklabels))) + 0.5
    pos_y = np.array(range(len(y_ticklabels))) + 0.5
    if len(x_ticklabels) > 10:
        pos_x = pos_x[::10]
        x_ticklabels = x_ticklabels[::10]
    if len(y_ticklabels) > 10:
        pos_y = pos_y[::10]
        y_ticklabels = y_ticklabels[::10]

    plt.xticks(pos_x, x_ticklabels, fontsize=16)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(pos_y, y_ticklabels, fontsize=16)  # 将标签印在y轴坐标上
    plt.ylabel(y_label, fontsize=16)
    plt.xlabel(x_label, fontsize=16)

    plt.savefig(filepath)
    plt.clf()