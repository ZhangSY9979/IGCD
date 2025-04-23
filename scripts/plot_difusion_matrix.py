import os

os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import torch
import numpy as np
from utils.args import get_args
from datasets import get_dataset

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as pl

pl.rcParams['figure.dpi'] = 300  # 图片像素
pl.rcParams['figure.figsize'] = (8.0, 6.0)
pl.rc('font', family='Times New Roman')
from matplotlib.pyplot import MultipleLocator
from sklearn import metrics


# 绘制混淆矩阵

def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.3, axis_labels=None):
    # 利用sklearn中的函数生成混淆矩阵并归一化
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵 
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    pl.imshow(cm, interpolation='nearest', cmap=pl.get_cmap('Blues'))
    pl.colorbar()  # 绘制图例

    # 图像标题
    if title is not None:
        pl.title(title)
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name

    major_locator = MultipleLocator(dataset.nc / 10)
    pl.xticks(num_local, axis_labels, fontsize=16)  # 将标签印在x轴坐标上， 并倾斜45度
    pl.yticks(num_local, axis_labels, fontsize=16)  # 将标签印在y轴坐标上
    ax = pl.gca()
    ax.xaxis.set_major_locator(major_locator)
    ax.yaxis.set_major_locator(major_locator)
    pl.ylabel('True classes', fontsize=16)
    pl.xlabel('Predicted classes', fontsize=16)

    # 保存图片
    pl.savefig(root + '/difussion_' + args.dataset + '.png')


# 设置参数
args = get_args()
args.dataset = 'seq-tinyimg'
root = 'img/bmkpv2_standard/'

# 读取模型和数据集
model = torch.load(root + args.dataset + '.pt')

dataset = get_dataset(args)
for t in range(dataset.nt):
    train_loader, test_loader = dataset.get_data_loaders()

# 获取预测结果
preds = []
labels = []
categories = list(range(dataset.t_c_arr[model.current_task][-1] + 1))
for k, test_loader in enumerate(dataset.test_loaders):
    for i, data in enumerate(test_loader):
        inputs, label = data[0], data[1]

        logit = model(inputs)
        _, pred = torch.max(logit.data, 1)

        preds.append(pred)
        labels.append(label)
preds = torch.cat(preds, dim=0).cpu().numpy()
labels = torch.cat(labels, dim=0).cpu().numpy()

# 绘制混淆矩阵
plot_matrix(labels, preds, list(range(dataset.nc)), title=None, thresh=0.3, axis_labels=None)
