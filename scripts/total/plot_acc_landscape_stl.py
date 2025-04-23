import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import copy
from utils.args import get_args
import seaborn as sns
from datasets import get_dataset



plt.rcParams['figure.dpi'] = 300 #图片像素
plt.rcParams['figure.figsize'] = (7.0, 6.0)
plt.rc('font',family='Times New Roman')
matplotlib.rcParams.update({'font.size':16 })


args = get_args()
args.dataset = 'seq-cifar100'
args.print_freq = 10
args.n_epochs = 100
args.classifier = 'linear'
args.scheduler_step = 99
args.ssl_leaner = 'moco'

args.lr = 0.005
args.clslr = 0.2
args.batch_size = 256
args.ssl_weight = 1
args.sl_weight = 10
args.weight_decay = 0
args.momentum = 0

loss_fn = F.cross_entropy

levels = [0.0, 1.0, 2.0, 3.0, 4.0]


def plot_loss(data, net_0, net_1, net_2, classifier, test_task, file_name):
    # 生成横纵坐标网格
    x_min, x_max = -0.1, 1.1
    y_min, y_max = -0.1, 1.1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    pos = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

    # 计算对应位置损失
    loss = [] # 分别对应任务1和2的损失
    for i in range(pos.shape[0]):
        pos_x, pos_y = pos[i, 0], pos[i, 1]
        net_temp = copy.deepcopy(net_0)

        for net_temp_params, net_0_params, net_1_params, net_2_params in zip(net_temp.parameters(), net_0.parameters(), net_1.parameters(), net_2.parameters()):
            net_temp_weight, net_0_weight, net_1_weight, net_2_weight = net_temp_params.data, net_0_params.data, net_1_params.data, net_2_params.data
            net_temp_params.data = net_2_weight + pos_x * (net_0_weight - net_2_weight) + pos_y * (net_1_params - net_2_weight)


        inputs, labels = data[0].to('cuda'), data[1].to('cuda')

        feat = net_temp.features(inputs)
        pred = classifier(feat)

        if test_task == 0:
            loss.append(loss_fn(pred[:, :10], labels).detach().cpu().item())
        else:
            loss.append(loss_fn(pred[:, 10:20], labels-10).detach().cpu().item())


    # 绘制contour
    z = np.array(loss).reshape(xx.shape)
    print(z)
    plt.contourf(xx, yy, z, alpha=0.3, cmap='RdYlGn_r', levels=levels, extend='both')
    C=plt.contour(xx,yy, z, linewidth=.3, alpha=1.0, levels=levels, extend='both', color='gray')
    plt.clabel(C,inline=True,fontsize=13)

    #plt.scatter(0, 0, c='blue', label='multi_task') #两个新类别
    plt.scatter(1, 0, s=200, c='red', label='Task 1 extractor') #两个新类别
    plt.scatter(0, 1, s=200, c='green', label='Task 2 extractor') #两个新类别
    plt.scatter(0.5, 0.5, s=200,  c='blue', label='Averaged extractor') #两个新类别
    plt.legend()
    plt.axis('off')

    plt.savefig(file_name)
    plt.clf()

# 设置参数
args = get_args()
args.dataset = 'seq-cifar100'
args.batch_size = 128
args.nt = 10
args.device = 'cuda'
root = 'img/accstl/'

# 读取模型和数据集
model = torch.load(root + args.dataset + '.pt')
dataset = get_dataset(args)

loader_0, _ = dataset.get_data_loaders()
for data in loader_0:
    data_0 = data
    break
loader_1, _ = dataset.get_data_loaders()
for data in loader_1:
    data_1 = data
    break


######### CIFAR100
net_0 = model.net_arr[0]
net_1 = model.net_arr[1]
net_2 = model.net_arr[2]
classifier = model.classifier

plot_loss(data_0, net_0, net_1, net_2, classifier, test_task=0, file_name='loss_landscape_task0.jpg')
plot_loss(data_1, net_0, net_1, net_2, classifier, test_task=1, file_name='loss_landscape_task1.jpg')









