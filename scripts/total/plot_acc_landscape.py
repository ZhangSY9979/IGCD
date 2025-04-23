import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import torch
import torch.nn.functional as F
import numpy as np
import copy
from utils.args import get_args
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import get_dataset
import matplotlib.pyplot as plt
import matplotlib

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


def plot_loss():
    dataset = get_dataset(args)
    data_0, data_1 = None, None
    for t in range(dataset.nt):
       train_loader, test_loader = dataset.get_data_loaders()
       for data in train_loader:
           if t == 0:
               data_0 = data
           elif t == 1:
               data_1 = data
           break
       if t == 1: break
    data = data_0 if test_task == 0 else data_1


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
        net_temp = copy.deepcopy(net_m)
        cls_temp = copy.deepcopy(classifier)

        for net_temp_params, net_m_params, net_c_params, net_0_params in zip(net_temp.parameters(), net_m.parameters(), net_c.parameters(), net_0.parameters()):
            net_temp_weight, net_m_weight, net_c_weight, net_0_weight = net_temp_params.data, net_m_params.data, net_c_params.data, net_0_params.data
            net_temp_params.data = net_m_weight + pos_x * (net_c_weight - net_m_weight) + pos_y * (net_0_params - net_m_weight)

        for cls_temp_params, cls_params, cls_c_params, cls_0_params in zip(cls_temp.parameters(), classifier.parameters(), classifier_c.parameters(), classifier_0.parameters()):
            cls_temp_weight, cls_weight, cls_c_weight, cls_0_weight = cls_temp_params.data, cls_params.data, cls_c_params.data, cls_0_params.data
            cls_temp_params.data = cls_weight + pos_x * (cls_c_weight - cls_weight) + pos_y * (cls_0_weight - cls_weight)


        inputs, labels = data[0].to('cuda'), data[1].to('cuda')

        feat = net_temp.features(inputs)
        pred = cls_temp(feat)

        if test_task == 0:
            loss.append(loss_fn(pred[:, :10], labels).detach().cpu().item())
        else:
            loss.append(loss_fn(pred[:, 10:20], labels-10).detach().cpu().item())


    # 绘制contour
    z = np.array(loss).reshape(xx.shape)
    #print(loss_0)
    plt.contourf(xx, yy, z, alpha=0.3, cmap='RdYlGn_r', levels=levels, extend='both')
    C=plt.contour(xx,yy, z, linewidth=.3, alpha=1.0, levels=levels, extend='both', color='gray')
    plt.clabel(C,inline=True,fontsize=13)

    #plt.scatter(0, 0, c='blue', label='multi_task') #两个新类别
    plt.scatter(0, 1, s=200, c='green', label='balance_learner_task0') #两个新类别
    plt.scatter(1, 0, s=200, c='red', label='online_learner_task1') #两个新类别
    plt.scatter(0.5, 0.5, s=200,  c='blue', label='balance_learner_task1') #两个新类别
    plt.legend()

    plt.savefig(file_name)
    plt.clf()

######### CIFAR100

file_name = 'loss_landscape_ssl_loss_task0.jpg'
test_task = 0
net_0 = torch.load('img/acc/sslconti_seq-cifar100task_0net.pt')     # 基任务模型
net_c = torch.load('img/acc/sslconti_seq-cifar100task_1online.pt')  # 增量模型
net_m = torch.load('img/acc/sslmulti_seq-cifar100task_1online.pt')  # 多任务模型 上界
classifier = torch.load('img/acc/sslmulti_seq-cifar100task_1classifier.pt')
classifier_0 = torch.load('img/acc/sslconti_seq-cifar100task_0classifier.pt')
classifier_c = torch.load('img/acc/sslconti_seq-cifar100task_1classifier.pt')
plot_loss()


file_name = 'loss_landscape_ssl_loss_task1.jpg'
test_task= 1
net_0 = torch.load('img/acc/sslconti_seq-cifar100task_0net.pt')     # 基任务模型
net_c = torch.load('img/acc/sslconti_seq-cifar100task_1online.pt')  # 增量模型
net_m = torch.load('img/acc/sslmulti_seq-cifar100task_1online.pt')  # 多任务模型 上界
classifier = torch.load('img/acc/sslmulti_seq-cifar100task_1classifier.pt')
classifier_0 = torch.load('img/acc/sslconti_seq-cifar100task_0classifier.pt')
classifier_c = torch.load('img/acc/sslconti_seq-cifar100task_1classifier.pt')
plot_loss()

file_name = 'loss_landscape_nossl_loss_task0.jpg'
test_task = 0
net_0 = torch.load('img/acc/nosslconti_seq-cifar100task_0net.pt')     # 基任务模型
net_c = torch.load('img/acc/nosslconti_seq-cifar100task_1online.pt')  # 增量模型
net_m = torch.load('img/acc/nosslmulti_seq-cifar100task_1online.pt')  # 多任务模型 上界
classifier = torch.load('img/acc/nosslmulti_seq-cifar100task_1classifier.pt')
classifier_0 = torch.load('img/acc/nosslconti_seq-cifar100task_0classifier.pt')
classifier_c = torch.load('img/acc/nosslconti_seq-cifar100task_1classifier.pt')
plot_loss()

file_name = 'loss_landscape_nossl_loss_task1.jpg'
test_task = 1
net_0 = torch.load('img/acc/nosslconti_seq-cifar100task_0net.pt')     # 基任务模型
net_c = torch.load('img/acc/nosslconti_seq-cifar100task_1online.pt')  # 增量模型
net_m = torch.load('img/acc/nosslmulti_seq-cifar100task_1online.pt')  # 多任务模型 上界
classifier = torch.load('img/acc/nosslmulti_seq-cifar100task_1classifier.pt')
classifier_0 = torch.load('img/acc/nosslconti_seq-cifar100task_0classifier.pt')
classifier_c = torch.load('img/acc/nosslconti_seq-cifar100task_1classifier.pt')
plot_loss()

######### Tiny_ImageNet


# file_name = 'loss_landscape_ssl_loss_task0.jpg'
# test_task = 0
# net_0 = torch.load('img/acc/sslconti_seq-tinyimgtask_0net.pt')     # 基任务模型
# net_c = torch.load('img/acc/sslconti_seq-tinyimgtask_1online.pt')  # 增量模型
# net_m = torch.load('img/acc/sslmulti_seq-tinyimgtask_1online.pt')  # 多任务模型 上界
# classifier = torch.load('img/acc/sslmulti_seq-tinyimgtask_1classifier.pt')
# classifier_0 = torch.load('img/acc/sslconti_seq-tinyimgtask_0classifier.pt')
# classifier_c = torch.load('img/acc/sslconti_seq-tinyimgtask_1classifier.pt')
# plot_loss()
#
#
# file_name = 'loss_landscape_ssl_loss_task1.jpg'
# test_task= 1
# net_0 = torch.load('img/acc/sslconti_seq-tinyimgtask_0net.pt')     # 基任务模型
# net_c = torch.load('img/acc/sslconti_seq-tinyimgtask_1online.pt')  # 增量模型
# net_m = torch.load('img/acc/sslmulti_seq-tinyimgtask_1online.pt')  # 多任务模型 上界
# classifier = torch.load('img/acc/sslmulti_seq-tinyimgtask_1classifier.pt')
# classifier_0 = torch.load('img/acc/sslconti_seq-tinyimgtask_0classifier.pt')
# classifier_c = torch.load('img/acc/sslconti_seq-tinyimgtask_1classifier.pt')
# plot_loss()
#
# file_name = 'loss_landscape_nossl_loss_task0.jpg'
# test_task = 0
# net_0 = torch.load('img/acc/nosslconti_seq-tinyimgtask_0net.pt')     # 基任务模型
# net_c = torch.load('img/acc/nosslconti_seq-tinyimgtask_1online.pt')  # 增量模型
# net_m = torch.load('img/acc/nosslmulti_seq-tinyimgtask_1online.pt')  # 多任务模型 上界
# classifier = torch.load('img/acc/nosslmulti_seq-tinyimgtask_1classifier.pt')
# classifier_0 = torch.load('img/acc/nosslconti_seq-tinyimgtask_0classifier.pt')
# classifier_c = torch.load('img/acc/nosslconti_seq-tinyimgtask_1classifier.pt')
# plot_loss()
#
# file_name = 'loss_landscape_nossl_loss_task1.jpg'
# test_task = 1
# net_0 = torch.load('img/acc/nosslconti_seq-tinyimgtask_0net.pt')     # 基任务模型
# net_c = torch.load('img/acc/nosslconti_seq-tinyimgtask_1online.pt')  # 增量模型
# net_m = torch.load('img/acc/nosslmulti_seq-tinyimgtask_1online.pt')  # 多任务模型 上界
# classifier = torch.load('img/acc/nosslmulti_seq-tinyimgtask_1classifier.pt')
# classifier_0 = torch.load('img/acc/nosslconti_seq-tinyimgtask_0classifier.pt')
# classifier_c = torch.load('img/acc/nosslconti_seq-tinyimgtask_1classifier.pt')
# plot_loss()




