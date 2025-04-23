import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import numpy as np
from scipy.linalg import svd

from utils.args import get_args
from datasets import get_dataset
from torch.nn.functional import avg_pool2d, relu


'''
计算分类器针对各特征提取器的激活情况
'''


# 设置参数
args = get_args()
args.dataset = 'seq-cifar100'
args.batch_size = 512
args.device = 'cuda'
root = 'img/accstlfm2/'

# 读取模型和数据集
model = torch.load(root + args.dataset + '.pt')
dataset = get_dataset(args)

logit_arr = []
for t in range(dataset.nt):
    cls = dataset.t_c_arr[t]
    train_loader, test_loader = dataset.get_data_loaders()
    for i, data in enumerate(test_loader):
        inputs = data[0].to(args.device)
        logit_arr_t= []
        with torch.no_grad():
            for net in model.net_arr: # 按特征提取器遍历

                feat = net.features(inputs)
                # 去掉分类器无关的部分
                logit = model.classifier(feat)
                logit = logit[:, cls]
                logit, _ = torch.max(logit, dim=1)
                logit = torch.mean(logit)
                logit_arr_t.append(logit.item())

        print(logit_arr_t)
        logit_arr.append(logit_arr_t)
        break

for logit in logit_arr:
    print(logit)



