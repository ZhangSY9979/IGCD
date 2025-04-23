import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import numpy as np
from scipy.linalg import svd

from utils.args import get_args
from datasets import get_dataset
from scripts.plot_heat_map import heat_map
from torch.nn.functional import avg_pool2d, relu

np.set_printoptions(threshold=np.inf)  # 防止np省略长数据


def principal_angles(A, B):
    """
    Compute the principal angles between two subspaces A and B.
    Parameters:
    A, B : array_like
        The matrices A and B are of shape (m, n) and (m, p) respectively,
        and each column is a vector that spans the subspace.
    Returns:
    angles : array_like
        The principal angles between the two subspaces, in radians.
    """
    # Orthonormalize the columns of A and B
    Q_A = np.linalg.qr(A)[0]
    Q_B = np.linalg.qr(B)[0]
    # Compute the SVD of the matrix product of the orthonormal bases
    _, singular_values, _ = svd(Q_A.T @ Q_B)
    # The singular values are the cosines of the principal angles
    angles = np.clip(singular_values, -1, 1)
    # Compute the angles and return them
    # angles = np.arccos(angles)
    # 注意，一般取第一个作为主角
    return angles

    # 保存图片
    pl.savefig(root + '/difussion_' + args.dataset + '.png')

# 设置参数
args = get_args()
args.dataset = 'seq-cifar100'
args.batch_size = 128
args.nt = 10
args.device = 'cuda'
root = 'img/accstlfm/'

# 读取模型和数据集
model = torch.load(root + args.dataset + '.pt')
dataset = get_dataset(args)


# 获取特征
# feats = []
# for k, test_loader in enumerate(dataset.test_loaders):
#     for i, data in enumerate(test_loader):
#         inputs = data[0].to(args.device)
#         with torch.no_grad():
#             net = model.net
#             feat = net.features(inputs)
#
#         feats.append(feat.cpu().numpy().T)
#         break

# 获取预测结果  STL版本
# for t in range(dataset.nt):
#     train_loader, test_loader = dataset.get_data_loaders()
# feats = []
# for k, test_loader in enumerate(dataset.test_loaders):
#     task_id = int(k // 10)
#     for i, data in enumerate(test_loader):
#         inputs = data[0].to(args.device)
#         with torch.no_grad():
#             net = model.net_arr[task_id]
#             feat = net.features(inputs)
#
#             # 去掉分类器无关的部分
#             logit = model.classifier(feat)
#
#             feat = feat.cpu().numpy()
#             logit = logit.detach().cpu().numpy()
#             U, S, Vh = np.linalg.svd(logit, full_matrices=False)
#             feat = np.dot(np.dot(
#                     U,
#                     U.transpose()
#                 ), feat)
#
#         feats.append(feat.T)
#         break
#
#
# angles_principle = np.zeros((dataset.nt, dataset.nt))
# angles_prototype = np.zeros((dataset.nt, dataset.nt))
# for i, feat_i in enumerate(feats):
#     for j, feat_j in enumerate(feats):
#         angle = principal_angles(feat_i, feat_j)
#         angles_principle[i, j] = angle[0]
#
#         mean_i, mean_j = np.mean(feat_i, axis=0), np.mean(feat_j, axis=0)
#         mean_i = mean_i / np.sqrt(np.sum(mean_i**2))
#         mean_j = mean_j / np.sqrt(np.sum(mean_j**2))
#         cos_angle = mean_i.T @ mean_j
#         # print(cos_angle)
#         angles_prototype[i, j] = cos_angle
#
# print(angles_principle)
# print(angles_prototype)



# 按类别对分类器原型绘制主角
angles_principle = np.zeros((dataset.nt, dataset.nt))
angles_prototype = np.zeros((dataset.nt, dataset.nt))
weight = model.classifier.linear.weight.detach().cpu().numpy().T
print(weight.shape)
for i in range(dataset.nt):
    for j in range(dataset.nt):
        cls_i, cls_j = dataset.t_c_arr[i], dataset.t_c_arr[j]
        weight_i, weight_j = weight[:, cls_i], weight[:, cls_j]

        angle = principal_angles(weight_i, weight_j)
        angles_principle[i, j] = np.mean(angle)

        mean_i, mean_j = np.mean(weight_i, axis=1), np.mean(weight_j, axis=1)
        print(mean_i.shape)
        mean_i = mean_i / np.sqrt(np.sum(mean_i**2))
        mean_j = mean_j / np.sqrt(np.sum(mean_j**2))
        cos_angle = mean_i.T @ mean_j
        # print(cos_angle)
        angles_prototype[i, j] = cos_angle

print(angles_principle)
print(angles_prototype)




# # 分类器原型间角度
# weight = model.classifier.linear.weight.detach().cpu().numpy()
# print(weight.shape)
# angles_prototype = np.zeros((dataset.nc, dataset.nc))
# for i in range(dataset.nc):
#     for j in range(dataset.nc):
#         weight_i, weight_j = weight[i, :], weight[j, :]
#         weight_i = weight_i / np.sqrt(np.sum(weight_i ** 2))
#         weight_j = weight_j / np.sqrt(np.sum(weight_j ** 2))
#         cos_angle = weight_i.T @ weight_j
#         angles_prototype[i, j] = cos_angle
#
# np.set_printoptions(suppress=True)
# # print(angles_prototype)
#
# labels = []
# for i in range(dataset.nc):
#     labels.append(str(i+1))
# heat_map(angles_prototype,
#          'angles_prototype_heatmap.png',
#          x_ticklabels=labels,
#          y_ticklabels=labels,
#          max=1,
#          min=0)
#
#
# # 计算特征在分类器张成子空间比例
# ratios = []
# ratios_avg = 0.0
# with torch.no_grad():
#     for t in range(dataset.nt):
#         train_loader, test_loader = dataset.get_data_loaders()
#         ratios_t = []
#         for i, data in enumerate(test_loader):
#             inputs = data[0].to(args.device)
#             cls = dataset.t_c_arr[t]
#             weight_t1 = weight[cls, :]
#             basis = weight_t1 / np.sqrt(np.sum(weight_t1**2, axis=1)).reshape(weight_t1.shape[0], 1)
#
#             for t_2 in range(dataset.nt):
#                 net = model.net_arr[t_2]
#                 feat = net.features(inputs).detach().cpu().numpy()
#
#                 feat_proj = feat.dot(basis.transpose()).dot(basis)
#
#                 # feat_res = feat - feat_proj
#
#                 ratio = (np.sum(feat_proj**2)) / (np.sum(feat ** 2))
#
#                 ratios_t.append(ratio)
#                 if t == t_2:
#                     ratios_avg += ratio
#             break
#         print(ratios_t)
#         ratios.append(ratios_t)
# print(ratios)
# print('Average_ratio:', ratios_avg / dataset.nt)
# labels = []
# for i in range(dataset.nt):
#     labels.append(str(i+1))
# heat_map(np.array(ratios),
#          'angles_feature_ratio.png',
#          x_ticklabels=labels,
#          y_ticklabels=labels,
#          max=0.4,
#          min=0,)





