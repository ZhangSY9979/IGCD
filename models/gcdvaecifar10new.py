from copy import copy
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
import torch
from torch.optim import SGD, Adam
from backbone.VAE import VAE
from backbone.VAE_MLP import VAE_MLP
from backbone.VaDE import VaDE, VaE
from torch.utils.data import DataLoader, Dataset
from models.utils.incremental_model import IncrementalModel
from sklearn.metrics import roc_auc_score
from backbone.processing1 import TrainerVaDE
from models.utils.K_means import KMeans
from utils.util import BCE, PairEnum, accuracy, cluster_acc, AverageMeter, seed_torch, cluster_pred_2_gt, \
    pred_2_gt_proj_acc
from functools import partial
from sklearn.mixture import GaussianMixture

def model_size_in_MB(model_list):
    param_size = 0
    for i in range(len(model_list)):
        model = model_list[i]
        for param in model.parameters():
            param_size += param.numel() * param.element_size()  # 每个参数的大小（字节）
    return param_size / (1024 ** 2)  # 转换为 MB


class GCDVAECIFAR10NEW(IncrementalModel):
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, args):
        super(GCDVAECIFAR10NEW, self).__init__(args)

        self.nets = []

        self.lambda2 = self.args.lambda2  # 损失系数
        self.lambda3 = self.args.lambda3
        self.lambda4 = self.args.lambda4
        self.eps = self.args.eps
        self.embedding_dim = self.args.embedding_dim  # 隐藏层维度
        self.weight_decay = self.args.weight_decay  # 超参之一
        self.lambda1 = self.args.lambda1  # 损失系数
        self.r_inter = self.args.r_inter  # 类间
        self.r_intra = self.args.r_intra  # 类内
        self.kld_ratio = self.args.kld_ratio  # 损失系数
        self.isPseudo = self.args.isPseudo  # 伪标签？？？

        self.bs = self.args.batch_size

        self.current_task = -1  # 当前任务数
        self.nc = None  # 总类别数
        self.t_c_arr = []  # 每个任务的类别标签
        self.nf = self.args.nf  # 总任务数？？？
        self.isPrint = self.args.isPrint  # 是否打印
        self.bate = torch.zeros(10).to(self.device)

        self.mus = []  # 均值
        self.log_vars = []  # 对数方差
        self.thresholds = []  # 阈值？？?
        self.cluster_net = None

        # 任务初始化

    def begin_il(self, dataset):  # 先把所有类别的VAE都生成好，也可以啊，因为我们反正也是默认知道新类的类别数目的

        self.nc = dataset.nc
        self.t_c_arr = dataset.t_c_arr
        # self.cluster_net = VaDE(input_dim=1000, hidden_dims=[800, 500], latent_dim=10, n_classes=10)  # 需要根据不同任务进行修改
        for i in range(self.nc):  # 总类别数目
            if self.args.dataset == 'seq-mnist':  # minst数据集
                # 0.98m
                net = VAE_MLP(latent_dim=self.embedding_dim, device=self.device, hidden_dims=[100, 100]).to(
                    self.device)
            elif self.args.dataset == 'seq-tinyimg' or self.args.dataset == 'seq-imagenet':
                if self.args.featureNet:
                    # 9.96m
                    net = VAE_MLP(input_dim=1000, latent_dim=self.embedding_dim, device=self.device,
                                  hidden_dims=[800, 500], is_mnist=False).to(self.device)
                else:
                    # 13.49 VAE  85.23M resnet
                    net = VAE(in_channels=3, latent_dim=self.embedding_dim, device=self.device,
                              hidden_dims=[32, 64, 128, 256, 512]).to(self.device)
            elif self.args.dataset == 'seq-cifar100' or self.args.dataset == 'seq-cifar10':
                if self.args.featureNet == 'pre_conv':
                    # 9.96m
                    net = VAE_MLP(input_dim=1024, latent_dim=self.embedding_dim, device=self.device,
                                  hidden_dims=[800, 500], is_mnist=False).to(self.device)
                elif self.args.featureNet:
                    # 9.96m
                    net = VAE_MLP(input_dim=1000, latent_dim=self.embedding_dim, device=self.device,
                                  hidden_dims=[800, 500], is_mnist=False).to(self.device)
                else:
                    # 13.49 VAE  85.23M resnet
                    net = VAE(in_channels=3, latent_dim=self.embedding_dim, device=self.device,
                              hidden_dims=[self.nf, self.nf * 2, self.nf * 4, self.nf * 8]).to(self.device)
            elif self.args.dataset == 'seq-20news':
                # 9.96m
                mid_dim = int(self.args.middle_dim)
                net = VAE_MLP(input_dim=2000, latent_dim=self.embedding_dim, device=self.device,
                              hidden_dims=[mid_dim, mid_dim], is_mnist=False).to(self.device)
            else:
                # 9.96m
                net = VAE_MLP(input_dim=1000, latent_dim=self.embedding_dim, device=self.device,
                              hidden_dims=[800, 500], is_mnist=False).to(self.device)
            # print(net)
            self.nets.append(
                net
            )
            self.mus.append(None)
            self.log_vars.append(None)
            self.thresholds.append(None)
            print(f"Model size: {model_size_in_MB(self.nets):.2f} MB")

    def train_first(self, dataset, train_loader):
        self.current_task += 1
        temp = self.t_c_arr[self.current_task]
        categories = temp[len(temp) - 1][0]
        # print(categories)
        for category in categories:  # 类别标号，但是我不能用啊，因为我要当做是无label的数据，要手动分配；第一个阶段可以用，后续不可以

            losses = []

            for epoch in range(self.args.n_epochs):

                avg_loss, posloss_arr, negloss_arr, kldloss_arr, maxloss_arr, pseudoloss_arr = self.train_category(
                    train_loader, category, epoch)

                losses.append(avg_loss)
                if False and epoch == 0 or (epoch + 1) % 10 == 0:
                    avg_maxloss = 0.0
                    avg_pseudoloss = 0.0
                    if self.current_task > 1:
                        avg_maxloss = np.mean(maxloss_arr)
                    if self.lambda3 != 0:
                        avg_pseudoloss = np.mean(pseudoloss_arr)
                    print(
                        "epoch: %d\t task: %d \t category: %d \t loss: %f \t posloss: %f \t negloss: %f \t kldloss: %f \t maxloss: %f \t pseudoloss: %f" % (
                            epoch + 1, self.current_task, category, avg_loss, np.mean(posloss_arr),
                            np.mean(negloss_arr), np.mean(kldloss_arr), avg_maxloss, avg_pseudoloss))
        for category in categories:
            self.bate[category] = self.set_score(train_loader, category, self.args.ratio_first)
            # self.bate.requires_grad = False
            # print(self.bate[category])

        torch.save(self, "data/" + self.args.model + "first_state.pt")

    def set_score(self, train_loader, c_index, ratio):
        network = self.nets[c_index].to(self.device)
        network.eval()
        ans = torch.tensor([]).to(self.device)
        for i, data in enumerate(train_loader):  # 一个任务全部的样本，因此有正样本，有负样本
            inputs = data[0].to(self.device)
            if self.current_task > 0:
                labels = data[3].to(self.device)
            else:
                labels = data[1].to(self.device)
            recons, _, mu, log_var = network(inputs)
            input_flat = inputs.view(inputs.shape[0], -1)
            recons_flat = recons.view(recons.shape[0], -1)
            dist = torch.sum((input_flat - recons_flat) ** 2, dim=1)
            scores = self.get_score(dist, i)  # 异常分数
            ans = torch.cat([ans, scores[labels == c_index]], dim=0)
        # ans,_ = torch.sort(ans)
        # res = copy(torch.mean(ans))
        ans, _ = torch.sort(ans)
        # print(f'c_index:{c_index}, ans: {len(ans)}')
        res = copy(ans[int(ratio * len(ans))])
        # self.args.rotiol += 0.01
        return res

    def test_first(self, dataset, test_loaders):
        all_outputs, all_labels = [], []
        preds = []
        # temp = self.t_c_arr[self.current_task]
        temp = self.t_c_arr[0]
        categories = temp[len(temp) - 1][0]
        samples = 0
        correct = 0
        W_label = []
        W_pre = []
        for k, test_loader in enumerate(dataset.test_loaders):
            print(test_loader.dataset.data.shape)
            for data in test_loader:
                inputs = data[0]
                labels = data[1]
                # print(inputs.shape)

                scores, dists = self.predict(inputs, categories)
                _, pred = torch.max(scores, 1)

                all_outputs.append(dists.detach().cpu())
                all_labels.append(labels.detach().cpu())
                preds.append(pred.detach().cpu())
                samples += inputs.shape[0]
                for i in range(inputs.shape[0]):
                    if pred[i] == labels[i]:
                        correct += 1
                    else:
                        W_label.append(labels[i])
                        W_pre.append(pred[i])
                        # print("true label:", labels[i], "wrong label", pred[i])
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0).numpy()
        preds = torch.cat(preds, dim=0).numpy()
        for i in range(len(categories)):
            label = np.where(all_labels == i, -1, 1)
            aoc = roc_auc_score(label, all_outputs[:, i].detach().cpu().numpy())
            print('类别', i, 'AOC:', aoc)
        print("first stage all_acc:", correct / samples)

    def train_second(self, dataset, train_loader):
        self.current_task += 1
        # self.args.ratio1 *= self.current_task
        temp = self.t_c_arr[self.current_task]
        categories = temp[len(temp) - 1][0]
        prev_categories = list(range(categories[0]))
        print(self.bate)
        print(train_loader.dataset.data.shape)
        self.distinguish(train_loader, prev_categories)
        print(train_loader.dataset.data.shape)

        # 第二步，聚类，先普通聚类吧
        k = len(categories)
        start_index = categories[0]
        self.get_new_labels(train_loader, k, start_index)
        print(train_loader.dataset.data.shape)
        #  第三步，当作有标签的数据进行训练

        self.reset_train_loader(train_loader, prev_categories)  # 保存用于计算损失
        print('==========\t task: %d\t categories:' % self.current_task, categories, '\t==========')
        for category in categories:  # 类别标号，但是我不能用啊，因为我要当做是无label的数据，要手动分配；第一个阶段可以用，后续不可以
            losses = []

            for epoch in range(self.args.n_epochs):

                avg_loss, posloss_arr, negloss_arr, kldloss_arr, maxloss_arr, pseudoloss_arr = self.train_category(
                    train_loader, category, epoch)
                # print("train_second")

                losses.append(avg_loss)
                if False and epoch == 0 or (epoch + 1) % 10 == 0:
                    avg_maxloss = 0.0
                    avg_pseudoloss = 0.0
                    if self.current_task > 1:
                        avg_maxloss = np.mean(maxloss_arr)
                    if self.lambda3 != 0:
                        avg_pseudoloss = np.mean(pseudoloss_arr)
                    print(
                        "epoch: %d\t task: %d \t category: %d \t loss: %f \t posloss: %f \t negloss: %f \t kldloss: %f \t maxloss: %f \t pseudoloss: %f" % (
                            epoch + 1, self.current_task, category, avg_loss, np.mean(posloss_arr),
                            np.mean(negloss_arr), np.mean(kldloss_arr), avg_maxloss, avg_pseudoloss))
        for category in categories:
            self.bate[category] = self.set_score(train_loader, category, self.args.ratio_bate)

    def distinguish(self, train_loader, prev_categories, cnt=0):  # 初始阶段就没有这个，这个是增量阶段
        dataset = train_loader.dataset
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)  # 这个肯定也是要修改的
        new_dist = []
        cnt_new_save = 0
        cnt_old_save = 0
        print('reset dataset with prev_categories', prev_categories)  # 一个列表，之前的全部类别标号
        for i, data in enumerate(loader):  # 训练集中的每个样本都和之前的类别的分类器进行一次预测，并保存结果
            input = data[0].to(self.device)
            target = data[1].to(self.device)

            with torch.no_grad():  # 梯度不更新
                out, pred = self.predict(input, prev_categories)
                _, pred = torch.max(out, 1)
                # print(pred.shape)
                for j in range(input.shape[0]):
                    temp = out[j]  # 一个样本
                    # 认为异常的编码器个数，越多越好，理想情况下是70；
                    if len(temp[temp < self.bate[:len(temp)]]) >= (
                            len(prev_categories) * self.args.ratio - self.args.num):
                        new_dist.append(i * self.bs + j)
                        if target[j] < len(prev_categories):
                            cnt_old_save += 1
                        else:
                            cnt_new_save += 1

        dataset.data = dataset.data[new_dist]
        dataset.targets = np.array(dataset.targets)[new_dist]
        print("old:", cnt_old_save)
        print("new:", cnt_new_save)

    def get_new_labels(self, train_loader, k, start_index):
        dataset = train_loader.dataset
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)  # 这个肯定也是要修改的
        kmeans = KMeans(k)
        labels_kmeans = kmeans(torch.cat([data[0] for i, data in enumerate(loader)]))
        true_label = np.concatenate([data[1] for i, data in enumerate(loader)])
        labels_kmeans = labels_kmeans.numpy()
        proj_all_new = cluster_pred_2_gt(labels_kmeans.astype(int), true_label.astype(int))  # 计算聚类准确率
        pacc_fun_all_new = partial(pred_2_gt_proj_acc, proj_all_new)
        pacc_labeled_all_new = pacc_fun_all_new(true_label.astype(int), labels_kmeans.astype(int))  # 有类别的
        print("Kmeans acc:", pacc_labeled_all_new)

        dataset = deepcopy(train_loader.dataset)
        dataset.data = (dataset.data - np.min(dataset.data, axis=0, keepdims=True)) / (
                    np.max(dataset.data, axis=0, keepdims=True) - np.min(dataset.data, axis=0, keepdims=True))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)  # 这个肯定也是要修改的
        self.args.k = k
        vade = TrainerVaDE(self.args, self.args.device, loader)
        vade.train()
        test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        vade.VaDE.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for x, true, _ in test_loader:
                x = x.to(self.device)
                x_hat, mu, log_var, z = vade.VaDE(x)
                gamma = vade.compute_gamma(z, vade.VaDE.pi_prior)
                pred = torch.argmax(gamma, dim=1)
                y_pred.extend(pred.cpu().detach().numpy())
            y_pred = np.array(y_pred)
            y_pred += start_index
        train_loader.dataset.set_att("Pseudo", y_pred)
        for i in range(start_index, start_index+k):
            temp_model = self.nets[i]
            # for name, param in temp_model.named_parameters():
            #     print(name)
            # print(temp_model)
            temp_model.encoder[0][0].weight.data.copy_(vade.VaDE.fc1.weight.data)
            temp_model.encoder[0][0].bias.data.copy_(vade.VaDE.fc1.bias.data)
            temp_model.encoder[1][0].weight.data.copy_(vade.VaDE.fc2.weight.data)
            temp_model.encoder[1][0].bias.data.copy_(vade.VaDE.fc2.bias.data)

            temp_model.decoder[1][0].weight.data.copy_(vade.VaDE.fc6.weight.data)
            temp_model.decoder[1][0].bias.data.copy_(vade.VaDE.fc6.bias.data)
            temp_model.final_layer.weight.data.copy_(vade.VaDE.fc7.weight.data)
            temp_model.final_layer.bias.data.copy_(vade.VaDE.fc7.bias.data)


    def test_second(self, dataset, test_loaders):
        # all_outputs, all_labels = [], []
        temp = self.t_c_arr[self.current_task]
        categories = list(range(temp[len(temp) - 1][0][-1] + 1))
        # print(categories)
        print("stage:", self.current_task)

        for k, test_loader in enumerate(dataset.test_loaders):
            if k == 0:
                samples = 0
                correct = 0
                for data in test_loader:
                    inputs = data[0]
                    labels = data[1]
                    samples += inputs.shape[0]

                    scores, dists = self.predict(inputs, categories)  # 修改吗？还是全部的
                    _, pred = torch.max(scores, 1)

                    for i in range(inputs.shape[0]):
                        if pred[i] == labels[i]:
                            correct += 1
                print("task ", k, " acc:", correct / samples)
            else:
                preds, all_outputs, all_labels = np.array([]), np.array([]), np.array([])
                for data in test_loader:
                    inputs = data[0]
                    labels = data[1]

                    scores, dists = self.predict(inputs, categories)  # 修改吗？还是全部的
                    _, pred = torch.max(scores, 1)

                    all_labels = np.append(all_labels, labels.detach().cpu().numpy())
                    preds = np.append(preds, pred.detach().cpu().numpy())
                proj_all_new = cluster_pred_2_gt(preds.astype(int), all_labels.astype(int))  # 计算聚类准确率
                pacc_fun_all_new = partial(pred_2_gt_proj_acc, proj_all_new)
                pacc_all_new = pacc_fun_all_new(all_labels.astype(int), preds.astype(int))
                print("task ", k, " acc:", pacc_all_new)

    def test_cluster(self, dataset, test_loader):
        temp = self.t_c_arr[self.current_task]
        categories = list(range(temp[len(temp) - 1][0][-1] + 1))
        # print(categories)
        print("stage:", self.current_task)
        preds, all_outputs, all_labels = np.array([]), np.array([]), np.array([])
        for k, test_loader in enumerate(dataset.test_loaders):
            for data in test_loader:
                inputs = data[0]
                labels = data[1]
                # print(labels)

                scores, dists = self.predict(inputs, categories)  # 修改吗？还是全部的
                _, pred = torch.max(scores, 1)
                all_labels = np.append(all_labels, labels.detach().cpu().numpy())
                preds = np.append(preds, pred.detach().cpu().numpy())
        proj_all_new = cluster_pred_2_gt(preds.astype(int), all_labels.astype(int))  # 计算聚类准确率
        pacc_fun_all_new = partial(pred_2_gt_proj_acc, proj_all_new)
        # label
        labeled_class = self.t_c_arr[0][0][0][-1]
        print("labeled:", labeled_class)
        selected_mask = all_labels <= labeled_class
        # print(selected_mask[selected_mask==1].shape)

        pacc_labeled_all_new = pacc_fun_all_new(all_labels[selected_mask].astype(int),
                                                preds[selected_mask].astype(int))  # 有类别的
        print("cluster,task 0 :", pacc_labeled_all_new)
        for i in range(self.current_task):
            lower = temp[i + 1][0][0]
            upper = temp[i + 1][0][-1]
            print(lower, "  ", upper)
            selected_mask = (all_labels >= lower) * (all_labels <= upper)
            # print(selected_mask[selected_mask==1].shape)
            pacc_all_new = pacc_fun_all_new(all_labels[selected_mask].astype(int), preds[selected_mask].astype(int))
            print("cluster,task ", i + 1, " :", pacc_all_new)

        selected_mask = all_labels > labeled_class
        # print(selected_mask[selected_mask==1].shape)

        pacc_labeled_all_new = pacc_fun_all_new(all_labels[selected_mask].astype(int),
                                                preds[selected_mask].astype(int))  # 有类别的
        print("cluster,all novel :", pacc_labeled_all_new)
        if self.current_task == 3:
            scores, all_outputs, all_labels = np.array([]), np.array([]), np.array([])
            for k, test_loader in enumerate(dataset.test_loaders):
                for data in test_loader:
                    inputs = data[0]
                    labels = data[1]
                    # print(labels)

                    score, dists = self.predict(inputs, categories)  # bs * 10
                    all_labels = np.append(all_labels, labels.detach().cpu().numpy())
                    scores = np.append(scores, score.detach().cpu().numpy())
            np.save("cifar_10_labels", all_labels)
            np.save("cifar_10_scores", scores)
    def reset_train_loader(self, train_loader, prev_categories):  # 初始阶段就没有这个，这个是增量阶段

        dataset = train_loader.dataset
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)  # 这个肯定也是要修改的
        prev_dists = []
        # features = []

        print('reset dataset with prev_categories', prev_categories)  # 一个列表，之前的全部类别标号
        for i, data in enumerate(loader):  # 训练集中的每个样本都和之前的类别的分类器进行一次预测，并保存结果
            input = data[0].to(self.device)

            with torch.no_grad():  # 梯度不更新
                if len(prev_categories) > 0:
                    _, prev_dist = self.predict(input, prev_categories)
                    prev_dists.append(prev_dist.detach().cpu())

        if len(prev_categories) > 0:
            prev_dists = torch.cat(prev_dists, dim=0)
            dataset.set_att("prev_dists", prev_dists)  # 重构误差

    def train_category(self, data_loader, category: int, epoch_id):  # 这个是要修改的，这个目前的都是第一阶段的训练方式，而且

        network = self.nets[category].to(self.device)
        network.train()
        optimizer = Adam(network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        avg_loss = 0.0
        sample_num = 0

        posloss_arr = []
        negloss_arr = []
        kldloss_arr = []
        maxloss_arr = []
        pseudoloss_arr = []

        temp = self.t_c_arr[self.current_task]
        categories = temp[len(temp) - 1][0]
        prev_categories = list(range(categories[0]))

        for i, data in enumerate(data_loader):  # 一个任务全部的样本，因此有正样本，有负样本
            inputs = data[0].to(self.device)
            print(inputs[:,:10])
            # print(inputs.shape)
            if self.current_task > 0:
                labels = data[3].to(self.device)
            else:
                labels = data[1].to(self.device)

            recons, _, mu, log_var = network(inputs)

            input_flat = inputs.view(inputs.shape[0], -1)
            recons_flat = recons.view(recons.shape[0], -1)
            dist = torch.sum((input_flat - recons_flat) ** 2, dim=1)

            pos_loss = self.lambda1*torch.relu(dist[labels == category] - self.r_intra)  # 类内损失
            posloss_arr.append(pos_loss.detach().cpu().data.numpy())
            pos_loss_mean = torch.mean(pos_loss)

            neg_loss = self.lambda2 * torch.relu(self.r_inter - dist[labels != category])  # 类间损失，其实就是负样本
            negloss_arr.append(neg_loss.detach().cpu().data.numpy())
            neg_loss_mean = torch.mean(neg_loss)

            mu_pos = mu[labels == category]  # VAE损失
            log_var_pos = log_var[labels == category]
            kld_loss = self.kld_ratio * -0.5 * torch.sum(1 + log_var_pos - mu_pos ** 2 - log_var_pos.exp(), dim=1)
            kldloss_arr.append(kld_loss.detach().cpu().data.numpy())
            kld_loss_mean = torch.mean(kld_loss, dim=0)

            if self.current_task > 0:
                prev_dists = data[4].to(self.device)  # 这是之前保存的样本在前面的VAE的结果
                max_scores = torch.relu(
                    dist[labels == category].view(-1, 1) - prev_dists[labels == category])  # 正样本，新类低于旧类
                max_loss = torch.sum(max_scores, dim=1) * self.lambda4 / len(prev_categories)
                maxloss_arr.append(max_loss.detach().cpu().data.numpy())
                max_loss_mean = torch.mean(max_loss)

                if self.isPseudo:
                    pseudo_input = []
                    for p_c in prev_categories:
                        p_net = self.nets[p_c]
                        p_input = p_net.sample(self.args.p)
                        pseudo_input.append(p_input)

                    pseudo_input = torch.cat(pseudo_input)
                    pseudo_recons, _, mu_p, log_var_p = network(pseudo_input)  # 旧样本在新的分类器的分数
                    pseudo_input_flat = pseudo_input.view(pseudo_input.shape[0], -1)
                    pseudo_recons_flat = pseudo_recons.view(pseudo_recons.shape[0], -1)

                    pseudo_dist = torch.sum((pseudo_input_flat - pseudo_recons_flat) ** 2, dim=1)
                    pseudo_loss = self.lambda3 * torch.relu(self.r_inter - pseudo_dist)  # 越大越好
                    pseudoloss_arr.append(pseudo_loss.detach().cpu().data.numpy())
                    pseudo_loss_mean = torch.mean(pseudo_loss)
                else:
                    pseudo_loss_mean = 0

                loss = pos_loss_mean + neg_loss_mean + kld_loss_mean + max_loss_mean + pseudo_loss_mean
                # print("task>1")

            else:
                loss = pos_loss_mean + neg_loss_mean + kld_loss_mean
                # print(loss)
                # print("task<1:",loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            sample_num += inputs.shape[0]
            # print(avg_loss)

        avg_loss /= sample_num
        # print("avg_loss:",avg_loss)
        posloss_arr = np.hstack(posloss_arr)
        negloss_arr = np.hstack(negloss_arr)
        kldloss_arr = np.hstack(kldloss_arr)
        if len(maxloss_arr) > 0:
            maxloss_arr = np.hstack(maxloss_arr)
        if len(pseudoloss_arr) > 0:
            pseudoloss_arr = np.hstack(pseudoloss_arr)
        return avg_loss, posloss_arr, negloss_arr, kldloss_arr, maxloss_arr, pseudoloss_arr

    def reset_mu(self, loader, category):
        network = self.nets[category].to(self.device)
        network.eval()

        mu_arr = []
        log_var_arr = []
        with torch.no_grad():
            for i, data in enumerate(loader):
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                recons, _, mu, log_var = network(inputs)

                mu_arr.append(mu[labels == category])
                log_var_arr.append(log_var[labels == category])

            mu_arr = torch.cat(mu_arr)
            log_var_arr = torch.cat(log_var_arr)

            mu_mean = torch.mean(mu_arr, dim=0)
            log_var_mean = torch.mean(log_var_arr, dim=0)

            self.mus[category] = mu_mean
            self.log_vars[category] = log_var_mean

    def get_score(self, dist, category):
        score = 1 / (dist + 1e-6)

        return score

    def test_task(self, dataset, test_loader):
        self.evaluate_4case(dataset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temp = self.t_c_arr[self.current_task]
        categories = temp[len(temp) - 1][0]
        categories = list(range(categories[-1] + 1))
        return self.predict(x, categories)[0]

    def predict(self, inputs: torch.Tensor, categories):
        inputs = inputs.to(self.device)
        # inputs = self.feat_net(inputs)
        outcome, dists = [], []
        with torch.no_grad():
            for i in categories:
                net = self.nets[i]  # 获取第i类的样本
                # print(net)
                net.to(self.device)
                net.eval()  # 模式

                recons, _, mu, log_var = net(inputs)  # 网络输出
                input_flat = inputs.view(inputs.shape[0], -1)
                recons_flat = recons.view(recons.shape[0], -1)
                dist = torch.sum((input_flat - recons_flat) ** 2, dim=1)

                scores = self.get_score(dist, i)  # 异常分数

                outcome.append(scores.view(-1, 1))
                dists.append(dist.view(-1, 1))

        outcome = torch.cat(outcome, dim=1)
        dists = torch.cat(dists, dim=1)
        return outcome, dists  # 返回的是一个输入在所有分类器的预测结果

    def evaluate_aoc(self, test_loaders):

        all_outputs, all_labels = [], []
        categories = list(range(self.t_c_arr[self.current_task][-1] + 1))

        for k, test_loader in enumerate(test_loaders):
            for data in test_loader:
                inputs = data[0]
                labels = data[1]

                _, dists = self.predict(inputs, categories)

                all_outputs.append(dists.detach().cpu())
                all_labels.append(labels.detach().cpu())

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0).numpy()
        for i in range(len(categories)):
            label = np.where(all_labels == i, -1, 1)
            aoc = roc_auc_score(label, all_outputs[:, i].detach().cpu().numpy())
            print('类别', i, 'AOC:', aoc)

    def compute_loss(self, x, x_hat, mu, log_var, z):
        p_c = self.cluster_net.pi_prior
        gamma = self.compute_gamma(z, p_c)
        # print(gamma[0])

        gamma_sum = torch.sum(gamma, dim=1) / gamma.shape[0]
        gamma_sum = gamma_sum / torch.sum(gamma_sum)
        loss_pi = -torch.sum(gamma_sum * torch.log(gamma_sum))
        # print(gamma_sum.shape)

        # log_p_x_given_z = F.binary_cross_entropy(x_hat, x, reduction='sum')  # 损失1，可以理解
        log_p_x_given_z = torch.mean((x_hat - x) ** 2)
        h = log_var.exp().unsqueeze(1) + (mu.unsqueeze(1) - self.cluster_net.mu_prior).pow(2)
        h = torch.sum(self.cluster_net.log_var_prior + h / self.cluster_net.log_var_prior.exp(), dim=2)
        log_p_z_given_c = 0.5 * torch.sum(gamma * h)  # 负号呢？没有2pi吗
        log_p_c = torch.sum(gamma * torch.log(p_c + 1e-9))  # 正确的
        log_q_c_given_x = torch.sum(gamma * torch.log(gamma + 1e-9))  # 正确的
        log_q_z_given_x = 0.5 * torch.sum(1 + log_var)  # 没有负号

        loss = log_p_x_given_z + 0.1 * log_p_z_given_c - log_p_c + log_q_c_given_x - log_q_z_given_x + loss_pi
        # loss = log_p_x_given_z + 0.1 * (log_p_z_given_c - log_p_c + log_q_c_given_x - log_q_z_given_x)

        loss /= x.size(0)
        return loss

    def compute_gamma(self, z, p_c):  # 完成正确，公式推导出来的
        h = (z.unsqueeze(1) - self.cluster_net.mu_prior).pow(2) / self.cluster_net.log_var_prior.exp()
        h += self.cluster_net.log_var_prior
        h += torch.Tensor([np.log(np.pi * 2)]).to(self.device)
        p_z_c = torch.exp(torch.log(p_c + 1e-9).unsqueeze(0) - 0.5 * torch.sum(h, dim=2)) + 1e-9
        gamma = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True)
        return gamma
