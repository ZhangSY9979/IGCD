from copy import copy
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
from torch.optim import SGD, Adam
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from backbone.VAE import VAE
from backbone.VAE_MLP import VAE_MLP
import torchvision.models as models
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


class CIFAR10(IncrementalModel):
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, args):
        super(CIFAR10, self).__init__(args)

        self.nets = []

        self.lambda2 = self.args.lambda2
        self.lambda3 = self.args.lambda3
        self.lambda4 = self.args.lambda4
        self.eps = self.args.eps
        self.embedding_dim = self.args.embedding_dim
        self.weight_decay = self.args.weight_decay
        self.lambda1 = self.args.lambda1
        self.r_inter = self.args.r_inter
        self.r_intra = self.args.r_intra
        self.kld_ratio = self.args.kld_ratio
        self.isPseudo = self.args.isPseudo

        self.bs = self.args.batch_size

        self.current_task = -1
        self.nc = None
        self.t_c_arr = []
        self.nf = self.args.nf
        self.isPrint = self.args.isPrint
        self.bate = torch.zeros(10).to(self.device)

        self.mus = []
        self.log_vars = []
        self.thresholds = []
        self.cluster_net = None
        self.extra = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device)
        self.classify = nn.Linear(1000, 4).to(self.device)


    def begin_il(self, dataset):
        self.nc = dataset.nc
        self.t_c_arr = dataset.t_c_arr
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

            self.nets.append(
                net
            )
            self.mus.append(None)
            self.log_vars.append(None)
            self.thresholds.append(None)


    def merge_model(self, model, original_model):
        alpha = 0.99
        merged_model = deepcopy(original_model)
        for (name_old, param_old), (name_new, param_new), (name_m, param_m) in zip(
                original_model.named_parameters(), model.named_parameters(), merged_model.named_parameters()
        ):
            assert name_old == name_new == name_m

            if name_m.startswith("layer4") and "bn" not in name_m.lower():
                param_m.data.copy_(alpha * param_new.data + (1 - alpha) * param_old.data)

        return merged_model

    def train_extra(self, dataset, train_loader):
        criterion = nn.CrossEntropyLoss()
        dataset1 = train_loader.dataset
        loader = DataLoader(dataset1, batch_size=128, shuffle=True)  # 这个肯定也是要修改的
        original_state = deepcopy(self.extra)
        for param in self.extra.parameters():
            param.requires_grad = False
        for param in self.extra.layer4.parameters():
            param.requires_grad = True
        optimizer = SGD([{'params': self.classify.parameters(), 'lr': 1e-5},
            {'params': self.extra.layer4.parameters(), 'lr': 1e-6}], weight_decay=0)

        # 训练模型
        for epoch in range(10):
            self.extra.train()
            running_loss = 0.0
            for i, data in enumerate(loader):
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                optimizer.zero_grad()
                outputs = self.extra(inputs)
                outputs_final = self.classify(outputs)
                loss = criterion(outputs_final, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.extra.parameters(), max_norm=1e-4)
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.args.n_epochs}, Loss: {running_loss / len(loader)}")

        self.extra = self.merge_model(self.extra, original_state)

        torch.save(self.extra , "/home/zhangsiyu/data/extra/"+self.args.extra_name+".pt")

    def train_first(self, dataset, train_loader):
        self.current_task += 1
        temp = self.t_c_arr[self.current_task]
        categories = temp[len(temp) - 1][0]
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


    def set_score(self, train_loader, c_index, ratio):
        network = self.nets[c_index].to(self.device)
        network.eval()
        ans = torch.tensor([]).to(self.device)
        for i, data in enumerate(train_loader):
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
        ans, _ = torch.sort(ans)
        res = copy(ans[int(ratio * len(ans))])
        return res

    def test_first(self, dataset, test_loaders):
        all_outputs, all_labels = [], []
        preds = []
        temp = self.t_c_arr[0]
        categories = temp[len(temp) - 1][0]
        samples = 0
        correct = 0
        for k, test_loader in enumerate(dataset.test_loaders):
            print(test_loader.dataset.data.shape)
            for data in test_loader:
                inputs = data[0]
                labels = data[1]

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
                        continue
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0).numpy()
        for i in range(len(categories)):
            label = np.where(all_labels == i, -1, 1)
            aoc = roc_auc_score(label, all_outputs[:, i].detach().cpu().numpy())
            print('类别', i, 'AOC:', aoc)
        print("first stage all_acc:", correct / samples)

    def train_second(self, dataset, train_loader):
        self.current_task += 1
        temp = self.t_c_arr[self.current_task]
        categories = temp[len(temp) - 1][0]
        prev_categories = list(range(categories[0]))
        self.distinguish(train_loader, prev_categories)

        k = len(categories)
        start_index = categories[0]
        self.get_new_labels(train_loader, k, start_index)

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

    def distinguish(self, train_loader, prev_categories, cnt=0):
        dataset = train_loader.dataset
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)
        new_dist = []
        cnt_new_save = 0
        cnt_old_save = 0
        print('reset dataset with prev_categories', prev_categories)
        for i, data in enumerate(loader):
            input = data[0].to(self.device)
            target = data[1].to(self.device)

            with torch.no_grad():
                out, pred = self.predict(input, prev_categories)
                _, pred = torch.max(out, 1)
                for j in range(input.shape[0]):
                    temp = out[j]
                    if len(temp[temp < self.bate[:len(temp)]]) >= (
                            len(prev_categories) * self.args.ratio - self.args.num):
                        new_dist.append(i * self.bs + j)
                        if target[j] < len(prev_categories):
                            cnt_old_save += 1
                        else:
                            cnt_new_save += 1

        dataset.data = dataset.data[new_dist]
        dataset.targets = np.array(dataset.targets)[new_dist]
        # print("old:", cnt_old_save)
        # print("new:", cnt_new_save)

    def get_new_labels(self, train_loader, k, start_index):
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
            temp_model.encoder[0][0].weight.data.copy_(vade.VaDE.fc1.weight.data)
            temp_model.encoder[0][0].bias.data.copy_(vade.VaDE.fc1.bias.data)
            temp_model.encoder[1][0].weight.data.copy_(vade.VaDE.fc2.weight.data)
            temp_model.encoder[1][0].bias.data.copy_(vade.VaDE.fc2.bias.data)

            temp_model.decoder[1][0].weight.data.copy_(vade.VaDE.fc6.weight.data)
            temp_model.decoder[1][0].bias.data.copy_(vade.VaDE.fc6.bias.data)
            temp_model.final_layer.weight.data.copy_(vade.VaDE.fc7.weight.data)
            temp_model.final_layer.bias.data.copy_(vade.VaDE.fc7.bias.data)


    def test_cluster(self, dataset, test_loader):
        temp = self.t_c_arr[self.current_task]
        categories = list(range(temp[len(temp) - 1][0][-1] + 1))
        print("stage:", self.current_task)
        preds, all_outputs, all_labels = np.array([]), np.array([]), np.array([])
        for k, test_loader in enumerate(dataset.test_loaders):
            for data in test_loader:
                inputs = data[0]
                labels = data[1]
                # print(labels)

                scores, dists = self.predict(inputs, categories)
                _, pred = torch.max(scores, 1)
                all_labels = np.append(all_labels, labels.detach().cpu().numpy())
                preds = np.append(preds, pred.detach().cpu().numpy())
        proj_all_new = cluster_pred_2_gt(preds.astype(int), all_labels.astype(int))
        pacc_fun_all_new = partial(pred_2_gt_proj_acc, proj_all_new)
        labeled_class = self.t_c_arr[0][0][0][-1]
        selected_mask = all_labels <= labeled_class

        pacc_labeled_all_new = pacc_fun_all_new(all_labels[selected_mask].astype(int),
                                                preds[selected_mask].astype(int))
        print("cluster,task 0 :", pacc_labeled_all_new)
        for i in range(self.current_task):
            lower = temp[i + 1][0][0]
            upper = temp[i + 1][0][-1]
            print(lower, "  ", upper)
            selected_mask = (all_labels >= lower) * (all_labels <= upper)
            pacc_all_new = pacc_fun_all_new(all_labels[selected_mask].astype(int), preds[selected_mask].astype(int))

        selected_mask = all_labels > labeled_class

        pacc_labeled_all_new = pacc_fun_all_new(all_labels[selected_mask].astype(int),
                                                preds[selected_mask].astype(int))
        print("cluster,all novel :", pacc_labeled_all_new)

    def reset_train_loader(self, train_loader, prev_categories):

        dataset = train_loader.dataset
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)
        prev_dists = []

        print('reset dataset with prev_categories', prev_categories)
        for i, data in enumerate(loader):
            input = data[0].to(self.device)

            with torch.no_grad():
                if len(prev_categories) > 0:
                    _, prev_dist = self.predict(input, prev_categories)
                    prev_dists.append(prev_dist.detach().cpu())

        if len(prev_categories) > 0:
            prev_dists = torch.cat(prev_dists, dim=0)
            dataset.set_att("prev_dists", prev_dists)

    def train_category(self, data_loader, category: int, epoch_id):

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

        for i, data in enumerate(data_loader):
            inputs = data[0].to(self.device)
            if self.current_task > 0:
                labels = data[3].to(self.device)
            else:
                labels = data[1].to(self.device)

            recons, _, mu, log_var = network(inputs)

            input_flat = inputs.view(inputs.shape[0], -1)
            recons_flat = recons.view(recons.shape[0], -1)
            dist = torch.sum((input_flat - recons_flat) ** 2, dim=1)

            pos_loss = self.lambda1*torch.relu(dist[labels == category] - self.r_intra)
            posloss_arr.append(pos_loss.detach().cpu().data.numpy())
            pos_loss_mean = torch.mean(pos_loss)

            neg_loss = self.lambda2 * torch.relu(self.r_inter - dist[labels != category])
            negloss_arr.append(neg_loss.detach().cpu().data.numpy())
            neg_loss_mean = torch.mean(neg_loss)

            mu_pos = mu[labels == category]
            log_var_pos = log_var[labels == category]
            kld_loss = self.kld_ratio * -0.5 * torch.sum(1 + log_var_pos - mu_pos ** 2 - log_var_pos.exp(), dim=1)
            kldloss_arr.append(kld_loss.detach().cpu().data.numpy())
            kld_loss_mean = torch.mean(kld_loss, dim=0)

            if self.current_task > 0:
                prev_dists = data[4].to(self.device)
                max_scores = torch.relu(
                    dist[labels == category].view(-1, 1) - prev_dists[labels == category])
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
                    pseudo_recons, _, mu_p, log_var_p = network(pseudo_input)
                    pseudo_input_flat = pseudo_input.view(pseudo_input.shape[0], -1)
                    pseudo_recons_flat = pseudo_recons.view(pseudo_recons.shape[0], -1)

                    pseudo_dist = torch.sum((pseudo_input_flat - pseudo_recons_flat) ** 2, dim=1)
                    pseudo_loss = self.lambda3 * torch.relu(self.r_inter - pseudo_dist)
                    pseudoloss_arr.append(pseudo_loss.detach().cpu().data.numpy())
                    pseudo_loss_mean = torch.mean(pseudo_loss)
                else:
                    pseudo_loss_mean = 0

                loss = pos_loss_mean + neg_loss_mean + kld_loss_mean + max_loss_mean + pseudo_loss_mean


            else:
                loss = pos_loss_mean + neg_loss_mean + kld_loss_mean


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            sample_num += inputs.shape[0]


        avg_loss /= sample_num
        posloss_arr = np.hstack(posloss_arr)
        negloss_arr = np.hstack(negloss_arr)
        kldloss_arr = np.hstack(kldloss_arr)
        if len(maxloss_arr) > 0:
            maxloss_arr = np.hstack(maxloss_arr)
        if len(pseudoloss_arr) > 0:
            pseudoloss_arr = np.hstack(pseudoloss_arr)
        return avg_loss, posloss_arr, negloss_arr, kldloss_arr, maxloss_arr, pseudoloss_arr

    def get_score(self, dist, category):
        score = 1 / (dist + 1e-6)

        return score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temp = self.t_c_arr[self.current_task]
        categories = temp[len(temp) - 1][0]
        categories = list(range(categories[-1] + 1))
        return self.predict(x, categories)[0]

    def predict(self, inputs: torch.Tensor, categories):
        inputs = inputs.to(self.device)
        outcome, dists = [], []
        with torch.no_grad():
            for i in categories:
                net = self.nets[i]

                net.to(self.device)
                net.eval()

                recons, _, mu, log_var = net(inputs)
                input_flat = inputs.view(inputs.shape[0], -1)
                recons_flat = recons.view(recons.shape[0], -1)
                dist = torch.sum((input_flat - recons_flat) ** 2, dim=1)

                scores = self.get_score(dist, i)

                outcome.append(scores.view(-1, 1))
                dists.append(dist.view(-1, 1))

        outcome = torch.cat(outcome, dim=1)
        dists = torch.cat(dists, dim=1)
        return outcome, dists




