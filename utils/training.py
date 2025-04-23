import torch
import numpy as np
import os
import sys
import time

from models.utils import get_il_model
from utils.conf import set_random_seed
from argparse import Namespace
from models.utils.incremental_model import IncrementalModel
from datasets.utils.incremental_dataset import IncrementalDataset
from typing import Tuple
from datasets import get_dataset
from backbone.VaDE import VaDE


def mask_classes(outputs: torch.Tensor, dataset: IncrementalDataset, k: int) -> None:
    cats = dataset.t_c_arr[k]
    outputs[:, 0:cats[0]] = -float('inf')
    outputs[:, cats[-1] + 1:] = -float('inf')


def evaluate(model: IncrementalModel, dataset: IncrementalDataset, last=False):
    accs_taskil, accs_classil, acc_taskil_pertask, acc_classil_pertask = [], [], [], []
    correct, correct_mask_classes, total = 0.0, 0.0, 0.0
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue

        correct_k, correct_mask_classes_k, total_k = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs = data[0]
            labels = data[1]
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                outputs = model(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            correct_k += torch.sum(pred == labels).item()
            total_k += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
                correct_mask_classes_k += torch.sum(pred == labels).item()

        acc_classil_pertask.append(correct_k / total_k * 100)
        acc_taskil_pertask.append(correct_mask_classes_k / total_k * 100)

    accs_classil.append(correct / total * 100)
    accs_taskil.append(correct_mask_classes / total * 100)

    return accs_classil, accs_taskil, acc_classil_pertask, acc_taskil_pertask

def forward_transfer(results, random_results):
    n_tasks = len(results)
    li = []
    for i in range(1, n_tasks):
        li.append(results[i - 1][i] - random_results[i])

    return np.mean(li)


def backward_transfer(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return round(np.mean(li), 2)


def forgetting(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return round(np.mean(li), 2)


def train_il(args: Namespace) -> None:
    print(args)

    final_acc_class, final_acc_task = [], []
    final_bwt_class, final_bwt_task = [], []
    for run_id in range(args.repeat):
        print('================================= repeat {}/{} ================================='
              .format(run_id+1, args.repeat, ), file=sys.stderr)

        if args.seed is not None:
            set_random_seed(args.seed)
        if not os.path.exists(args.img_dir):
            os.makedirs(args.img_dir)

        dataset = get_dataset(args)
        model = get_il_model(args)

        nt = dataset.nt
        acc_track = []
        for i in range(nt):
            acc_track.append([0.0])

        model.begin_il(dataset)
        for t in range(dataset.nt):

            train_loader, test_loader = dataset.get_data_loaders()  # 不不，这是根据要求变化的
            if t == 0:
                start_time = time.time()
                model.train_first(dataset, train_loader)
                # 这里进行数据集的更新，# 添加一步重新处理数据集的操作------dataset更新，主要是为了方便训练
                train_time = time.time() - start_time

                model.test_first(dataset, test_loader)
                model.test_cluster(dataset, test_loader)


            else:
                start_time = time.time()
                model.train_second(dataset, train_loader)
                train_time = time.time() - start_time

                model.test_second(dataset, test_loader)
                # model.end_il(dataset)
                model.test_cluster(dataset, test_loader)
        model.end_il(dataset)


def train_il_faster(args: Namespace) -> None:
    print(args)

    final_acc_class, final_acc_task = [], []
    final_bwt_class, final_bwt_task = [], []
    for run_id in range(args.repeat):
        print('================================= repeat {}/{} ================================='
              .format(run_id+1, args.repeat, ), file=sys.stderr)

        if args.seed is not None:
            set_random_seed(args.seed)
        if not os.path.exists(args.img_dir):
            os.makedirs(args.img_dir)

        dataset = get_dataset(args)
        model = torch.load("./data/gcdvaefirst_state.pt")

        nt = dataset.nt
        acc_track = []
        for i in range(nt):
            acc_track.append([0.0])

        # model.begin_il(dataset)
        # model.args.VaDE_epochs = 100
        model.bate.requires_grad = False
        # model.args.pretrain_lr = 0.001
        # model.args.VaDE_lr = 1e-8
        # model.args.VaDE_epochs = 50
        # model.cluster_net = VaDE(1000, [800, 500], 50, 10)
        # model.args.ratio_bate = 0.2
        # model.bs = 32
        # model.args.n_epochs = 1
        # model.args.patience = 50
        # model.args.pretrain = True
        model.args.pretrained_path = 'weights/pretrained_parameter_softmax.pth'

        for t in range(dataset.nt):
            train_loader, test_loader = dataset.get_data_loaders()  # 不不，这是根据要求变化的
            if t == 0:
                # for i in range(70):
                #     model.bate[i] = model.set_score(train_loader, i, 0.3)
                continue
                # start_time = time.time()
                # model.train_first(dataset, train_loader)
                # train_time = time.time() - start_time
                #
                # model.test_first(dataset, test_loader)
                # model.test_cluster(dataset, test_loader)
            else:
                # model.bate.requires_grad = False
                start_time = time.time()
                # model.get_new_labels(test_loader,1,70)
                model.train_second(dataset, train_loader)
                train_time = time.time() - start_time

                model.test_second(dataset, test_loader)
                model.test_cluster(dataset, test_loader)
                # model.bate.requires_grad = False
        model.end_il(dataset)
def train_il_faster_kmeans(args: Namespace) -> None:
    print(args)

    final_acc_class, final_acc_task = [], []
    final_bwt_class, final_bwt_task = [], []
    for run_id in range(args.repeat):
        print('================================= repeat {}/{} ================================='
              .format(run_id+1, args.repeat, ), file=sys.stderr)

        if args.seed is not None:
            set_random_seed(args.seed)
        if not os.path.exists(args.img_dir):
            os.makedirs(args.img_dir)

        dataset = get_dataset(args)
        model = torch.load("./data/good/gcdvaecifar10first_state.pt")

        nt = dataset.nt
        acc_track = []
        for i in range(nt):
            acc_track.append([0.0])

        # model.begin_il(dataset)
        # model.args.VaDE_epochs = 300
        # model.bate.requires_grad = False
        # print(model.args)
        # print(model.bs)
        # model.args.ratio1 = 0.35
        # model.args.n_epochs = 100
        # model.bs = 128
        model.bate.requires_grad = False
        # model.args.pretrain_lr = 0.001
        # model.args.VaDE_lr = 1e-8
        model.args.pretrain_epochs = 50
        # model.cluster_net = VaDE(1000, [800, 500], 50, 10)
        # model.args.ratio_bate = 0.2
        # model.bs = 32
        # model.args.n_epochs = 1
        # model.args.patience = 50
        # model.args.pretrain = True
        model.args.pretrained_path = 'weights/pretrained_parameter_softmax_cifar10_faster.pth'

        for t in range(dataset.nt):
            train_loader, test_loader = dataset.get_data_loaders()  # 不不，这是根据要求变化的
            if t == 0:
                # print(model.bate)
                # for i in range(70):
                #     model.bate[i] = model.set_score(train_loader, i, 0.02)
                continue
                # start_time = time.time()
                # model.train_first(dataset, train_loader)
                # train_time = time.time() - start_time
                #
                # model.test_first(dataset, test_loader)
                # model.test_cluster(dataset, test_loader)
            else:
                # model.bate.requires_grad = False
                start_time = time.time()
                model.train_second(dataset, train_loader)
                train_time = time.time() - start_time

                model.test_second(dataset, test_loader)
                model.test_cluster(dataset, test_loader)
                # model.bate.requires_grad = False
        model.end_il(dataset)
def train_il_faster_kmeans_new(args: Namespace) -> None:
    print(args)

    final_acc_class, final_acc_task = [], []
    final_bwt_class, final_bwt_task = [], []
    for run_id in range(args.repeat):
        print('================================= repeat {}/{} ================================='
              .format(run_id+1, args.repeat, ), file=sys.stderr)

        if args.seed is not None:
            set_random_seed(args.seed)
        if not os.path.exists(args.img_dir):
            os.makedirs(args.img_dir)

        dataset = get_dataset(args)
        model = torch.load("./data/gcdnew1first_state.pt")

        nt = dataset.nt
        acc_track = []
        for i in range(nt):
            acc_track.append([0.0])

        # model.begin_il(dataset)
        # model.args.VaDE_epochs = 300
        model.bate.requires_grad = False
        # print(model.args)
        # print(model.bs)
        # model.args.ratio1 = 0.35
        model.args.n_epochs_again = 100
        # model.bs = 128

        for t in range(dataset.nt):
            train_loader, test_loader = dataset.get_data_loaders()  # 不不，这是根据要求变化的
            if t == 0:
                # print(model.bate)
                # for i in range(70):
                #     model.bate[i] = model.set_score(train_loader, i, 0.05)
                continue
                # start_time = time.time()
                # model.train_first(dataset, train_loader)
                # train_time = time.time() - start_time
                #
                # model.test_first(dataset, test_loader)
                # model.test_cluster(dataset, test_loader)
            else:
                # model.bate.requires_grad = False
                start_time = time.time()
                model.train_second(dataset, train_loader)
                train_time = time.time() - start_time

                model.test_second(dataset, test_loader)
                model.test_cluster(dataset, test_loader)
                # model.bate.requires_grad = False
        model.end_il(dataset)


def print_accuracy(accs_classil, accs_taskil, task_number: int) -> None:
    mean_acc_class_il = np.mean(accs_classil)
    mean_acc_task_il = np.mean(accs_taskil)
    print('Accuracy for {} task(s): \t [Class-IL]: {} %'
          ' \t [Task-IL]: {} %'.format(task_number, round(
        mean_acc_class_il, 2), round(mean_acc_task_il, 2), file=sys.stderr))
