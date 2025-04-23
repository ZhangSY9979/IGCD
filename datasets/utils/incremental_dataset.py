import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import datasets
from abc import abstractmethod
from argparse import Namespace
from typing import Tuple
import numpy as np
from pathlib import Path
import random


class IncrementalDataset:
    NAME = None
    SETTING = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0  # current task
        self.args = args
        self.nt = args.nt if args.nt else self.nt  # number of tasks
        self.nc = self.nc  # number of classes
        t_c_arr = args.t_c_arr
        # relationship between tasks and classes
        # t_c_arr = args.t_c_arr if args.t_c_arr else self.get_balance_classes()
        # if self.args.task_shuffle:
        #     random.shuffle(t_c_arr)
        self.t_c_arr = t_c_arr

    def get_balance_classes(self):
        class_arr = list(range(self.nc))
        if self.args.class_shuffle:
            random.shuffle(class_arr)
        cpt = self.nc // self.nt

        order = [class_arr[i:i + cpt] for i in range(0, len(class_arr), cpt)]
        for cls in order:
            cls.sort()
        return order

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @abstractmethod
    def not_aug_dataloader(self, batch_size: int) -> DataLoader:
        """
        Returns the dataloader of the current task,
        not applying data augmentation.
        :param batch_size: the batch size of the loader
        :return: the current training loader
        """
        pass


def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                         setting: IncrementalDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.  # 根据类别号对数据进行分类
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_mask = np.zeros_like(np.array(train_dataset.targets)) == 1
    test_mask = np.zeros_like(np.array(test_dataset.targets)) == 1
    c_arr_all = setting.t_c_arr[setting.i]  #
    for x in range(len(c_arr_all)):
        c_arr, partition, _ = c_arr_all[x]
        for cat in c_arr:
            temp = np.array(train_dataset.targets) == cat
            num = np.size(temp[temp == True])
            # print("num:",num)
            lower = int(num * partition[0])
            upper = int(num * partition[1])
            cnt = 0
            for p in range(np.size(temp)):
                if temp[p]:
                    if cnt < lower or cnt >= upper:
                        temp[p] = False
                    cnt += 1
                if cnt == num:
                    break
            train_mask = np.logical_or(train_mask, temp)
    c_arr = c_arr_all[len(c_arr_all) - 1][0]
    for cat in c_arr:
        test_mask = np.logical_or(test_mask,
                                  np.array(test_dataset.targets) == cat)
    # print(np.size(train_mask[train_mask == True]))
    # print(np.size(test_mask[test_mask == True]))
    if setting.NAME == 'seq-imagenet':
        # ImageNet need
        def convert_data(data, mask):
            converted = []
            for i, keep in enumerate(list(mask)):
                if keep:
                    converted.append(data[i])
            return converted

        train_dataset.data = convert_data(train_dataset.data, train_mask)
        test_dataset.data = convert_data(test_dataset.data, test_mask)

        train_dataset.targets = convert_data(train_dataset.targets, train_mask)
        test_dataset.targets = convert_data(test_dataset.targets, test_mask)
    else:
        train_dataset.data = train_dataset.data[train_mask]
        test_dataset.data = test_dataset.data[test_mask]

        train_dataset.targets = np.array(train_dataset.targets)[train_mask]
        test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=8)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += 1
    return train_loader, test_loader


def get_previous_train_loader(train_dataset: datasets, batch_size: int,
                              setting: IncrementalDataset) -> DataLoader:  # 获得之前的训练集
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
                                setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets)
                                < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def get_not_train_dataset(train_dataset: datasets, batch_size: int,
                              setting: IncrementalDataset) -> DataLoader:  # 获得之前的训练集
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.zeros_like(np.array(train_dataset.targets)) == 1
    c_arr_all = setting.t_c_arr[setting.i]  #
    for x in range(len(c_arr_all)):
        c_arr, partition, _ = c_arr_all[x]
        for cat in c_arr:
            temp = np.array(train_dataset.targets) == cat
            num = np.size(temp[temp == True])
            # print("num:",num)
            lower = int(num * partition[0])
            upper = int(num * partition[1])
            cnt = 0
            for p in range(np.size(temp)):
                if temp[p]:
                    if cnt < lower or cnt > upper:
                        temp[p] = False
                    cnt += 1
                if cnt == num:
                    break
            train_mask = np.logical_or(train_mask, temp)

    if setting.NAME == 'seq-imagenet':
        # ImageNet need
        def convert_data(data, mask):
            converted = []
            for i, keep in enumerate(list(mask)):
                if keep:
                    converted.append(data[i])
            return converted

        train_dataset.data = convert_data(train_dataset.data, train_mask)

        train_dataset.targets = convert_data(train_dataset.targets, train_mask)
    else:
        train_dataset.data = train_dataset.data[train_mask]
        train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    return train_dataset


class ILDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.attributes = []
        self.trans = []
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def set_att(self, att_name, att_data, att_transform=None):
        self.attributes.append(att_name)
        self.trans.append(att_transform)
        setattr(self, att_name, att_data)

    def get_att_names(self):
        return self.attributes

    def __getitem__(self, index):
        x_data = self.data[index]
        target_data = self.targets[index]
        if self.transform:
            x_data = self.transform(x_data)
        if self.target_transform:
            target_data = self.target_transform(target_data)
        ret_tuple = (x_data, target_data, x_data)
        for i, att in enumerate(self.attributes):
            att_data = getattr(self, att)[index]
            trans = self.trans[i]
            if trans:
                att_data = trans(att_data)

            ret_tuple += (att_data,)
        return ret_tuple


def getfeature_loader(train_dataset: datasets, test_dataset: datasets, setting: IncrementalDataset,train_transforms,test_transforms):
    if setting.args.featureNet:
        my_file = Path(
            setting.args.root + "/" + setting.args.dataset + "-" + setting.args.featureNet + "-train-data.npy")
        if my_file.exists():
            print("feature already extracted")
            train_data = np.load(
                setting.args.root + "/" + setting.args.dataset + "-" + setting.args.featureNet + "-train-data.npy",
                allow_pickle=True)
            train_label = np.load(
                setting.args.root + "/" + setting.args.dataset + "-" + setting.args.featureNet + "-train-label.npy",
                allow_pickle=True)
            test_data = np.load(
                setting.args.root + "/" + setting.args.dataset + "-" + setting.args.featureNet + "-test-data.npy",
                allow_pickle=True)
            test_label = np.load(
                setting.args.root + "/" + setting.args.dataset + "-" + setting.args.featureNet + "-test-label.npy",
                allow_pickle=True)
        else:
            print("feature file not found !!  extracting feature ...")
            train_data, train_label = get_feature_by_extractor(train_dataset, setting.extractor, setting)
            test_data, test_label = get_feature_by_extractor(test_dataset, setting.extractor, setting)

            np.save(setting.args.root + "/" + setting.args.dataset + "-" + setting.args.featureNet + "-train-data.npy",
                    train_data, allow_pickle=True)
            np.save(setting.args.root + "/" + setting.args.dataset + "-" + setting.args.featureNet + "-train-label.npy",
                    train_label, allow_pickle=True)
            np.save(setting.args.root + "/" + setting.args.dataset + "-" + setting.args.featureNet + "-test-data.npy",
                    test_data, allow_pickle=True)
            np.save(setting.args.root + "/" + setting.args.dataset + "-" + setting.args.featureNet + "-test-label.npy",
                    test_label, allow_pickle=True)

        # train_dataset = ILDataset(train_data, train_label,transform=train_transforms)
        # test_dataset = ILDataset(test_data, test_label,transform=test_transforms)
        train_dataset = ILDataset(train_data, train_label)
        test_dataset = ILDataset(test_data, test_label)

    train, test = store_masked_loaders(train_dataset, test_dataset, setting=setting)

    return train, test


def get_feature_by_extractor(train_dataset: datasets, extractor, setting: IncrementalDataset):
    extractor = extractor.to(setting.args.device).eval()
    train_loader = DataLoader(train_dataset,
                              batch_size=128, shuffle=False, num_workers=0)
    features, labels = [], []
    for data in train_loader:
        # print(data)
        img = data[0]
        label = data[1]
        img = img.to(setting.args.device)
        with torch.no_grad():
            feature = extractor(img)

        feature = feature.to('cpu')
        img = img.to('cpu')

        features.append(feature)
        labels.append(label)

    feature = torch.cat(features).numpy()
    label = torch.cat(labels).numpy()

    return feature, label


def get_feature_extractor(args):
    extractor = None
    if args.featureNet == 'resnet18':
        extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif args.featureNet == 'vgg11':
        extractor = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
    elif args.featureNet == 'swint':
        extractor = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    elif args.featureNet == 'vitb16':
        extractor = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    elif args.featureNet == "fine-tune":
        extractor = torch.load("/home/zhangsiyu/data/extra/"+args.extra_name+".pt")
    return extractor
