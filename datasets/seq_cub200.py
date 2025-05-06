import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.incremental_dataset import getfeature_loader, IncrementalDataset, get_feature_extractor
from datasets.utils.incremental_dataset import get_not_train_dataset
from argparse import Namespace
from datasets.transforms.denormalization import DeNormalize
from models.utils.putil import make_transform


def ori_transform():
    resnet_sz_resize = 256
    resnet_sz_crop = 224
    resnet_mean = [0.485, 0.456, 0.406]
    resnet_std = [0.229, 0.224, 0.225]
    resnet_transform = transforms.Compose([
        transforms.Resize(resnet_sz_resize),
        transforms.CenterCrop(resnet_sz_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=resnet_mean, std=resnet_std)
    ])

    return resnet_transform


class MyDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.data = []  # 修改的，不如初始化的时候就全部都读进来分成数据和label????
        self.targets = []
        fh = open(root, 'r')
        for line in fh:
            line = line.rstrip()
            words = line.split()
            # imgs.append((words[0], int(words[1])))
            # img = Image.open(words[0]).convert('RGB')
            self.data.append((words[0]))
            self.targets.append(int(words[1]))
        # self.imgs = imgs
        # self.data = np.vstack(self.data)
        self.transform = transform
        self.target_transform = target_transform
        self.not_aug_transform = ori_transform()
        self.attributes = []
        self.trans = []

    def set_att(self, att_name, att_data, att_transform=None):
        self.attributes.append(att_name)
        self.trans.append(att_transform)
        setattr(self, att_name, att_data)

    def get_att_names(self):
        return self.attributes

    def __getitem__(self, index):
        fn, target = self.data[index], self.targets[index]
        img = Image.open(fn).convert('RGB')
        # img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()
        not_aug_img = self.not_aug_transform(original_img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        ret_tuple = (img, target, not_aug_img)
        for i, att in enumerate(self.attributes):
            att_data = getattr(self, att)[index]

            trans = self.trans[i]
            if trans:
                att_data = trans(att_data)

            ret_tuple += (att_data,)

        return ret_tuple

    def __len__(self):
        return len(self.imgs)


class SequentialCUB200(IncrementalDataset):
    NAME = 'seq-cub200'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 5
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.nc = 100
        self.nt = 10
        self.n_channel = 3
        self.n_imsize1 = 32
        self.n_imsize2 = 32
        super(SequentialCUB200, self).__init__(args)

        if self.args.featureNet:
            self.args.transform = 'pytorch'
            self.extractor = get_feature_extractor(args)

        if self.args.transform == 'pytorch':
            self.normalization_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.dnormalization_transform = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalization_transform])
            self.test_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                self.normalization_transform])
        else:
            self.normalization_transform = None
            self.dnormalization_transform = None
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
            self.test_transform = transforms.Compose([transforms.ToTensor()])

    def get_data_loaders(self):

        train_dataset = MyDataset(self.args.root + 'train.txt', train=True, transform=make_transform(True))
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        self.test_transform, self.NAME)
        else:
            test_dataset = MyDataset(self.args.root + 'test.txt', train=False, transform=make_transform(False))

        train, test = getfeature_loader(train_dataset, test_dataset, setting=self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        train_dataset = MyDataset(self.args.root + 'CIFAR10', train=True, transform=make_transform(False))
        train_dataset = get_not_train_dataset(train_dataset, batch_size, self)

        return train_dataset

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.train_transform])
        return transform
