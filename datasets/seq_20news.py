import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from argparse import Namespace
from datasets.utils.incremental_dataset import IncrementalDataset
from datasets.utils.incremental_dataset import ILDataset, getfeature_loader
from datasets.utils.validation import get_train_val


def _prepare_twentynews_data(root):
    tnews_train = fetch_20newsgroups(data_home=root + '/twentynews/', subset='train',
                                     remove=('headers', 'footers', 'quotes'))
    tnews_test = fetch_20newsgroups(data_home=root + '/twentynews/', subset='test',
                                    remove=('headers', 'footers', 'quotes'))
    train_texts = tnews_train['data']
    train_y = tnews_train['target']
    test_texts = tnews_test['data']
    test_y = tnews_test['target']

    vectorizer = TfidfVectorizer(max_features=2000)
    train_x = vectorizer.fit_transform(train_texts).todense()
    test_x = vectorizer.transform(test_texts).todense()

    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y)
    test_x = torch.from_numpy(test_x).float()
    test_y = torch.from_numpy(test_y)

    return train_x, train_y, test_x, test_y


class Sequential20NEWS(IncrementalDataset):
    NAME = 'seq-20news'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 10
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.nc = 20
        self.nt = 10
        super(Sequential20NEWS, self).__init__(args)

        self.normalization_transform = None
        self.dnormalization_transform = None
        self.train_transform = None
        self.test_transform = None

        self._train_x, self._train_y, self._test_x, self._test_y = _prepare_twentynews_data(args.root)

    def get_data_loaders(self):
        train_dataset = ILDataset(self._train_x, self._train_y)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        self.test_transform, self.NAME)
        else:
            test_dataset = ILDataset(self._test_x, self._test_y)

        train, test = getfeature_loader(train_dataset, test_dataset, setting=self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        pass

    def get_transform(self):
        return None
