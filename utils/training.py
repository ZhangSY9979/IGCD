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


def train_il(args: Namespace) -> None:
    print(args)
    for run_id in range(args.repeat):
        print('================================= repeat {}/{} ================================='
              .format(run_id+1, args.repeat, ), file=sys.stderr)

        if args.seed is not None:
            set_random_seed(args.seed)
        if not os.path.exists(args.img_dir):
            os.makedirs(args.img_dir)

        dataset = get_dataset(args)
        model = get_il_model(args)
        train_loader, test_loader = dataset.get_data_loaders()
        model.train_extra(dataset, train_loader)

        args.featureNet = "fine-tune"
        dataset = get_dataset(args)
        model.begin_il(dataset)
        for t in range(dataset.nt):

            train_loader, test_loader = dataset.get_data_loaders()  # 不不，这是根据要求变化的
            if t == 0:
                model.train_first(dataset, train_loader)
                model.test_cluster(dataset, test_loader)
            else:
                model.train_second(dataset, train_loader)
                model.test_cluster(dataset, test_loader)
        model.end_il(dataset)

