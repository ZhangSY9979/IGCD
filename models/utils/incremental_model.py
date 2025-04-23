import torch
import torch.nn as nn
from argparse import Namespace

from backbone.MNISTMLP import MNISTMLP
from backbone.ResNet import resnet18, resnet34


class IncrementalModel(nn.Module):
    """
    Incremental learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, args: Namespace) -> None:
        super(IncrementalModel, self).__init__()
        self.args = args
        self.device = self.args.device
        self.n_epochs = self.args.n_epochs
        self.batch_size = self.args.batch_size
        self.learning_rate = self.args.lr
        self.lr = self.args.lr
        assert self.n_epochs is not None, 'no n_epochs'
        assert self.batch_size is not None, 'no batch_size'
        assert self.lr is not None, 'no lr'

        self.net = None

    def set_model(self, dataset):
        if self.args.dataset == 'seq-mnist':
            self.net = MNISTMLP(28 * 28, dataset.nc).to(self.device)
        else:
            if self.args.featureNet:
                self.net = MNISTMLP(1000, dataset.nc, hidden_dim=[800, 500]).to(self.device)
            elif self.args.backbone == 'None' or self.args.backbone == 'resnet18':
                self.net = resnet18(dataset.nc).to(self.device)
            elif self.args.backbone == 'resnet34':
                self.net = resnet34(dataset.nc).to(self.device)

    # 生命周期函数
    def begin_il(self, dataset):
        self.set_model(dataset)

    def begin_task(self, dataset, train_loader):
        pass

    # 生命周期函数
    def train_task(self, dataset, train_loader):
        self.begin_task(dataset, train_loader)

        for epoch_id in range(self.args.n_epochs):
            total_loss, batch_num = 0.0, 0
            for batch_id, data in enumerate(train_loader):
                loss = self.observe(data, epoch_id, batch_id)
                total_loss += loss
                batch_num += 1
            avg_loss = total_loss / batch_num
            if epoch_id % self.args.print_freq == 0:
                print('epoch:%d, loss:%.5f' % (epoch_id, avg_loss))

        self.end_task(dataset, train_loader)

    def observe(self, data, epoch_id, batch_id):
        pass

    def end_task(self, dataset, train_loader):
        pass

    # 生命周期函数
    def test_task(self, dataset, test_loader):
        pass

    # 生命周期函数
    def end_il(self, dataset):
        torch.save(self,"data/"+self.args.model1+".pt")
        pass

    def forward(self, x):
        self.net.eval()
        x = x.to(self.device)
        with torch.no_grad():
            outputs = self.net(x)
        return outputs
