import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import copy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from utils.args import get_args
from datasets import get_dataset

def mask_classes(outputs: torch.Tensor, cats) -> None:
    outputs[:, 0:cats[0]] = -float('inf')
    outputs[:, cats[-1] + 1:] = -float('inf')

def linear_probe(args, net, classifier, train_loader, test_loader, cur_class):
    loss = F.cross_entropy
    print(cur_class[0], cur_class[-1] + 1)

    # self.net.eval()
    opt = torch.optim.SGD([
        {'params': classifier.parameters()}
    ], lr=args.lr)

    scheduler = StepLR(opt, step_size=args.scheduler_step, gamma=0.1)
    for epoch in range(args.n_epochs):
        for step, data in enumerate(train_loader):
            inputs, labels = data[0].to(args.device), data[1].to(args.device)

            feat = net.features(inputs)

            pred = classifier(feat)
            loss_ce = loss(
                pred[:, cur_class[0]: cur_class[-1] + 1],
                labels - cur_class[0]
            )

            opt.zero_grad()
            loss_ce.backward()
            opt.step()

        scheduler.step()
        if epoch % args.print_freq == 0:
            print('epoch:%d, feat_extract_loss:%.5f, classifier_loss:%.5f' % (
                epoch, loss_ce.to('cpu').item(), loss_ce.to('cpu').item()))

    correct, total= 0.0, 0.0
    for data in test_loader:
        inputs, labels = data[0].to(args.device), data[1].to(args.device)

        feat = net.features(inputs)
        outputs = classifier(feat)

        total += labels.shape[0]
        mask_classes(outputs, cur_class)
        _, pred = torch.max(outputs.data, 1)
        correct += torch.sum(pred == labels).item()

    acc = correct / total * 100
    return acc

# 设置参数
args = get_args()

args.dataset = 'seq-cifar100'
args.print_freq = 10
args.n_epochs = 100
args.device = 'cuda'

args.lr = 0.01
args.batch_size = 32
args.scheduler_step = 99

root = 'img/accstl/'

# 读取模型和数据集
model = torch.load(root + args.dataset + '.pt')
dataset = get_dataset(args)

accs = []
for t in range(dataset.nt):
    cls = dataset.t_c_arr[t]
    train_loader, test_loader = dataset.get_data_loaders()

    accs_t = []
    for k, net in enumerate(model.net_arr):
        classifier = copy.deepcopy(net.classifier)
        acc = linear_probe(args, net, classifier, train_loader, test_loader, cls)
        accs_t.append(acc)
        print('task', str(t), 'model', k, 'linear probe:', acc)
    accs.append(accs_t)
    print('task', str(t), 'linear probe:', accs_t)
print('all task linear probe')
print(accs)





