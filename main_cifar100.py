import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from utils.args import get_args
from utils.training import train_il
from utils.conf import set_random_seed
import torch


def main():
    args = get_args()

    args.model = 'cifar100'
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.featureNet = None
    args.transform = "pytorch"
    args.dataset = 'seq-cifar100'
    args.lr = 5e-5
    args.pretrain_lr = 0.003  # 0.005
    args.VaDE_lr = 1e-9  # 1e-6
    args.batch_size = 64
    args.n_epochs = 100
    args.pretrain_epochs = 50
    args.VaDE_epochs = 30
    args.root1 = "data/"
    args.root = "/home/zhangsiyu/data/"
    args.extra_name = "resenet_cifar100"

    args.ratio = 1
    args.num = 0
    args.ratio_bate = 0.3
    args.lambda1 = 1
    args.lambda2 = 20
    args.lambda3 = 20
    args.lambda4 = 10
    args.prato = 0
    args.kld_ratio = 0.5
    args.eps = 1
    args.embedding_dim = 250  # 可以调
    args.weight_decay = 1e-2
    args.r_inter = 1500
    args.r_intra = 800  # 重新调整
    args.isPseudo = True
    args.nt = 4
    args.patience = 50
    args.pretrain = True
    args.pretrained_path = 'weights/pretrained_parameter_softmax_cifar100.pth'

    args.model1 = "model/gcd"  # 模型保存的路径
    args.nf = 64  # 不调
    args.isPrint = False
    args.t_c_arr = {
        0: ((list(range(70)), (0, 0.87), '0-70'),),
        1: ((list(range(70)), (0.87, 0.95), '0-70'),
            (list(range(70, 80)), (0, 0.7), '70-80'),),
        2: ((list(range(70)), (0.95, 0.97), '0-70'),
            (list(range(70, 80)), (0.7, 0.9), '70-80'),
            (list(range(80, 90)), (0, 0.9), '89-90'),),
        3: ((list(range(70)), (0.97, 1.0), '0-70'),
            (list(range(70, 80)), (0.9, 1.0), '70-80'),
            (list(range(80, 90)), (0.9, 1.0), '80-90'),
            (list(range(90, 100)), (0, 1.0), '90-100'),),
    }


    for conf in [42, 0, 9999, 2025, 918]:  # 下面应该调整一下r_intar,感觉影响大，异常检测能力很弱，这是为啥呢
        print("")
        print("=================================================================")
        print("==========================", args.dataset, "nt:", conf, "==========================")
        print("=================================================================")
        print("")
        args.seed = conf
        if args.seed is not None:
            set_random_seed(args.seed)
        args.featureNet = None
        train_il(args)



if __name__ == '__main__':
    main()
