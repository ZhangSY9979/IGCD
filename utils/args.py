from argparse import ArgumentParser

from utils.conf import set_random_seed


def get_args():
    parser = ArgumentParser(description='deep inc params', allow_abbrev=False)
    # 数据集参数
    parser.add_argument('--root', type=str, default='../../swj/code/visintIncre/data/',
                        help='dictionary of dataset')
    parser.add_argument('--transform', type=str, default='default',
                        help='default or pytorch.')
    parser.add_argument('--featureNet', type=str, default=None,
                        help='feature extractor')
    parser.add_argument('--nt', type=int, default=None,
                        help='task number')
    parser.add_argument('--t_c_arr', type=str, default=None,
                        help='class array for each task')
    parser.add_argument('--validation', type=bool, default=False,
                        help='is test with the validation set')
    parser.add_argument('--class_shuffle', type=bool, default=False,
                        help='is random shuffle the classes order')
    parser.add_argument('--task_shuffle', type=bool, default=False,
                        help='is random shuffle the task order')
    # 模型参数
    parser.add_argument('--backbone', type=str, default='None',
                        help='the backbone of model')
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=None,
                        help='number of epoch')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='batch size')
    # 其他参数
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print loss frequency')
    parser.add_argument('--img_dir', type=str, default='img/',
                        help='image dir')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed if None')
    parser.add_argument('--repeat', type=int, default=1,
                        help='repeat number')

    parser.add_argument('--LOG_DIR', default='./logs', help='Path to log folder')
    parser.add_argument('--dataset', default='cub', help='Training dataset, e.g. cub, cars, SOP, Inshop') # cub # mit # dog # air
    parser.add_argument('--embedding-size', default=512, type=int, dest='sz_embedding', help='Size of embedding that is appended to backbone model.')
    # parser.add_argument('--batch-size', default=120, type=int, dest='sz_batch', help='Number of samples per batch.')  # 150
    # parser.add_argument('--epochs', default=60, type=int, dest='nb_epochs', help='Number of training epochs.')

    parser.add_argument('--gpu-id', default=0, type=int, help='ID of GPU that is used for training.')

    parser.add_argument('--workers', default=0, type=int, dest='nb_workers', help='Number of workers for dataloader.')
    parser.add_argument('--basemodel', default='resnet18', help='Model for training')  # resnet50 #resnet18  VIT
    parser.add_argument('--loss', default='Proxy_Anchor', help='Criterion for training') #Proxy_Anchor #Contrastive
    parser.add_argument('--optimizer', default='adamw', help='Optimizer setting')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate setting')  #1e-4
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay setting')
    parser.add_argument('--lr-decay-step', default=5, type=int, help='Learning decay step setting')  #
    parser.add_argument('--lr-decay-gamma', default=0.5, type=float, help='Learning decay gamma setting')
    parser.add_argument('--alpha', default=32, type=float, help='Scaling Parameter setting')
    parser.add_argument('--mrg', default=0.1, type=float, help='Margin parameter setting')
    parser.add_argument('--warm', default=5, type=int, help='Warmup training epochs')  # 1
    parser.add_argument('--bn-freeze', default=True, type=bool, help='Batch normalization parameter freeze')
    parser.add_argument('--l2-norm', default=True, type=bool, help='L2 normlization')
    parser.add_argument('--remark', default='', help='Any reamrk')

    parser.add_argument('--use_split_modlue', type=bool, default=True)
    parser.add_argument('--use_GM_clustering', type=bool, default=True) # False

    parser.add_argument('--exp', type=str, default='0')
    args = parser.parse_known_args()[0]

    if args.seed is not None:
        set_random_seed(args.seed)

    return args