import argparse

def get_args_parser(subparsers):
    subparsers.add_argument('--batch-size', default=16, type=int, help='Batch size per device')


    # Model parameters。模型选择+输入尺寸+预训练+两种正则化
    subparsers.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL', help='Name of model to train')
    subparsers.add_argument('--input-size', default=224, type=int, help='images input size')  # 图像缩放到边长224*224。
    subparsers.add_argument('--pretrained', default=True, help='Load pretrained model or not')
    subparsers.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    subparsers.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

    # Optimizer parameters。配置优化器+梯度更新策略。
    subparsers.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    subparsers.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    subparsers.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    subparsers.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',  help='Clip gradient norm (default: None, no clipping)')
    subparsers.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    subparsers.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    subparsers.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')

    # Learning rate schedule parameters。学习率调度器
    subparsers.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
    subparsers.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.03)')  # 初始学习率
    subparsers.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')  # LR曲线叠噪声，目的逃离局部最优。
    subparsers.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    subparsers.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    subparsers.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')  # 在预热epoch里，LR从warmup-lr升到基础--lr。
    subparsers.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')  # 学习率下限。
    subparsers.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')  # 阶梯式衰减参数
    subparsers.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    subparsers.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')  # 主调度结束，维持min-lr再训练这么多epoch。
    subparsers.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')  # plateau调度器
    subparsers.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
    subparsers.add_argument('--unscale_lr', type=bool, default=True, help='scaling lr by batch size (default: True)')

    # Augmentation parameters。配置数据增广/损失细节。
    subparsers.add_argument('--color-jitter', type=float, default=None, metavar='PCT', help='Color jitter factor (default: 0.3)')  # 颜色抖动强度，None不启用此增广。
    subparsers.add_argument('--aa', type=str, default=None, metavar='NAME', help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),  # 在训练时自动套用一组更强的图形增广。
    subparsers.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')  # 标签平滑系数。
    subparsers.add_argument('--train-interpolation', type=str, default='bicubic', help='Training interpolation (random, bilinear, bicubic default: "bicubic")')  # 重采样算法。

    # * Random Erase params 随机擦除
    subparsers.add_argument('--reprob', type=float, default=0.0, metavar='PCT', help='Random erase prob (default: 0.25)')  # 对每张训练图像应用随机擦除的概率。
    subparsers.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')  # 被抹去区域用什么值填充。
    subparsers.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')  # 擦除次数。

    # Data parameters
    subparsers.add_argument('--data-path', default='/local_datasets/', type=str, help='dataset path')
    subparsers.add_argument('--dataset', default='ImageNet-R', type=str, help='dataset name')
    subparsers.add_argument('--shuffle', default=False, help='shuffle the data order')  # 是否打乱顺序
    subparsers.add_argument('--output_dir', default='output/', help='path where to save, empty for no saving')
    subparsers.add_argument('--device', default='cuda', help='device to use for training / testing')
    subparsers.add_argument('--seed', default=42, type=int)
    subparsers.add_argument('--eval', action='store_true', help='Perform evaluation only')
    subparsers.add_argument('--num_workers', default=2, type=int)
    subparsers.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    subparsers.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    subparsers.set_defaults(pin_mem=True)



    # Continual learning parameters

    subparsers.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')  # 类别掩码
    subparsers.add_argument('--task_inc', default=False, type=bool, help='if doing task incremental')  # 任务增量测评


    # pool_size
    subparsers.add_argument('--size', default=20, type=int,)
    subparsers.add_argument('--length', default=15,type=int, )
    subparsers.add_argument('--top_k', default=1, type=int, )
    subparsers.add_argument('--initializer', default='uniform', type=str,)
    subparsers.add_argument('--prompt_key', default=True, type=bool,)
    subparsers.add_argument('--prompt_key_init', default='uniform', type=str)
    subparsers.add_argument('--use_prompt_mask', default=False, type=bool)
    subparsers.add_argument('--shared_prompt_pool', default=False, type=bool)
    subparsers.add_argument('--shared_prompt_key', default=False, type=bool)
    subparsers.add_argument('--batchwise_prompt', default=True, type=bool)
    subparsers.add_argument('--embedding_key', default='cls', type=str)
    subparsers.add_argument('--predefined_key', default='', type=str)
    subparsers.add_argument('--pull_constraint', default=True)
    subparsers.add_argument('--pull_constraint_coeff', default=0.1, type=float)

    # ViT parameters
    subparsers.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')
    subparsers.add_argument('--head_type', default='prompt', choices=['token', 'gap', 'prompt', 'token+prompt'], type=str, help='input type of classification head')
    subparsers.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*', type=list, help='freeze part in backbone model')

    # Misc parameters
    subparsers.add_argument('--print_freq', type=int, default=10, help = 'The frequency of printing')

    # Ours
    subparsers.add_argument('--method', type=str, default='delay', help='The method of Prompt')
    subparsers.add_argument('--e_prompt_layer_idx', default=[2, 3, 4], type=int, nargs="+",
                            help='the layer index of the E-Prompt')
    subparsers.add_argument('--client_num', default=10, type=int, nargs="+",
                            help='the num of client')
    subparsers.add_argument('--task_num', default=10, type=int, nargs="+",
                            help='the num of tasks')
    subparsers.add_argument('--private_class_num', default=40, type=int, nargs="+",
                            help='the num of client')
    subparsers.add_argument('--surrogate_num', default=20, type=int, nargs="+",
                            help='the num of client')
    subparsers.add_argument('--global_epoch', default=5, type=int, nargs="+",
                            help='the num of client')
    subparsers.add_argument('--local_epoch', default=20, type=int, nargs="+",
                            help='the num of client')
    subparsers.add_argument('--threshold', default=0.1, type=int, nargs="+",
                            help='the num of client')
    subparsers.add_argument('--data_name',default='ImageNet-R',type=str)
    subparsers.add_argument('--model_name', default='Tail_Anchor', choices=['AlexNet', 'VGG16','ResNet18','Tail_Anchor'], type=str)




