import argparse

from pathlib import Path
import random

import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

from Models.Server_DF import Server_DF
from config.cifar100_delay import get_args_parser

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import Models.vision_transformer__l2p
from data.homeoffice_subset_spilter import officehome_spliter
from data.i5datasets import i5datasets
from data.iCIFAR100c import iCIFAR100c
from data.imagenet_r_subset_spliter import ImagenetR_spliter
from data.svhn_mnist_subset_spliter import svhn_mnist_Data_spliter

torch.set_printoptions(threshold=float('inf'))

from data.cifar100_subset_spliter import cifar100_Data_Spliter
from data.five_datasets_subset_spliter import five_datasets_Data_Spliter
import warnings
warnings.filterwarnings("ignore")

def main(args):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pretrained_cfg = create_model(args.model).default_cfg
    pretrained_cfg['file']='pretrain_model/ViT-B_16.npz'

    cudnn.benchmark = True  # 根据硬件尝试不同算法，选择最快的那一个。

    print(args.data_name)
    if args.data_name == 'cifar100':
        client_data, client_mask = cifar100_Data_Spliter(client_num=args.client_num, task_num=args.task_num,
                                                       private_class_num=args.private_class_num, input_size=args.input_size).random_split()
        surro_data, test_data = cifar100_Data_Spliter(client_num=args.client_num, task_num=args.task_num,
                                                     private_class_num=args.private_class_num, input_size=args.input_size).process_testdata(args.surrogate_num)
        surro_data = iCIFAR100c(subset=surro_data)
        args.nb_classes = 100

    elif args.data_name == '5datasets':
        data_spliter = five_datasets_Data_Spliter(client_num=args.client_num, task_num=args.task_num,
                                                  private_class_num=args.private_class_num, input_size=args.input_size)
        client_data, client_mask = data_spliter.random_split()
        surro_data, test_data = data_spliter.process_testdata(args.surrogate_num)
        surro_data = i5datasets(surro_data)
        args.nb_classes = 50

    elif args.data_name == 'ImageNet-R':
        data_spliter = ImagenetR_spliter(client_num=args.client_num, task_num=args.task_num,
                                                  private_class_num=args.private_class_num,
                                                  input_size=args.input_size)

        client_data, client_mask = data_spliter.random_split()
        args.nb_classes = 200

        surro_data, test_data = ImagenetR_spliter(client_num=args.client_num, task_num=args.task_num,
                                                  private_class_num=args.private_class_num, input_size=args.input_size).process_testdata(5)
        surro_data = iCIFAR100c(subset=surro_data)

    elif args.data_name =="svhn-mnist":
        data_spliter = svhn_mnist_Data_spliter(client_num=args.client_num,task_num=args.task_num,private_class_num=2,input_size=args.input_size)

        # data_spliter.get_data_ready()
        client_data, client_mask = data_spliter.random_split_mnist()
        client_data1, client_mask1 = data_spliter.random_split_svhn()

        surrodata,test_data=data_spliter.process_testdata_surro(20)

        surro_data = iCIFAR100c(subset=surrodata)
        # surro_data.getTrainData([0,1,2])
        # test_loader = DataLoader(surro_data, batch_size=16, shuffle=True, num_workers=2)
        # for iteration, (index, x, y) in enumerate(test_loader):
        #     print(x.shape)
        #     print(y.shape)

        args.nb_classes = 10


    else:
        data_spliter = officehome_spliter(client_num=args.client_num, task_num=args.task_num,
                                                          private_class_num=args.private_class_num,
                                                          input_size=args.input_size)
        args.nb_classes = 65

    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=True,
        pretrained_cfg=pretrained_cfg,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        # pretrained_cfg_overlay=dict(file='pretrain_model/pytorch_model.bin')
        # checkpoint_path='pretrain_model/original_model.pth'
        )  # 实例化一个基线ViT模型的配置

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        pretrained_cfg=pretrained_cfg,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,  # 每个prompt的token数，每层/每次注入的提示长度。
        embedding_key=args.embedding_key,  # 获取查询向量的方式，用于从样本特征里提取key，用于和prompt keys做相似度检索。
        prompt_init=args.prompt_key_init,
        prompt_pool=True,
        prompt_key=args.prompt_key,
        pool_size=args.size,  # 设置prompt池中共有多少个prompt
        top_k=args.top_k,  # 每个样本实际选用的prompt数，从池里按相似度取前k个。
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,  # 是否启用prompt mask（在某些层上屏蔽prompt，控制使用范围，减小干扰）。
        # e_prompt_layer_idx=args.e_prompt_layer_idx,  # 指定在哪些transformer层注入prompt
        # method=args.method,
        # pretrained_cfg_overlay=dict(file='pretrain_model/pytorch_model.bin')
        # checkpoint_path='pretrain_model/model.pth'
    )

    original_model.to(device)
    model.to(device)


    if args.freeze:  # 冻结主干模型。冻结参数。
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    print(args)
    for n, p in model.named_parameters():
        if p.requires_grad == True:
            print(n)


    # 实例化并启动联邦学习服务器
    if args.data_name=='cifar100' or args.data_name=='5datasets' or args.data_name=="ImageNet-R":
        myServer = Server_DF(id='Server',origin_model=original_model,model_name=args.model_name,client_num=args.client_num,task_num=args.task_num,
                     subset=client_data,class_mask=client_mask,lr=args.lr,global_epoch=args.global_epoch,local_epoch=args.local_epoch,
                     batch_size=args.batch_size,device=args.device,method=args.method,threshold=args.threshold,
                     surrogate_data=surro_data,test_data=None,args=args,model=model)
    else:
        myServer = Server_DF(id='Server',origin_model=original_model,model=model,client_num=args.client_num,task_num=args.task_num,
                         subset=data_spliter,class_mask=None,lr=args.lr,global_epoch=args.global_epoch,local_epoch=args.local_epoch,
                         batch_size=args.batch_size,device=args.device,method=args.method,threshold=args.threshold,
                         surrogate_data=None,test_data=None,args=args)

    # svhn-mnist的测试
    # myServer = Server_DF(id='Server', origin_model=original_model, model_name=args.model_name,
    #                      client_num=args.client_num, task_num=args.task_num,
    #                      subset1=client_data, class_mask1=client_mask,subset2=client_data1,class_mask2=client_mask1, lr=args.lr, global_epoch=args.global_epoch,
    #                      local_epoch=args.local_epoch,
    #                      batch_size=args.batch_size, device=args.device, method=args.method, threshold=args.threshold,
    #                      surrogate_data=surro_data, test_data=None, args=args, model=model)

    myServer.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('FedTA training and evaluation configs')

    config = 'imagenet_r_delay'

    subparser = parser.add_subparsers(dest='subparser_name')

    config_parser = None
    if config == 'cifar100_delay':
        from config.cifar100_delay import get_args_parser
        config_parser = subparser.add_parser('cifar100_delay', help='Split-CIFAR100 Delay configs')
    elif config == 'five_datasets_delay':
        from config.five_datasets_delay import get_args_parser
        config_parser = subparser.add_parser('five_datasets_delay', help='5datasets delay configs')
    elif config == 'imagenet_r_delay':
        from config.imagenet_r_delay import get_args_parser
        config_parser = subparser.add_parser('imagenet_r_delay', help='ImageNet-R delay configs')
    else:
        from config.five_datasets_delay import get_args_parser
        config_parser = subparser.add_parser('five_datasets_delay', help='5datasets delay configs')

    get_args_parser(config_parser)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)