# FedTA PyTorch Implementation

This repository contains PyTorch implementation code.

## Environment
The system I used and tested in
- Ubuntu 20.04.4 LTS
- Slurm 21.08.1
- NVIDIA GeForce RTX 3090
- Python 3.8

## Usage
First, install the packages below:
```
pytorch==1.12.1
torchvision==0.13.1
timm==0.6.7
pillow==9.2.0
matplotlib==3.5.3
```

## Pretrain models
Our method  loads pre-trained ViT locally. You can remove the following two lines of code from main.py to switch to online loading:
```
pretrained_cfg = create_model(args.model).default_cfg
pretrained_cfg['file']='pretrain_model/ViT-B_16.npz'
```

## Data preparation
If you already have CIFAR-100 or ImageNet-R, pass your dataset path to  `--data-path`.


The datasets aren't ready, change the download argument in `{dataset name}_subset_spliter.py` as follows

**CIFAR-100**
```
self.cifar100_dataset = CIFAR100(root='./local_datasets', train=True, download=True)
```

**ImageNet-R**
```
self.Imagenet_R = Imagenet_R(root='./local_datasets', train=True, download=True)
```

## Notice 

For every dataset, we creat a spliter function like 

```
data_spliter = cifar100_Data_Spliter(client_num, task_num, 15,224)
datas,mask = data_spliter.random_split()
```

Inside this function, we first set private_class_num = 15, meaning all 15 private classes are assigned to one client. Taking 5 clients as an example, this results in a total of 75 private classes. The remaining 25 classes are considered public classes, which are randomly distributed among the clients. However, each sample appears only once to ensure that the data among clients does not overlap.

This setting is to ensure strong spatial-temporal data heterogeneity.
## Training
To train a model via command line:

Single node with single gpu

config_file can be chosen from['cifar100_delay','imagenet_r_delay']. For example, the training process of CIFAR-100 can be started by:

```
python main.py \
       cifar100_delay \   #config_file
       --model vit_base_patch16_224 \
       --batch-size 4 \
       --data-path local_datasets/ \
       --output_dir ./output \
       --data_name cifar100
```

Notice that you can also run Imagenet-R with 'cifar100_delay' as the config_file.



you can direce

## License
This repository is released under the CC BY-NC-ND 4.0.


## Contact
you can directly contact me though the e-mail in the origin paper.

## Citation
```angular2html
@inproceedings{yu2025handling,
  title={Handling spatial-temporal data heterogeneity for federated continual learning via tail anchor},
  author={Yu, Hao and Yang, Xin and Zhang, Le and Gu, Hanlin and Li, Tianrui and Fan, Lixin and Yang, Qiang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={4874--4883},
  year={2025}
}
```


