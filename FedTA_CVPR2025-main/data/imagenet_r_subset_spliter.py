import random
import time
from typing import TypeVar, Sequence

import numpy as np

import torch
import torchvision
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision.datasets import MNIST,CIFAR100
from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image

from data.continual_datasets import Imagenet_R, ILSVRC
from utils import build_transform

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class ImagenetR_spliter():

    def __init__(self,client_num,task_num,private_class_num,input_size):
        self.client_num = client_num
        self.task_num = task_num
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)

        self.private_class_num = private_class_num
        self.input_size = input_size



    # 分成client_num数目个subset,每个subset里包含了task个subsubset
    def random_split(self):
        trans = build_transform(True,self.input_size)
        self.Imagenet_R = Imagenet_R(root='./local_datasets', train=True, download=True)
        trainset = self.Imagenet_R

        # 100个类别的数据分给三个客户端使用
        class_counts = torch.zeros(200) #每个类的数量
        class_label = [] # 每个类的index
        for i in range(200):
            class_label.append([])
        j = 0
        xs =[]
        js =[]
        for x, label in tqdm(trainset):
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1
            if label==72:
                js.append(j)
                xs.append(x)
                # plt.imshow(x)
                # plt.axis('off')
                # plt.show()

        shown_sample = xs[len(xs)-1]
        # plt.imshow(shown_sample)
        # plt.axis('off')
        # plt.show()

        # class_label 里保存了每个类的index

        # 分类
        total_private_class_num = self.client_num*self.private_class_num
        public_class_num = 200-total_private_class_num
        class_public = [i for i in range(200)]
        class_public = set(class_public)
        class_public.remove(72)
        class_public = list(class_public)

        class_p = random.sample(class_public, total_private_class_num-1)
        class_public = list(set(class_public) - set(class_p))

        class_private = [class_p[self.private_class_num*i : self.private_class_num*i+self.private_class_num] for i in range(0,self.client_num)]
        for i in range(0,self.client_num):
            if i==0:
                tem = class_private[i][0]
                class_private[i][0]=72
                class_public.append(tem)
            class_private[i].extend(class_public)
            # random.shuffle(class_private[i])
        # print(class_private)


        # 对每个客户端进行操作
        client_subset = [[] for i in range(0,self.client_num)]
        client_mask = [[] for i in range(0,self.client_num)]

        class_every_task = int((public_class_num+self.private_class_num)/self.task_num)
        dirichlet_perclass = {}
        for i in class_public:
            a = np.random.dirichlet(np.ones(self.client_num), 1)
            # while  (a < 0.1).any():
            #     a = np.random.dirichlet(np.ones(self.client_num), 1)
            dirichlet_perclass[i] = a[0]
        for i in range(0,self.client_num):
            for j in range(0,self.task_num):
                index = []
                class_this_task = class_private[i][j*class_every_task: j*class_every_task+class_every_task]
                client_mask[i].append(class_this_task)
                for k in class_private[i][j*class_every_task:j*class_every_task+class_every_task]:
                    if k in class_public:
                        # 是公共类
                        lenth = int(int(class_counts[k])*dirichlet_perclass[k][i])
                        unused_indice = set(class_label[k])
                        q = 0
                        while q < lenth:
                            random_index = random.choice(list(unused_indice))
                            index.append(random_index)
                            unused_indice.remove(random_index)
                            q += 1
                        class_label[k]=unused_indice
                    else: #是私有类
                        index.extend(class_label[k])
                random.shuffle(index)

                client_subset[i].append(CustomedSubset(trainset,index,trans,shown_sample))

        return client_subset,client_mask


    def process_testdata(self,surrogate_num):
        trans = build_transform(False,self.input_size)
        self.Imagenet_R_test = Imagenet_R(root='./local_datasets', train=False, download=True)
        testset = self.Imagenet_R_test
        # 100个类别的数据分给三个客户端使用

        class_counts = torch.zeros(200)  # 每个类的数量
        class_label = []  # 每个类的index
        for i in range(200):
            class_label.append([])
        j = 0
        for x, label in testset:
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1
        # class_label 里保存了每个类的index

        surro_index =[]
        test_index = []
        for i in tqdm(range(200)):
            q = 0
            unused_indice = set(class_label[i])

            while q < surrogate_num:
                random_index = random.choice(list(unused_indice))
                surro_index.append(random_index)
                unused_indice.remove(random_index)
                q += 1
            test_index.extend(list(unused_indice))
        surrodata = CustomedSubset(testset,surro_index,trans,None)
        testdata = CustomedSubset(testset,test_index,trans,None)
        return surrodata,testdata

    def random_split_synchron(self):
        trans = build_transform(True,self.input_size)
        self.Imagenet_R = Imagenet_R(root='./local_datasets', train=True, download=True)
        trainset = self.Imagenet_R

        # 100个类别的数据分给三个客户端使用
        class_counts = torch.zeros(200) #每个类的数量
        class_label = [] # 每个类的index
        for i in range(200):
            class_label.append([])
        j = 0
        for x, label in tqdm(trainset):
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1
        # class_label 里保存了每个类的index
        # 分类

        class_public = [i for i in range(200)]
        # 对每个客户端进行操作
        client_subset = [[] for i in range(0,self.client_num)]
        client_mask = [[] for i in range(0,self.client_num)]

        class_every_task = 10
        dirichlet_perclass = {}
        for i in class_public:
            a = np.random.dirichlet(np.ones(self.client_num), 1)
            # while  (a < 0.1).any():
            #     a = np.random.dirichlet(np.ones(self.client_num), 1)
            dirichlet_perclass[i] = a[0]
        for i in range(0,self.client_num):
            for j in range(0,self.task_num):
                index = []
                class_this_task = class_public[j*class_every_task: j*class_every_task+class_every_task]
                client_mask[i].append(class_this_task)

                # 是公共类
                for k in class_this_task:
                    lenth = int(int(class_counts[k])*0.8)
                    unused_indice = set(class_label[k])
                    q = 0
                    # print(unused_indice)
                    while q < lenth:
                        random_index = random.choice(list(unused_indice))
                        index.append(random_index)
                        unused_indice.remove(random_index)
                        q += 1
                    random.shuffle(index)
                client_subset[i].append(CustomedSubset(trainset,index,trans,None))

        return client_subset,client_mask



    def process_ILSVRC(self,surrogate_num):
        trans = build_transform(False,self.input_size)
        self.ILSVRC = ILSVRC(root='D:/datasets', train=False, download=True)
        testset = self.ILSVRC
        # 100个类别的数据分给三个客户端使用

        class_counts = torch.zeros(200)  # 每个类的数量
        class_label = []  # 每个类的index
        for i in range(200):
            class_label.append([])
        j = 0
        for x, label in testset:
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1
        # class_label 里保存了每个类的index

        surro_index =[]
        test_index = []
        for i in tqdm(range(200)):
            q = 0
            unused_indice = set(class_label[i])

            while q < surrogate_num:
                random_index = random.choice(list(unused_indice))
                surro_index.append(random_index)
                unused_indice.remove(random_index)
                q += 1
            test_index.extend(list(unused_indice))

        surrodata = CustomedSubset(testset,surro_index,trans,None)
        testdata = CustomedSubset(testset,test_index,trans,None)
        return surrodata,testdata



class CustomedSubset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int],trans,show_sample) -> None:

        self.indices = indices
        self.data = []
        self.targets = []
        self.dataset = dataset
        self.transform_pretrain = trans

        self.show_sample = show_sample

        for i in self.indices:
            self.data.append(dataset.data[i])
            self.targets.append(dataset.targets[i])
        # self.data = self.data
        self.targets = np.array(self.targets)
        self.transform=trans
        self.target_transform = None

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        if self.transform_pretrain is not None:
            img_pre = self.transform_pretrain(img)


        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_pre,target

    def get_test_sample(self):
        img = self.show_sample

        if self.transform_pretrain is not None:
            img_pre = self.transform_pretrain(img)


        return img


    def __len__(self):
        return len(self.indices)








