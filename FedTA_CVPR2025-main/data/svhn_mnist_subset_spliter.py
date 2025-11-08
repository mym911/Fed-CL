import random
import random
import time
from typing import TypeVar, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Subset, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

from tqdm import tqdm
from torch.utils.data import Subset, Dataset, DataLoader
# from data_process.continual_datasets import *
# from ResNet import resnet18_cbam
from torch.nn import CrossEntropyLoss

import warnings
from data.continual_datasets import SVHN

warnings.filterwarnings("ignore")

import utils


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

def build_transform(is_train, inputsize):
    resize_im = inputsize > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(inputsize, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * inputsize)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(inputsize))
    t.append(transforms.ToTensor())

    return transforms.Compose(t)


class svhn_mnist_Data_spliter():

    def __init__(self,client_num,task_num,private_class_num,input_size):
        self.client_num = client_num
        self.task_num = task_num
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        self.private_class_num = private_class_num
        self.input_size = input_size
        self.transform_train = build_transform(True, input_size)
        self.transform_val = build_transform(False, input_size)

        self.mnist_trans = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1))
        ])
        self.get_data_ready()

    def get_data_ready(self):
        print('getting data')
        # self.nmnist_data_train = NotMNIST('./local_datasets', train=True, download=False)
        # self.nmnist_data_test = NotMNIST('./local_datasets', train=False, download=False)
        # print('Not-MNIST DONE')
        # self.mnist_data_train =MNIST_RGB('./local_datasets', train=True, download=False)
        # self.mnist_data_test =MNIST_RGB('./local_datasets', train=False, download=False)
        # print('MNIST_rgb DONE')
        self.svhn_data_train =SVHN('./local_datasets', split='train' ,download=True,transform=self.transform_train)
        self.svhn_data_test =SVHN('./local_datasets', split='test', download=True,transform=self.transform_train)
        print('svhn DONE')
        # self.cifar10_data_train = datasets.CIFAR10('./local_datasets', train=True, download=True)
        # self.cifar10_data_test =datasets.CIFAR10('./local_datasets', train=False, download=True)
        # print('CIFAR-10 DONE')
        self.mnist_data_train =MNIST('./local_datasets', train=True, download=True,transform=self.mnist_trans)
        self.mnist_data_test =MNIST('./local_datasets', train=False, download=True,transform=self.mnist_trans)
        print('MNIST DONE')

        # self.train_sets = [self.cifar10_data_train,self.Fmnist_data_train,self.nmnist_data_train,self.mnist_data_train,self.svhn_data_train,]
        # self.test_sets = [self.cifar10_data_test,self.Fmnist_data_test,self.nmnist_data_test,self.mnist_data_test,self.svhn_data_test]

        self.train_sets = self.svhn_data_train
        self.test_set = self.svhn_data_test

    def random_split_svhn(self):
        # 对每个客户端进行操作
        trans = build_transform(True, self.input_size)
        trainset = self.svhn_data_train

        # 100个类别的数据分给三个客户端使用
        # 100个类别的数据分给三个客户端使用
        class_counts = torch.zeros(10)  # 每个类的数量
        class_label = []  # 每个类的index
        for i in range(10):
            class_label.append([])
        j = -1
        h=0
        for x, label in tqdm(trainset):
            if h >10000:
                break
            j += 1
            if class_counts[label] < 1005:
                class_counts[label] += 1
                class_label[label].append(j)
                h+=1
            else:continue
        # class_label 里保存了每个类的index

        # 分类
        total_private_class_num = self.client_num * self.private_class_num
        public_class_num = 10 - total_private_class_num
        class_public = [i for i in range(10)]
        class_p = random.sample(class_public, total_private_class_num)
        class_public = list(set(class_public) - set(class_p))

        class_private = [class_p[self.private_class_num * i: self.private_class_num * i + self.private_class_num] for i
                         in range(0, self.client_num)]
        for i in range(0, self.client_num):
            class_private[i].extend(class_public)
            random.shuffle(class_private[i])

        # 对每个客户端进行操作
        client_subset = [[] for i in range(0, self.client_num)]
        client_mask = [[] for i in range(0, self.client_num)]

        class_every_task = int((public_class_num + self.private_class_num) / self.task_num)
        dirichlet_perclass = {}
        for i in class_public:
            a = np.random.dirichlet(np.ones(self.client_num), 1)
            # while  (a < 0.1).any():
            #     a = np.random.dirichlet(np.ones(self.client_num), 1)
            dirichlet_perclass[i] = a[0]
        for i in range(0, self.client_num):
            # 每个客户端2个隐私类，一共6个类，每3个为一个task
            for j in range(0, self.task_num):
                index = []
                class_this_task = class_public[j*class_every_task: j*class_every_task+class_every_task]
                client_mask[i].append(class_this_task)
                for k in class_this_task:
                    if k in class_public:
                        # 是公共类
                        len = int(int(class_counts[k]) * dirichlet_perclass[k][i])
                        unused_indice = set(class_label[k])
                        q = 0
                        while q < len:
                            random_index = random.choice(list(unused_indice))
                            index.append(random_index)
                            unused_indice.remove(random_index)
                            q += 1
                    else:  # 是私有类
                        index.extend(class_label[k])
                random.shuffle(index)
                client_subset[i].append(CustomedSubset(trainset, index, trans,0))

        return client_subset, client_mask


    def random_split_mnist(self):
        # 对每个客户端进行操作
        trans = build_transform(True, self.input_size)
        trainset = self.mnist_data_train

        # 100个类别的数据分给三个客户端使用
        # 100个类别的数据分给三个客户端使用
        class_counts = torch.zeros(10)  # 每个类的数量
        class_label = []  # 每个类的index
        for i in range(10):
            class_label.append([])
        j = -1
        h=0
        for x, label in tqdm(trainset):
            if h>10000:
                break
            j += 1
            if class_counts[label] < 1005:
                class_counts[label] += 1
                class_label[label].append(j)
                h+=1
            else:
                continue
        # class_label 里保存了每个类的index

        # 分类
        total_private_class_num = self.client_num * self.private_class_num
        public_class_num = 10 - total_private_class_num
        class_public = [i for i in range(10)]
        class_p = random.sample(class_public, total_private_class_num)
        class_public = list(set(class_public) - set(class_p))

        class_private = [class_p[self.private_class_num * i: self.private_class_num * i + self.private_class_num] for i
                         in range(0, self.client_num)]
        for i in range(0, self.client_num):
            class_private[i].extend(class_public)
            random.shuffle(class_private[i])

        # 对每个客户端进行操作
        client_subset = [[] for i in range(0, self.client_num)]
        client_mask = [[] for i in range(0, self.client_num)]

        class_every_task = int((public_class_num + self.private_class_num) / self.task_num)
        dirichlet_perclass = {}
        for i in class_public:
            a = np.random.dirichlet(np.ones(self.client_num), 1)
            # while  (a < 0.1).any():
            #     a = np.random.dirichlet(np.ones(self.client_num), 1)
            dirichlet_perclass[i] = a[0]
        for i in range(0, self.client_num):
            # 每个客户端2个隐私类，一共6个类，每3个为一个task
            for j in range(0, self.task_num):
                index = []
                class_this_task = class_public[j*class_every_task: j*class_every_task+class_every_task]
                client_mask[i].append(class_this_task)
                for k in class_this_task:
                    if k in class_public:
                        # 是公共类
                        len = int(int(class_counts[k]) * dirichlet_perclass[k][i])
                        unused_indice = set(class_label[k])
                        q = 0
                        while q < len:
                            random_index = random.choice(list(unused_indice))
                            index.append(random_index)
                            unused_indice.remove(random_index)
                            q += 1
                    else:  # 是私有类
                        index.extend(class_label[k])
                random.shuffle(index)
                client_subset[i].append(CustomedSubset(trainset, index, trans,0))

        return client_subset, client_mask



    def process_testdata(self):
        trans = build_transform(False,self.input_size)

        self.test_sets = []

        mnist_test = self.mnist_data_test

        svhn_test = self.svhn_data_test

        class_counts_mnist = torch.zeros(10)  # 每个类的数量
        class_label_mnist = [[] for i in range(10)]  # 每个类的index
        j = 0
        for x, label in tqdm(mnist_test):
            label = int(label)
            class_counts_mnist[label] += 1
            class_label_mnist[label].append(j)
            j += 1

        class_counts_svhn = torch.zeros(10)  # 每个类的数量
        class_label_svhn = [[] for i in range(10)]  # 每个类的index
        j = 0
        for x, label in tqdm(svhn_test):
            label = int(label)
            class_counts_svhn[label] += 1
            class_label_svhn[label].append(j)
            j += 1

        index =[]
        index1 =[]
        for j in range(10):
            # for mnist
            num = int(class_counts_mnist[j])
            unused_indice1 = set(class_label_mnist[j])
            q = 0
            while q < num:
                random_index = random.choice(list(unused_indice1))
                index.append(random_index)
                unused_indice1.remove(random_index)
                q += 1

            # for svhn
            num = int(class_counts_svhn[j])
            unused_indice = set(class_label_svhn[j])
            q = 0
            while q < num :
                random_index = random.choice(list(unused_indice))
                index1.append(random_index)
                unused_indice.remove(random_index)
                q += 1

        return CustomedSubset(mnist_test, index, trans,0),CustomedSubset(svhn_test, index1, trans,0)

    def process_testdata_surro(self, surrogate_num):
        trans = build_transform(True, self.input_size)

        mnist_test = self.mnist_data_test
        svhn_test = self.svhn_data_test
        # 100个类别的数据分给三个客户端使用

        class_counts = torch.zeros(10)  # 每个类的数量
        class_label = []  # 每个类的index

        class_counts_svhn = torch.zeros(10)  # 每个类的数量
        class_label_svhn = []  # 每个类的index

        for i in range(10):
            class_label.append([])
            class_label_svhn.append([])

        j = -1
        h = 0
        for x, label in tqdm(mnist_test):
            if h > 5000:
                break
            j += 1
            if class_counts[label] < 501:
                class_counts[label] += 1
                class_label[label].append(j)
                h += 1
            else:
                continue


        j = -1
        h = 0
        for x, label in tqdm(svhn_test):
            if h > 5000:
                break
            j += 1
            if class_counts_svhn[label] < 505:
                class_counts_svhn[label] += 1
                class_label_svhn[label].append(j)
                h += 1
            else:
                continue


        # class_label 里保存了每个类的index

        surro_index_mnist = []
        test_index_mnist = []

        surro_index_svhn = []
        test_index_svhn = []
        for i in tqdm(range(10)):
            q = 0
            unused_indice = set(class_label[i])
            print(len(unused_indice))
            while q < surrogate_num:
                random_index = random.choice(list(unused_indice))
                surro_index_mnist.append(random_index)
                unused_indice.remove(random_index)
                q += 1
            test_index_mnist.extend(list(unused_indice))

        for i in tqdm(range(10)):
            q = 0
            unused_indice = set(class_label_svhn[i])
            print(len(unused_indice))
            while q < surrogate_num:
                random_index = random.choice(list(unused_indice))
                surro_index_svhn.append(random_index)
                unused_indice.remove(random_index)
                q += 1
            test_index_svhn.extend(list(unused_indice))


        surrodata = CustomedDualSubset(mnist_test, surro_index_mnist,svhn_test,surro_index_svhn, trans,0)
        testdata = CustomedDualSubset(mnist_test, surro_index_mnist,svhn_test,surro_index_svhn, trans,0)

        return surrodata, testdata

    def train_feature_extractor(self):
        trans = build_transform(True, self.input_size)
        svhn_data = self.svhn_data_train
        mnist_data = self.mnist_data_train

        self.feature_extractor = resnet18_cbam(True)
        self.feature_extractor.to('cuda')
        # for n, p in self.feature_extractor.named_parameters():
        #     if n.startswith('modify'):
        #         p.requires_grad =True
        #     else:
        #         p.requires_grad=False
        #
        # for n, p in self.feature_extractor.named_parameters():
        #     if p.requires_grad:
        #         print(n)
        #
        class_counts_mnist = torch.zeros(10)  # 每个类的数量
        class_label_mnist = [[] for i in range(10)]  # 每个类的index
        j = 0
        for x, label in tqdm(mnist_data):
            label = int(label)
            class_counts_mnist[label] += 1
            class_label_mnist[label].append(j)
            j += 1

        class_counts_svhn = torch.zeros(10)  # 每个类的数量
        class_label_svhn = [[] for i in range(10)]  # 每个类的index
        j = 0
        for x, label in tqdm(svhn_data):
            label = int(label)
            class_counts_svhn[label] += 1
            class_label_svhn[label].append(j)
            j += 1

        # 对每个客户端进行操作
        subset = []
        index = []
        index1 = []

        for j in range(10):
            # for mnist
            num = int(class_counts_mnist[j])
            unused_indice1 = set(class_label_mnist[j])
            q = 0
            while q < int(num * 0.05):
                random_index = random.choice(list(unused_indice1))
                index.append(random_index)
                unused_indice1.remove(random_index)
                q += 1

            #for svhn
            num = int(class_counts_svhn[j])
            unused_indice = set(class_label_svhn[j])
            q = 0
            while q < int(num * 0.05):
                random_index = random.choice(list(unused_indice))
                index1.append(random_index)
                unused_indice.remove(random_index)
                q += 1

        subset.append(CustomedDualSubset(mnist_data,index,svhn_data,index1,trans,0))
        subset = subset[0]


        trainloader = DataLoader(subset,batch_size=32,shuffle=True)

        test_data = self.process_testdata()

        testloader = DataLoader(test_data,batch_size=32,shuffle=True)
        # 训练feature_extractor
        optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=0.0001,weight_decay=1e-03)
        loss_function = CrossEntropyLoss()
        for epoch in range(100):
            self.feature_extractor.to("cuda")
            self.feature_extractor.train()
            for batchidx, (x, label) in enumerate(trainloader):
                x=x.to('cuda')
                label = label.to('cuda')
                _,logits = self.feature_extractor(x)  # logits: [b, 10]
                loss =loss_function(logits, label)  # 标量

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch%10==0:
                print(epoch, 'loss:', loss.item())

            self.feature_extractor.eval()  # 测试模式
            with torch.no_grad():

                total_correct = 0  # 预测正确的个数
                total_num = 0
                for x, label in testloader:
                    # x: [b, 3, 32, 32]
                    # label: [b]
                    x = x.to("cuda")
                    label = label.to("cuda")
                    _,logits = self.feature_extractor(x)  # [b, 10]
                    pred = logits.argmax(dim=1)  # [b]

                    # [b] vs [b] => scalar tensor
                    correct = torch.eq(pred, label).float().sum().item()
                    total_correct += correct
                    total_num += x.size(0)

                acc = 100 * total_correct / total_num
                print(epoch, 'test acc:', acc)
        torch.save(self.feature_extractor,'./pretrain_models/resnet-for_SVHN_MNIST.pth')

        return subset


    def train_feature_extractor_mnist(self):
        self.feature_extractor = resnet18_cbam(True)
        self.feature_extractor.to('cuda')
        trans = build_transform(True, self.input_size)
        mnist_data = self.mnist_data_train

        class_counts_mnist = torch.zeros(10)  # 每个类的数量
        class_label_mnist = [[] for i in range(10)]  # 每个类的index
        j = 0
        for x, label in tqdm(mnist_data):
            label = int(label)
            class_counts_mnist[label] += 1
            class_label_mnist[label].append(j)
            j += 1

        index = []
        for j in range(10):
            # for mnist
            num = int(class_counts_mnist[j])
            unused_indice1 = set(class_label_mnist[j])
            q = 0
            while q < int(num * 0.05):
                random_index = random.choice(list(unused_indice1))
                index.append(random_index)
                unused_indice1.remove(random_index)
                q += 1

        subset =  CustomedSubset(mnist_data,index,trans,0)
        trainloader = DataLoader(subset, batch_size=32, shuffle=True)

        # test_data = self.process_testdata()
        #
        # testloader = DataLoader(test_data, batch_size=32, shuffle=True)
        # 训练feature_extractor
        optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=0.0001, weight_decay=1e-03)
        loss_function = CrossEntropyLoss()
        for epoch in range(100):
            self.feature_extractor.to("cuda")
            self.feature_extractor.train()
            for batchidx, (x, label) in enumerate(trainloader):
                x = x.to('cuda')
                label = label.to('cuda')
                _, logits = self.feature_extractor(x)  # logits: [b, 10]
                loss = loss_function(logits, label)  # 标量

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        torch.save(self.feature_extractor, './pretrain_models/resnet-for_MNIST.pth')
        print(f'mnist_pretrain_finish...')


    def train_feature_extractor_svhn(self):
        self.feature_extractor = resnet18_cbam(True)
        self.feature_extractor.to('cuda')
        trans = build_transform(True, self.input_size)
        mnist_data = self.svhn_data_train

        class_counts_mnist = torch.zeros(10)  # 每个类的数量
        class_label_mnist = [[] for i in range(10)]  # 每个类的index
        j = 0
        for x, label in tqdm(mnist_data):
            label = int(label)
            class_counts_mnist[label] += 1
            class_label_mnist[label].append(j)
            j += 1

        index = []
        for j in range(10):
            # for mnist
            num = int(class_counts_mnist[j])
            unused_indice1 = set(class_label_mnist[j])
            q = 0
            while q < int(num * 0.05):
                random_index = random.choice(list(unused_indice1))
                index.append(random_index)
                unused_indice1.remove(random_index)
                q += 1

        subset =   CustomedSubset(mnist_data,index,trans,0)
        trainloader = DataLoader(subset, batch_size=32, shuffle=True)

        # test_data = self.process_testdata()
        #
        # testloader = DataLoader(test_data, batch_size=32, shuffle=True)
        # 训练feature_extractor
        optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=0.0001, weight_decay=1e-03)
        loss_function = CrossEntropyLoss()
        for epoch in range(100):
            self.feature_extractor.to("cuda")
            self.feature_extractor.train()
            for batchidx, (x, label) in enumerate(trainloader):
                x = x.to('cuda')
                label = label.to('cuda')
                _, logits = self.feature_extractor(x)  # logits: [b, 10]
                loss = loss_function(logits, label)  # 标量

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        torch.save(self.feature_extractor, './pretrain_models/resnet-for_svhn.pth')
        print(f'svhn_pretrain_finish...')




class CustomedSubset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int],trans,set_count) -> None:

        self.indices = indices
        self.data = []
        self.targets = []
        self.dataset = dataset
        self.transform = trans
        for i in self.indices:
            self.data.append(dataset.data[i])
            self.targets.append(dataset.targets[i])
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.target_transform = None
        self.set_count = set_count

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        # if len(img.shape)==2:
        #     img = img.unsqueeze(0)

        if type(img) == type(torch.Tensor(0)):
            img = img.cpu().numpy()
            img = Image.fromarray(img).convert('RGB')
        elif img.shape== (3, 32, 32):
            img = Image.fromarray(img.transpose(1,2,0))
        else:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img,target

    def __len__(self):
        return len(self.indices)




class CustomedDualSubset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int],dataset1: Dataset[T_co], indices1: Sequence[int],trans,set_count) -> None:

        self.indices = indices
        self.indices1 =indices1

        self.dataset = dataset
        self.transform = trans
        self.data = []
        self.targets = []

        for i in self.indices:
            self.data.append(dataset.data[i])
            self.targets.append(dataset.targets[i])


        for i in self.indices1:
            self.data.append(dataset1[i])
            self.targets.append(dataset1.targets[i])


        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.target_transform = None
        self.set_count = set_count

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        # if len(img.shape)==2:
        #     img = img.unsqueeze(0)

        if type(img) == type(torch.Tensor(0)):
            img = img.cpu().numpy()
            img = Image.fromarray(img).convert('RGB')

        elif img.shape== (3, 32, 32):
            img = Image.fromarray(img.transpose(1,2,0))

        elif img.shape== (1, 32, 32):
            img = Image.fromarray(img).convert('RGB')

        else:
            img = Image.fromarray(img).convert('RGB')


        if self.transform is not None:
            img = Image.fromarray(img).convert('RGB')
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img,target

    def __len__(self):
        return len(self.indices)






