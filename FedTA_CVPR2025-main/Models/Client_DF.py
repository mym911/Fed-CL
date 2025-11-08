import time
from copy import deepcopy
import random

import numpy as np
import torch
from timm.optim import create_optimizer
from torch import nn
from torch.autograd import Variable
from torch.utils.data import random_split, DataLoader, Dataset
from tqdm import tqdm

# from Models.ResNet import resnet18_cbam
from Models.Tail_Anchor import Tail_Anchor

from Models.classification_head import Chead

from data.cifar100_subset_spliter import CustomedSubset
from data.iCIFAR100c import iCIFAR100c
from utils import accuracy, CosineSimilarityClassifier
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.manifold import MDS
# import pandas as pd
#
# import seaborn as sns


# 每个客户端拥有的
class Client_DF(object):

    def __init__(self, id, original_model, model_name, task_per_global_epoch, subset, local_epoch, batch_size, lr, device, method, class_mask, args, vit):
        self.id = id
        self.original_model = original_model
        self.vit = vit  # 输入增强

        self.task_id = -1
        self.task_per_global_epoch = task_per_global_epoch
        self.test_loader = []
        # subset应该是一个【】，其中包含了num_task个数据以及类别，以[[(类别)：[数据]]，{}]的形式保存
        self.train_data = subset
        self.local_epoch = local_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.method = method
        self.nb_classes = args.nb_classes
        self.model = self.init_local_model(model_name)
        if args.data_name == 'cifar100' or args.data_name=='5datasets' or args.data_name=='ImageNet-R' or args.data_name=='svhn-mnist':
            self.class_mask = class_mask
        else:
            self.class_mask = []

        self.local_protos = None
        self.global_protos = None
        self.prompts = None
        self.heads = [None,None,None,None,None,None,None,None,None,None]
        self.vit_heads = [None,None,None,None,None,None,None,None,None,None]


        self.head = Chead(args.nb_classes)


    def init_local_model(self, model_name):
        if model_name=='Tail_Anchor':
            return Tail_Anchor(10, 768, 200)
        else:
            return resnet18_cbam(pretrained=False)

    def get_data_office_home(self, task_id, data, mask):
        self.train_dataset = data
        self.current_class = mask
        self.class_mask.append(mask)
        # self.train_dataset = self.train_data[task_id]
        # self.current_class = self.class_mask[task_id]
        print(f'{self.id} client，{task_id} task has {len(self.current_class)} classes:{self.current_class}')
        trainset = self.train_dataset
        traindata, testdata = random_split(trainset,
                                           [int(len(trainset) * 0.7), len(trainset) - int(len(trainset) * 0.7)])
        testdata = deepcopy(testdata)
        self.test_loader.append(testdata)

        self.traindata = traindata

    def get_data(self, task_id):
        self.train_dataset = self.train_data[task_id]
        self.current_class = self.class_mask[task_id]
        print(f'{self.id} client，{task_id} task has {len(self.current_class)} classes:{self.current_class}')
        trainset = self.train_dataset
        traindata, testdata = random_split(trainset,
                                           [int(len(trainset) * 0.7), len(trainset) - int(len(trainset) * 0.7)])

        self.test_loader.append(testdata)

        self.traindata = traindata
        print(len(traindata))


    def update_data(self, round, args):
        task = round // self.task_per_global_epoch
        flag = True
        if self.task_id != task:
            flag = False
            if args.data_name == 'cifar100' or args.data_name == '5datasets' or args.data_name == 'ImageNet-R' or args.data_name == 'svhn-mnist':
                self.get_data(task)
            self.task_id = task


    def train(self, round, args):
        self.original_model.eval()

        if self.prompts is not None:
            self.vit.load_prompts(self.prompts)
        else:
            self.vit.init_prompts()

        ###train###
        # update train data

        self.model.to(self.device)
        self.vit.to(self.device)
        train_loader = DataLoader(self.traindata, batch_size=16, num_workers=args.num_workers, shuffle=True)
        print(f'Client {self.id} on Task {self.task_id} is training prompts')

        # training input enhancement
        optimizer = torch.optim.Adam(self.vit.parameters(), lr=self.lr, weight_decay=1e-03)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        cos = nn.CosineEmbeddingLoss()
        for epoch in tqdm(range(self.local_epoch)):
            for iteration, (input, target) in enumerate(train_loader):
                input, target = Variable(input, requires_grad=False).to(self.device, non_blocking=True), target.long().to(self.device, non_blocking=True)

                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(input)
                        cls_features = output['pre_logits']
                output = self.vit(input, task_id=self.task_id, cls_features=cls_features, train=True)
                # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
                # logits = pre
                logits = output['logits']  # 仅由vit得到的分类输出
                # output_mixed = output['pre_logits']
                pull_off = output['reduce_sim']  # 做拉进约束使用
                # class_mask
                mask = self.current_class
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                loss = criterion(logits, target) - 0.1 * pull_off
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                optimizer.step()

        ###evaluate###

        # self.evaluate_only_prompts(0,args.nb_classes)
        # self.evaluate_only_prompts(self.task_id,args.nb_classes)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-03)
        for epoch in tqdm(range(self.local_epoch)):
            for iteration, (input, target) in enumerate(train_loader):
                input, target = Variable(input, requires_grad=False).to(self.device, non_blocking=True), target.long().to(self.device, non_blocking=True)

                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(input)
                        cls_features = output['pre_logits']
                        output = self.vit(input, task_id=self.task_id, cls_features=cls_features, train=True)
                pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))  # 锚点的前向函数
                logits = pre
                # NCE loss
                if self.global_protos is None:
                    loss_InfoNCE = 0
                else:
                    count = 0
                    loss_InfoNCE = None
                    for i, label in enumerate(target):
                        if label.item() in self.global_protos.keys():
                            count += 1
                            feature = output_mixed[i].unsqueeze(0)
                            loss_instance = self.calculate_infonce(feature, label.item(), (round+1) % self.task_per_global_epoch==0)
                            if loss_InfoNCE is None:
                                loss_InfoNCE = loss_instance
                            else:
                                loss_InfoNCE += loss_instance
                    if count != 0:
                        loss_InfoNCE = loss_InfoNCE / count
                    else:
                        loss_InfoNCE = 0
                loss_InfoNCE = loss_InfoNCE

                mask = self.current_class
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                loss = criterion(logits, target) + 0.2 * self.task_per_global_epoch * loss_InfoNCE - 0.1 * pull_off2

                if round == 16 and self.id == 0:
                    print(loss_InfoNCE)
                    print(criterion(logits, target))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        target_list = []
        feature_list = []
        for idx, (input, target) in enumerate(train_loader):
            input, target = Variable(input, requires_grad=False).to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input.to(self.device))
                    output = output['pre_logits'].requires_grad_(False)
                output = self.vit(input, task_id=self.task_id, cls_features=output, train=True)
                _, output_mixed, _ = self.model(output['feat'].to(self.device),target.to(self.device))

            # print(np.isnan(output_mixed.cpu().detach().numpy()).any())
            if not np.isnan(output_mixed.cpu().detach().numpy()).any():
                target_list.append(target)
                feature_list.append(output_mixed)
        local_protos = {}
        if target_list !=[]:
            target_list = torch.cat(target_list, dim=0)
            feature_list = torch.cat(feature_list, dim=0)

            local_protos = {}
            for class_index in self.current_class:
                data_index = (target_list == class_index).nonzero().squeeze(-1)
                if data_index.shape[0] != 0:
                    all_features = feature_list[data_index]
                    proto = all_features.mean(0).cpu().detach().numpy()
                    local_protos[class_index] = proto

        self.local_protos =local_protos

        # save local classifier
        self.heads[self.task_id] = deepcopy(self.model.get_head())

        self.evaluate( 0,args.nb_classes)
        self.evaluate(self.task_id, args.nb_classes)

        # save local input enhancement
        self.prompts = deepcopy(self.vit.get_prompts())

    def load_global_weights(self,weights):
        self.model.load_state_dict(weights)
        self.evaluate(self.task_id,self.nb_classes)

    def get_global_proto_and_head(self, proto, head, prompt, round):  # 从服务器获取全局原型/分类头/prompt后加载，并做评估。
        self.global_protos = deepcopy(proto)

        self.global_head = head

        self.prompts = prompt
        self.vit.load_prompts(self.prompts)

        self.evaluate(0, self.nb_classes)
        self.evaluate(self.task_id,self.nb_classes)

    def get_global_proto_and_head_no_test(self, proto, head, prompt, round):
        self.global_protos = deepcopy(proto)

        self.global_head = head

        self.prompts = prompt
        self.vit.load_prompts(self.prompts)


    def get_head(self,head):
        self.heads[self.task_id] =deepcopy(head)
        self.evaluate_only_heads(0, self.nb_classes)
        self.evaluate_only_heads(self.task_id, self.nb_classes)

    def train_only_heads(self, round, args):

        ###train###
        # update train data
        task = round // self.task_per_global_epoch
        flag = True
        if self.task_id != task:
            flag = False
            if args.data_name == 'cifar100' or args.data_name == '5datasets' or args.data_name == 'ImageNet-R':
                self.get_data(task)
            self.task_id = task

        if self.heads[self.task_id] is not None:
            self.head.load_head(self.heads[self.task_id])

        self.head.to(self.device)
        train_loader = DataLoader(self.traindata, batch_size=16, num_workers=args.num_workers,
            pin_memory=args.pin_mem, shuffle=True)
        print(f'Client {self.id} on Task {self.task_id} is training prompts')

        # Input-Enhancement
        optimizer = torch.optim.Adam(self.head.parameters(), lr=self.lr,weight_decay=1e-03)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        for epoch in range(self.local_epoch):
            for iteration, (input,target) in enumerate(train_loader):
                input, target = Variable(input, requires_grad=False).to(self.device, non_blocking=True), target.to(self.device,non_blocking=True)
                # input走一遍获得cls——token
                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(input)
                output = self.head(output['feat'])
                logits = output
                # 加了class_mask
                mask = self.current_class
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                loss = criterion(logits,target)
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                optimizer.step()

        ###evaluation###
        self.heads[self.task_id] = deepcopy(self.head.get_head())
        self.evaluate_only_heads(0,args.nb_classes)
        self.evaluate_only_heads(self.task_id,args.nb_classes)


    def evaluate(self, task=0, nb_classes=None):
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data, batch_size=8, shuffle=True)
        correct = 0
        total = 0

        self.model.load_head(self.heads[task])
        self.model.to(self.device)

        for iteration, (input,target) in enumerate(test_loader):
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            # if iteration==0:
            #     input_ = input[0]
            #     self.model.forward_visual(input_)
            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    output = output['pre_logits'].requires_grad_(False)
                    output = self.vit(input, task_id=self.task_id, cls_features=output, train=True)
                    pre, output_mixed, _ = self.model(output['feat'].to(self.device), target.to(self.device))
            logits = pre

            # logits = output['logits']

            # class_mask
            mask = self.class_mask[task]
            not_mask = np.setdiff1d(np.arange(nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            predicts = torch.max(logits, dim=1)[1].cpu()
            # print(predicts)
            # print(target)
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total

        print(f'{acc}')


    def evaluate_cosin_similarity(self, task=0, nb_classes=None):
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=2)
        correct = 0
        total = 0

        for iteration, (input, target) in enumerate(test_loader):
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)

                _, output_mix, _ = self.model(output, None)
            for i, label in enumerate(target):
                predicts = CosineSimilarityClassifier(output_mix[i].squeeze(0),self.global_protos,self.current_class)
                if predicts ==label:
                    correct +=1
            total += len(target)
        acc = 100 * correct / total
        print(f'Client {self.id} on Task {task} acc is {acc}')

    def evaluate_only_prompts(self, task=0, nb_classes=None):  # 针对的加强效果评估。
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=2)

        correct = 0
        total = 0
        for iteration, (input, target) in enumerate(test_loader):
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            # if iteration==0:
            #     input_ = input[0]
            #     self.model.forward_visual(input_)
            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    output = output['pre_logits'].requires_grad_(False)
                output = self.vit(input, task_id=self.task_id, cls_features=output, train=True)
            #     pre, output_mixed, _ = self.model(output['feat'].to(self.device), target.to(self.device))
            # logits = pre

            logits = output['logits']
            # class_mask
            mask = self.class_mask[task]
            not_mask = np.setdiff1d(np.arange(nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            predicts = torch.max(logits, dim=1)[1].cpu()
            # print(predicts)
            # print(target)
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total

        print(f'{acc}')


    def evaluate_only_heads(self, task=0, nb_classes=None):
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=2)
        self.vit.load_head(self.heads[task])
        self.vit.to(self.device)
        correct = 0
        total = 0
        for iteration, (input, target) in enumerate(test_loader):
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)

                output = self.head(output['feat'])
            logits = output
            # 加了class_mask
            mask = self.class_mask[task]
            not_mask = np.setdiff1d(np.arange(nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
            predicts = torch.max(logits, dim=1)[1].cpu()
            # print(predicts)
            # print(target)
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total

        print(f'{acc}')

    # Contrastive Learning
    def calculate_infonce(self, feature, label,is_last):
        # print(self.global_protos.keys())
        # print(label)

        all_global_protos_keys = np.array(list(self.global_protos.keys()))
        all_protos = []
        for protos_key in all_global_protos_keys:
            all_protos.append(self.global_protos[protos_key])
        all_protos = np.vstack(all_protos)


        pos_index = np.where(all_global_protos_keys == label)[0]
        neg_index = np.where(
            (all_global_protos_keys != label)
        )[0]
        f_pos = torch.from_numpy(all_protos[pos_index]).to(self.device)
        f_neg = torch.from_numpy(all_protos[neg_index]).to(self.device)
        f_proto = torch.cat((f_pos, f_neg), dim=0)

        l = torch.cosine_similarity(feature, f_proto, dim=1)


        l = l / 0.2

        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [
            0 for _ in range(f_neg.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
        pos_mask = pos_mask.view(1, -1)
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        if is_last:
            infonce_loss = 1-torch.log(sum_pos_l)
        else:
            infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
        return infonce_loss


    def show_embeddings(self,train_loader,task):
        target_list = []
        feature_list = []
        keep = [48, 71, 84, 89, 93]

        self.model.to(self.device)
        for idx, (input,target) in enumerate(train_loader):
            input, target = Variable(input, requires_grad=False).to(self.device, non_blocking=True), target.to(
                self.device, non_blocking=True)
            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    output = output['pre_logits'].requires_grad_(False)
                    output = self.vit(input, task_id=self.task_id, cls_features=output, train=True)
                    pre, output_mixed, _ = self.model(output['feat'].to(self.device), target.to(self.device))
                    for i in range(len(target)):
                        if target[i] in keep:
                            target_list.append(target[i])
                            feature_list.append(output_mixed[i])

        target_list = torch.Tensor(target_list)
        feature_list = torch.cat(feature_list,dim=0)

        target_list = target_list.cpu()
        feature_list = feature_list.reshape(-1, 1536).cpu()

        # print(target_list.shape)
        # print(feature_list.shape)
        tsne = TSNE(n_components=2, early_exaggeration=2.0,metric='cosine', random_state=42)
        x_tsne = tsne.fit_transform(feature_list)
        self.plot_xy(x_tsne, target_list,task)

    def plot_xy(self, x_values, label, task):

        df = pd.DataFrame(x_values, columns=['x', 'y'])
        df['label'] = np.array(label)
        df['label'].astype(str)
        colors = ['darkorange','darkgreen','darkblue','darkred','darkorchid']
        sns.scatterplot(data=df,x="x", y="y", hue='label',palette=colors)

        plt.axis('off')
        # plt.legend(False)
        plt.savefig(f'{task} scattor_plot client {self.id}', bbox_inches='tight', pad_inches=0.0)
        plt.show()


    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm


    def train_only_prompts(self, round, args):
        self.original_model.eval()

        if self.prompts is not None:
            self.vit.load_prompts(self.prompts)
        else:
            self.vit.init_prompts()

        if self.heads[self.task_id] is not None:
            self.vit.load_head(self.heads[self.task_id])

        ###train###
        # update train data
        task = round // self.task_per_global_epoch
        flag = True
        if self.task_id != task:
            flag = False
            if args.data_name == 'cifar100' or args.data_name=='5datasets' or args.data_name == 'ImageNet-R':
                self.get_data(task)
            self.task_id = task

        self.model.to(self.device)
        self.vit.to(self.device)
        train_loader = DataLoader(self.traindata, batch_size=16, num_workers=args.num_workers,
            pin_memory=args.pin_mem, shuffle=True)
        print(f'Client {self.id} on Task {self.task_id} is training prompts')

        # Input-Enhancement
        optimizer = torch.optim.Adam(self.vit.parameters(), lr=self.lr,weight_decay=1e-03)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        cos = nn.CosineEmbeddingLoss()
        for epoch in tqdm(range(self.local_epoch)):
            for iteration, (input,target) in enumerate(train_loader):
                input, target = Variable(input, requires_grad=False).to(self.device, non_blocking=True), target.to(self.device,non_blocking=True)

                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(input)
                        cls_features = output['pre_logits']
                output = self.vit(input, task_id=self.task_id, cls_features=cls_features, train=True)
                # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
                # logits = pre
                logits = output['logits']
                # output_mixed = output['pre_logits']
                pull_off = output['reduce_sim']
                # class_mask
                mask = self.current_class
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                loss = criterion(logits,target)  - 0.1 * pull_off
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                optimizer.step()

        self.heads[self.task_id] = deepcopy(self.vit.head)
        self.prompts = deepcopy(self.vit.get_prompts())
        self.evaluate_only_prompts(0,args.nb_classes)
        self.evaluate_only_prompts(self.task_id,args.nb_classes)



    def get_global_prompt_head(self,head,prompt):
        # self.global_protos = deepcopy(proto)
        self.heads[self.task_id] = deepcopy(head)
        self.model.head.to(self.device)
        self.prompts = prompt
        self.vit.load_prompts(self.prompts)
        self.evaluate_only_prompts(0, self.nb_classes)
        self.evaluate_only_prompts(self.task_id,self.nb_classes)




    def evaluate_on_global_testset(self,testdata):

        test_loader = DataLoader(testdata, batch_size=16, shuffle=True, num_workers=2)

        self.vit.load_head(self.global_head)
        self.vit.to(self.device)

        correct = 0
        total = 0
        for iteration, (input, target) in enumerate(test_loader):
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    output = output['pre_logits'].requires_grad_(False)
                    output = self.vit(input, task_id=self.task_id, cls_features=output, train=True)
                    pre, output_mixed, _ = self.model(output['feat'].to(self.device), target.to(self.device))

                output = self.head(output['feat'])
            logits = output

            # 加了class_mask

            predicts = torch.max(logits, dim=1)[1].cpu()
            # print(predicts)
            # print(target)
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total

        print(f'{acc}')
