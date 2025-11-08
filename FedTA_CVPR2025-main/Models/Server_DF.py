import random
import time
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from Models.Client_DF import Client_DF
from Models.classification_head import Chead

from utils import accuracy, global_distillation_loss
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity

class Server_DF(object):

    def __init__(self,id,origin_model,model_name,client_num,task_num,subset,class_mask,lr,global_epoch,local_epoch,batch_size,device,method,threshold,surrogate_data,test_data,args,model):
        self.id = id
        self.model_name = model_name
        self.origin_model = origin_model
        self.model = model

        self.client_num=client_num
        self.task_num = task_num
        self.clients =[]

        # Every client has a subset and mask
        self.client_data = subset
        self.class_mask = class_mask


        self.surrogate_data = surrogate_data
        self.lr = lr
        self.batch_size = batch_size
        self.global_epoch = global_epoch
        self.local_epoch = local_epoch
        self.device = device
        self.method = method
        self.threshold = threshold
        self.test_data = test_data
        self.args = args

        self.task_id = -1

        self.existing_class =set()
        self.global_head = Chead(args.nb_classes)

        self.global_protos = None
        self.temp_protos =None

        self.fix_keys = []
    def init_client(self):
        print('Initialize clients')
        for i in range(self.client_num):    #
            # id,original_model,vit,task_per_global_epoch,subset,local_epoch,batch_size,lr,device,method
            if self.args.data_name == 'cifar100' or self.args.data_name == '5datasets' or self.args.data_name == 'ImageNet-R':
                self.clients.append(Client_DF(i,self.origin_model,self.model_name,self.global_epoch,self.client_data[i],
                                              self.local_epoch,self.batch_size,self.lr,self.device,self.method,self.class_mask[i],self.args,self.model
                                           ))
            else:
                self.clients.append(Client_DF(i, self.origin_model, self.model_name, self.global_epoch, None,
                                              self.local_epoch, self.batch_size, self.lr, self.device, self.method,
                                              None,self.args
                                              ))
        print("Initialization completes")

    def fedavg(self, sample_nums):
        training_num = 0
        for idx in range(len(sample_nums)):
            samplenum = sample_nums[idx]
            training_num += samplenum
        sample_num = sample_nums[0]
        averaged_params = self.clients[0].model.cpu().state_dict()
        for k in averaged_params.keys():
            for i in range(0, self.client_num):
                local_sample_number = sample_nums[i]
                local_model_params = self.clients[i].model.cpu().state_dict()
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params




    def train_clients(self):
        used = []
        for i in range(self.task_num * self.global_epoch):
            self.thisclients = [i for i in range(self.client_num)]

            # if len(used) ==0:
            #     used = self.thisclients
            # else:
            #     used.extend(self.thisclients)
            #     used = list(set(used))
            # if len(used) !=20 and i >0:
            #     not_used = [j for j in range(self.client_num) if j not in used]
            #     chosen = random.choice(not_used)
            #     for j in range(len(self.thisclients)):
            #         if j in used:
            #             self.thisclients[j] = chosen
            #             used.append(chosen)
            #             break
            # print(used)


            for j in range(self.client_num):
                self.clients[j].update_data(round=i, args=self.args)


            if self.task_id != i//self.global_epoch:
                self.task_id = i//self.global_epoch
                if self.args.data_name=='office_home':
                    datas,mask = self.client_data.random_split(domain=self.task_id)
                    for j in range(self.client_num):
                        self.clients[j].get_data_office_home(self.task_id,datas[j],mask[j])


            print(f"--------round {i},task number {i//self.global_epoch}-----------")
            for j in self.thisclients:
                self.clients[j].train(round=i,args=self.args)

            # FedTA
            # if i ==15 or i==16:
            #     for j in self.thisclients:
            #         print(f'{j} client')
            #         for q in self.clients[j].local_protos.keys():
            #             print(f'{q} is {self.clients[j].local_protos[q]}')

            self.choose_best_proto_greedy_similarity_fixed_key((i+2)%self.global_epoch==0,threshold=0.25,round=i)

            if i%self.global_epoch !=4:
                self.kd_fusion_prompt(self.thisclients)

            self.fed_avg_head(self.thisclients)

            print('Server aggregation Compelte, results are ()：')
            for j in range(self.client_num):
                # FedTA
                if j in self.thisclients:
                    self.clients[j].get_global_proto_and_head(self.global_protos,self.global_head,self.prompt,i)
                    print('-------')

                else:
                    self.clients[j].get_global_proto_and_head_no_test(self.global_protos, self.global_head, self.prompt, i)



        print("All Process completes")

    def fuse_protos(self):
        global_protos = dict()
        temp = dict()
        for i in self.clients:
            for label in i.local_protos.keys():
                if label in temp:
                    temp[label].append(i.local_protos[label])
                else:
                    temp[label] = [i.local_protos[label]]
        for label in temp.keys():
            global_protos[label] = np.mean(temp[label], axis=0)
            temp[label] = np.vstack(temp[label])
        self.global_protos = global_protos

        self.temp_protos = temp





    def choose_best_proto_greedy_similarity_fixed_key(self,is_fix=False,threshold=0.2,round=15):

        if self.global_protos != None:
            global_protos = self.global_protos  # global protos from last round
        else:
            global_protos = dict()


        temp = dict()
        this_round = dict()
        # {1:[x,x,x], 2:[q,q,q]}


        for i in self.thisclients:
            for label in self.clients[i].local_protos.keys():
                if label not in self.fix_keys:
                    if label in temp:
                        temp[label].append(self.clients[i].local_protos[label])
                    else:
                        temp[label] = [self.clients[i].local_protos[label]]

                    if label in this_round:
                        if not np.isnan(self.clients[i].local_protos[label]).any():
                            this_round[label].append(self.clients[i].local_protos[label])
                    else:
                        if not np.isnan(self.clients[i].local_protos[label]).any():
                            this_round[label] = [self.clients[i].local_protos[label]]



        if len(this_round.keys()) !=0 and self.fix_keys !=[]:
            keys = list(this_round.keys())
            keys.extend(self.fix_keys)
        elif len(this_round.keys())==0  and self.fix_keys != []:

            keys = self.fix_keys
        else:
            keys = list(this_round.keys())


        matrix = None
        first_shape = None
        for key, value in global_protos.items():
            if first_shape is None:
                first_shape = value.shape
                break
        different_shapes = {}
        print(first_shape)
        for key, value in global_protos.items():
            if value.shape != first_shape:
                different_shapes[key] = value.shape
            if len(value.shape) == 3:
                global_protos[key] = global_protos[key].squeeze()
        if different_shapes:
            print("Some values have different shapes:")
            for key, shape in different_shapes.items():
                print(f"Key: {key}, Shape: {shape}")
        else:
            print("All values have the same shape.")


        num = []
        for i in range(len(keys)):
            if keys[i] not in self.fix_keys:
                if matrix is None:
                    matrix = np.array(this_round[keys[i]])
                else:
                    matrix = np.concatenate([matrix, this_round[keys[i]]])

                if keys[i] in global_protos.keys():

                    matrix = np.concatenate([matrix, global_protos[keys[i]].unsqueeze(0)])
                    num.append(len(this_round[keys[i]]) + 1)
                else:
                    num.append(len(this_round[keys[i]]))
            else:
                if matrix is None:
                    matrix = np.array(self.global_protos[keys[i]].unsqueeze(0))
                else:
                    matrix = np.concatenate([matrix, self.global_protos[keys[i]].unsqueeze(0)])
                num.append(1)

        # if round ==15:
        #     contains_nan_rows = np.isnan(matrix).any(axis=1)
        #     print(f"Rows containing NaN: {contains_nan_rows}")
        #     print(matrix[contains_nan_rows])
        #     if self.global_protos:
        #         for q in self.global_protos.keys():
        #             print(f'{q} global proto is {self.global_protos[q]}')
        #     for q in this_round.keys():
        #         print(f'{q} proto is {this_round[q]}')


        # matrix = self.l2_normalize(torch.Tensor(matrix).to(self.device),dim=1)


        # adjecent matrix
        adj_matrix = torch.tensor(cosine_similarity(matrix))
        print(adj_matrix.shape)
        matrix = torch.Tensor(matrix).to(self.device)

        # set the distance of the same class' protos to 1
        low_bound = 0
        for j in num:
            high_bound = low_bound + j
            for h in range(low_bound, high_bound):
                for k in range(low_bound, high_bound):
                    adj_matrix[h][k] = 1.0
            low_bound = high_bound


        # print(adj_matrix)
        remain_keys = set(temp.keys())


        low_bound = 0
        map = {}
        for j in range(len(num)):
            if keys[j] not in self.fix_keys:
                high_bound = low_bound + num[j]
                min = 999
                chose = None
                for h in range(low_bound, high_bound):
                    aver_simi = torch.mean(adj_matrix[h])
                    if aver_simi < min:
                        min = aver_simi
                        chose = h
                if min <=threshold:
                    self.fix_keys.append(keys[j])
                map[keys[j]] = min
                global_protos[keys[j]] = matrix[chose].cpu()
                low_bound = high_bound

            else:
                high_bound = low_bound + num[j]
                aver_simi = torch.mean(adj_matrix[low_bound])
                map[keys[j]] = aver_simi
                global_protos[keys[j]] = self.global_protos[keys[j]]
                low_bound = high_bound

        # print(map)

        # print(global_protos.keys())
        self.global_protos = global_protos
        if is_fix:
            if self.fix_keys !=[]:
                self.fix_keys.extend(list(global_protos.keys()))
                self.fix_keys = list(set(self.fix_keys))
            else:
                self.fix_keys = list(global_protos.keys())


        print(len(self.fix_keys))

        first_shape = None
        for key, value in self.global_protos.items():
            if first_shape is None:
                first_shape = value.shape
                break
        different_shapes = {}
        for key, value in self.global_protos.items():
            if value.shape != first_shape:
                different_shapes[key] = value.shape
            if len(value.shape) == 3:
                self.global_protos[key] = self.global_protos[key].squeeze()

        if different_shapes:
            print("Some values have different shapes:")
            for key, shape in different_shapes.items():
                print(f"Key: {key}, Shape: {shape}")
        else:
            print("All values have the same shape.")

    def choose_best_proto_greedy_similarity_fixed_key_UPDATE(self,is_fix=False,threshold=0.25):

        if self.global_protos != None:
            global_protos = self.global_protos  # global protos from last round
        else:
            global_protos = dict()


        this_round = dict()
        # {1:[x,x,x], 2:[q,q,q]}


        temp = dict()


        for i in self.thisclients:
            for label in self.clients[i].local_protos.keys():
                if label not in self.fix_keys:
                    if label in temp:
                        temp[label].append(self.clients[i].local_protos[label])
                    else:
                        temp[label] = [self.clients[i].local_protos[label]]


                    if label in this_round:
                        this_round[label].append(self.clients[i].local_protos[label])
                    else:
                        this_round[label] = [self.clients[i].local_protos[label]]


        # for label in self.global_protos.keys():
        #     if label in self.fix_keys:
        #         temp[label] = self.global_protos[label]
        #         this_round[label] = self.global_protos[label]



        if len(this_round.keys()) !=0 and self.fix_keys !=[]:
            keys = list(this_round.keys())
            keys.extend(self.fix_keys)

        elif len(this_round.keys())==0  and self.fix_keys != []:

            keys = self.fix_keys
        else:
            keys = list(this_round.keys())



        first_shape = None
        for key, value in global_protos.items():
            if first_shape is None:
                first_shape = value.shape
                break
        different_shapes = {}
        print(first_shape)
        for key, value in global_protos.items():
            if value.shape != first_shape:
                different_shapes[key] = value.shape
            if len(value.shape) == 3:
                global_protos[key] = global_protos[key].squeeze()
        if different_shapes:
            print("Some values have different shapes:")
            for key, shape in different_shapes.items():
                print(f"Key: {key}, Shape: {shape}")
        else:
            print("All values have the same shape.")



        num = []
        for i in range(len(keys)):

            if keys[i] not in self.fix_keys:

                if matrix is None:
                    matrix = np.array(this_round[keys[i]])
                else:
                    matrix = np.concatenate([matrix, this_round[keys[i]]])

                if keys[i] in global_protos.keys():
                    matrix = np.concatenate([matrix, global_protos[keys[i]].unsqueeze(0)])
                    num.append(len(this_round[keys[i]]) + 1)
                else:
                    num.append(len(this_round[keys[i]]))



            else:
                if matrix is None:
                    matrix = np.array(self.global_protos[keys[i]].unsqueeze(0))
                else:
                    matrix = np.concatenate([matrix, self.global_protos[keys[i]].unsqueeze(0)])
                num.append(1)

        matrix =self.l2_normalize(torch.Tensor(matrix).to(self.device),dim=1)

        # adjecent matrix
        adj_matrix = torch.matmul(matrix,matrix.t())

        # set the distance of the same class' protos to 1
        low_bound = 0
        for j in num:
            high_bound = low_bound + j
            for h in range(low_bound, high_bound):
                for k in range(low_bound, high_bound):
                    adj_matrix[h][k] = 1.0
            low_bound = high_bound
        # print(adj_matrix)

        remain_keys = set(temp.keys())

        print(adj_matrix)

        low_bound = 0
        map = {}
        for j in range(len(num)):
            if keys[j] not in self.fix_keys:
                high_bound = low_bound + num[j]
                min = 999
                chose = None
                for h in range(low_bound, high_bound):
                    aver_simi = torch.mean(adj_matrix[h])
                    if aver_simi < min:
                        min = aver_simi
                        chose = h
                if min <=threshold:
                    self.fix_keys.append(keys[j])
                map[keys[j]] = min
                global_protos[keys[j]] = matrix[chose].cpu()
                low_bound = high_bound
            else:
                high_bound = low_bound + num[j]
                aver_simi = torch.mean(adj_matrix[low_bound])
                map[keys[j]] = aver_simi
                global_protos[keys[j]] = self.global_protos[keys[j]]
                low_bound = high_bound

        # print(map)

        # print(global_protos.keys())
        self.global_protos = global_protos
        if is_fix:
            if self.fix_keys !=[]:
                self.fix_keys.extend(list(global_protos.keys()))
                self.fix_keys = list(set(self.fix_keys))
            else:
                self.fix_keys = list(global_protos.keys())

        print(self.fix_keys)
        print(len(self.fix_keys))


        first_shape = None
        for key, value in self.global_protos.items():
            if first_shape is None:
                first_shape = value.shape
                break
        different_shapes = {}
        for key, value in self.global_protos.items():
            if value.shape != first_shape:
                different_shapes[key] = value.shape
            if len(value.shape) == 3:
                self.global_protos[key] = self.global_protos[key].squeeze()

        if different_shapes:
            print("Some values have different shapes:")
            for key, shape in different_shapes.items():
                print(f"Key: {key}, Shape: {shape}")
        else:
            print("All values have the same shape.")


    def kd_fusion_prompt(self,chosen_clients):
        my_result = None
        if len(chosen_clients)==1:
            my_result= deepcopy(self.clients[chosen_clients[0]].global_prompt)
        else:
            classes = set()
            for i in chosen_clients:
                temp = set(self.clients[i].current_class)
                classes = set.union(classes,temp)

            self.surrogate_data.getTrainData(list(classes))


            my_result = deepcopy(self.clients[chosen_clients[0]].prompts)
            test_loader = DataLoader(self.surrogate_data, batch_size=self.batch_size, shuffle=True, num_workers=2)
            self.model.to(self.device)

            optimizer = torch.optim.Adam(my_result.parameters(), lr=self.lr, weight_decay=1e-03)
            for h in tqdm(range(10)):
                for iteration, (index,x,y) in enumerate(test_loader):
                    x = Variable(x, requires_grad=True).to(self.device, non_blocking=True)
                    y = y.long().to(self.device)
                    my_result.to(self.device)
                    with torch.no_grad():
                        if self.origin_model is not None:
                            output = self.origin_model(x)
                            cls_features = output['pre_logits']
                        else:
                            cls_features = None

                    self.model.load_prompts(my_result)
                    output = self.model.forward_features(x, cls_features=cls_features, train=False)['x']
                    outputs = []
                    with torch.no_grad():
                        for localmodel in chosen_clients[1:]:
                            prompt = self.clients[localmodel].prompts
                            self.model.load_prompts(prompt)
                            my = self.model.forward_features(x, cls_features=cls_features, train=False)['x']
                            outputs.append(my)

                    loss = global_distillation_loss(output,outputs)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            self.prompt = my_result
            print('prompt fusion complete')
        return my_result



    def start(self):
        self.init_client()
        self.train_clients()
        # self.test_surro()


    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm


    def fed_avg_prompt(self, chosen_clients):
        prompts = []
        for i in chosen_clients:
            prompts.append(self.clients[i].prompts)

        # key 和 prompt都要平均

        result_prompt = deepcopy(prompts[0].state_dict())

        for k in result_prompt.keys():
            for i in range(len(prompts)):
                local_model_params = prompts[i].state_dict()
                if i == 0:
                    result_prompt[k] = local_model_params[k]
                else:
                    result_prompt[k] += local_model_params[k]

            result_prompt[k] = result_prompt[k] / len(chosen_clients)

        self.prompt = deepcopy(self.clients[0].prompts)
        self.prompt.load_state_dict(result_prompt)
        return self.prompt



    def fed_avg_head(self, chosen_clients):
        heads = []
        for i in chosen_clients:
            heads.append(self.clients[i].vit.head)

        # key 和 prompt都要平均

        result_head = deepcopy(heads[0].state_dict())

        for k in result_head.keys():
            for i in range(len(heads)):
                local_model_params = heads[i].state_dict()
                if i == 0:
                    result_head[k] = local_model_params[k]
                else:
                    result_head[k] += local_model_params[k]

            result_head[k] = result_head[k] / len(chosen_clients)

        self.global_head = deepcopy(self.clients[0].vit.head)
        self.global_head.load_state_dict(result_head)





