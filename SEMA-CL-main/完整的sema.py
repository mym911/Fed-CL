import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
import math
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ..utils.inc_net import SEMAVitNet
from .base import BaseLearner
from ..utils.toolkit import tensor2numpy
from ..backbone.sema_block import SEMAModules

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SEMAVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]

        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        if self._cur_task == 0:
            self._network.fc = nn.Linear(768, data_manager.nb_classes)
            nn.init.kaiming_uniform_(self._network.fc.weight, a=math.sqrt(5))
            nn.init.zeros_(self._network.fc.bias)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                              source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size,
                                                    shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):  # 任务级总控

        self._network.to(self._device)

        if self._cur_task == 0:  # 当前任务在首个增量任务（第0个任务）。
            # show total parameters and trainable parameters
            total_params = sum(p.numel() for p in self._network.parameters())  # 统计参数总数
            print(f'{total_params:,} total parameters.')

            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)  # 只统计requires_grad=True的参数，即参与训练的参数数目
            print(f'{total_trainable_params:,} training parameters.')
            self._train_new(train_loader, test_loader)
        else:  # 非首任务
            for module in self._network.backbone.modules():
                if isinstance(module, SEMAModules):
                    module.detecting_outlier = True  # SEMAModules的detecting_outlier=True，让前向时返回是否需要扩展新适配器的标记
            detect_loader = DataLoader(train_loader.dataset, batch_size=self.args["detect_batch_size"], shuffle=True,
                                       num_workers=num_workers)  # 用于新模式检测的数据加载器。
            added = self._detect_outlier(detect_loader, train_loader, test_loader, 0)  # 本任务总共新增了多少次、层、模块。

            for module in self._network.backbone.modules():
                if isinstance(module, SEMAModules):
                    module.detecting_outlier = False
            if added == 0:
                self.update_optimizer_and_scheduler(num_epoch=self.args['func_epoch'], lr=self.init_lr)
                self._init_train(self.args['func_epoch'], train_loader, test_loader, self.optimizer, self.scheduler,
                                 phase='func')

        for module in self._network.backbone.modules():
            if isinstance(module, SEMAModules):
                module.end_of_task_training()

    def _train_new(self, train_loader, test_loader):  # 首任务调用。两阶段专训器，各自调用_init_train
        self.update_optimizer_and_scheduler(num_epoch=self.args['func_epoch'], lr=self.init_lr)
        self._init_train(self.args['func_epoch'], train_loader, test_loader, self.optimizer, self.scheduler,
                         phase='func')
        self.update_rd_optimizer_and_scheduler(num_epoch=self.args['rd_epoch'], lr=self.args['rd_lr'])
        self._init_train(self.args['rd_epoch'], train_loader, test_loader, self.rd_optimizer, self.rd_scheduler,
                         phase='rd')

    def _detect_outlier(self, detect_loader, train_loader, test_loader, added):  # 侦测分布调用_train_new只训练新加模块
        is_added = False
        for i, (_, inputs, targets) in enumerate(detect_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            model_outcome = self._network(inputs)
            added_record = model_outcome["added_record"]

            if sum(added_record) > 0:  # 说明本批触发了新增适配器。
                added += 1  # 暂时把所有层的 detecting_outlier=False（避免训练时继续触发扩展）。
                is_added = True
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.detecting_outlier = False
                self._train_new(train_loader, test_loader)
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.detecting_outlier = True
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.freeze_functional()
                        module.freeze_rd()
                        module.reset_newly_added_status()

        if is_added:
            return self._detect_outlier(detect_loader, train_loader, test_loader, added)
        else:
            return added

    def _init_train(self, total_epoch, train_loader, test_loader, optimizer, scheduler, phase='func'):  # 单阶段的epoch循环
        prog_bar = tqdm(range(total_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outcome = self._network(inputs)

                logits = outcome["logits"]
                logits = logits[:, :self._total_classes]  # 只保留已出现的类别
                if self._cur_task > 0:
                    logits[:, :self._known_classes] = -float('inf')  # 非首任务时，屏蔽旧类。

                if phase == "func":  # 用交叉熵训练功能模块
                    loss = F.cross_entropy(logits, targets)
                elif phase == "rd":  # 用outcome["rd_loss"]训练RD,分类头不参与优化
                    logits = outcome["logits"]
                    loss = outcome["rd_loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "{} Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                phase,
                self._cur_task,
                epoch + 1,
                total_epoch,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outcome = self._network(inputs)
                logits = outcome["logits"]
                outputs = logits[:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outcome = self._network(inputs)
                logits = outcome["logits"]
                outputs = logits[:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def update_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        lr = self.args["init_lr"] if lr is None else lr
        func_params = [p for n, p in self._network.named_parameters() if
                       ('functional' in n or 'router' in n or 'fc' in n) and p.requires_grad]
        if self.args['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(func_params, momentum=0.9, lr=lr, weight_decay=self.args["weight_decay"])
        elif self.args['optimizer'] == 'adam':
            self.optimizer = optim.AdamW(func_params, lr=lr, weight_decay=self.args["weight_decay"])

        min_lr = self.args['min_lr'] if self.args['min_lr'] is not None else 1e-8
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epoch, eta_min=min_lr)

    def update_rd_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        lr = self.args["rd_lr"] if lr is None else lr
        rd_params = [p for n, p in self._network.named_parameters() if 'rd' in n and p.requires_grad]
        if self.args['optimizer'] == 'sgd':
            self.rd_optimizer = optim.SGD(rd_params, momentum=0.9, lr=lr, weight_decay=self.args["weight_decay"])
        elif self.args['optimizer'] == 'adam':
            self.rd_optimizer = optim.AdamW(rd_params, lr=lr, weight_decay=self.args["weight_decay"])

        min_lr = self.args['min_lr'] if self.args['min_lr'] is not None else 1e-8
        self.rd_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.rd_optimizer, T_max=num_epoch,
                                                                 eta_min=min_lr) if self.rd_optimizer else None

    def save_checkpoint(self, filename):
        state_dict = self._network.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if 'adapter' in k or ('fc' in k and 'block' not in k):
                save_dict[k] = v
        torch.save(save_dict, "{}.pth".format(filename))

    def load_checkpoint(self, filename):
        self._network.load_state_dict(torch.load(filename), strict=False)


import torch
from torch import nn
from typing import List
import copy
import logging
from backbone.sema_components import Adapter, AE, Records

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AdapterModule(nn.Module):  # 适配器
    def __init__(self, config, adapter_id, writer):
        super().__init__()
        self.config = config
        self.functional = Adapter(self.config, adapter_id, dropout=0.1, bottleneck=self.config.ffn_num,
                                  init_option=self.config.ffn_adapter_init_option,
                                  adapter_scalar=self.config.ffn_adapter_scalar,
                                  adapter_layernorm_option=self.config.ffn_adapter_layernorm_option,
                                  )
        layer_id = int(adapter_id.split('.')[0])
        self.not_addition_layer = layer_id < config.adapt_start_layer or layer_id > config.adapt_end_layer
        if self.not_addition_layer:
            self.rd = None
        else:
            self.rd = AE(self.config)
        self.activation = nn.ReLU()
        self.newly_added = True
        self.adapter_id = adapter_id
        self.writer = writer
        self.rd_loss_record = Records(max_len=config.buffer_size)

    def forward(self, x):
        func_out = self.functional(x)
        if self.not_addition_layer:
            rd_loss = torch.tensor(0.).to(device)
            return func_out, rd_loss, torch.zeros_like(rd_loss).to(device)
        else:
            rd_loss = self.rd.compute_reconstruction_loss(x)
        z_score = self.get_z_score_deviation(rd_loss)
        if self.training:
            self.add_z_score_record(rd_loss)
        return func_out, rd_loss, z_score

    def get_z_score_deviation(self, rd_loss):  # 把RD重构损失转化成标准化偏差分数，用于判新/离群检测的函数
        mean, stddev = self.rd_loss_record.mean, self.rd_loss_record.stddev  # 均值，标准差
        if not self.rd_loss_record.length > 2:
            return torch.zeros_like(rd_loss).to(device)
        z_score = (rd_loss - mean) / stddev
        z_score = torch.abs(z_score)
        return z_score

    def add_z_score_record(self, rd_loss):
        self.rd_loss_record.add_record(rd_loss.detach().cpu())


class SEMAModules(nn.Module):  # 一层里的判新器+扩展器+路由加权器,把多适配器如何增、怎么选、何时训组织成可控的前向与扩展流程.
    def __init__(self, config, layer_id, writer):
        super().__init__()
        self.adapters: List[Adapter] = nn.ModuleList()
        self.config = config
        self.act_func = nn.ReLU()
        self.layer_id = layer_id
        self.writer = writer
        self.newly_added = True
        self.added_for_task = True
        self.adapt_start_layer = config.adapt_start_layer
        self.adapt_end_layer = config.adapt_end_layer
        # initialize with one adapter
        self.add_adapter(initialize=True)
        self.added_adapter = 0

        self.router = nn.Linear(config.d_model, 1).cuda()
        self.new_router = None
        self.detecting_outlier = False

    @property
    def num_adapters(self):
        return len(self.adapters)

    def set_new_router(self):
        self.new_router = nn.Linear(self.config.d_model, 1).cuda()

    def fix_router(self):  # 把训练好的新增路由器并入主路由器,让本层路由器的输出层从K→K+1，与新增适配器的数量对齐.
        trained_router = nn.Linear(self.config.d_model, len(self.adapters)).cuda()
        old_router = self.router
        weight = copy.deepcopy(old_router.weight.data)  # 取出旧路由器old_router的权重
        new_weight = copy.deepcopy(self.new_router.weight.data)  # 新路由器new_router的权重
        weight = torch.cat([weight, new_weight])  # 将新旧路由器的权重，在输出维度上做拼接
        trained_router.weight = nn.Parameter(weight)
        bias = copy.deepcopy(old_router.bias.data)
        new_bias = copy.deepcopy(self.new_router.bias.data)
        bias = torch.cat([bias, new_bias])  # 将新旧路由器的偏置在输出维度上做拼接
        trained_router.bias = nn.Parameter(bias)
        self.router = trained_router
        self.new_router = None

    def add_adapter(self, initialize=False):  # 当RD判断新分布需要扩展时,创建并挂载一个新适配器.
        adapter_id = f"{self.layer_id}.{len(self.adapters)}"  # 给新适配器命名ID
        new_adapter = AdapterModule(self.config, adapter_id, self.writer).to(device)
        self.newly_added = True
        self.added_for_task = True
        self.adapters.append(new_adapter)  # 把新适配器加入本层适配器列表（K→K+1)
        if not initialize:
            self.set_new_router()
        logging.info(f"Adapter {adapter_id} added at block {self.layer_id}")

    def forward(self, x):
        rd_loss = torch.tensor(0.).to(device)

        added = False
        not_addition_layer = self.layer_id < self.adapt_start_layer or self.layer_id > self.adapt_end_layer
        if not_addition_layer:
            func_out, _, _ = self.adapters[-1](x)  # 拆包并丢弃后两个值,只拿功能输出func_out用于后续计算.
        else:
            func_outs, rd_losses, z_scores = [], [], []
            for adapter in self.adapters:
                func_out, rd_loss, z_score = adapter(x)
                func_outs.append(func_out)
                rd_losses.append(rd_loss)
                z_scores.append(z_score)

            func_outs = torch.stack(func_outs)  # 形状约为[K,B,T,D]
            rd_losses = torch.stack(rd_losses)  # [K,B]
            z_scores = torch.stack(z_scores)  # [K,B]

            addition_criteria = (
                        z_scores.mean(dim=1).min() > self.config.exp_threshold  # 若最小的平均值都大于阈值,说明所有适配器不适配这批数据→判定为新分布.
                        and self.adapt_start_layer <= self.layer_id <= self.adapt_end_layer  # 只允许在可扩展层范围内加新适配器.
                        and not self.added_for_task and self.detecting_outlier  # 本层在本任务里未添加.
                        )

            if addition_criteria:
                self.add_adapter()
                out = {"func_out": torch.zeros_like(func_outs[0]).to(device), "rd_loss": torch.tensor(0.).to(device),
                       "added": True}
                return out
            else:
                logits = self.router(x.mean(dim=1))
                if self.new_router is not None:
                    new_logits = self.new_router(x.mean(dim=1))
                    logits = torch.cat([logits, new_logits], dim=1)
                mask = torch.softmax(logits, dim=1)
                func_out = (func_outs * mask.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)).sum(dim=0)
                if self.adapters[-1].newly_added:
                    rd_loss = rd_losses[-1].mean()
                else:
                    rd_loss = torch.tensor(0.).to(device)

        out = {"func_out": func_out, "rd_loss": rd_loss, "added": added}
        return out

    def end_of_task_training(self):
        self.freeze_functional()
        self.freeze_rd()
        self.reset_newly_added_status()
        self.added_for_task = False

    def reset_newly_added_status(self):
        self.newly_added = False
        for adapter in self.adapters:
            adapter.newly_added = False

    def freeze_functional(self):
        adapter_ls = self.adapters
        for adapter in adapter_ls:
            for param in adapter.functional.parameters():
                param.requires_grad = False
                param._grad = None
        if self.new_router is not None:
            self.fix_router()
        for param in self.router.parameters():
            param.requires_grad = False
            param._grad = None

    def freeze_rd(self):
        adapter_ls = self.adapters
        for adapter in adapter_ls:
            if adapter.rd is not None:
                for param in adapter.rd.parameters():
                    param.requires_grad = False
                    param._grad = None
                adapter.rd_loss_record.updating = False


# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import timm
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from timm.models.layers import DropPath
from timm.models.registry import register_model
from collections import OrderedDict
from backbone.sema_block import SEMAModules


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None, writer=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        if config.ffn_adapt:
            self.adapter_module = SEMAModules(self.config, layer_id=layer_id, writer=writer)
        self.layer_id = layer_id
        self.writer = writer

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        out = self.adapter_module(x)
        adapt_x = out["func_out"]

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        if self.config.ffn_adapt:
            if self.config.ffn_option == 'sequential':
                out = self.adapter_module(x)
                x = out["func_out"]
            elif self.config.ffn_option == 'parallel':
                x = x + adapt_x
            else:
                raise ValueError(self.config.ffn_adapt)

        x = residual + x
        out.update({"blk_out": x})
        return out


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None, optim_config=None, writer=None):
        super().__init__()

        print("I'm using ViT with adapters.")
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i, writer=writer
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # self.init_weights(weight_init)

        ######### MAE begins ############
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        ######## Adapter begins #########
        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            # properly registered
            self.embeddings = nn.ParameterList(  # batch, num_prompt, embed_dim
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _ in
                 range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

        self.optim_config = optim_config

    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    @property
    def feature_dim(self):
        return self.out_dim

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        rd_losses = 0
        added_record = []

        for idx, blk in enumerate(self.blocks):
            if self.tuning_config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)
            blk_ret = blk(x)
            x = blk_ret["blk_out"]
            rd_loss, added = blk_ret["rd_loss"], blk_ret["added"]
            rd_losses += rd_loss
            added_record.append(added)
            if self.tuning_config.vpt_on:
                x = x[:, self.tuning_config.vpt_num:, :]
            if added:
                break

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        out = {"features": outcome, "rd_loss": rd_losses, "added_record": added_record}
        return out

    def forward(self, x):
        out = self.forward_features(x)
        x = out["features"]
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        out.update({"features": x})
        return out


def vit_base_patch16_224_sema(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    checkpoint_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768 * 2]
            v_weight = qkv_weight[768 * 2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768 * 2]
            v_bias = qkv_bias[768 * 2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
    model.out_dim = 768
    return model


def vit_base_patch16_224_in21k_sema(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    checkpoint_model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768 * 2]
            v_weight = qkv_weight[768 * 2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768 * 2]
            v_bias = qkv_bias[768 * 2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
    model.out_dim = 768
    return model


import copy
import logging
import torch
from torch import nn
from backbone.linears import SimpleLinear, SplitCosineLinear, CosineLinear, SimpleContinualLinear
from backbone.prompt import CodaPrompt
import timm


def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    # SimpleCIL or SimpleCIL w/ Finetune
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()

    elif '_memo' in name:
        if args["model_name"] == "memo":
            from backbone import vit_memo
            _basenet, _adaptive_net = timm.create_model("vit_base_patch16_224_memo", pretrained=True, num_classes=0)
            _basenet.out_dim = 768
            _adaptive_net.out_dim = 768
            return _basenet, _adaptive_net
    # SSF
    elif '_ssf' in name:
        if args["model_name"] == "aper_ssf":
            from backbone import vit_ssf
            if name == "pretrained_vit_b16_224_ssf":
                model = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
                model.out_dim = 768
            elif name == "pretrained_vit_b16_224_in21k_ssf":
                model = timm.create_model("vit_base_patch16_224_in21k_ssf", pretrained=True, num_classes=0)
                model.out_dim = 768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    # VPT
    elif '_vpt' in name:
        if args["model_name"] == "aper_vpt":
            from backbone.vpt import build_promptmodel
            if name == "pretrained_vit_b16_224_vpt":
                basicmodelname = "vit_base_patch16_224"
            elif name == "pretrained_vit_b16_224_in21k_vpt":
                basicmodelname = "vit_base_patch16_224_in21k"

            print("modelname,", name, "basicmodelname", basicmodelname)
            VPT_type = "Deep"
            if args["vpt_type"] == 'shallow':
                VPT_type = "Shallow"
            Prompt_Token_num = args["prompt_token_num"]

            model = build_promptmodel(modelname=basicmodelname, Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type)
            prompt_state_dict = model.obtain_prompt()
            model.load_prompt(prompt_state_dict)
            model.out_dim = 768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    elif '_adapter' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "sema":
            from backbone import vit_sema
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                ffn_adapter_type=args["ffn_adapter_type"],
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
                exp_threshold=args["exp_threshold"],
                adapt_start_layer=args["adapt_start_layer"],
                adapt_end_layer=args["adapt_end_layer"],
                rd_dim=args["rd_dim"],
                buffer_size=args["buffer_size"],
            )
            if name == "pretrained_vit_b16_224_adapter":
                model = vit_sema.vit_base_patch16_224_sema(num_classes=0,
                                                           global_pool=False, drop_path_rate=0.0,
                                                           tuning_config=tuning_config)
                model.out_dim = 768
            elif name == "pretrained_vit_b16_224_in21k_adapter":
                model = vit_sema.vit_base_patch16_224_in21k_sema(num_classes=0,
                                                                 global_pool=False, drop_path_rate=0.0,
                                                                 tuning_config=tuning_config)
                model.out_dim = 768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        elif args["model_name"] == "aper_adapter":
            from backbone import vit_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
            )
            if name == "pretrained_vit_b16_224_adapter":
                model = vit_adapter.vit_base_patch16_224_adapter(num_classes=0,
                                                                 global_pool=False, drop_path_rate=0.0,
                                                                 tuning_config=tuning_config)
                model.out_dim = 768
            elif name == "pretrained_vit_b16_224_in21k_adapter":
                model = vit_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                                                                       global_pool=False, drop_path_rate=0.0,
                                                                       tuning_config=tuning_config)
                model.out_dim = 768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # L2P
    elif '_l2p' in name:
        if args["model_name"] == "l2p":
            from backbone import vit_l2p
            model = timm.create_model(
                args["backbone_type"],
                pretrained=args["pretrained"],
                num_classes=args["nb_classes"],
                drop_rate=args["drop"],
                drop_path_rate=args["drop_path"],
                drop_block_rate=None,
                prompt_length=args["length"],
                embedding_key=args["embedding_key"],
                prompt_init=args["prompt_key_init"],
                prompt_pool=args["prompt_pool"],
                prompt_key=args["prompt_key"],
                pool_size=args["size"],
                top_k=args["top_k"],
                batchwise_prompt=args["batchwise_prompt"],
                prompt_key_init=args["prompt_key_init"],
                head_type=args["head_type"],
                use_prompt_mask=args["use_prompt_mask"],
            )
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # dualprompt
    elif '_dualprompt' in name:
        if args["model_name"] == "dualprompt":
            from backbone import vit_dualprompt
            model = timm.create_model(
                args["backbone_type"],
                pretrained=args["pretrained"],
                num_classes=args["nb_classes"],
                drop_rate=args["drop"],
                drop_path_rate=args["drop_path"],
                drop_block_rate=None,
                prompt_length=args["length"],
                embedding_key=args["embedding_key"],
                prompt_init=args["prompt_key_init"],
                prompt_pool=args["prompt_pool"],
                prompt_key=args["prompt_key"],
                pool_size=args["size"],
                top_k=args["top_k"],
                batchwise_prompt=args["batchwise_prompt"],
                prompt_key_init=args["prompt_key_init"],
                head_type=args["head_type"],
                use_prompt_mask=args["use_prompt_mask"],
                use_g_prompt=args["use_g_prompt"],
                g_prompt_length=args["g_prompt_length"],
                g_prompt_layer_idx=args["g_prompt_layer_idx"],
                use_prefix_tune_for_g_prompt=args["use_prefix_tune_for_g_prompt"],
                use_e_prompt=args["use_e_prompt"],
                e_prompt_layer_idx=args["e_prompt_layer_idx"],
                use_prefix_tune_for_e_prompt=args["use_prefix_tune_for_e_prompt"],
                same_key_value=args["same_key_value"],
            )
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # Coda_Prompt
    elif '_coda_prompt' in name:
        if args["model_name"] == "coda_prompt":
            from backbone import vit_coda_promtpt
            model = timm.create_model(args["backbone_type"], pretrained=args["pretrained"])
            # model = vision_transformer_coda_prompt.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
            #                 num_heads=12, ckpt_layer=0,
            #                 drop_path_rate=0)
            # from timm.models import vit_base_patch16_224
            # load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            # del load_dict['head.weight']; del load_dict['head.bias']
            # model.load_state_dict(load_dict)
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class IncrementalNet(BaseNet):  # 增量场景下的分类头扩展与对齐、前向输出打包、以及可选的可视化支持.
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):  # 重建分类头到新的类别数
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):  # 新类平均范数 ≈ 旧类平均范数，经验性校正
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):  # 取特征→过分类头→组装输出字典.
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x["features"])
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations
        return out

    def unset_gradcam_hook(self):  # 暂停手机激活和梯度，用在backbone.last_conv
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):  # 在目标层last_conv上挂前/反向钩子，做grad-cam可视化激活和梯度缓存
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):  # 反向时被调用，把该层输出梯度保存
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):  # 前向时被调用，把输出特征图保存
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.backbone.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.backbone.last_conv.register_forward_hook(
            forward_hook
        )


class CosineIncrementalNet(BaseNet):  # 余弦分类器，负载在新增类别时无缝扩展/迁移分类头.
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:  # 第一次从单头切到分头时.
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc


class DERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(DERNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.backbones = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.backbones)

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)

        out = self.fc(features)  # {logits: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim:])["logits"]

        out.update({"aux_logits": aux_logits, "features": features})
        return out
        """
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        """

    def update_fc(self, nb_classes):  # 每新任务时，扩容并迁移分类头.
        if len(self.backbones) == 0:  # 还有没骨干
            self.backbones.append(get_backbone(self.args, self.pretrained))
        else:
            self.backbones.append(get_backbone(self.args, self.pretrained))
            self.backbones[-1].load_state_dict(self.backbones[-2].state_dict())  # 把上一条骨干的参数拷贝给新加的骨干

        if self.out_dim is None:
            self.out_dim = self.backbones[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):  # 调用SimpleLinear构建线性分类器,in_dim拼接后的特征维度,out_dim类别数.
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_backbone(self):
        for param in self.backbones.parameters():
            param.requires_grad = False
        self.backbones.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.backbones) == 1
        self.backbones[0].load_state_dict(model_infos['backbone'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        if self.RP_dim is not None:
            feature_dim = self.RP_dim
        else:
            feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x)
        out.update({"features": x})
        return out


class SEMAVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.fc = None
        self.args = args

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        out = self.backbone(x)
        x = out["features"]
        out.update({"logits": self.fc(x)})
        return out


# l2p and dualprompt
class PromptVitNet(nn.Module):
    def __init__(self, args, pretrained):
        super(PromptVitNet, self).__init__()
        self.backbone = get_backbone(args, pretrained)
        if args["get_original_backbone"]:
            self.original_backbone = self.get_original_backbone(args)
        else:
            self.original_backbone = None

    def get_original_backbone(self, args):
        return timm.create_model(
            args["backbone_type"],
            pretrained=args["pretrained"],
            num_classes=args["nb_classes"],
            drop_rate=args["drop"],
            drop_path_rate=args["drop_path"],
            drop_block_rate=None,
        ).eval()

    def forward(self, x, task_id=-1, train=False):
        with torch.no_grad():
            if self.original_backbone is not None:
                cls_features = self.original_backbone(x)['pre_logits']
            else:
                cls_features = None

        x = self.backbone(x, task_id=task_id, cls_features=cls_features, train=train)
        return x


# coda_prompt
class CodaPromptVitNet(nn.Module):
    def __init__(self, args, pretrained):
        super(CodaPromptVitNet, self).__init__()
        self.args = args
        self.backbone = get_backbone(args, pretrained)
        self.fc = nn.Linear(768, args["nb_classes"])
        self.prompt = CodaPrompt(768, args["nb_tasks"], args["prompt_param"])

    # pen: get penultimate features
    def forward(self, x, pen=False, train=False):
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.backbone(x)
                q = q[:, 0, :]
            out, prompt_loss = self.backbone(x, prompt=self.prompt, q=q, train=train)
            out = out[:, 0, :]
        else:
            out, _ = self.backbone(x)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.fc(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out


class MultiBranchCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

        # no need the backbone.

        print(
            'Clear the backbone in MultiBranchCosineIncrementalNet, since we are using self.backbones with dual branches')
        self.backbone = torch.nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbones = nn.ModuleList()
        self.args = args

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self._feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self._feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]

        features = torch.cat(features, 1)
        # import pdb; pdb.set_trace()
        out = self.fc(features)
        out.update({"features": features})
        return out

    def construct_dual_branch_network(self, tuned_model):
        if 'ssf' in self.args['backbone_type']:
            newargs = copy.deepcopy(self.args)
            newargs['backbone_type'] = newargs['backbone_type'].replace('_ssf', '')
            print(newargs['backbone_type'])
            self.backbones.append(get_backbone(newargs))  # pretrained model without scale
        elif 'vpt' in self.args['backbone_type']:
            newargs = copy.deepcopy(self.args)
            newargs['backbone_type'] = newargs['backbone_type'].replace('_vpt', '')
            print(newargs['backbone_type'])
            self.backbones.append(get_backbone(newargs))  # pretrained model without vpt
        elif 'adapter' in self.args['backbone_type']:
            newargs = copy.deepcopy(self.args)
            newargs['backbone_type'] = newargs['backbone_type'].replace('_adapter', '')
            print(newargs['backbone_type'])
            self.backbones.append(get_backbone(newargs))  # pretrained model without adapter
        else:
            self.backbones.append(get_backbone(self.args))  # the pretrained model itself

        self.backbones.append(tuned_model.backbone)  # adappted tuned model

        self._feature_dim = self.backbones[0].out_dim * len(self.backbones)
        self.fc = self.generate_fc(self._feature_dim, self.args['init_cls'])


class FOSTERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(FOSTERNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.backbones = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None
        self.args = args

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.backbones)

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim:])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        self.backbones.append(get_backbone(self.args, self.pretrained))
        if self.out_dim is None:
            self.out_dim = self.backbones[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.backbones[-1].load_state_dict(self.backbones[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_backbone(self):
        for param in self.backbones.parameters():
            param.requires_grad = False
        self.backbones.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format(
                args["dataset"],
                args["seed"],
                args["backbone_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.backbones) == 1
        self.backbones[0].load_state_dict(model_infos['backbone'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class AdaptiveNet(nn.Module):
    def __init__(self, args, pretrained):
        super(AdaptiveNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.TaskAgnosticExtractor, _ = get_backbone(args, pretrained)  # Generalized blocks
        self.TaskAgnosticExtractor.train()
        self.AdaptiveExtractors = nn.ModuleList()  # Specialized Blocks
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.AdaptiveExtractors)

    def extract_vector(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        out = self.fc(features)  # {logits: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim:])["logits"]

        out.update({"aux_logits": aux_logits, "features": features})
        out.update({"base_features": base_feature_map})
        return out

        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''

    def update_fc(self, nb_classes):
        _, _new_extractor = get_backbone(self.args, self.pretrained)
        if len(self.AdaptiveExtractors) == 0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
            self.AdaptiveExtractors[-1].load_state_dict(self.AdaptiveExtractors[-2].state_dict())

        if self.out_dim is None:
            # logging.info(self.AdaptiveExtractors[-1])
            self.out_dim = self.AdaptiveExtractors[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, :self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = (torch.norm(weights[-increment:, :], p=2, dim=1))
        oldnorm = (torch.norm(weights[:-increment, :], p=2, dim=1))
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print('alignweights,gamma=', gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format(
                args["dataset"],
                args["seed"],
                args["backbone_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        checkpoint_name = checkpoint_name.replace("memo_", "")
        model_infos = torch.load(checkpoint_name)
        model_dict = model_infos['backbone']
        assert len(self.AdaptiveExtractors) == 1

        base_state_dict = self.TaskAgnosticExtractor.state_dict()
        adap_state_dict = self.AdaptiveExtractors[0].state_dict()

        pretrained_base_dict = {
            k: v
            for k, v in model_dict.items()
            if k in base_state_dict
        }

        pretrained_adap_dict = {
            k: v
            for k, v in model_dict.items()
            if k in adap_state_dict
        }

        base_state_dict.update(pretrained_base_dict)
        adap_state_dict.update(pretrained_adap_dict)

        self.TaskAgnosticExtractor.load_state_dict(base_state_dict)
        self.AdaptiveExtractors[0].load_state_dict(adap_state_dict)
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


import torch
from torch import nn
from torch.nn import functional as F
import math


class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 adapter_id=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.adapter_id = adapter_id
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        output = self.up_proj(down)
        return output


class AE(nn.Module):
    def __init__(self, config):
        super(AE, self).__init__()
        self.input_dim = config.d_model
        self.config = config
        self.encoder = nn.Linear(self.input_dim, config.rd_dim)
        self.decoder = nn.Linear(config.rd_dim, self.input_dim)
        self.weight_initialize()

    def forward(self, x):
        encoded = self.encoder(x)
        reconstruction = self.decoder(encoded)
        return reconstruction

    def compute_reconstruction_loss(self, x):
        x = x.mean(dim=1)
        reconstruction = self.forward(x)
        reconstruction_losses = []
        B = x.shape[0]
        for i in range(B):
            reconstruction_losses.append(self.reconstruction_loss(reconstruction[i], x[i]))
        reconstruction_losses = torch.stack(reconstruction_losses)
        return reconstruction_losses

    def reconstruction_loss(self, reconstruction, x):
        reconstruction_loss = F.mse_loss(reconstruction, x)
        return reconstruction_loss

    def weight_initialize(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
            nn.init.zeros_(self.encoder.bias)
            nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
            nn.init.zeros_(self.decoder.bias)


class Records:
    def __init__(self, max_len=500) -> None:
        self._max_len = max_len
        self._curr_len = 0
        self.record = torch.zeros(self._max_len)
        self._mean = 0
        self._var = 0
        self._powersumavg = 0
        self.updating = True

    @property
    def length(self):
        return self._curr_len

    @property
    def mean(self):
        return self._mean

    @property
    def stddev(self):
        return math.sqrt(self._var)

    def add_record(self, v):
        if not self.updating:
            return
        if self._curr_len < self._max_len:
            place_left = self._max_len - self._curr_len
            if place_left > len(v):
                self.record[self._curr_len:self._curr_len + len(v)] = v
                self._curr_len += len(v)
            else:
                self.record[self._curr_len:] = v[:place_left]
                self._curr_len = self._max_len
        else:
            self.record = torch.cat([self.record, v])
            self.record = self.record[len(v):]
        self._mean = torch.mean(self.record[:self._curr_len])
        self._var = torch.var(self.record[:self._curr_len])
