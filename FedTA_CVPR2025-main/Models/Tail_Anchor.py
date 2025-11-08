from copy import deepcopy

import torch
import torch.nn as nn

from Models.classification_head import Chead


# 直接在尾部加参数
class Tail_Anchor(nn.Module):  # 给每个类别维护一对“键-锚”向量。
    def __init__(self,anchor_size,key_size,nb_class):
        super(Tail_Anchor,self).__init__()
        self.size = anchor_size
        self.key_size = key_size
        self.nb_class = nb_class

        key_pool_size = (nb_class, key_size)
        self.key = nn.Parameter(torch.randn(key_pool_size))
        nn.init.uniform_(self.key, -1, 1)

        self.head = Chead(200)  # 类别数写死为200。
        anchor_pool_size = (nb_class, key_size)
        self.anchor_pool = nn.Parameter(torch.randn(anchor_pool_size))
        # nn.init.uniform_(self.anchor_pool, -1, 1)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):  # L2归一化。
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    # x是预训练给的768维，增强E经过冻结的预训练ViT后得到的特征。大小为[B,C]，B是一次前向里的样本数，C是设置的维度768维。
    def forward(self, x, class_mask):
        # 先找key
        # x_embed_norm 就是key
        x_embed_norm = self.l2_normalize(x, dim=1)  # B, C。把样本特征单位化，作为查询键。

        tem_key = self.key.reshape(-1, self.key_size)

        tem_key = tem_key.squeeze(0)
        # print(x_embed_norm.shape)

        key_norm = self.l2_normalize(tem_key, dim=1)  # Pool_size, C    把键池中的每个键单位化，便于用点积近似余弦相似度。

        similarity = torch.matmul(x_embed_norm, key_norm.t().to('cuda'))  # B, Pool_size   计算每个样本与每个键的相似度。

        q, index = torch.topk(similarity, k=1)  # q是每行最大的那个相似度值，形状[B,1].index对应最大值所在的列下标，形状[B,1].
        # print(q,index)


        anchor = self.anchor_pool[index]  # 根据index在锚池里取对应锚；形状为[B,1,C]
        # print(anchor.shape)

        anchor = anchor.reshape(-1, 768)  # 把[B,1,C]修改成[B,C]，这里的C是768
        # print(x.shape)
        # print(anchor.shape)
        x1 = torch.stack((x, anchor), dim=1).view(-1, 768*2)  # 将原始特征x与锚anchor在新维度拼成[B,2,C]，再拉平成[B,2C]。
        x = self.head(x1)  # 传入分类头，输出logits；

        # Put pull_constraint loss calculation inside
        batched_key_norm = key_norm[index]  # 取出被选中的键向量
        x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
        sim = batched_key_norm * x_embed_norm  # B, top_k, C  与样本单位向量逐元素相乘
        reduce_sim = torch.sum(sim) / self.key_size  # Scalar

        return x, x1, reduce_sim

    def load_head(self, head):
        self.head = deepcopy(head)

    def get_head(self):
        return self.head







