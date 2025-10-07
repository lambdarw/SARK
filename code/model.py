from enum import Enum, auto
import random
from typing import List
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from sklearn.metrics import f1_score
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class WeightsCalculator:
    """
    多任务学习动态权重计算器
    基于梯度相似度矩阵实现任务权重的自适应调整，解决多任务优化中的不平衡问题
    """
    def __init__(self, model_device) -> None:
        """
        初始化权重计算器
        参数:
            model_device: 模型所在设备，用于张量设备一致性
        """
        self.device = model_device
        self.MAX_N_ITER = 100          # 最大迭代次数，防止陷入无限循环
        self.EPS = 1e-3                # 收敛阈值，用于判断优化是否稳定
        self.gamma_model = self.Gamma().to(model_device)  # 初始化Gamma搜索模型

    def reset(self):
        """重置权重计算模型的初始状态，将搜索空间重置为[0,1]"""
        self.gamma_model.set_gamma(0, 1, self.device)

    class Gamma(nn.Module):
        """
        Gamma搜索模型：定义权重优化的目标函数
        通过线性插值方式在[alpha, e_t]区间搜索最优权重组合
        """
        def __init__(self):
            """初始化Gamma参数，创建100个均匀分布的插值点"""
            super().__init__()
            self.gamma = nn.Parameter(torch.linspace(0, 1, 100).unsqueeze(-1))
            # 注：gamma表示插值系数，控制从alpha到e_t的权重过渡

        def forward(self, M, alpha, e_t):
            """
            计算权重插值的目标函数值
            参数:
                M: 梯度相似度矩阵(n_loss x n_loss)
                alpha: 当前权重向量
                e_t: 基向量(仅第t维为1，其余为0)
            返回:
                目标函数值（用于衡量权重组合的优劣）
            """
            # 生成100个插值权重向量：(1-gamma)*alpha + gamma*e_t
            temp = (1 - self.gamma) * alpha.expand(100, -1) + self.gamma * e_t.expand(100, -1)
            # 计算权重组合的梯度相似度损失（矩阵对角线元素和）
            loss = temp.matmul(M).matmul(temp.T).diag().sum()
            return loss

        def set_gamma(self, start, end, device):
            """重置gamma参数的搜索区间"""
            self.gamma.data = torch.linspace(start, end, 100).unsqueeze(-1).to(device)

    def search_1d(self, M, alpha, e_t, gamma_model: Gamma):
        """
        一维搜索算法：在[alpha, e_t]插值区间寻找最优gamma值
        参数:
            M: 梯度相似度矩阵
            alpha: 当前权重向量
            e_t: 基向量
            gamma_model: Gamma模型实例
        返回:
            最优插值系数gamma
        """
        loss_rec = 0.0            # 记录上一轮损失值
        start, end = 0, 1         # 搜索区间初始化
        for i in range(4):        # 限制搜索迭代次数
            gamma_model.set_gamma(start, end, M.device)  # 设置搜索区间
            loss = gamma_model(M, alpha, e_t)            # 计算目标函数值
            # 收敛判断：损失变化小于阈值时终止
            if abs(loss.item() - loss_rec) < self.EPS:
                break
            # 梯度清零并反向传播
            gamma_model.gamma.grad = None
            loss.backward()
            loss_rec = loss.item()
            # 获取当前gamma值和梯度
            gamma = gamma_model.gamma.data
            grad = gamma_model.gamma.grad
            # 梯度符号一致时，取梯度最小点作为最优解
            if grad[0] * grad[-1] >= 0:
                return gamma[torch.argmin(abs(grad))].clamp(0, 0.9999)
            # 梯度符号变化时，收缩搜索区间
            try:
                end = gamma[grad > 0].min()
                start = gamma[grad < 0].max()
            except:
                raise Exception("梯度计算出现NaN，可能是损失函数异常")
        # 超出迭代次数时返回区间均值
        return gamma_model.gamma.data.mean()

    def calc_weights(self, submodels: List[nn.Module], losses):
        """
        计算多任务损失的动态权重
        参数:
            submodels: 参与权重计算的子模型列表
            losses: 各任务的损失值列表
        返回:
            任务权重向量(长度等于任务数)
        """
        n_loss = len(losses)                      # 任务数量
        alpha = torch.tensor([1 / n_loss] * n_loss)  # 初始等权重

        # 损失不可导时直接返回初始权重
        if not losses[0].requires_grad:
            return alpha.to(losses[0].device)
        
        # 收集所有可训练参数
        params = [p for model in submodels for p in model.parameters() if p.requires_grad]
        
        # 构建梯度矩阵g (n_loss x total_params)
        g = []
        for loss in losses:
            # 计算损失对参数的梯度
            temp_g = torch.autograd.grad(loss, params, retain_graph=True)
            # 展平所有梯度并拼接成向量
            g.append(torch.cat([i.reshape(-1) for i in temp_g]))
        
        # 构建梯度相似度矩阵M (n_loss x n_loss)
        M = []
        for i in range(n_loss):
            for j in range(n_loss):
                # 计算梯度向量的内积作为相似度度量
                M.append(g[i].matmul(g[j]))
        M = torch.stack(M).reshape(n_loss, n_loss)

        device = M.device
        alpha = alpha.to(device)  # 权重向量移至目标设备

        # 迭代优化权重向量
        for i in range(self.MAX_N_ITER):
            # 计算当前权重下各任务的梯度重要性
            t = torch.argmin(torch.sum(alpha.expand(n_loss, n_loss) * M, 1))
            e_t = torch.zeros(n_loss).to(device)
            e_t[t] = 1.0  # 构建基向量（仅第t维为1）
            
            # 一维搜索获取最优插值系数
            gamma = self.search_1d(M, alpha, e_t, self.gamma_model)
            # 更新权重向量：向基向量e_t插值
            alpha = (1 - gamma) * alpha + gamma * e_t
            
            # 收敛条件判断
            if gamma < self.EPS or abs(1 - alpha.max()) < self.EPS:
                break

        return alpha


class BgeJoinedModelLoss(Enum):
    ClaasificationLoss = auto()
    RankLoss = auto()
    ContrastiveLoss = auto()


class BgeJoinedModel(nn.Module):
    def __init__(self, pretrained_model_path, loss_types: List[BgeJoinedModelLoss]):
        assert len(loss_types) > 0
        super(BgeJoinedModel, self).__init__()
        self.loss_types = loss_types

        # 加载 config 并设置更大的 dropout
        config = AutoConfig.from_pretrained(pretrained_model_path)
        config.hidden_dropout_prob = 0.3                # FFN dropout
        config.attention_probs_dropout_prob = 0.3       # Attention dropout

        self.bge = AutoModel.from_pretrained(pretrained_model_path, config=config)

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        h = self.bge(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).last_hidden_state[:, 0, :]  # 取 [CLS] 向量
        logits = self.classifier(h)
        return logits

