import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
import logging

import csv
from typing import Dict
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AutoModelForSequenceClassification,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from datetime import datetime
from joined_dataset import JoinedDataset, Collater
from model import BgeJoinedModel, BgeJoinedModelLoss, WeightsCalculator


import torch
import torch.nn as nn


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        loss_types,
        optimizer,
        lr_scheduler,
        device,
        writer,
        now,
        # uncertainty_weighting
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.loss_types = loss_types
        self.optimizer=optimizer       # 传入已经配置好的优化器
        # self.uncertainty_weighting=uncertainty_weighting # 将权重层也传入
        
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.writer = writer
        self.weights_calculator = WeightsCalculator(self.device)
        self.now = now
        self.global_step = 0
    # 交叉熵损失
    def calc_cls_loss(self, cls_tokens, labels):
        outputs = self.model(**cls_tokens)
        loss_cls = nn.functional.cross_entropy(outputs, labels)
        return loss_cls
    
    # 建议的向量化高效实现
    def calc_rank_loss(self, rank_tokens, batch_size):
        outputs = self.model(**rank_tokens)[:, 1]
        
        # 将输出变形为 (batch_size, num_samples_per_group)，这里是 (batch_size, 6)
        grouped_outputs = outputs.view(batch_size, -1)

        # 使用广播机制计算每个组内两两之间的得分差
        # 维度变化: (batch_size, 6, 1) - (batch_size, 1, 6) -> (batch_size, 6, 6)
        diffs = grouped_outputs.unsqueeze(2) - grouped_outputs.unsqueeze(1)

        # RankLoss 的目标是最大化 (output[i] - output[j])，其中 i 的排序应该高于 j
        # L = -log(sigmoid(oi - oj))
        loss_matrix = -nn.functional.logsigmoid(diffs)
        
        # 我们只关心 i < j 的配对，这对应于矩阵的下三角部分
        # 使用 tril 获取下三角部分（不包括对角线），然后求和
        loss_rank = torch.tril(loss_matrix, diagonal=-1).sum()

        # 每组有 6 个样本，配对数为 C(6, 2) = 15
        loss_rank /= (15 * batch_size)
        return loss_rank

    def cal_loss(self, data):
        """
        输入数据形式：
        {
            'formal': {
                "classification": [tokenizer([[question, text], ...]), tensor([label, ...])],
                "rank_list": tokenizer([[question, doc], ...])
            },
            'informal': {
                "classification": [...],
                "rank_list": ...
            }
        }
        """
        losses = []
        loss_cls = None
        loss_rank = None

        # 分类损失（正式 + 非正式 的均值）
        if BgeJoinedModelLoss.ClaasificationLoss in self.loss_types:
            cls_formal = self.calc_cls_loss(*data['formal']['classification'])
            cls_informal = self.calc_cls_loss(*data['informal']['classification'])
            loss_cls = (cls_formal + cls_informal) / 2
            losses.append(loss_cls)

        # 排序损失（正式 + 非正式 的均值）
        if BgeJoinedModelLoss.RankLoss in self.loss_types:
            rank_formal = data['formal']['rank_list']
            rank_informal = data['informal']['rank_list']
            assert len(rank_formal['input_ids']) % 6 == 0
            assert len(rank_informal['input_ids']) % 6 == 0

            loss_rank_formal = self.calc_rank_loss(rank_formal, batch_size=len(rank_formal["input_ids"]) // 6)
            loss_rank_informal = self.calc_rank_loss(rank_informal, batch_size=len(rank_informal["input_ids"]) // 6)
            loss_rank = (loss_rank_formal + loss_rank_informal) / 2
            losses.append(loss_rank)

        # 动态加权合并损失
        weights = self.weights_calculator.calc_weights(
            [self.model.module.bge.embeddings, self.model.module.bge.encoder], losses
        )

        loss = torch.stack(losses).matmul(weights)
        
        # 不确定加权类
        # losses_tensor = torch.stack(losses)
        # loss = self.uncertainty_weighting(losses_tensor)

        # 权重日志记录
        log_file = os.path.join(self.writer.log_dir, "weight_logs.csv")
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if self.global_step == 0:
                header = ["step", "cls_weight", "rank_weight"]
                writer.writerow(header)

            row = [self.global_step]
            if BgeJoinedModelLoss.ClaasificationLoss in self.loss_types:
                row.append(weights[0].item())
            else:
                row.append(None)
            if BgeJoinedModelLoss.RankLoss in self.loss_types:
                row.append(weights[1].item() if len(weights) > 1 else None)
            else:
                row.append(None)
            writer.writerow(row)

        self.global_step += 1
        return loss, loss_cls, loss_rank


    def train_loop(self, epoch, total_loss):
        progress_bar = tqdm(range(len(self.train_dataloader)))
        progress_bar.set_description(f"loss: {0:>7f}")
        finish_step_num = (epoch - 1) * len(self.train_dataloader)

        gradient_accumulation_steps = 4  #8
        self.model.train()
        for step, sample in enumerate(self.train_dataloader, start=1):

            def move_to_device(data, device):
                if isinstance(data, torch.Tensor):
                    return data.to(device)
                elif isinstance(data, dict):
                    return {k: move_to_device(v, device) for k, v in data.items()}
                elif isinstance(data, list):
                    return [move_to_device(item, device) for item in data]
                elif isinstance(data, tuple):
                    return tuple(move_to_device(item, device) for item in data)
                else:
                    return data  # 保留原样（如字符串、int、None）

            sample = move_to_device(sample, self.device) 
            loss, clsloss, rankloss = self.cal_loss(sample)

            loss = loss / gradient_accumulation_steps

            # 添加 None 值检查
            if loss is not None:
                self.writer.add_scalar("loss", loss.item(), step + finish_step_num)
            if clsloss is not None:
                self.writer.add_scalar("clsloss", clsloss.item(), step + finish_step_num)
            if rankloss is not None:
                self.writer.add_scalar("rankloss", rankloss.item(), step + finish_step_num)

            # self.writer.add_scalar("loss", loss, step + finish_step_num)
            # self.writer.add_scalar("clsloss", clsloss, step + finish_step_num)
            # self.writer.add_scalar("rankloss", rankloss, step + finish_step_num)

            loss.backward()
            if (step % gradient_accumulation_steps == 0) or (step == len(self.train_dataloader)):
                if isinstance(self.model, nn.DataParallel):
                    torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # 参数更新
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # 梯度清零
                self.optimizer.zero_grad()

            total_loss += loss.item()*gradient_accumulation_steps
            progress_bar.set_description(
                f"loss: {total_loss/(finish_step_num + step):>7f}"
            )
            progress_bar.update(1)
        return total_loss
    def test_loop(self, dataloader, dataset_type="Train"):
        """
        用于评估模型在验证集或测试集上的性能，包括分类的准确性、精确率、召回率和F1。
        同时记录分类损失。
        """
        assert dataset_type in ["Train", "Valid", "Test"]
        self.model.eval()

        tp, fp, fn = 0, 0, 0     # 统计用于计算precision和recall
        y0 = y1 = 0              # 预测为0和1的数量
        total_loss = 0.0        # 累计分类损失
        total_samples = 0       # 累计样本数

        with torch.no_grad():
            for sample in dataloader:
                # 将所有数据移动到设备上
                def move_to_device(data, device):
                    if isinstance(data, torch.Tensor):
                        return data.to(device)
                    elif isinstance(data, dict):
                        return {k: move_to_device(v, device) for k, v in data.items()}
                    elif isinstance(data, list):
                        return [move_to_device(item, device) for item in data]
                    elif isinstance(data, tuple):
                        return tuple(move_to_device(item, device) for item in data)
                    else:
                        return data

                sample = move_to_device(sample, self.device)

                # 只计算分类相关指标与损失
                cls_formal_tokens, cls_formal_labels = sample['formal']['classification']
                cls_informal_tokens, cls_informal_labels = sample['informal']['classification']

                # 正式数据推理
                outputs_formal = self.model(**cls_formal_tokens)
                preds_formal = outputs_formal.argmax(dim=1)

                tp += torch.sum((preds_formal == 1) & (cls_formal_labels == 1)).item()
                fp += torch.sum((preds_formal == 1) & (cls_formal_labels == 0)).item()
                fn += torch.sum((preds_formal == 0) & (cls_formal_labels == 1)).item()
                y0 += torch.sum(preds_formal == 0).item()
                y1 += torch.sum(preds_formal == 1).item()
                
                loss_formal = nn.functional.cross_entropy(outputs_formal, cls_formal_labels)
                total_loss += loss_formal.item() * cls_formal_labels.size(0)
                total_samples += cls_formal_labels.size(0)

                # 非正式数据推理
                outputs_informal = self.model(**cls_informal_tokens)
                preds_informal = outputs_informal.argmax(dim=1)

                tp += torch.sum((preds_informal == 1) & (cls_informal_labels == 1)).item()
                fp += torch.sum((preds_informal == 1) & (cls_informal_labels == 0)).item()
                fn += torch.sum((preds_informal == 0) & (cls_informal_labels == 1)).item()
                y0 += torch.sum(preds_informal == 0).item()
                y1 += torch.sum(preds_informal == 1).item()

                loss_informal = nn.functional.cross_entropy(outputs_informal, cls_informal_labels)
                total_loss += loss_informal.item() * cls_informal_labels.size(0)
                total_samples += cls_informal_labels.size(0)

        # 平均损失
        avg_loss = total_loss / total_samples if total_samples > 0 else 0

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        except ZeroDivisionError as e:
            print(e)
            print(f"n_y == 0: {y0}\nn_y == 1: {y1}")
            return 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{dataset_type} dataset precision: {(100*precision):.1f}%")
        print(f"{dataset_type} dataset recall: {(100*recall):.1f}%")
        print(f"{dataset_type} dataset f1: {(100*f1):.1f}%")
        print(f"n_y == 0: {y0}\nn_y == 1: {y1}")
        print(f"{dataset_type} dataset avg classification loss: {avg_loss:.4f}")

        if dataset_type == "Train":
            return f1
        print(dataset_type)
        return f1, avg_loss

    def train(self, epoch_num, outdir):
        total_loss = 0.0
        best_f1 = 0.0
        outdir = os.path.join(outdir, self.now)
        os.makedirs(outdir, exist_ok=True)
        print('创建目录', outdir)
        for t in range(epoch_num):
            print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
            total_loss = self.train_loop(t + 1, total_loss)
            train_f1 = self.test_loop(self.train_dataloader, dataset_type="Train")
            self.writer.add_scalar("f1/train_acc", train_f1, t + 1)
            logging.info(f"Epoch {t+1} train_f1: {train_f1:.4f}")
            valid_f1, valid_loss = self.test_loop(self.valid_dataloader, dataset_type="Valid")
            self.writer.add_scalar("f1/valid_f1", valid_f1, t + 1)
            self.writer.add_scalar("Loss/valid", valid_loss, t + 1)

            logging.info(f"Epoch {t+1} valid_f1: {valid_f1:.4f}")
            
            print("保存模型至", outdir)
            torch.save(
                    self.model.module.state_dict(),
                    os.path.join(
                        outdir,
                        f"epoch_{t+1}_valid_f1_{(100*valid_f1):0.1f}_model_weights.bin",
                    ),
            )
            
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--valid_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--tensorboard_log_dir", type=str)
    parser.add_argument("--cls_loss", action="store_true", default=False)
    parser.add_argument("--rank_loss", action="store_true", default=False)
    parser.add_argument("--scl_loss", action="store_true")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epoch_num", type=int)
    parser.add_argument("--ttype", type=str)
    args = parser.parse_args()
    return args
  
def seed_everything(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = get_args()
    seed_everything(42)
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epoch_num = args.epoch_num

    now = datetime.now().strftime("%Y%m%d_%H%M")
    tensorboard_log_dir = os.path.join(
        args.tensorboard_log_dir,
        f"{now}")
    writer = SummaryWriter(tensorboard_log_dir)

    logging.basicConfig(
        filename=os.path.join(tensorboard_log_dir, "training.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = args.pretrained_model_path
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=512)

    train_data = JoinedDataset(args.train_data_path)
    collater = Collater(tokenizer, flag=0)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collater
    )

    valid_data = JoinedDataset(args.valid_data_path)
    collater = Collater(tokenizer, flag=1)
    valid_dataloader = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, collate_fn=collater
    )

    test_data = JoinedDataset(args.test_data_path)
    collater = Collater(tokenizer, flag=2)
    test_dataloader = DataLoader(
        test_data, batch_size=32, shuffle=False, collate_fn=collater
    )
    loss_types = []
    if args.cls_loss:
        loss_types.append(BgeJoinedModelLoss.ClaasificationLoss)
    if args.rank_loss:
        loss_types.append(BgeJoinedModelLoss.RankLoss)
    model = BgeJoinedModel(checkpoint, loss_types)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    lr_scheduler = None

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epoch_num * len(train_dataloader),
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        loss_types=loss_types,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        writer=writer,
        now = now
    )
    trainer.train(epoch_num=epoch_num, outdir=args.outdir)
    print("Train Done!")  

    def test(checkpoint_path, output_path, dataloader=test_dataloader, dataset_type='test_data', device=device, loss_types=loss_types): 
        model = BgeJoinedModel(args.pretrained_model_path, loss_types=loss_types)

        
        print('加载训练好的模型')
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()

        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个 GPU 进行推理")
            model = nn.DataParallel(model)  
        
        # 每个段落得分
        all_scores = []
        all_docs = []
        
        with torch.no_grad():
            for sample in dataloader:
                X, docs= sample
                X = X.to(device)
                outputs = model(
                    input_ids=X["input_ids"],
                    attention_mask=X.get("attention_mask", None),
                    token_type_ids=X.get("token_type_ids", None),
                )[:, 1]
                
                all_scores.extend(outputs.cpu().tolist())
                all_docs.extend(docs) 
        
        sorted_docs_per_query = []
        for i in range(0, len(all_scores), 50):
            query_scores = all_scores[i:i + 50]
            query_docs = all_docs[i:i + 50]
            sorted_indices = sorted(range(50), key=lambda j: query_scores[j], reverse=True)
            sorted_docs = [query_docs[j] for j in sorted_indices]
            sorted_docs_per_query.append(sorted_docs)

        res = []
        i = 0
        with open(args.test_data_path, 'r') as f:
            for line in f:
                dic = {}
                sample = json.loads(line)
                dic['question'] = sample['question']
                dic['answers'] = sample['answers']
                dic['top50'] = sorted_docs_per_query[i]
                # dic['DPR'] = sample['rank_list']
                res.append(dic)
                i += 1

        with open(output_path, 'w') as f:
            json.dump(res, f, indent=4)

    # checkpoint_path = './output/ours_best/train1_epoch_27_valid_f1_72.2_model_weights.bin'
    # output_path = f'./Data/results/{args.ttype}/temple/test_rerank_style_{args.ttype}.json'
    # test(checkpoint_path, output_path)

if __name__ == "__main__":
    main()
