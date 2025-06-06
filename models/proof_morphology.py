import copy
import logging
import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.inc_net import Proof_Net
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, get_attribute, ClipLoss

num_workers = 4

class MorphologyMemoryPool(nn.Module):
    """
    增强的形态演化记忆池
    核心功能：
    1. 存储和学习形态0→形态1的演化模式
    2. 支持跨任务的演化知识迁移
    3. 智能原型管理和演化回放
    """
    def __init__(self, feature_dim, num_classes_per_pair, pool_size=512, hidden_dim=1024):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes_per_pair = num_classes_per_pair
        self.pool_size = pool_size
        self.hidden_dim = hidden_dim
        
        # 核心组件
        self.memory_keys = nn.Parameter(torch.randn(pool_size, feature_dim))
        self.memory_values = nn.Parameter(torch.randn(pool_size, feature_dim))
        
        # 形态演化变换网络
        self.evolution_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        
        # 原型演化历史存储
        self.evolution_history = {}  # {class_idx: [(morph0_proto, morph1_proto), ...]}
        self.evolution_patterns = {}  # {pattern_id: evolution_vector}
        
        # 演化模式相似度网络
        self.pattern_similarity = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self._init_parameters()

    def _init_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.memory_keys)
        nn.init.xavier_uniform_(self.memory_values)
        
        for m in self.evolution_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, morph0_features, mode='evolve'):
        """
        前向传播
        Args:
            morph0_features: 形态0特征 [batch_size, feature_dim]
            mode: 'evolve' (预测演化) 或 'update' (更新记忆)
        """
        if mode == 'evolve':
            return self._predict_evolution(morph0_features)
        elif mode == 'retrieve':
            return self._retrieve_similar_patterns(morph0_features)
        else:
            return self._update_memory_patterns(morph0_features)

    def _predict_evolution(self, morph0_features):
        """预测形态演化"""
        batch_size = morph0_features.size(0)
        
        # 1. 检索相似的演化模式
        similar_patterns, similarity_scores = self._retrieve_similar_patterns(morph0_features)
        
        # 2. 使用注意力机制聚合相似模式
        if similar_patterns is not None:
            attended_patterns, attention_weights = self.attention(
                morph0_features.unsqueeze(1),  # query
                similar_patterns,  # key & value
                similar_patterns
            )
            context_features = attended_patterns.squeeze(1)
        else:
            context_features = morph0_features
            attention_weights = None
        
        # 3. 基于上下文预测演化
        evolution_input = morph0_features + context_features
        predicted_morph1 = self.evolution_net(evolution_input)
        
        # 4. 残差连接
        predicted_morph1 = morph0_features + predicted_morph1
        
        return predicted_morph1, attention_weights

    def _retrieve_similar_patterns(self, query_features):
        """检索相似的演化模式"""
        if not self.evolution_history:
            return None, None
            
        # 收集所有演化模式
        all_morph0_patterns = []
        all_morph1_patterns = []
        
        for class_idx, history in self.evolution_history.items():
            for morph0_proto, morph1_proto in history:
                all_morph0_patterns.append(morph0_proto)
                all_morph1_patterns.append(morph1_proto)
        
        if not all_morph0_patterns:
            return None, None
            
        # 转换为tensor
        stored_morph0 = torch.stack(all_morph0_patterns).to(query_features.device)
        stored_morph1 = torch.stack(all_morph1_patterns).to(query_features.device)
        
        # 计算相似度 (使用余弦相似度)
        query_norm = F.normalize(query_features, dim=1)
        stored_norm = F.normalize(stored_morph0, dim=1)
        
        similarity_matrix = torch.matmul(query_norm, stored_norm.t())  # [batch, num_stored]
        
        # 选择top-k最相似的模式
        top_k = min(5, similarity_matrix.size(1))
        top_similarities, top_indices = torch.topk(similarity_matrix, k=top_k, dim=1)
        
        # 检索对应的演化模式
        batch_size = query_features.size(0)
        selected_patterns = torch.zeros(batch_size, top_k, self.feature_dim).to(query_features.device)
        
        for i in range(batch_size):
            for j, idx in enumerate(top_indices[i]):
                # 使用演化向量 (morph1 - morph0)
                evolution_vector = stored_morph1[idx] - stored_morph0[idx]
                selected_patterns[i, j] = evolution_vector
        
        return selected_patterns, top_similarities

    def update_evolution_history(self, class_idx, morph0_prototype, morph1_prototype):
        """更新演化历史"""
        if class_idx not in self.evolution_history:
            self.evolution_history[class_idx] = []
        
        # 添加新的演化对
        self.evolution_history[class_idx].append(
            (morph0_prototype.detach().clone(), morph1_prototype.detach().clone())
        )
        
        # 限制历史长度，保持最新的演化模式
        max_history = 10
        if len(self.evolution_history[class_idx]) > max_history:
            self.evolution_history[class_idx] = self.evolution_history[class_idx][-max_history:]

    def update_memory(self, morph0_features, morph1_features, learning_rate=0.01):
        """在线更新记忆池 - 增强版"""
        with torch.no_grad():
            # 计算演化向量
            evolution_vectors = morph1_features - morph0_features
            
            # 找到最相似的记忆位置
            morph0_norm = F.normalize(morph0_features, dim=1)
            keys_norm = F.normalize(self.memory_keys, dim=1)
            
            similarities = torch.matmul(morph0_norm, keys_norm.t())
            _, best_indices = torch.max(similarities, dim=1)
            
            # 更新对应的记忆
            for i, idx in enumerate(best_indices):
                # 使用移动平均更新
                self.memory_keys[idx] = (1 - learning_rate) * self.memory_keys[idx] + learning_rate * morph0_features[i]
                self.memory_values[idx] = (1 - learning_rate) * self.memory_values[idx] + learning_rate * evolution_vectors[i]

    def compute_evolution_loss(self, morph0_features, morph1_features, predicted_morph1):
        """计算演化损失 - 多层次损失"""
        if predicted_morph1 is None:
            return torch.tensor(0.0, device=morph1_features.device)
        
        # 1. 直接重建损失
        reconstruction_loss = F.mse_loss(predicted_morph1, morph1_features)
        
        # 2. 演化方向损失
        true_evolution = morph1_features - morph0_features
        predicted_evolution = predicted_morph1 - morph0_features
        direction_loss = F.cosine_embedding_loss(
            predicted_evolution, true_evolution,
            torch.ones(predicted_evolution.size(0)).to(morph1_features.device)
        )
        
        # 3. 特征分布损失
        true_norm = torch.norm(morph1_features, dim=1)
        pred_norm = torch.norm(predicted_morph1, dim=1)
        norm_loss = F.mse_loss(pred_norm, true_norm)
        
        return reconstruction_loss + 0.5 * direction_loss + 0.1 * norm_loss

    def compute_rehearsal_loss(self, current_features, target_classes):
        """计算演化回放损失 - 简化版，专注于相关类别"""
        if not self.evolution_history:
            return torch.tensor(0.0, device=current_features.device)
        
        rehearsal_loss = torch.tensor(0.0, device=current_features.device)
        count = 0
        
        # 只对当前batch中出现的类别进行演化回放
        unique_classes = torch.unique(target_classes)
        
        for class_idx in unique_classes:
            class_idx = class_idx.item()
            if class_idx in self.evolution_history:
                # 获取该类别的演化历史
                class_history = self.evolution_history[class_idx]
                
                # 只使用最新的演化对，避免过多噪声
                for morph0_proto, morph1_proto in class_history[-1:]:  # 只用最新1个演化
                    # 预测演化
                    morph0_proto = morph0_proto.to(current_features.device).unsqueeze(0)
                    predicted_morph1, _ = self._predict_evolution(morph0_proto)
                    
                    # 与真实形态1比较
                    morph1_proto = morph1_proto.to(current_features.device).unsqueeze(0)
                    loss = self.compute_evolution_loss(morph0_proto, morph1_proto, predicted_morph1)
                    
                    rehearsal_loss += loss
                    count += 1
        
        return rehearsal_loss / count if count > 0 else rehearsal_loss

    def get_memory_status(self):
        """获取记忆池状态"""
        status = {
            'pool_size': self.pool_size,
            'evolution_history_classes': len(self.evolution_history),
            'total_evolution_pairs': sum(len(history) for history in self.evolution_history.values()),
            'memory_utilization': len([k for k in self.memory_keys if torch.norm(k) > 0.1]) / self.pool_size
        }
        return status


class MorphologyEvolutionLearner(BaseLearner):
    """
    形态演化增量学习器
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = Proof_Net(args, False)
        self._network.extend_task()
        self._network.update_context_prompt()
        
        # 形态演化相关参数
        self.batch_size = get_attribute(args, "batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.tuned_epoch = get_attribute(args, "tuned_epoch", 5)
        
        # 形态管理
        self._known_classes = 0
        self._current_morphology_pair = -1  # 当前形态对索引
        self._morphology_stage = 0  # 0: 形态0, 1: 形态1
        self._classes_per_pair = get_attribute(args, "classes_per_pair", 5)  # 每对包含的类别数
        
        # Memory Pool配置
        feature_dim = self._network.feature_dim
        self.memory_pool = MorphologyMemoryPool(
            feature_dim=feature_dim,
            num_classes_per_pair=self._classes_per_pair,
            pool_size=get_attribute(args, "memory_pool_size", 512),
            hidden_dim=get_attribute(args, "memory_pool_hidden", 1024)
        ).to(self._device)
        
        # 形态演化原型存储 {pair_idx: {class_idx: {morph_stage: prototype}}}
        self._morphology_prototypes = {}
        
        # 损失权重配置 - 重新调整为合理权重
        self.lambda_evolution = get_attribute(args, "lambda_evolution", 1.0)  # 演化损失权重
        self.lambda_rehearsal = get_attribute(args, "lambda_rehearsal", 0.5)  # 原型回放权重 (适中)
        self.lambda_memory = get_attribute(args, "lambda_memory", 0.3)  # Memory Pool损失权重 (适中)
        
        logging.info(f"形态演化学习器初始化完成:")
        logging.info(f"  每对类别数: {self._classes_per_pair}")
        logging.info(f"  Memory Pool大小: {self.memory_pool.pool_size}")
        logging.info(f"  特征维度: {feature_dim}")
        logging.info(f"  防遗忘权重 - 原型回放: {self.lambda_rehearsal}, Memory Pool: {self.lambda_memory}")

    def incremental_train(self, data_manager, task_idx=None, morphology_stage=None, pair_idx=None):
        """
        形态演化增量训练
        
        Args:
            data_manager: 数据管理器
            task_idx: 任务索引
            morphology_stage: 形态阶段 (0: 形态0, 1: 形态1)
            pair_idx: 形态对索引
        """
        # 更新当前状态
        if task_idx is not None:
            self._cur_task = task_idx
        if morphology_stage is not None:
            self._morphology_stage = morphology_stage
        if pair_idx is not None:
            self._current_morphology_pair = pair_idx
            
        logging.info(f"开始形态演化训练:")
        logging.info(f"  任务索引: {self._cur_task}")
        logging.info(f"  形态阶段: {self._morphology_stage}")
        logging.info(f"  形态对索引: {self._current_morphology_pair}")
        
        # 形态演化的类别数量管理
        if self._morphology_stage == 0:
            # 形态0：开始学习新的classes_per_pair个类别
            # total_classes包含所有已知类别 + 当前要学习的新类别
            self._total_classes = self._known_classes + self._classes_per_pair
            logging.info(f"形态0开始：total_classes = {self._known_classes} + {self._classes_per_pair} = {self._total_classes}")
        else:
            # 形态1：继续学习相同的类别（不增加新类别数量）
            self._total_classes = self._known_classes + self._classes_per_pair
            logging.info(f"形态1继续：total_classes = {self._total_classes}")
            
        # 确保total_classes至少为classes_per_pair（避免为0的情况）
        if self._total_classes == 0:
            self._total_classes = self._classes_per_pair
            logging.warning(f"修正total_classes从0到{self._classes_per_pair}")
        
        logging.info(f"  已知类别: {self._known_classes} -> {self._total_classes}")
        
        # 更新网络原型
        self._network.update_prototype(self._total_classes)
        
        # 获取训练数据
        train_dataset = data_manager.get_multimodal_dataset(
            self._cur_task, source="train", mode="train", 
            appendent=self._get_memory()
        )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        
        # 保存旧网络
        self._old_network = copy.deepcopy(self._network).to(self._device)
        self._old_network.eval()
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=num_workers
        )
        
        # 创建测试加载器
        self.current_test_loader, self.test_loader = self._create_test_loaders(data_manager)
        
        # 计算原型
        train_dataset_for_proto = data_manager.get_multimodal_dataset(
            self._cur_task, source="train", mode="test"
        )
        train_loader_for_proto = DataLoader(
            train_dataset_for_proto, batch_size=self.batch_size,
            shuffle=True, num_workers=num_workers
        )
        
        # GPU并行处理
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            self.memory_pool = nn.DataParallel(self.memory_pool, self._multiple_gpus)
            
        self._network.to(self._device)
        self.memory_pool.to(self._device)
        
        # 计算当前任务原型
        self._compute_morphology_prototypes(train_loader_for_proto)
        
        # 执行形态演化训练
        if self._morphology_stage == 0:
            # 形态0：标准增量学习
            self._train_morphology_0()
        else:
            # 形态1：演化学习（利用形态0的知识和Memory Pool）
            self._train_morphology_1()
            
        # 构建回放记忆
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        
        # 恢复单GPU模式
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
            self.memory_pool = self.memory_pool.module
            
        # 更新known_classes（仅在形态1完成后）
        if self._morphology_stage == 1:
            # 形态1完成后，将当前学到的类别加入known_classes
            self._known_classes = self._total_classes
            logging.info(f"形态对{self._current_morphology_pair}完成，更新known_classes: {self._known_classes}")
        else:
            logging.info(f"形态0完成，known_classes保持不变: {self._known_classes}")

    def _compute_morphology_prototypes(self, train_loader):
        """计算形态原型"""
        self._network.eval()
        
        # 初始化当前形态对的原型存储
        if self._current_morphology_pair not in self._morphology_prototypes:
            self._morphology_prototypes[self._current_morphology_pair] = {}
            
        embedding_list, label_list = [], []
        
        with torch.no_grad():
            for batch in train_loader:
                if isinstance(batch[1], dict) and 'image' in batch[1]:
                    _, data_dict, labels = batch
                    images = data_dict['image'].to(self._device)
                else:
                    _, images, labels = batch
                    images = images.to(self._device)
                    
                labels = labels.to(self._device)
                
                # 提取特征
                embeddings = self._network.convnet.encode_image(images, normalize=True)
                
                embedding_list.append(embeddings)
                label_list.append(labels)
        
        # 合并所有特征
        all_embeddings = torch.cat(embedding_list, dim=0)
        all_labels = torch.cat(label_list, dim=0)
        
        # 计算每个类别的原型
        # 无论形态0还是形态1，都处理当前要学习的新类别
        current_classes = range(self._known_classes, self._total_classes)
        logging.info(f"计算原型的类别范围: {self._known_classes} 到 {self._total_classes}")
        
        for class_idx in current_classes:
            class_mask = (all_labels == class_idx)
            if class_mask.sum() > 0:
                class_embeddings = all_embeddings[class_mask]
                prototype = class_embeddings.mean(0)
                
                # 存储原型
                if class_idx not in self._morphology_prototypes[self._current_morphology_pair]:
                    self._morphology_prototypes[self._current_morphology_pair][class_idx] = {}
                    
                self._morphology_prototypes[self._current_morphology_pair][class_idx][self._morphology_stage] = prototype
                
                logging.info(f"计算原型 - 形态对{self._current_morphology_pair}, 类别{class_idx}, 形态{self._morphology_stage}")
                
                # 如果是形态1，更新Memory Pool的演化历史
                if self._morphology_stage == 1:
                    # 检查是否有对应的形态0原型
                    if 0 in self._morphology_prototypes[self._current_morphology_pair][class_idx]:
                        morph0_proto = self._morphology_prototypes[self._current_morphology_pair][class_idx][0]
                        morph1_proto = prototype
                        
                        # 更新Memory Pool的演化历史
                        self.memory_pool.update_evolution_history(class_idx, morph0_proto, morph1_proto)
                        logging.info(f"更新Memory Pool演化历史: 类别{class_idx}")

    def _train_morphology_0(self):
        """训练形态0（基础学习）"""
        logging.info("开始形态0训练（基础学习）")
        
        # 训练设置
        self._setup_optimizer_and_scheduler()
        
        # 训练循环
        for epoch in range(self.tuned_epoch):
            self._network.train()
            self.memory_pool.train()
            
            total_loss_sum = 0.0
            memory_rehearsal_loss_sum = 0.0
            correct, total = 0, 0
            
            pbar = tqdm(self.train_loader, desc=f"形态0-Epoch {epoch+1}/{self.tuned_epoch}")
            
            for i, batch in enumerate(pbar):
                # 数据准备
                if isinstance(batch[1], dict) and 'image' in batch[1]:
                    _, data_dict, targets = batch
                    inputs = data_dict['image'].to(self._device)
                else:
                    _, inputs, targets = batch
                    inputs = inputs.to(self._device)
                    
                targets = targets.to(self._device).long()
                
                # 提取特征
                features = self._network.encode_image(inputs)
                
                # 分类损失
                class_to_label = self.data_manager._class_to_label
                templates = self.data_manager._data_to_prompt[0]
                total_labels = class_to_label[:max(self._total_classes, len(class_to_label))]
                
                text_batch = [templates.format(lbl) for lbl in total_labels[:self._total_classes]]
                cls_logits = self._forward_for_classification(inputs, text_batch)
                ce_loss = F.cross_entropy(cls_logits, targets)
                
                # Memory Pool演化回放损失（利用之前学到的演化知识）
                memory_rehearsal_loss = self.memory_pool.compute_rehearsal_loss(features, targets)
                memory_rehearsal_loss_sum += memory_rehearsal_loss.item()
                
                # 原型回放损失
                rehearsal_loss = self._compute_morphology_rehearsal_loss()
                
                # 知识蒸馏损失（简单有效的防遗忘）
                distillation_loss = self._compute_distillation_loss(inputs, targets)
                
                # 总损失 - 降低权重避免过度拟合
                total_loss = (ce_loss + 
                            0.1 * memory_rehearsal_loss +  # 降低Memory Pool权重
                            0.2 * rehearsal_loss +  # 降低原型回放权重
                            0.1 * distillation_loss)  # 轻量级知识蒸馏
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # 统计
                total_loss_sum += total_loss.item()
                _, preds = torch.max(cls_logits, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
                # 更新进度条
                if i % 10 == 0:
                    acc = correct / total * 100
                    pbar.set_postfix({
                        'Loss': f'{total_loss.item():.4f}',
                        'CE': f'{ce_loss.item():.4f}',
                        'MemReh': f'{memory_rehearsal_loss.item():.4f}',
                        'Acc': f'{acc:.2f}%'
                    })
            
            # 学习率调度
            self.scheduler.step()
            
            # Epoch统计
            epoch_loss = total_loss_sum / len(self.train_loader)
            epoch_memory_loss = memory_rehearsal_loss_sum / len(self.train_loader)
            epoch_acc = correct / total * 100
            
            logging.info(f"形态0 Epoch {epoch+1}: Loss={epoch_loss:.4f}, "
                        f"MemReh={epoch_memory_loss:.4f}, Acc={epoch_acc:.2f}%")

    def _train_morphology_1(self):
        """训练形态1（演化学习）"""
        logging.info("开始形态1训练（演化学习）")
        
        # 获取形态0的原型用于演化学习
        morph0_prototypes = self._get_morphology_0_prototypes()
        
        if not morph0_prototypes:
            logging.warning("未找到形态0原型，使用标准训练")
            self._train_morphology_0()
            return
            
        # 训练设置
        self._setup_optimizer_and_scheduler()
        
        # 训练循环
        for epoch in range(self.tuned_epoch):
            self._network.train()
            self.memory_pool.train()
            
            total_loss_sum = 0.0
            evolution_loss_sum = 0.0
            memory_loss_sum = 0.0
            memory_rehearsal_loss_sum = 0.0
            correct, total = 0, 0
            
            pbar = tqdm(self.train_loader, desc=f"形态1-Epoch {epoch+1}/{self.tuned_epoch}")
            
            for i, batch in enumerate(pbar):
                # 数据准备
                if isinstance(batch[1], dict) and 'image' in batch[1]:
                    _, data_dict, targets = batch
                    inputs = data_dict['image'].to(self._device)
                else:
                    _, inputs, targets = batch
                    inputs = inputs.to(self._device)
                    
                targets = targets.to(self._device).long()
                
                # 提取当前形态1特征
                morph1_features = self._network.encode_image(inputs)
                
                # 使用Memory Pool预测形态演化
                predicted_morph1, attention_weights = self._predict_evolution_from_prototypes(
                    targets, morph0_prototypes
                )
                
                # 分类损失
                class_to_label = self.data_manager._class_to_label
                templates = self.data_manager._data_to_prompt[0]
                total_labels = class_to_label[:max(self._total_classes, len(class_to_label))]
                
                text_batch = [templates.format(lbl) for lbl in total_labels[:self._total_classes]]
                cls_logits = self._forward_for_classification(inputs, text_batch)
                ce_loss = F.cross_entropy(cls_logits, targets)
                
                # 演化预测损失（核心：学习形态演化）- 修复维度不匹配问题
                if predicted_morph1 is not None:
                    # 确保predicted_morph1与当前batch维度匹配
                    batch_size = morph1_features.size(0)
                    pred_size = predicted_morph1.size(0)
                    
                    if pred_size == batch_size:
                        # 维度匹配，计算演化损失
                        morph0_batch = []
                        for target in targets:
                            class_idx = target.item()
                            if class_idx in morph0_prototypes:
                                morph0_batch.append(morph0_prototypes[class_idx])
                            else:
                                morph0_batch.append(torch.zeros_like(morph1_features[0]))
                        
                        if morph0_batch:
                            morph0_features_batch = torch.stack(morph0_batch).to(self._device)
                            evolution_loss = self.memory_pool.compute_evolution_loss(
                                morph0_features_batch, morph1_features, predicted_morph1
                            )
                            evolution_loss_sum += evolution_loss.item()
                        else:
                            evolution_loss = torch.tensor(0.0, device=self._device)
                    else:
                        # 维度不匹配，跳过演化损失
                        logging.warning(f"演化预测维度不匹配: pred={pred_size}, batch={batch_size}")
                        evolution_loss = torch.tensor(0.0, device=self._device)
                else:
                    evolution_loss = torch.tensor(0.0, device=self._device)
                
                # Memory Pool对比损失
                memory_loss = self._compute_memory_pool_loss(morph1_features, targets)
                memory_loss_sum += memory_loss.item()
                
                # Memory Pool演化回放损失（关键：防止遗忘之前的演化模式）
                memory_rehearsal_loss = self.memory_pool.compute_rehearsal_loss(morph1_features, targets)
                memory_rehearsal_loss_sum += memory_rehearsal_loss.item()
                
                # 原型回放损失
                rehearsal_loss = self._compute_morphology_rehearsal_loss()
                
                # 知识蒸馏损失（简单有效的防遗忘）
                distillation_loss = self._compute_distillation_loss(inputs, targets)
                
                # 总损失 - 平衡权重避免训练不稳定
                total_loss = (ce_loss + 
                            0.3 * evolution_loss +  # 适中的演化学习权重
                            0.1 * memory_loss +  # 降低Memory Pool对比损失权重
                            0.1 * memory_rehearsal_loss +  # 降低演化回放权重
                            0.2 * rehearsal_loss +  # 保持原型回放权重
                            0.1 * distillation_loss)  # 轻量级知识蒸馏
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # 统计
                total_loss_sum += total_loss.item()
                _, preds = torch.max(cls_logits, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
                # 更新进度条
                if i % 10 == 0:
                    acc = correct / total * 100
                    pbar.set_postfix({
                        'Loss': f'{total_loss.item():.4f}',
                        'Evo': f'{evolution_loss.item():.4f}',
                        'Mem': f'{memory_loss.item():.4f}',
                        'MemReh': f'{memory_rehearsal_loss.item():.4f}',
                        'Acc': f'{acc:.2f}%'
                    })
                    
                # 频繁更新Memory Pool（每5个batch而不是20个）
                if i % 5 == 0:
                    self._update_memory_pool_online(targets, morph0_prototypes, morph1_features)
            
            # 学习率调度
            self.scheduler.step()
            
            # Epoch统计
            epoch_loss = total_loss_sum / len(self.train_loader)
            epoch_evolution_loss = evolution_loss_sum / len(self.train_loader)
            epoch_memory_loss = memory_loss_sum / len(self.train_loader)
            epoch_memory_rehearsal_loss = memory_rehearsal_loss_sum / len(self.train_loader)
            epoch_acc = correct / total * 100
            
            logging.info(f"形态1 Epoch {epoch+1}: Loss={epoch_loss:.4f}, "
                        f"Evo={epoch_evolution_loss:.4f}, Mem={epoch_memory_loss:.4f}, "
                        f"MemReh={epoch_memory_rehearsal_loss:.4f}, Acc={epoch_acc:.2f}%")

    def _get_morphology_0_prototypes(self):
        """获取形态0的原型"""
        if self._current_morphology_pair not in self._morphology_prototypes:
            return {}
            
        morph0_prototypes = {}
        pair_prototypes = self._morphology_prototypes[self._current_morphology_pair]
        
        for class_idx, stages in pair_prototypes.items():
            if 0 in stages:  # 形态0
                morph0_prototypes[class_idx] = stages[0].to(self._device)
                
        return morph0_prototypes

    def _predict_evolution_from_prototypes(self, targets, morph0_prototypes):
        """使用Memory Pool从形态0原型预测形态1 - 确保维度匹配"""
        batch_size = targets.size(0)
        device = targets.device
        feature_dim = 512  # CLIP特征维度
        
        # 为整个batch创建演化预测，确保维度匹配
        predicted_morph1_full = torch.zeros(batch_size, feature_dim, device=device)
        attention_weights_full = None
        has_valid_predictions = False
        
        # 收集有形态0原型的样本
        morph0_batch = []
        valid_indices = []
        
        for i, target in enumerate(targets):
            class_idx = target.item()
            if class_idx in morph0_prototypes:
                morph0_batch.append(morph0_prototypes[class_idx])
                valid_indices.append(i)
        
        if morph0_batch:
            # 转换为tensor
            morph0_features = torch.stack(morph0_batch).to(device)
            
            # 使用Memory Pool进行演化预测
            predicted_subset, attention_weights = self.memory_pool(morph0_features, mode='evolve')
            
            if predicted_subset is not None:
                # 将预测结果填入对应位置
                for j, idx in enumerate(valid_indices):
                    predicted_morph1_full[idx] = predicted_subset[j]
                
                has_valid_predictions = True
                attention_weights_full = attention_weights
        
        # 返回完整batch维度的预测结果
        if has_valid_predictions:
            return predicted_morph1_full, attention_weights_full
        else:
            return None, None

    def _compute_memory_pool_loss(self, morph1_features, targets):
        """计算Memory Pool的对比学习损失"""
        # 简单的特征对比损失
        normalized_features = F.normalize(morph1_features, dim=1)
        
        # 计算批内相似度矩阵
        similarity_matrix = torch.matmul(normalized_features, normalized_features.t())
        
        # 创建标签矩阵
        targets_expanded = targets.unsqueeze(1)
        label_matrix = (targets_expanded == targets_expanded.t()).float()
        
        # 对比损失（简化版）
        positive_pairs = similarity_matrix * label_matrix
        negative_pairs = similarity_matrix * (1 - label_matrix)
        
        loss = torch.clamp(0.5 - positive_pairs.mean() + negative_pairs.mean(), min=0)
        
        return loss

    def _update_memory_pool_online(self, targets, morph0_prototypes, morph1_features):
        """智能在线更新Memory Pool - 增强版"""
        with torch.no_grad():
            for i, target in enumerate(targets):
                class_idx = target.item()
                if class_idx in morph0_prototypes:
                    morph0_proto = morph0_prototypes[class_idx].unsqueeze(0)
                    morph1_feat = morph1_features[i].unsqueeze(0)
                    
                    # 更新Memory Pool的键值对
                    self.memory_pool.update_memory(morph0_proto, morph1_feat, learning_rate=0.01)
                    
                    # 检查是否可以更新演化历史
                    if hasattr(self, '_morphology_prototypes') and self._current_morphology_pair in self._morphology_prototypes:
                        if class_idx in self._morphology_prototypes[self._current_morphology_pair]:
                            # 如果有完整的形态对，更新演化历史
                            stages = self._morphology_prototypes[self._current_morphology_pair][class_idx]
                            if 0 in stages and self._morphology_stage == 1:
                                # 计算新的形态1原型（在线平均）
                                old_morph1 = stages.get(1, morph1_feat.squeeze(0))
                                new_morph1 = 0.9 * old_morph1 + 0.1 * morph1_feat.squeeze(0)
                                
                                # 更新演化历史
                                self.memory_pool.update_evolution_history(
                                    class_idx, stages[0], new_morph1
                                )

    def _compute_morphology_rehearsal_loss(self):
        """计算形态演化的原型回放损失 - 简化版"""
        if not hasattr(self, '_morphology_prototypes') or not self._morphology_prototypes:
            return torch.tensor(0.0, device=self._device)
            
        rehearsal_loss = torch.tensor(0.0, device=self._device)
        count = 0
        
        # 只对已完成的形态对进行回放（不包括当前正在训练的）
        for pair_idx in range(self._current_morphology_pair):
            if pair_idx in self._morphology_prototypes:
                pair_prototypes = self._morphology_prototypes[pair_idx]
                
                for class_idx, stages in pair_prototypes.items():
                    # 只对每个类别选择一个代表性原型（避免过多回放）
                    selected_stage = 1 if 1 in stages else (0 if 0 in stages else None)
                    
                    if selected_stage is not None:
                        prototype = stages[selected_stage]
                        if prototype is not None:
                            # 计算原型的分类损失
                            proto_gpu = prototype.to(self._device).unsqueeze(0)
                            
                            try:
                                class_to_label = self.data_manager._class_to_label
                                templates = self.data_manager._data_to_prompt[0]
                                total_labels = class_to_label[:self._total_classes]
                                
                                text_batch = [templates.format(lbl) for lbl in total_labels]
                                proto_logits = self._forward_for_classification_features(proto_gpu, text_batch)
                                
                                target = torch.tensor([class_idx], device=self._device, dtype=torch.long)
                                
                                if class_idx < proto_logits.size(1):
                                    proto_loss = F.cross_entropy(proto_logits, target)
                                    rehearsal_loss += proto_loss
                                    count += 1
                            except Exception as e:
                                # 如果计算失败，跳过这个原型
                                logging.debug(f"原型回放计算失败: {e}")
                                continue
        
        return rehearsal_loss / count if count > 0 else rehearsal_loss

    def _setup_optimizer_and_scheduler(self):
        """设置优化器和调度器"""
        # 冻结backbone，只训练必要部分
        for name, param in self._network.convnet.named_parameters():
            if 'logit_scale' not in name:
                param.requires_grad = False
                
        self._network.freeze_projection_weight_new()
        
        # 收集需要优化的参数
        params_to_optimize = list(self._network.parameters()) + list(self.memory_pool.parameters())
        
        if self.args['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(
                params_to_optimize, momentum=0.9, 
                lr=self.init_lr, weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            self.optimizer = torch.optim.AdamW(
                params_to_optimize, lr=self.init_lr, 
                weight_decay=self.weight_decay
            )
        else:
            self.optimizer = torch.optim.AdamW(
                params_to_optimize, lr=self.init_lr, 
                weight_decay=self.weight_decay
            )
            
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.tuned_epoch, eta_min=self.min_lr
        )

    def _forward_for_classification(self, images, text_list):
        """分类前向传播 - 使用分批文本编码避免内存问题"""
        image_features = self._network.encode_image(images)
        image_features = F.normalize(image_features, dim=1)
        
        # 分批处理文本编码，避免一次性编码过多文本
        text_batch_size = 8  # 每次最多编码8个文本
        text_features_list = []
        
        with torch.no_grad():
            for i in range(0, len(text_list), text_batch_size):
                batch_texts = text_list[i:i + text_batch_size]
                try:
                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    texts_tokenized = self._network.tokenizer(batch_texts).to(self._device)
                    batch_text_features = self._network.encode_text(texts_tokenized)
                    batch_text_features = F.normalize(batch_text_features, dim=1)
                    text_features_list.append(batch_text_features)
                    
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        logging.warning(f"文本编码CUDA错误，跳过批次 {i//text_batch_size}: {e}")
                        # 使用零向量填充
                        dummy_features = torch.zeros(len(batch_texts), image_features.size(1)).to(self._device)
                        text_features_list.append(dummy_features)
                    else:
                        raise e
            
            # 合并所有文本特征
            if text_features_list:
                text_features = torch.cat(text_features_list, dim=0)
            else:
                # 如果所有批次都失败，返回零特征
                text_features = torch.zeros(len(text_list), image_features.size(1)).to(self._device)
            
        logits = image_features @ text_features.t()
        return logits

    def _forward_for_classification_features(self, image_features, text_list):
        """使用给定特征进行分类前向传播"""
        image_features = F.normalize(image_features, dim=1)
        
        with torch.no_grad():
            texts_tokenized = self._network.tokenizer(text_list).to(self._device)
            text_features = self._network.encode_text(texts_tokenized)
            text_features = F.normalize(text_features, dim=1)
            
        logits = image_features @ text_features.t()
        return logits

    def _create_test_loaders(self, data_manager):
        """创建测试数据加载器"""
        try:
            # 当前任务测试集
            current_test_dataset = data_manager.get_multimodal_dataset(
                self._cur_task, source="test", mode="test"
            )
            current_test_loader = DataLoader(
                current_test_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=num_workers
            )
            
            # 累积测试集
            test_datasets = []
            for task_idx in range(self._cur_task + 1):
                try:
                    task_test_dataset = data_manager.get_multimodal_dataset(
                        task_idx, source="test", mode="test"
                    )
                    if len(task_test_dataset) > 0:
                        test_datasets.append(task_test_dataset)
                except Exception as e:
                    logging.warning(f"获取任务{task_idx}测试数据失败: {e}")
                    continue
                    
            if test_datasets:
                from torch.utils.data import ConcatDataset
                cumulative_test_dataset = ConcatDataset(test_datasets)
                cumulative_test_loader = DataLoader(
                    cumulative_test_dataset, batch_size=self.batch_size,
                    shuffle=False, num_workers=num_workers
                )
            else:
                cumulative_test_loader = current_test_loader
                
            return current_test_loader, cumulative_test_loader
            
        except Exception as e:
            logging.error(f"创建测试加载器失败: {e}")
            return None, None

    def eval_morphology_evolution(self, data_manager, pair_idx):
        """评估形态演化性能"""
        if pair_idx not in self._morphology_prototypes:
            return 0.0
            
        # 获取形态0和形态1的原型
        pair_prototypes = self._morphology_prototypes[pair_idx]
        
        correct_predictions = 0
        total_predictions = 0
        
        for class_idx, stages in pair_prototypes.items():
            if 0 in stages and 1 in stages:
                morph0_proto = stages[0].to(self._device).unsqueeze(0)
                morph1_true = stages[1].to(self._device)
                
                # 使用Memory Pool预测演化
                predicted_morph1, _ = self.memory_pool(morph0_proto, mode='evolve')
                
                if predicted_morph1 is not None:
                    # 计算相似度
                    similarity = F.cosine_similarity(
                        predicted_morph1.squeeze(0), morph1_true, dim=0
                    )
                    
                    # 阈值判断（相似度>0.8认为预测正确）
                    if similarity > 0.8:
                        correct_predictions += 1
                    total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0

    def report_memory_pool_status(self, pair_idx):
        """报告Memory Pool状态"""
        status = self.memory_pool.get_memory_status()
        
        logging.info(f"=== Memory Pool状态报告 (形态对 {pair_idx}) ===")
        logging.info(f"  记忆池大小: {status['pool_size']}")
        logging.info(f"  演化历史类别数: {status['evolution_history_classes']}")
        logging.info(f"  总演化对数: {status['total_evolution_pairs']}")
        logging.info(f"  记忆利用率: {status['memory_utilization']:.2%}")
        
        # 输出每个类别的演化历史
        if hasattr(self.memory_pool, 'evolution_history'):
            logging.info("  演化历史详情:")
            for class_idx, history in self.memory_pool.evolution_history.items():
                logging.info(f"    类别{class_idx}: {len(history)}个演化对")

    @torch.no_grad()
    def _compute_accuracy(self, model, loader):
        """计算准确率"""
        model.eval()
        correct, total = 0, 0
        
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt[0]
        all_labels = class_to_label[:self._total_classes]
        
        for batch in loader:
            if isinstance(batch[1], dict) and 'image' in batch[1]:
                _, data_dict, targets = batch
                inputs = data_dict['image'].to(self._device)
            else:
                _, inputs, targets = batch
                inputs = inputs.to(self._device)
                
            targets = targets.long().to(self._device)
            
            text_list = [templates.format(lbl) for lbl in all_labels]
            logits = self._forward_for_classification(inputs, text_list)
            
            _, preds = torch.max(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
        return np.around((correct / total) * 100, decimals=2)

    def _eval_cnn(self, loader):
        """评估时使用Memory Pool增强预测"""
        self._network.eval()
        y_pred, y_true = [], []
        total_samples = 0
        
        logging.info(f"开始测试评估...")
        
        for batch in loader:
            # Handle different batch formats
            if len(batch) == 3:
                _, _inputs, _targets = batch
            elif len(batch) == 4:
                _, _inputs, _targets, _stages = batch
            else:
                logging.error(f"Unexpected batch format with {len(batch)} elements")
                continue
                
            # 处理多种输入格式
            if isinstance(_inputs, dict) and 'image' in _inputs:
                _inputs = _inputs['image'].to(self._device)
            elif isinstance(_inputs, dict) and 'stage_id' in _inputs:
                data_dict = _inputs
                _inputs = data_dict['image'].to(self._device)
            else:
                _inputs = _inputs.to(self._device)
                
            _targets = _targets.to(self._device)
            
            with torch.no_grad():
                # 准备文本模板进行分类
                class_to_label = self.data_manager._class_to_label
                templates = self.data_manager._data_to_prompt[0]
                total_labels = class_to_label[:max(self._total_classes, len(class_to_label))]
                text_batch = [templates.format(lbl) for lbl in total_labels[:self._total_classes]]
                
                # 获取基础分类预测
                base_logits = self._forward_for_classification(_inputs, text_batch)
                
                # 获取top-k预测结果，符合_evaluate期望的格式
                k = min(self.topk, base_logits.size(1))
                if k > 0:
                    topk_preds = torch.topk(base_logits, k=k, dim=1)[1]
                    
                    # 如果k小于self.topk，需要填充
                    if k < self.topk:
                        padding = torch.zeros(base_logits.size(0), self.topk - k, 
                                            device=base_logits.device, dtype=torch.long)
                        topk_preds = torch.cat([topk_preds, padding], dim=1)
                else:
                    # 如果没有类别，返回零预测
                    topk_preds = torch.zeros(base_logits.size(0), self.topk, 
                                           device=base_logits.device, dtype=torch.long)
                
                # 如果Memory Pool存在且有演化历史，尝试使用它增强预测
                if (hasattr(self, 'memory_pool') and 
                    hasattr(self.memory_pool, 'evolution_history') and 
                    len(self.memory_pool.evolution_history) > 0):
                    try:
                        # 提取图像特征用于Memory Pool查询
                        image_features = self._network.encode_image(_inputs)
                        
                        # 使用Memory Pool的演化预测功能
                        predicted_evolution, attention_weights = self.memory_pool.forward(
                            image_features, mode='evolve'
                        )
                        
                        if predicted_evolution is not None:
                            # 将演化预测转换为分类logits
                            evolution_logits = self._forward_for_classification_features(
                                predicted_evolution, text_batch
                            )
                            
                            # 确保维度匹配
                            if evolution_logits.size(1) == base_logits.size(1):
                                # 融合预测：基础预测 + Memory Pool演化增强
                                alpha = 0.15  # 轻微增强权重
                                enhanced_logits = (1 - alpha) * base_logits + alpha * evolution_logits
                                
                                # 重新计算top-k预测
                                k = min(self.topk, enhanced_logits.size(1))
                                if k > 0:
                                    topk_preds = torch.topk(enhanced_logits, k=k, dim=1)[1]
                                    if k < self.topk:
                                        padding = torch.zeros(enhanced_logits.size(0), self.topk - k, 
                                                            device=enhanced_logits.device, dtype=torch.long)
                                        topk_preds = torch.cat([topk_preds, padding], dim=1)
                                
                                # 记录增强效果
                                if hasattr(self, '_memory_enhancement_count'):
                                    self._memory_enhancement_count += _inputs.size(0)
                                else:
                                    self._memory_enhancement_count = _inputs.size(0)
                    except Exception as e:
                        # Memory Pool出错时回退到基础预测，但不记录警告（避免日志过多）
                        pass
            
            y_pred.append(topk_preds.cpu().numpy())
            y_true.append(_targets.cpu().numpy())
            total_samples += _targets.size(0)
            
        # 确保返回正确的格式
        if y_pred and y_true:
            y_pred_array = np.concatenate(y_pred, axis=0)  # 二维数组 [N, topk]
            y_true_array = np.concatenate(y_true, axis=0)  # 一维数组 [N]
            
            # 计算Top-1准确率
            top1_correct = (y_pred_array[:, 0] == y_true_array).sum()
            top1_accuracy = (top1_correct / total_samples) * 100
            
            # 显示测试结果
            logging.info(f"测试完成:")
            logging.info(f"  测试样本数量: {total_samples}")
            logging.info(f"  Top-1准确率: {top1_accuracy:.2f}%")
            
            # 如果Memory Pool被使用，显示增强效果
            if hasattr(self, '_memory_enhancement_count') and self._memory_enhancement_count > 0:
                enhancement_rate = (self._memory_enhancement_count / total_samples) * 100
                logging.info(f"  Memory Pool增强样本: {self._memory_enhancement_count}/{total_samples} ({enhancement_rate:.1f}%)")
            
            return y_pred_array, y_true_array
        else:
            # 返回空数组但保持正确的维度
            logging.info(f"测试完成: 无有效样本")
            empty_pred = np.zeros((0, self.topk), dtype=np.int64)
            empty_true = np.zeros((0,), dtype=np.int64)
            return empty_pred, empty_true

    def eval_task(self):
        """评估任务性能 - 增强版，包含完整的增量学习指标"""
        # 标准评估
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        # 增量学习特定评估
        incremental_metrics = self._compute_incremental_metrics()

        if self.args["convnet_type"].lower() != "clip" or self.args["model_name"].lower() == "l2p" or self.args["model_name"].lower() == "dualprompt":
            return cnn_accy, nme_accy, None, None, None, None, incremental_metrics
        else:
            y_pred, y_true = self._eval_zero_shot()
            zs_acc = self._evaluate_zs(y_pred, y_true)
            zs_seen, zs_unseen, zs_harmonic, zs_total = zs_acc["grouped"]["old"], zs_acc["grouped"]["new"], zs_acc["grouped"]["harmonic"], zs_acc["grouped"]["total"]

        return cnn_accy, nme_accy, zs_seen, zs_unseen, zs_harmonic, zs_total, incremental_metrics

    def _compute_incremental_metrics(self):
        """计算增量学习指标"""
        if not hasattr(self, '_task_accuracies'):
            self._task_accuracies = []
            
        # 计算当前任务在所有已知类别上的准确率
        current_acc = self._compute_accuracy_on_all_tasks()
        self._task_accuracies.append(current_acc)
        
        # 计算各种增量学习指标
        metrics = {
            'current_task_acc': current_acc,
            'average_incremental_acc': self._compute_average_incremental_accuracy(),
            'forgetting_rate': self._compute_forgetting_rate(),
            'backward_transfer': self._compute_backward_transfer(),
            'forward_transfer': self._compute_forward_transfer(),
            'all_task_accuracies': self._task_accuracies.copy()
        }
        
        return metrics

    def _compute_accuracy_on_all_tasks(self):
        """计算在所有已学习任务上的累积准确率"""
        if not hasattr(self, 'data_manager') or self.data_manager is None:
            return 0.0
            
        # 收集所有已学习任务的数据
        all_test_datasets = []
        for task_idx in range(self._cur_task + 1):
            try:
                test_dataset = self.data_manager.get_multimodal_dataset(
                    task_idx, source="test", mode="test"
                )
                if len(test_dataset) > 0:
                    all_test_datasets.append(test_dataset)
            except Exception as e:
                logging.warning(f"获取任务{task_idx}测试数据失败: {e}")
                continue
        
        if not all_test_datasets:
            return 0.0
        
        # 合并所有测试数据
        from torch.utils.data import ConcatDataset
        combined_test_dataset = ConcatDataset(all_test_datasets)
        combined_test_loader = DataLoader(
            combined_test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=num_workers
        )
        
        # 计算累积准确率
        return self._compute_accuracy(self._network, combined_test_loader)

    def _compute_average_incremental_accuracy(self):
        """计算平均增量准确率"""
        if not hasattr(self, '_task_accuracies') or len(self._task_accuracies) == 0:
            return 0.0
        return sum(self._task_accuracies) / len(self._task_accuracies)

    def _compute_forgetting_rate(self):
        """计算遗忘率"""
        if not hasattr(self, '_task_accuracies') or len(self._task_accuracies) <= 1:
            return 0.0
        
        # 计算每个任务从学习完成到现在的性能下降
        forgetting_rates = []
        for i in range(len(self._task_accuracies) - 1):
            max_acc = max(self._task_accuracies[i:])  # 该任务的最高准确率
            current_acc = self._compute_task_specific_accuracy(i)  # 当前在该任务上的准确率
            forgetting = max_acc - current_acc
            forgetting_rates.append(max(0, forgetting))  # 遗忘率不能为负
        
        return sum(forgetting_rates) / len(forgetting_rates) if forgetting_rates else 0.0

    def _compute_task_specific_accuracy(self, task_idx):
        """计算在特定任务上的准确率"""
        if not hasattr(self, 'data_manager') or self.data_manager is None:
            return 0.0
        
        try:
            test_dataset = self.data_manager.get_multimodal_dataset(
                task_idx, source="test", mode="test"
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=num_workers
            )
            return self._compute_accuracy(self._network, test_loader)
        except Exception as e:
            logging.warning(f"计算任务{task_idx}准确率失败: {e}")
            return 0.0

    def _compute_backward_transfer(self):
        """计算后向迁移（之前任务的性能提升）"""
        if not hasattr(self, '_task_accuracies') or len(self._task_accuracies) <= 1:
            return 0.0
        
        # 计算每个之前任务的性能变化
        transfers = []
        for i in range(len(self._task_accuracies) - 1):
            initial_acc = self._task_accuracies[i]  # 任务i学习完成时的准确率
            current_acc = self._compute_task_specific_accuracy(i)  # 当前在任务i上的准确率
            transfer = current_acc - initial_acc
            transfers.append(transfer)
        
        return sum(transfers) / len(transfers) if transfers else 0.0

    def _compute_forward_transfer(self):
        """计算前向迁移（对新任务的帮助）"""
        # 这个指标通常需要在学习新任务之前评估，暂时返回0
        return 0.0

    def _compute_distillation_loss(self, inputs, targets):
        """计算知识蒸馏损失 - 简单有效的防遗忘方法"""
        if self._old_network is None:
            return torch.tensor(0.0, device=self._device)
        
        try:
            # 当前网络的输出
            current_features = self._network.encode_image(inputs)
            
            # 旧网络的输出（teacher）
            with torch.no_grad():
                old_features = self._old_network.encode_image(inputs)
            
            # 特征级别的知识蒸馏
            distillation_loss = F.mse_loss(current_features, old_features.detach())
            
            return distillation_loss
        except Exception as e:
            logging.debug(f"知识蒸馏计算失败: {e}")
            return torch.tensor(0.0, device=self._device)

    def _get_memory(self):
        """获取历史样本用于回放 - 核心防遗忘机制"""
        if self._cur_task == 0:
            return None
        
        # 收集所有历史任务的代表性样本
        memory_data = []
        memory_targets = []
        memory_stages = []
        
        # 每个历史类别保存少量样本
        samples_per_class = max(2, self.samples_per_class // 2)  # 每类至少2个样本
        
        try:
            for task_idx in range(self._cur_task):
                # 获取历史任务数据
                task_data, task_targets, task_stages, _ = self.data_manager.get_dataset(
                    cil_task_idx=task_idx,
                    source="train", 
                    mode="test",  # 使用test模式避免数据增强
                    ret_data=True
                )
                
                if len(task_data) > 0:
                    # 按类别采样
                    unique_classes = np.unique(task_targets)
                    for class_idx in unique_classes:
                        class_mask = (task_targets == class_idx)
                        class_data = task_data[class_mask]
                        class_targets = task_targets[class_mask]
                        class_stages = task_stages[class_mask]
                        
                        # 随机采样代表性样本
                        if len(class_data) > samples_per_class:
                            indices = np.random.choice(len(class_data), samples_per_class, replace=False)
                            class_data = class_data[indices]
                            class_targets = class_targets[indices]
                            class_stages = class_stages[indices]
                        
                        memory_data.append(class_data)
                        memory_targets.append(class_targets)
                        memory_stages.append(class_stages)
            
            if memory_data:
                # 合并所有回放样本
                memory_data = np.concatenate(memory_data)
                memory_targets = np.concatenate(memory_targets)
                memory_stages = np.concatenate(memory_stages)
                
                logging.info(f"构建回放记忆: {len(memory_data)}个历史样本，来自{self._cur_task}个历史任务")
                
                return (memory_data, memory_targets, memory_stages)
                
        except Exception as e:
            logging.warning(f"构建回放记忆失败: {e}")
            
        return None

# 为了保持兼容性，创建一个别名
Learner = MorphologyEvolutionLearner 