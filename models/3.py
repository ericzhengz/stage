import copy
import logging
import math
import torch
import numpy as np
import random
import time
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.inc_net import Proof_Net
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, get_attribute, ClipLoss

num_workers = 4

class LRUCache:
    """LRU缓存实现"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        
    def get(self, key):
        if key in self.cache:
            # 移到末尾（最近使用）
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # 删除最久未使用的
                self.cache.popitem(last=False)
        self.cache[key] = value

class EnhancedEvolutionCache:
    """多层级演化预测缓存系统"""
    def __init__(self, max_cache_size=1000):
        # 原型级缓存：{morph0_hash: predicted_morph1}
        self.prototype_cache = LRUCache(max_cache_size // 2)
        
        # 模式级缓存：{pattern_signature: evolution_vector}
        self.pattern_cache = LRUCache(max_cache_size // 4)
        
        # 批次级缓存：{batch_signature: batch_predictions}
        self.batch_cache = LRUCache(max_cache_size // 4)
        
        # 缓存统计
        self.hit_count = 0
        self.miss_count = 0
        
    def get_cache_key(self, tensor):
        """生成高效的缓存键"""
        # 使用tensor的hash值作为缓存键，避免存储大量数据
        return hash(tensor.detach().cpu().numpy().tobytes())
        
    def get_batch_signature(self, targets, morph0_prototypes):
        """生成批次签名用于批次级缓存"""
        class_indices = sorted([t.item() for t in targets if t.item() in morph0_prototypes])
        return tuple(class_indices)
    
    def get_hit_rate(self):
        """获取缓存命中率"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def clear_cache(self):
        """清空所有缓存"""
        self.prototype_cache = LRUCache(self.prototype_cache.capacity)
        self.pattern_cache = LRUCache(self.pattern_cache.capacity)
        self.batch_cache = LRUCache(self.batch_cache.capacity)
        self.hit_count = 0
        self.miss_count = 0

class AdaptiveEvolutionManager:
    """自适应演化对管理器 - 优化版"""
    def __init__(self):
        # 降低质量阈值，确保至少保留一些演化对
        self.base_history_size = 2  # 减少到2个
        self.max_history_size = 5   # 减少到5个
        self.quality_threshold = 0.3  # 大幅降低质量阈值（从0.75降到0.3）
        
        # 演化对质量评估
        self.evolution_quality = {}  # {class_idx: [quality_scores]}
        self.evolution_count = {}    # {class_idx: count} 跟踪实际存储的演化对数量
        
    def _compute_evolution_quality(self, evolution_vector):
        """计算演化向量的质量分数"""
        # 基于演化向量的模长和方向稳定性评估质量
        magnitude = torch.norm(evolution_vector).item()
        
        # 归一化到[0,1]范围
        normalized_magnitude = min(1.0, magnitude / 2.0)  # 假设正常演化向量模长在2以内
        
        # 可以加入更多质量评估标准
        quality_score = normalized_magnitude
        
        return quality_score
        
    def should_keep_evolution_pair(self, class_idx, morph0_proto, morph1_proto):
        """判断是否应该保留演化对 - 修复版"""
        # 计算演化向量的质量分数
        evolution_vector = morph1_proto - morph0_proto
        quality_score = self._compute_evolution_quality(evolution_vector)
        
        if class_idx not in self.evolution_quality:
            self.evolution_quality[class_idx] = []
        if class_idx not in self.evolution_count:
            self.evolution_count[class_idx] = 0
            
        # 记录质量分数
        self.evolution_quality[class_idx].append(quality_score)
        
        # 修复逻辑：确保至少保留基础数量的演化对
        current_count = self.evolution_count[class_idx]
        
        # 如果当前数量少于基础数量，强制保留
        if current_count < self.base_history_size:
            self.evolution_count[class_idx] += 1
            return True
        
        # 如果质量足够高，保留
        if quality_score > self.quality_threshold:
            self.evolution_count[class_idx] += 1
            return True
        
        # 如果当前数量少于最大数量，且质量不是太差，也保留
        if current_count < self.max_history_size and quality_score > 0.2:
            self.evolution_count[class_idx] += 1
            return True
        
        return False
            
    def adaptive_history_limit(self, class_idx):
        """自适应历史限制"""
        if class_idx not in self.evolution_quality or len(self.evolution_quality[class_idx]) == 0:
            return self.base_history_size
            
        avg_quality = np.mean(self.evolution_quality[class_idx])
        if avg_quality > 0.9:
            return self.max_history_size  # 高质量类别保留更多
        elif avg_quality > 0.8:
            return (self.base_history_size + self.max_history_size) // 2
        else:
            return self.base_history_size
    
    def get_quality_stats(self):
        """获取质量统计信息"""
        stats = {}
        for class_idx, qualities in self.evolution_quality.items():
            if qualities:
                stats[class_idx] = {
                    'avg_quality': np.mean(qualities),
                    'max_quality': np.max(qualities),
                    'min_quality': np.min(qualities),
                    'count': len(qualities)
                }
        return stats

class HierarchicalRehearsalStrategy:
    """分层演化回放策略"""
    def __init__(self):
        # 重要性分层
        self.critical_classes = set()  # 关键类别
        self.stable_classes = set()    # 稳定类别
        self.recent_classes = set()    # 最近学习的类别
        
        # 回放频率控制
        self.rehearsal_frequency = {
            'critical': 1.0,   # 每次都回放
            'recent': 0.8,     # 80%概率回放
            'stable': 0.4      # 40%概率回放
        }
        
    def update_class_importance(self, class_idx, importance_type):
        """更新类别重要性"""
        # 清除其他分类
        self.critical_classes.discard(class_idx)
        self.stable_classes.discard(class_idx)
        self.recent_classes.discard(class_idx)
        
        # 添加到新分类
        if importance_type == 'critical':
            self.critical_classes.add(class_idx)
        elif importance_type == 'stable':
            self.stable_classes.add(class_idx)
        elif importance_type == 'recent':
            self.recent_classes.add(class_idx)
        
    def select_rehearsal_classes(self, current_targets, all_history_classes):
        """智能选择需要回放的类别"""
        rehearsal_classes = set()
        
        # 1. 当前batch中出现的类别（高优先级）
        rehearsal_classes.update(current_targets.tolist())
        
        # 2. 关键类别（必须回放）
        rehearsal_classes.update(self.critical_classes)
        
        # 3. 按概率选择其他类别
        other_classes = set(all_history_classes) - rehearsal_classes
        
        for class_idx in other_classes:
            if class_idx in self.recent_classes:
                if random.random() < self.rehearsal_frequency['recent']:
                    rehearsal_classes.add(class_idx)
            elif class_idx in self.stable_classes:
                if random.random() < self.rehearsal_frequency['stable']:
                    rehearsal_classes.add(class_idx)
        
        # 4. 限制总数量（避免计算过载）
        max_rehearsal_classes = 5
        if len(rehearsal_classes) > max_rehearsal_classes:
            # 保留重要类别，随机选择其他
            important_classes = rehearsal_classes.intersection(
                self.critical_classes.union(set(current_targets.tolist()))
            )
            other_classes = rehearsal_classes - important_classes
            
            remaining_slots = max_rehearsal_classes - len(important_classes)
            if remaining_slots > 0 and other_classes:
                selected_others = random.sample(
                    list(other_classes), 
                    min(remaining_slots, len(other_classes))
                )
                rehearsal_classes = important_classes.union(set(selected_others))
            else:
                rehearsal_classes = important_classes
        
        return list(rehearsal_classes)

class MorphologyMemoryPool(nn.Module):
    """
    重新设计的形态演化记忆池 - 固定大小模式池
    核心思路：使用固定大小的演化模式池，避免动态增长
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
        
        # 重新设计：固定大小的演化模式池
        self.evolution_patterns = nn.Parameter(torch.randn(50, feature_dim))  # 固定50个演化模式
        self.pattern_usage_count = torch.zeros(50)  # 使用次数统计
        self.pattern_quality = torch.zeros(50)  # 质量分数
        
        # 简化的缓存
        self.evolution_cache = EnhancedEvolutionCache(max_cache_size=500)
        
        # 性能监控
        self.performance_tracker = {
            'cache_hit_rate': 0.0,
            'compute_time': [],
            'pattern_usage': torch.zeros(50)
        }
        
        self._init_parameters()

    def _init_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.memory_keys)
        nn.init.xavier_uniform_(self.memory_values)
        nn.init.xavier_uniform_(self.evolution_patterns)
        
        for m in self.evolution_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, morph0_features, mode='evolve'):
        """
        前向传播 - 简化版本
        """
        if mode == 'evolve':
            return self.fast_predict_evolution(morph0_features)
        else:
            return self._update_memory_patterns(morph0_features)

    def fast_predict_evolution(self, morph0_features):
        """快速演化预测 - 使用固定模式池"""
        start_time = time.time()
        batch_size = morph0_features.size(0)
        
        # 1. 检查缓存
        cached_results = []
        cache_mask = []
        
        for i in range(batch_size):
            cache_key = self.evolution_cache.get_cache_key(morph0_features[i])
            cached_result = self.evolution_cache.prototype_cache.get(cache_key)
            
            if cached_result is not None:
                cached_results.append(cached_result)
                cache_mask.append(True)
                self.evolution_cache.hit_count += 1
            else:
                cache_mask.append(False)
                self.evolution_cache.miss_count += 1
        
        # 2. 只对未缓存的样本进行计算
        uncached_indices = [i for i, cached in enumerate(cache_mask) if not cached]
        
        final_predictions = torch.zeros_like(morph0_features)
        attention_weights = None
        
        if uncached_indices:
            uncached_features = morph0_features[uncached_indices]
            
            # 使用固定模式池进行快速预测
            uncached_predictions, attention_weights = self._predict_with_fixed_patterns(uncached_features)
            
            # 更新缓存
            for idx, pred in zip(uncached_indices, uncached_predictions):
                cache_key = self.evolution_cache.get_cache_key(morph0_features[idx])
                self.evolution_cache.prototype_cache.put(cache_key, pred.detach())
        
        # 3. 合并缓存和新计算的结果
        cached_idx = 0
        uncached_idx = 0
        
        for i in range(batch_size):
            if cache_mask[i]:
                final_predictions[i] = cached_results[cached_idx]
                cached_idx += 1
            else:
                if uncached_idx < len(uncached_predictions):
                    final_predictions[i] = uncached_predictions[uncached_idx]
                    uncached_idx += 1
                else:
                    final_predictions[i] = morph0_features[i]
        
        # 更新性能统计
        compute_time = time.time() - start_time
        self.performance_tracker['compute_time'].append(compute_time)
        self.performance_tracker['cache_hit_rate'] = self.evolution_cache.get_hit_rate()
        
        return final_predictions, attention_weights

    def _predict_with_fixed_patterns(self, morph0_features):
        """使用固定模式池预测演化"""
        batch_size = morph0_features.size(0)
        feature_dim = morph0_features.size(-1)
        
        # 1. 快速选择top-3演化模式（固定复杂度）
        query_norm = F.normalize(morph0_features, dim=-1)
        patterns_norm = F.normalize(self.evolution_patterns, dim=-1)
        
        # 计算相似度矩阵 [batch_size, 50]
        similarity_matrix = torch.matmul(query_norm, patterns_norm.t())
        
        # 选择top-3模式
        top_similarities, top_indices = torch.topk(similarity_matrix, k=3, dim=1)
        
        # 2. 使用注意力机制聚合模式
        selected_patterns = torch.zeros(batch_size, 3, feature_dim, 
                                      device=morph0_features.device,
                                      dtype=morph0_features.dtype)
        
        for i in range(batch_size):
            for j in range(3):
                pattern_idx = top_indices[i, j]
                selected_patterns[i, j] = self.evolution_patterns[pattern_idx]
                
                # 更新使用统计
                self.performance_tracker['pattern_usage'][pattern_idx] += 1
        
        # 3. 注意力机制聚合
        try:
            query = morph0_features.unsqueeze(1)  # [batch_size, 1, feature_dim]
            key = selected_patterns  # [batch_size, 3, feature_dim]
            value = selected_patterns  # [batch_size, 3, feature_dim]
            
            attended_patterns, attention_weights = self.attention(query, key, value)
            context_features = attended_patterns.squeeze(1)  # [batch_size, feature_dim]
        except Exception as e:
            logging.warning(f"注意力机制计算失败: {e}")
            context_features = morph0_features
            attention_weights = None
        
        # 4. 预测演化
        try:
            evolution_input = morph0_features + context_features
            predicted_morph1 = self.evolution_net(evolution_input)
            predicted_morph1 = morph0_features + predicted_morph1  # 残差连接
        except Exception as e:
            logging.error(f"演化预测失败: {e}")
            predicted_morph1 = morph0_features
            attention_weights = None
        
        return predicted_morph1, attention_weights

    def update_evolution_patterns(self, morph0_features, morph1_features, learning_rate=0.01):
        """更新演化模式池 - 在线学习"""
        with torch.no_grad():
            # 计算演化向量
            evolution_vectors = morph1_features - morph0_features
            
            # 找到最相似的模式位置
            morph0_norm = F.normalize(morph0_features, dim=1)
            patterns_norm = F.normalize(self.evolution_patterns, dim=1)
            
            similarities = torch.matmul(morph0_norm, patterns_norm.t())
            _, best_indices = torch.max(similarities, dim=1)
            
            # 更新对应的模式
            for i, idx in enumerate(best_indices):
                # 使用移动平均更新
                self.evolution_patterns[idx] = (1 - learning_rate) * self.evolution_patterns[idx] + learning_rate * evolution_vectors[i]
                
                # 更新质量分数（基于相似度）
                self.pattern_quality[idx] = max(self.pattern_quality[idx], similarities[i, idx].item())

    def compute_evolution_loss(self, morph0_features, morph1_features, predicted_morph1):
        """计算演化损失"""
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
        
        return reconstruction_loss + 0.5 * direction_loss

    def smart_rehearsal_loss(self, current_features, target_classes):
        """简化的演化回放损失"""
        # 使用固定模式池进行回放
        batch_size = current_features.size(0)
        
        # 随机选择一些模式进行回放
        num_rehearsal = min(5, batch_size)
        rehearsal_indices = torch.randperm(len(self.evolution_patterns))[:num_rehearsal]
        
        rehearsal_patterns = self.evolution_patterns[rehearsal_indices]
        
        # 计算回放损失
        rehearsal_loss = torch.tensor(0.0, device=current_features.device)
        
        for i in range(num_rehearsal):
            pattern = rehearsal_patterns[i].unsqueeze(0)
            predicted = self._predict_with_fixed_patterns(pattern)[0]
            rehearsal_loss += F.mse_loss(predicted, pattern)
        
        return rehearsal_loss / num_rehearsal if num_rehearsal > 0 else rehearsal_loss

    def compute_rehearsal_loss(self, current_features, target_classes):
        """计算演化回放损失 - 保持向后兼容"""
        return self.smart_rehearsal_loss(current_features, target_classes)

    def get_memory_status(self):
        """获取记忆池状态"""
        status = {
            'pool_size': self.pool_size,
            'fixed_patterns': len(self.evolution_patterns),
            'cache_hit_rate': self.evolution_cache.get_hit_rate(),
            'avg_compute_time': np.mean(self.performance_tracker['compute_time']) if self.performance_tracker['compute_time'] else 0.0,
            'pattern_usage_stats': {
                'max_usage': torch.max(self.performance_tracker['pattern_usage']).item(),
                'min_usage': torch.min(self.performance_tracker['pattern_usage']).item(),
                'avg_usage': torch.mean(self.performance_tracker['pattern_usage']).item()
            }
        }
        return status
    
    def report_optimization_status(self):
        """报告优化状态"""
        status = self.get_memory_status()
        
        logging.info("=== 重新设计的Memory Pool状态报告 ===")
        logging.info(f"  固定模式池大小: {status['fixed_patterns']}")
        logging.info(f"  缓存命中率: {status['cache_hit_rate']:.2%}")
        logging.info(f"  平均计算时间: {status['avg_compute_time']:.4f}s")
        logging.info(f"  模式使用统计: 最大={status['pattern_usage_stats']['max_usage']:.0f}, "
                    f"最小={status['pattern_usage_stats']['min_usage']:.0f}, "
                    f"平均={status['pattern_usage_stats']['avg_usage']:.1f}")

    # 保持向后兼容的方法
    def update_evolution_history(self, class_idx, morph0_prototype, morph1_prototype):
        """更新演化历史 - 现在直接更新模式池"""
        # 将演化对转换为tensor并更新模式池
        morph0_tensor = morph0_prototype.unsqueeze(0)
        morph1_tensor = morph1_prototype.unsqueeze(0)
        self.update_evolution_patterns(morph0_tensor, morph1_tensor, learning_rate=0.01)

    def update_memory(self, morph0_features, morph1_features, learning_rate=0.01):
        """在线更新记忆池"""
        self.update_evolution_patterns(morph0_features, morph1_features, learning_rate)

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
        
        # 损失权重配置 - 优化后的权重，降低计算密集组件权重
        self.lambda_evolution = get_attribute(args, "lambda_evolution", 0.5)  # 从1.0降到0.5
        self.lambda_rehearsal = get_attribute(args, "lambda_rehearsal", 0.3)  # 从0.5降到0.3  
        self.lambda_memory = get_attribute(args, "lambda_memory", 0.2)        # 从0.3降到0.2
        
        logging.info(f"形态演化学习器初始化完成:")
        logging.info(f"  每对类别数(将由DataManager动态决定): {self._classes_per_pair}")
        logging.info(f"  Memory Pool大小: {self.memory_pool.pool_size}")
        logging.info(f"  特征维度: {feature_dim}")
        logging.info(f"  优化后权重 - 演化: {self.lambda_evolution}, 回放: {self.lambda_rehearsal}, Memory: {self.lambda_memory}")
        
        # 严格检查配置参数，确保与IIMinsects202.json一致
        logging.info(f"=== 内存管理配置检查 ===")
        logging.info(f"  memory_size: {self._memory_size}")
        logging.info(f"  memory_per_class: {self._memory_per_class}")  
        logging.info(f"  fixed_memory: {self._fixed_memory}")
        logging.info(f"  预期配置: memory_size=400, fixed_memory=false")
        
        if self._memory_size != 400:
            logging.warning(f"memory_size配置异常: 期望400, 实际{self._memory_size}")
        if self._fixed_memory != False:
            logging.warning(f"fixed_memory配置异常: 期望false, 实际{self._fixed_memory}")

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
        
        # 从DataManager获取当前任务的实际类别数
        num_classes_in_task = data_manager.get_task_size(self._cur_task)
        logging.info(f"从DataManager获取到当前任务的类别数: {num_classes_in_task}")

        # 形态演化的类别数量管理
        if self._morphology_stage == 0:
            # 形态0: 学习新的类别
            self._total_classes = self._known_classes + num_classes_in_task
        else:
            # 形态1: 学习已有类别的演化，网络结构应与形态0保持一致
            # 因此，总类别数也需要包含当前正在学习演化的类别
            self._total_classes = self._known_classes + num_classes_in_task
        
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
            
        # 更新known_classes（仅在形态1完成后）
        if self._morphology_stage == 1:
            self._known_classes = self._total_classes
        else:
            # 形态0完成后，不要立即更新known_classes，等待形态1完成
            pass

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
                memory_rehearsal_loss = self.memory_pool.smart_rehearsal_loss(features, targets)
                memory_rehearsal_loss_sum += memory_rehearsal_loss.item()
                
                # 原型回放损失
                rehearsal_loss = self._compute_morphology_rehearsal_loss()
                
                # 知识蒸馏损失（简单有效的防遗忘）
                distillation_loss = self._compute_distillation_loss(inputs, targets)
                
                # 总损失 - 使用优化后的权重
                total_loss = (ce_loss + 
                            0.05 * memory_rehearsal_loss +  # 进一步降低Memory Pool权重
                            0.15 * rehearsal_loss +         # 适度降低原型回放权重
                            0.1 * distillation_loss)        # 轻量级知识蒸馏
                
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
                memory_rehearsal_loss = self.memory_pool.smart_rehearsal_loss(morph1_features, targets)
                memory_rehearsal_loss_sum += memory_rehearsal_loss.item()
                
                # 原型回放损失
                rehearsal_loss = self._compute_morphology_rehearsal_loss()
                
                # 知识蒸馏损失（简单有效的防遗忘）
                distillation_loss = self._compute_distillation_loss(inputs, targets)
                
                # 总损失 - 使用优化后的权重，专注于演化学习
                total_loss = (ce_loss + 
                            0.25 * evolution_loss +         # 适中的演化学习权重
                            0.08 * memory_loss +            # 降低Memory Pool对比损失权重
                            0.05 * memory_rehearsal_loss +  # 大幅降低演化回放权重
                            0.15 * rehearsal_loss +         # 保持适中的原型回放权重
                            0.1 * distillation_loss)        # 轻量级知识蒸馏
                
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
            
            # 定期报告Memory Pool优化状态
            if epoch == self.tuned_epoch - 1:  # 最后一个epoch
                self.memory_pool.report_optimization_status()

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
        """报告Memory Pool状态 - 适配新架构"""
        status = self.memory_pool.get_memory_status()
        
        logging.info(f"=== 重新设计的Memory Pool状态报告 (形态对 {pair_idx}) ===")
        logging.info(f"  记忆池大小: {status['pool_size']}")
        logging.info(f"  固定模式池大小: {status['fixed_patterns']}")
        logging.info(f"  缓存命中率: {status['cache_hit_rate']:.2%}")
        logging.info(f"  平均计算时间: {status['avg_compute_time']:.4f}s")
        logging.info(f"  模式使用统计: 最大={status['pattern_usage_stats']['max_usage']:.0f}, "
                    f"最小={status['pattern_usage_stats']['min_usage']:.0f}, "
                    f"平均={status['pattern_usage_stats']['avg_usage']:.1f}")
        
        # 优化效果统计
        avg_usage = status['pattern_usage_stats']['avg_usage']
        if avg_usage > 0:
            logging.info(f"  优化效果: 使用固定50个模式池，避免动态增长")
            logging.info(f"  计算复杂度: O(1) 恒定，不再随演化对数量增长")
            logging.info(f"  预期训练时间: 所有任务保持一致，不再线性增长")

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
                
                # 如果Memory Pool存在，尝试使用它增强预测
                if hasattr(self, 'memory_pool'):
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
        """获取历史样本用于回放 - 严格按照base.py的指示和配置参数"""
        if self._cur_task == 0:
            return None
        
        # 检查是否有存储的记忆数据
        if not hasattr(self, '_data_memory') or self._data_memory is None:
            logging.info("没有存储的记忆数据，返回None")
            return None
        
        # 严格按照配置参数获取记忆大小限制
        # IIMinsects202.json: "memory_size": 400, "fixed_memory": false
        memory_limit = self._memory_size  # 应该是400
        
        logging.info(f"回放内存配置: memory_size={memory_limit}, fixed_memory={self._fixed_memory}")
        
        # 如果存储的记忆数据 <= 限制，全部返回
        if len(self._data_memory) <= memory_limit:
            logging.info(f"记忆样本数({len(self._data_memory)}) <= 限制({memory_limit})，全部用于回放")
            return (self._data_memory, self._targets_memory, self._stages_memory)
        
        # 如果存储的记忆数据 > 限制，需要采样到限制数量
        logging.info(f"记忆样本数({len(self._data_memory)}) > 限制({memory_limit})，进行平衡采样")
        
        # 严格按照base.py的逻辑：平衡采样确保每个类别都有代表
        unique_classes = np.unique(self._targets_memory)
        samples_per_class = max(1, memory_limit // len(unique_classes))
        
        logging.info(f"采样策略: {len(unique_classes)}个类别，每类最多{samples_per_class}个样本")
        
        selected_indices = []
        
        for class_idx in unique_classes:
            class_mask = (self._targets_memory == class_idx)
            class_indices = np.where(class_mask)[0]
            
            num_to_sample = min(len(class_indices), samples_per_class)
            if num_to_sample > 0:
                sampled_indices = np.random.choice(class_indices, num_to_sample, replace=False)
                selected_indices.extend(sampled_indices)
        
        # 如果采样数量不足memory_limit，随机补充剩余样本
        if len(selected_indices) < memory_limit:
            remaining_indices = list(set(range(len(self._data_memory))) - set(selected_indices))
            if remaining_indices:
                additional_needed = min(len(remaining_indices), memory_limit - len(selected_indices))
                additional_indices = np.random.choice(remaining_indices, additional_needed, replace=False)
                selected_indices.extend(additional_indices)
        
        # 确保不超过memory_limit
        if len(selected_indices) > memory_limit:
            selected_indices = np.random.choice(selected_indices, memory_limit, replace=False)
        
        # 返回采样后的数据
        sampled_data = self._data_memory[selected_indices]
        sampled_targets = self._targets_memory[selected_indices]
        sampled_stages = self._stages_memory[selected_indices]
        
        logging.info(f"最终回放样本数: {len(sampled_data)} (限制: {memory_limit})")
        return (sampled_data, sampled_targets, sampled_stages)

    def build_rehearsal_memory(self, data_manager, per_class):
        """构建回放记忆 - 严格按照base.py的指示和配置参数"""
        # 调用父类方法更新原型
        super().build_rehearsal_memory(data_manager, per_class)
        
        # 获取当前任务的数据用于存储
        try:
            current_data, current_targets, current_stages, _ = data_manager.get_dataset(
                cil_task_idx=self._cur_task,
                source="train",
                mode="test",  # 不做数据增强
                ret_data=True
            )
            
            if len(current_data) == 0:
                logging.warning(f"当前任务 {self._cur_task} 没有可用数据")
                return
            
            # 严格按照base.py中samples_per_class的计算规则
            # 当fixed_memory=false时：samples_per_class = memory_size // total_classes
            # 当fixed_memory=true时：samples_per_class = memory_per_class
            if self._fixed_memory:
                samples_per_class = self._memory_per_class
                logging.info(f"固定内存模式: 每类存储 {samples_per_class} 个样本")
            else:
                # IIMinsects202.json: "fixed_memory": false, "memory_size": 400
                samples_per_class = max(1, self._memory_size // self._total_classes)
                logging.info(f"动态内存模式: memory_size={self._memory_size}, total_classes={self._total_classes}")
                logging.info(f"计算得出每类存储 {samples_per_class} 个样本")
            
            # 直接从当前任务数据中获取实际存在的类别
            actual_classes = np.unique(current_targets)
            
            logging.info(f"任务 {self._cur_task}: 准备存储记忆样本，每类 {samples_per_class} 个")
            logging.info(f"当前任务实际类别: {sorted(actual_classes.tolist())}")
            
            # 为每个类别选择代表性样本
            selected_data = []
            selected_targets = []
            selected_stages = []
            
            for class_idx in actual_classes:
                # 获取当前类别的所有样本
                class_mask = (current_targets == class_idx)
                if not np.any(class_mask):
                    logging.warning(f"类别 {class_idx} 在当前任务中没有样本")
                    continue
                    
                class_data = current_data[class_mask]
                class_targets = current_targets[class_mask]
                class_stages = current_stages[class_mask]
                
                # 随机选择样本（确保不超过该类别的样本总数）
                num_samples = min(len(class_data), samples_per_class)
                if num_samples > 0:
                    selected_indices = np.random.choice(len(class_data), num_samples, replace=False)
                    
                    selected_data.append(class_data[selected_indices])
                    selected_targets.append(class_targets[selected_indices])
                    selected_stages.append(class_stages[selected_indices])
                    
                    logging.info(f"  类别 {class_idx}: 从 {len(class_data)} 个样本中选择 {num_samples} 个")
            
            if selected_data:
                # 合并当前任务的选定样本
                new_memory_data = np.concatenate(selected_data)
                new_memory_targets = np.concatenate(selected_targets)
                new_memory_stages = np.concatenate(selected_stages)
                
                # 如果是第一个任务，直接存储
                if not hasattr(self, '_data_memory') or self._data_memory is None:
                    self._data_memory = new_memory_data
                    self._targets_memory = new_memory_targets
                    self._stages_memory = new_memory_stages
                    logging.info(f"初始化记忆存储: {len(new_memory_data)} 个样本")
                else:
                    # 合并新旧记忆数据
                    self._data_memory = np.concatenate([self._data_memory, new_memory_data])
                    self._targets_memory = np.concatenate([self._targets_memory, new_memory_targets])
                    self._stages_memory = np.concatenate([self._stages_memory, new_memory_stages])
                    logging.info(f"累积记忆存储: 新增 {len(new_memory_data)} 个样本")
                
                # 如果不是固定内存模式，且超过了memory_size，则需要重新平衡
                if not self._fixed_memory and len(self._data_memory) > self._memory_size:
                    logging.info(f"记忆样本数 ({len(self._data_memory)}) 超过配置限制 ({self._memory_size})")
                    self._balance_memory_size()
                
                logging.info(f"记忆存储完成: 当前记忆样本总数 = {len(self._data_memory)}")
                
        except Exception as e:
            logging.error(f"构建回放记忆失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _balance_memory_size(self):
        """平衡记忆大小，严格按照配置参数确保不超过限制"""
        if not hasattr(self, '_data_memory') or len(self._data_memory) <= self._memory_size:
            return
            
        logging.info(f"开始平衡记忆大小: 当前{len(self._data_memory)}个样本 -> 目标{self._memory_size}个样本")
            
        # 获取所有独特的类别
        unique_classes = np.unique(self._targets_memory)
        
        # 严格按照base.py的samples_per_class逻辑
        if self._fixed_memory:
            samples_per_class = self._memory_per_class
        else:
            # IIMinsects202.json: "fixed_memory": false, "memory_size": 400
            samples_per_class = max(1, self._memory_size // len(unique_classes))
        
        logging.info(f"平衡策略: {len(unique_classes)}个类别，每类最多{samples_per_class}个样本")
        
        selected_indices = []
        
        for class_idx in unique_classes:
            class_mask = (self._targets_memory == class_idx)
            class_indices = np.where(class_mask)[0]
            
            # 为每个类别选择样本
            num_to_select = min(len(class_indices), samples_per_class)
            if num_to_select > 0:
                selected_class_indices = np.random.choice(class_indices, num_to_select, replace=False)
                selected_indices.extend(selected_class_indices)
        
        # 如果按类别平衡后还有剩余空间，随机填充
        if len(selected_indices) < self._memory_size:
            remaining_indices = list(set(range(len(self._data_memory))) - set(selected_indices))
            if remaining_indices:
                additional_needed = min(len(remaining_indices), self._memory_size - len(selected_indices))
                additional_indices = np.random.choice(remaining_indices, additional_needed, replace=False)
                selected_indices.extend(additional_indices)
        
        # 确保不超过memory_size限制
        if len(selected_indices) > self._memory_size:
            selected_indices = np.random.choice(selected_indices, self._memory_size, replace=False)
        
        # 更新记忆数据
        self._data_memory = self._data_memory[selected_indices]
        self._targets_memory = self._targets_memory[selected_indices]
        self._stages_memory = self._stages_memory[selected_indices]
        
        # 验证结果
        unique_classes_after = len(np.unique(self._targets_memory))
        logging.info(f"记忆大小平衡完成: {len(self._data_memory)}个样本，覆盖{unique_classes_after}个类别")

# 为了保持兼容性，创建一个别名
Learner = MorphologyEvolutionLearner