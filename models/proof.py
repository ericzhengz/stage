import copy
import logging
import math
import torch
import numpy as np
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import Proof_Net
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, get_attribute, ClipLoss
# from utils.data_manager import LaionData # This might not be directly needed in proof.py itself unless used globally
import os

num_workers = 4

class StageSpecificMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.embed_dim = embed_dim
        if embed_dim == 0: # Prevent division by zero if embed_dim is not set
            logging.error("embed_dim is 0 in StageSpecificMultiHeadAttention. This will cause errors.")
            # Or raise ValueError("embed_dim cannot be 0")
            return # Or handle appropriately
        if num_heads == 0: # Prevent division by zero
            logging.error("num_heads is 0 in StageSpecificMultiHeadAttention. Defaulting to 1.")
            num_heads = 1 # Default to 1 head if 0 is passed, or handle as error

        if embed_dim % num_heads != 0:
            logging.warning(f"embed_dim ({embed_dim}) is not divisible by num_heads ({num_heads}). This can cause errors or suboptimal performance in MultiheadAttention.")
            # Optionally, adjust num_heads to a valid divisor or raise an error
            # For now, we proceed with the user's values and let PyTorch handle it or error out.

    def forward(self, query_feature, class_prototypes_dict, current_stage_id):
        """
        Args:
            query_feature (Tensor): The query feature for a single instance. Shape: [embed_dim].
            class_prototypes_dict (dict): Prototypes for the class of the query instance. 
                                          {stage_id: proto_tensor [embed_dim], ...}.
            current_stage_id (int): The stage ID of the query_feature.

        Returns:
            Tensor: Attended feature. Shape: [embed_dim].
        """
        device = query_feature.device
        
        if not class_prototypes_dict:
            return query_feature

        key_value_list = []
        # Ensure prototypes are on the correct device and correctly shaped
        for stage_id, proto in class_prototypes_dict.items():
            if proto is None: # Skip if a prototype is None
                logging.warning(f"Prototype for stage_id {stage_id} is None. Skipping.")
                continue
            # Prototypes should already be [embed_dim]. Unsqueeze to [1, embed_dim] for cat
            key_value_list.append(proto.to(device).unsqueeze(0)) 
            
        if not key_value_list: # If all prototypes were None or dict was empty after filtering
             return query_feature

        # Stack to form [num_prototypes, embed_dim]
        keys_values = torch.cat(key_value_list, dim=0) 
        
        # Reshape for MHA:
        # Query: [N, L, E] -> [1, 1, embed_dim] (L=target_seq_len=1)
        query = query_feature.unsqueeze(0).unsqueeze(0) 
        # Key/Value: [N, S, E] -> [1, num_prototypes, embed_dim] (S=source_seq_len=num_prototypes)
        kv_input = keys_values.unsqueeze(0)

        try:
            # output shape: [N, L, E] -> [1, 1, embed_dim]
            attended_output, _ = self.mha(query, kv_input, kv_input) 
            return attended_output.squeeze(0).squeeze(0) # Back to [embed_dim]
        except Exception as e:
            logging.error(f"Error in StageSpecificMultiHeadAttention: {e}. Query shape: {query.shape}, KV_input shape: {kv_input.shape}, embed_dim: {self.embed_dim}, num_heads: {self.mha.num_heads}. Falling back to query_feature.")
            return query_feature 

def stage_evolution_contrastive_loss(
    features, 
    labels, 
    state_ids, 
    all_stage_prototypes, 
    target_similarity_range, 
    device,
    stage_map = {0: 1, 1: None} 
):
    loss = torch.tensor(0.0, device=device)
    count = 0
    
    if not isinstance(target_similarity_range, (list, tuple)) or len(target_similarity_range) != 2:
        logging.error(f"target_similarity_range must be a list or tuple of two numbers. Got {target_similarity_range}")
        return loss # Return zero loss if config is wrong

    lower_bound, upper_bound = target_similarity_range

    for i in range(features.shape[0]):
        current_feat = features[i] 
        class_id = labels[i].item()
        current_stage_id = state_ids[i].item()

        if class_id not in all_stage_prototypes or not all_stage_prototypes[class_id]:
            continue # Skip if no prototypes for this class
        
        current_class_prototypes = all_stage_prototypes[class_id]
        related_stage_id = stage_map.get(current_stage_id)
        
        # Handle linear evolution: larva(0) -> pupa(1) -> terminal(None)
        if related_stage_id is None:
            continue
        elif related_stage_id in current_class_prototypes:
            related_proto = current_class_prototypes[related_stage_id]
            if related_proto is None: # Check if the specific related prototype is None
                logging.warning(f"Related prototype for class {class_id}, stage {related_stage_id} is None. Skipping.")
                continue
            related_proto = related_proto.to(device)
            

            
            similarity = F.cosine_similarity(current_feat.unsqueeze(0), related_proto.unsqueeze(0)).squeeze()
            
            # Ensure similarity is a scalar tensor before comparison
            if not torch.is_tensor(similarity) or similarity.ndim != 0:
                logging.error(f"Similarity is not a scalar tensor: {similarity}. Skipping this sample.")
                continue

            if similarity < lower_bound:
                loss += (lower_bound - similarity).pow(2) 
            elif similarity > upper_bound:
                loss += (similarity - upper_bound).pow(2)
            
            count += 1
            
    if count > 0:
        return loss / count
    return loss

# Ensure math, logging, and F (torch.nn.functional) are imported at the top of the file.

# II. 修改 unicl_loss 函数
def unicl_loss(image_features, text_features, state_features, labels, state_ids,
               temperature=0.07, epoch=None, max_epoch=None,
               stage_prototypes=None,  # Dict: {class_id: {stage_id: proto_tensor}}
               attention_module=None,  # Instance of StageSpecificMultiHeadAttention
               stage_evolution_loss_config=None):  # Dict with 'weight', 'target_range', etc.
    """三路对比学习损失函数，实现方案A+C混合架构中的特定逻辑"""
    device = image_features.device

    # 初始化损失值
    instance_loss_val = torch.tensor(0.0, device=device)
    category_loss_val = torch.tensor(0.0, device=device)
    stage_evolution_loss_val = torch.tensor(0.0, device=device)

    # 动态温度调整 (可选)
    if epoch is not None and max_epoch is not None and max_epoch > 0:
        progress = float(epoch) / float(max_epoch)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        dynamic_temperature = temperature * (0.5 + 0.5 * cosine_decay)
    else:
        dynamic_temperature = temperature

    batch_size = image_features.shape[0]
    if batch_size == 0:
        logging.warning("unicl_loss called with batch_size = 0.")
        return {
            'total_loss': torch.tensor(0.0, device=device),
            'instance_loss': 0.0,
            'category_loss': 0.0,
            'stage_evolution_loss': 0.0,
            'temperature': dynamic_temperature
        }

    # 确保所有张量在正确的设备上
    image_features = image_features.to(device)
    text_features = text_features.to(device)
    state_features = state_features.to(device)
    labels = labels.to(device)
    state_ids = state_ids.to(device) # 修正：添加括号

    # 确保特征是2D的 [batch_size, feature_dim]
    if len(text_features.shape) > 2: text_features = text_features.view(text_features.shape[0], -1)
    if len(state_features.shape) > 2: state_features = state_features.view(state_features.shape[0], -1)
    if len(image_features.shape) > 2: image_features = image_features.view(image_features.shape[0], -1)

    # 特征归一化
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    state_features_original_normalized = F.normalize(state_features, dim=1) # 先对原始state_features归一化

    current_state_features = state_features_original_normalized.clone()

    # 1. 使用StageSpecificMultiHeadAttention增强阶段特征
    if attention_module is not None and stage_prototypes is not None:
        state_features_enhanced_by_attention = current_state_features.clone()
        for i in range(batch_size):
            class_id = labels[i].item()
            current_stage_id = state_ids[i].item()
            if class_id in stage_prototypes and stage_prototypes[class_id]:
                class_protos_dict = stage_prototypes[class_id]
                query_feat_single = current_state_features[i] # 使用当前（可能已归一化）的state feature作为查询

                enhanced_feat = attention_module(
                    query_feat_single,
                    class_protos_dict,
                    current_stage_id
                )
                state_features_enhanced_by_attention[i] = F.normalize(enhanced_feat, dim=0) # 注意力输出后再次归一化
        current_state_features = state_features_enhanced_by_attention # 更新当前使用的state_features

    # 实例级对比损失 (样本内，跨模态)
    if batch_size >= 1:
        # 使用更新后的 current_state_features
        tri_feats = torch.stack([image_features, text_features, current_state_features], dim=1)  # [B, 3, D]
        current_instance_loss_sum = torch.tensor(0.0, device=device)
        for i in range(batch_size):
            feats_i = tri_feats[i]  # [3, D]
            sim_matrix = torch.matmul(feats_i, feats_i.t()) / dynamic_temperature  # [3, 3]
            for row_idx in range(3):
                row_sim_vals = sim_matrix[row_idx]
                positive_mask = torch.ones_like(row_sim_vals, device=device)
                positive_mask[row_idx] = 0  # 排除自身
                
                exp_row_sim_vals = torch.exp(row_sim_vals)
                sum_pos_sim = torch.sum(exp_row_sim_vals * positive_mask)
                sum_all_sim = torch.sum(exp_row_sim_vals)
                if sum_pos_sim > 0: # 确保有正样本对
                    current_instance_loss_sum -= torch.log(sum_pos_sim / (sum_all_sim + 1e-8))
        instance_loss_val = current_instance_loss_sum / (3 * batch_size) # 平均到每个模态和每个样本

    # 类别级对比损失 (样本间，模态内)
    if batch_size >= 2:
        labels_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        self_mask = 1 - torch.eye(batch_size, device=labels_matrix.device)
        labels_matrix = labels_matrix * self_mask

        # 以图像特征为例
        img_img_sim = torch.matmul(image_features, image_features.t()) / dynamic_temperature
        current_category_loss_sum = torch.tensor(0.0, device=device)
        valid_samples_for_category_loss = 0
        for i in range(batch_size):
            row_sim_vals = img_img_sim[i]
            max_val_stabilizer = torch.max(row_sim_vals) # 数值稳定性
            exp_sim_vals = torch.exp(row_sim_vals - max_val_stabilizer)
            
            sum_pos_class_sim = torch.sum(exp_sim_vals * labels_matrix[i])
            sum_all_other_sim = torch.sum(exp_sim_vals * self_mask[i]) 
            
            if sum_pos_class_sim > 0 and sum_all_other_sim > 0:
                current_category_loss_sum -= torch.log(sum_pos_class_sim / (sum_all_other_sim + 1e-8))
                valid_samples_for_category_loss += 1
        if valid_samples_for_category_loss > 0:
            category_loss_val = current_category_loss_sum / valid_samples_for_category_loss
    elif batch_size == 1:
        logging.debug("Batch size is 1, skipping category-level contrastive loss.")

    # 4. 阶段演化对比损失
    if stage_evolution_loss_config is not None and stage_prototypes is not None and batch_size >= 1:
        # 用实例的图像特征与目标演化阶段的原型进行比较
        stage_evolution_loss_val = stage_evolution_contrastive_loss(
            features=image_features, # 使用原始的图像特征
            labels=labels,
            state_ids=state_ids,
            all_stage_prototypes=stage_prototypes,
            target_similarity_range=stage_evolution_loss_config['target_range'],
            device=device
        )

    # 组合损失
    default_instance_weight = 1.0
    default_category_weight = 0.5
    default_stage_evolution_weight = 0.1 

    if stage_evolution_loss_config is not None:
        instance_weight = stage_evolution_loss_config.get("instance_loss_weight", default_instance_weight)
        category_weight = stage_evolution_loss_config.get("category_loss_weight", default_category_weight)
        stage_evolution_weight = stage_evolution_loss_config.get("weight", default_stage_evolution_weight)
    else:
        instance_weight = default_instance_weight
        category_weight = default_category_weight
        stage_evolution_weight = 0 # 如果没有配置，则此项损失权重为0

    current_total_loss = torch.tensor(0.0, device=device)
    if not torch.isnan(instance_loss_val) and instance_weight > 0:
        current_total_loss += instance_loss_val * instance_weight
    if not torch.isnan(category_loss_val) and category_weight > 0:
        current_total_loss += category_loss_val * category_weight
    # 只有在配置存在且权重为正时才添加演化损失
    if stage_evolution_loss_config is not None and not torch.isnan(stage_evolution_loss_val) and stage_evolution_weight > 0:
        current_total_loss += stage_evolution_loss_val * stage_evolution_weight
    
    if torch.isnan(current_total_loss):
        logging.error(f"Total loss is NaN. Components: Inst: {instance_loss_val.item():.4f}, Cat: {category_loss_val.item():.4f}, Evo: {stage_evolution_loss_val.item():.4f}")
        current_total_loss = torch.tensor(0.0, device=device) 
        logging.warning("Total loss was NaN, reset to 0. Check loss components and weights.")

    return {
        'total_loss': current_total_loss,
        'instance_loss': instance_loss_val.item() if torch.is_tensor(instance_loss_val) else instance_loss_val,
        'category_loss': category_loss_val.item() if torch.is_tensor(category_loss_val) else category_loss_val,
        'stage_evolution_loss': stage_evolution_loss_val.item() if torch.is_tensor(stage_evolution_loss_val) else stage_evolution_loss_val,
        'temperature': dynamic_temperature
    }

class Learner(BaseLearner):
    """
    对三路投影进行增量学习的核心类，主要修改点是拆分出 forward_for_classification，
    训练时区分对比逻辑与分类逻辑，评估时只用 forward_for_classification(或 _compute_accuracy)。
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = Proof_Net(args, False)
        # Call extend_task after Proof_Net is initialized
        self._network.extend_task() 
        # Initialize context prompts after Proof_Net is initialized
        self._network.update_context_prompt()

        # Initialize StageSpecificMultiHeadAttention if configured
        self.attention_num_heads = get_attribute(args, "attention_num_heads", 0)
        self.batch_size = get_attribute(args, "batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.frozen_layers = get_attribute(args, "frozen_layers", None)
        self.tuned_epoch = get_attribute(args, "tuned_epoch", 5)

        self._known_classes = 0
        self.use_cos = get_attribute(args, "use_cos", False)

        # 使用从_network获取的实际特征维度
        actual_feature_dim = self._network.feature_dim 
        
        self.stage_specific_attention_module = None
        if self.attention_num_heads > 0:
            feature_dim = self._network.feature_dim
            if feature_dim == 0:
                logging.error("Feature dimension is 0 when initializing StageSpecificMultiHeadAttention in Learner.")
            # Add check for num_heads > 0 before creating the module
            elif self.attention_num_heads <= 0:
                logging.error(f"attention_num_heads must be positive, got {self.attention_num_heads}. StageSpecificMultiHeadAttention will not be created.")
            else:
                self.stage_specific_attention_module = StageSpecificMultiHeadAttention(
                    embed_dim=feature_dim,
                    num_heads=self.attention_num_heads,
                    dropout=get_attribute(args, "attention_dropout", 0.1)
                )
                # Move the attention module to the correct device
                if self.stage_specific_attention_module is not None:
                    self.stage_specific_attention_module.to(self._device)

        # Stage evolution contrastive loss configuration
        self.stage_evolution_loss_config = {
            'weight': get_attribute(args, "stage_evolution_loss_weight", 0.1),
            'target_range': (
                get_attribute(args, "stage_evolution_target_min_sim", 0.3),
                get_attribute(args, "stage_evolution_target_max_sim", 0.7)
            ),
            "instance_loss_weight": get_attribute(args, "instance_loss_weight", 1.0),
            "category_loss_weight": get_attribute(args, "category_loss_weight", 0.5),
            # temperature for unicl_loss can be passed directly if needed,
            # or unicl_loss uses its own default/dynamic calculation.
        }
        self.unicl_temperature = get_attribute(args, "unicl_temperature", 0.07)

        # Prototype rehearsal configuration
        self.lambda_rehearsal = get_attribute(args, "lambda_rehearsal", 0.1)  # 原型回放损失权重
        self.alpha_noise = get_attribute(args, "alpha_noise", 0.1)  # 原型增强噪声标准差


    def after_task(self):
        self._known_classes = self._total_classes
        # 现在使用原型而不是样本，记录原型数量
        prototype_count = len(self._class_prototypes) if hasattr(self, '_class_prototypes') else 0
        logging.info("Prototype count: {}".format(prototype_count))

    def cal_prototype(self, trainloader, model):
        """
        计算每个类别（以及虫态）的原型，用于后续原型匹配。
        仅针对图像特征做平均，text/state 原型根据需求自行扩展。
        """
        model = model.eval()
        model = model.to(self._device)
        embedding_list, label_list, state_list = [], [], []

        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                if isinstance(batch[1], dict) and 'stage_id' in batch[1]:
                    _, data_dict, label = batch
                    data = data_dict['image'].to(self._device)
                    states = data_dict['stage_id'].to(self._device)
                else:
                    _, data, label = batch
                    data = data.to(self._device)
                    # 根据当前任务推断虫态：数据集只有0和1两种状态
                    try:
                        current_stage = self.data_manager.get_stage_for_cil_task(self._cur_task)
                        # 直接使用阶段作为虫态：阶段0->状态0, 阶段1->状态1
                        inferred_state = current_stage
                        states = torch.full((data.size(0),), inferred_state, dtype=torch.long).to(self._device)
                        logging.debug(f"Task {self._cur_task}, Stage {current_stage}: inferred state {inferred_state}")
                    except Exception as e:
                        logging.warning(f"Failed to infer state for task {self._cur_task}: {e}, using default state 0")
                        states = torch.full((data.size(0),), 0, dtype=torch.long).to(self._device)

                label = label.to(self._device)
                embedding = model.convnet.encode_image(data, normalize=True)

                embedding_list.append(embedding)
                label_list.append(label)
                state_list.append(states)

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        state_list = torch.cat(state_list, dim=0)

        class_list = list(range(self._known_classes, self._total_classes))
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            if len(data_index) > 0:
                embedding = embedding_list[data_index]
                proto = embedding.mean(0)
                self._network.img_prototypes[class_index] = proto.to(self._device)

                # 计算虫态级别原型
                states = state_list[data_index]
                if class_index not in self._network.img_prototypes_by_state:
                    self._network.img_prototypes_by_state[class_index] = {}

                unique_states = torch.unique(states)
                for state_id in unique_states:
                    st_mask = (states == state_id)
                    if st_mask.sum() > 0:
                        state_proto = embedding[st_mask].mean(0)
                        self._network.img_prototypes_by_state[class_index][state_id.item()] = state_proto.to(self._device)
                        
                        # Stage CIL: 同步存储到学习器的双层原型结构
                        if not hasattr(self, '_class_prototypes'):
                            self._class_prototypes = {}
                        if class_index not in self._class_prototypes:
                            self._class_prototypes[class_index] = {}
                        self._class_prototypes[class_index][state_id.item()] = state_proto.cpu()
                        
                        logging.info(f"Stage CIL: Synchronized prototype - Class {class_index}, Stage {state_id.item()}, Shape: {state_proto.shape}")

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

     
        if self._cur_task % 2 == 0: 
            pass 

        self._network.update_prototype(self._total_classes)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # 多模态数据集
        # Pass self._cur_task (scalar CIL task index) instead of an array of class labels
        train_dataset = data_manager.get_multimodal_dataset(
            self._cur_task, # Was: np.arange(self._known_classes, self._total_classes)
            source="train", mode="train", appendent=self._get_memory())
        self.train_dataset = train_dataset
        self.data_manager = data_manager

        self._old_network = copy.deepcopy(self._network).to(self._device)
        self._old_network.eval()

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        
        # Stage CIL: 创建分离的测试数据加载器用于分别监控性能
        self.current_test_loader, self.test_loader = self._create_test_loaders(data_manager)

        # 原型网络训练集
        # Pass self._cur_task for consistency; get_classes_for_cil_task(self._cur_task) will give the new classes.
        train_dataset_for_protonet = data_manager.get_multimodal_dataset(
            self._cur_task, # Was: np.arange(self._known_classes, self._total_classes)
            source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.to(self._device)

        # 计算原型
        self.cal_prototype(self.train_loader_for_protonet, self._network)

        # 训练
        self._train_proj_with_replay(self.train_loader, self.test_loader, self.train_loader_for_protonet)

        # 构建回放记忆
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train_proj_with_replay(self, train_loader, test_loader, train_loader_for_protonet):
        self._train_transformer = True
        self._network.to(self._device)

        # 冻结主干，只训练投影层
        for name, param in self._network.convnet.named_parameters():
            if 'logit_scale' not in name:
                param.requires_grad = False
        self._network.freeze_projection_weight_new()

        if self.args['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = torch.optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epoch, eta_min=self.min_lr)
        cliploss = ClipLoss()

        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt[0]  # 取第一个模板
        total_labels = class_to_label[:self._total_classes]

        prog_bar = tqdm(range(self.tuned_epoch))
        for epoch in range(self.tuned_epoch):
            self._network.train()
            losses, unicl_losses = 0.0, 0.0
            correct = torch.tensor(0, device=self._device)
            total = torch.tensor(0, device=self._device)

            for i, batch in enumerate(train_loader):
                if isinstance(batch[1], dict) and 'stage_id' in batch[1]:
                    _, data_dict, targets = batch
                    inputs = data_dict['image'].to(self._device)
                    state_ids = data_dict['stage_id'].to(self._device)
                else:
                    _, inputs, targets = batch
                    # 根据当前任务推断虫态：数据集只有0和1两种状态
                    try:
                        current_stage = self.data_manager.get_stage_for_cil_task(self._cur_task)
                        # 直接使用阶段作为虫态：阶段0->状态0, 阶段1->状态1
                        inferred_state = current_stage
                        state_ids = torch.full((inputs.size(0),), inferred_state, dtype=torch.long).to(self._device)
                        logging.debug(f"Training Task {self._cur_task}, Stage {current_stage}: inferred state {inferred_state}")
                    except Exception as e:
                        logging.warning(f"Failed to infer state for training task {self._cur_task}: {e}, using default state 0")
                        state_ids = torch.full((inputs.size(0),), 0, dtype=torch.long).to(self._device)
                    inputs = inputs.to(self._device)
                targets = targets.to(self._device).long()

                # 1) 分类分支：forward_for_classification
                with torch.no_grad():
                    # 构造全类别文本（所有类别）
                    text_batch = [templates.format(lbl) for lbl in total_labels]
                    cls_logits = self.forward_for_classification(inputs, text_batch)
                ce_loss = torch.nn.functional.cross_entropy(cls_logits, targets)

                # 2) 原型回放损失计算
                rehearsal_loss = self._compute_prototype_rehearsal_loss(total_labels, templates)

                # 3) 三路对比分支：先为本批生成对应文本
                labels_string = [class_to_label[int(t)] for t in targets]
                batch_texts = [templates.format(lbl) for lbl in labels_string]  # 生成与每个样本对应的文本描述
                image_feats, text_feats, state_feats, proto_feats, logit_scale = \
                    self._network.forward_tri_modal(inputs, batch_texts, state_ids)

                # 额外计算 clip 对比损失，使用同一 batch 文本输入
                batch_text_features = self._network.encode_text(self._network.tokenizer(batch_texts).to(self._device), normalize=True)
                batch_text_features = torch.nn.functional.normalize(batch_text_features, dim=1)
                img_norm = torch.nn.functional.normalize(self._network.encode_image(inputs), dim=1)
                clip_loss_val = cliploss(img_norm, batch_text_features, logit_scale)

                
                unicl_output_dict = unicl_loss( # 修改：接收字典输出
                    image_feats, text_feats, state_feats, 
                    targets, state_ids,
                    temperature=self.unicl_temperature, # 使用配置的温度
                    epoch=epoch, max_epoch=self.tuned_epoch,
                    stage_prototypes=self._network.img_prototypes_by_state,
                    attention_module=self.stage_specific_attention_module, # 新增：传递注意力模块
                    stage_evolution_loss_config=self.stage_evolution_loss_config # 新增：传递演化损失配置
                )
                unicl_val = unicl_output_dict['total_loss'] # 从字典中获取总损失
                

                # 使用配置的损失权重，减少过拟合风险
                unicl_weight = get_attribute(self.args, "unicl_loss_weight", 0.2)  # 降低UniCL权重
                total_loss = ce_loss + clip_loss_val + unicl_weight * unicl_val + self.lambda_rehearsal * rehearsal_loss
                
                # Stage CIL: 增强的损失监控和调试信息
                if i % 20 == 0:  # 每20个批次记录一次详细信息
                    logging.info(f"Stage CIL Loss Breakdown - Batch {i}:")
                    logging.info(f"  CE Loss: {ce_loss.item():.4f}")
                    logging.info(f"  CLIP Loss: {clip_loss_val.item():.4f}")  
                    logging.info(f"  UniCL Loss: {unicl_val.item():.4f} (weight: {unicl_weight})")
                    logging.info(f"  Rehearsal Loss: {rehearsal_loss.item():.4f} (weight: {self.lambda_rehearsal})")
                    logging.info(f"  Total Loss: {total_loss.item():.4f}")
                    
                    # 检查原型存储状态
                    prototype_count = sum(len(stages) for stages in self._class_prototypes.values()) if hasattr(self, '_class_prototypes') else 0
                    network_prototype_count = sum(len(stages) for stages in self._network.img_prototypes_by_state.values()) if hasattr(self._network, 'img_prototypes_by_state') else 0
                    logging.info(f"  Prototypes: Learner={prototype_count}, Network={network_prototype_count}")
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                losses += total_loss.item()
                unicl_losses += unicl_val.item()
                _, preds = torch.max(cls_logits, dim=1)
                correct += (preds == targets).sum()
                total += targets.size(0)
            scheduler.step()
            train_acc = np.around(correct.cpu().numpy() * 100 / total.cpu().numpy(), 2)
            
            # Stage CIL: 分别监控当前任务和累积性能
            try:
                current_test_loader, cumulative_test_loader = self._create_test_loaders(self.data_manager)
                
                # 当前任务测试性能（使用原型增强）
                current_test_acc = self._compute_accuracy_with_prototype_enhancement(
                    self._network, current_test_loader, use_prototype_boost=True)
                
                # 累积任务测试性能（使用原型增强）
                cumulative_test_acc = self._compute_accuracy_with_prototype_enhancement(
                    self._network, cumulative_test_loader, use_prototype_boost=True)
                
                # 为了向后兼容，保留原有的test_acc变量
                test_acc = cumulative_test_acc
                
                info = f"Task {self._cur_task}, Epoch {epoch+1}/{self.tuned_epoch} => " \
                       f"Loss {losses/len(train_loader):.3f}, UniCL {unicl_losses/len(train_loader):.3f}, " \
                       f"Train_acc {train_acc:.2f}, Current_test {current_test_acc:.2f}, Cumulative_test {cumulative_test_acc:.2f}"
                       
                # 详细的性能监控日志（每5个epoch记录一次）
                if (epoch + 1) % 5 == 0:
                    logging.info(f"Stage CIL Performance Monitor - Task {self._cur_task}, Epoch {epoch+1}:")
                    logging.info(f"  Training Accuracy: {train_acc:.2f}%")
                    logging.info(f"  Current Task Test Accuracy: {current_test_acc:.2f}%")
                    logging.info(f"  Cumulative Test Accuracy: {cumulative_test_acc:.2f}%")
                    logging.info(f"  Performance Gap (Train-Current): {train_acc - current_test_acc:.2f}%")
                    logging.info(f"  Forgetting (Current-Cumulative): {current_test_acc - cumulative_test_acc:.2f}%")
                    
            except Exception as e:
                # 如果分离测试失败，回退到原有方法
                logging.warning(f"Failed to create separated test loaders: {e}")
                test_acc = self._compute_accuracy_with_prototype_enhancement(self._network, test_loader, use_prototype_boost=True)
                info = f"Task {self._cur_task}, Epoch {epoch+1}/{self.tuned_epoch} => " \
                       f"Loss {losses/len(train_loader):.3f}, UniCL {unicl_losses/len(train_loader):.3f}, " \
                       f"Train_acc {train_acc:.2f}, Test_acc {test_acc:.2f}"
            
            prog_bar.set_description(info)
            
    def _compute_prototype_rehearsal_loss(self, total_labels, templates):
        """
        计算原型回放损失 - Stage CIL双层结构支持，增强时序处理。
        
        Args:
            total_labels: 所有类别的标签列表
            templates: 文本模板
            
        Returns:
            torch.Tensor: 原型回放损失
        """
        if not hasattr(self, '_class_prototypes') or len(self._class_prototypes) == 0:
            # 没有存储的原型，返回零损失
            return torch.tensor(0.0, device=self._device)
        
        rehearsal_loss = torch.tensor(0.0, device=self._device)
        prototype_count = 0
        
        # Stage CIL: 时序感知的原型回放 - 优先回放早期阶段的原型
        for class_idx, stages_dict in self._class_prototypes.items():
            if stages_dict is None or len(stages_dict) == 0:
                continue
                
            # 按阶段顺序排序，优先处理早期阶段（防止遗忘）
            sorted_stages = sorted(stages_dict.items())
            
            # 针对每个类别的每个阶段计算原型回放损失
            for stage_idx, prototype in sorted_stages:
                if prototype is None:
                    continue
                    
                # 时序权重：早期阶段给予更高权重以防止遗忘
                temporal_weight = 1.0 + (len(sorted_stages) - stage_idx) * 0.1
                
                # 将原型移到GPU并添加高斯噪声进行增强
                prototype_gpu = prototype.to(self._device)
                noise = torch.randn_like(prototype_gpu) * self.alpha_noise
                augmented_prototype = prototype_gpu + noise
                
                # 通过网络的分类头获得logits
                try:
                    # 优先尝试使用文本特征进行分类（CLIP风格）
                    with torch.no_grad():
                        # 构造所有类别的文本特征
                        text_batch = [templates.format(lbl) for lbl in total_labels[:self._total_classes]]
                        text_tokens = self._network.tokenizer(text_batch).to(self._device)
                        text_features = self._network.encode_text(text_tokens, normalize=True)
                        
                        # 计算原型与文本特征的相似度作为logits
                        augmented_prototype_norm = torch.nn.functional.normalize(augmented_prototype, dim=0)
                        prototype_logits = torch.matmul(text_features, augmented_prototype_norm.unsqueeze(-1)).squeeze(-1)
                        prototype_logits = prototype_logits.unsqueeze(0)  # 添加batch维度
                        
                except Exception as e:
                    logging.warning(f"Failed to compute prototype logits via text features: {e}")
                    # 备用方案：如果有其他分类器层
                    if hasattr(self._network, 'fc') and self._network.fc is not None:
                        prototype_logits = self._network.fc(augmented_prototype.unsqueeze(0))
                    elif hasattr(self._network, 'classifier') and self._network.classifier is not None:
                        prototype_logits = self._network.classifier(augmented_prototype.unsqueeze(0))
                    else:
                        logging.warning(f"No suitable classifier found for prototype rehearsal, skipping class {class_idx}, stage {stage_idx}")
                        continue
                
                # 计算原型回放损失
                target = torch.tensor([class_idx], device=self._device, dtype=torch.long)
                
                # 确保target索引不超出logits的范围
                if class_idx < prototype_logits.size(1):
                    loss = torch.nn.functional.cross_entropy(prototype_logits, target)
                    # 应用时序权重
                    weighted_loss = loss * temporal_weight
                    rehearsal_loss += weighted_loss
                    prototype_count += 1
                    logging.debug(f"Stage CIL: Added rehearsal loss for class {class_idx}, stage {stage_idx}, weight {temporal_weight:.2f}")
                else:
                    logging.warning(f"Class index {class_idx} exceeds logits dimension {prototype_logits.size(1)}")
        
        # 平均化损失
        if prototype_count > 0:
            rehearsal_loss = rehearsal_loss / prototype_count
            logging.info(f"Stage CIL: Computed temporal-aware prototype rehearsal loss from {prototype_count} prototypes")
        else:
            logging.info("Stage CIL: No valid prototypes found for rehearsal loss computation")
        
        return rehearsal_loss

    def forward_for_classification(self, images, text_list):
        """
        单独的分类前向：针对所有类别生成文本特征 (num_classes,D)，
        与图像特征 (batch_size,D) 做内积 -> [batch_size, num_classes]。
        text_list: 包含所有类别的描述字符串，长度等于 self._total_classes
        """
        # encode_image -> [B, D]
        image_features = self._network.encode_image(images)
        image_features = F.normalize(image_features, dim=1)

        with torch.no_grad():
            texts_tokenized = self._network.tokenizer(text_list).to(self._device)
            text_features = self._network.encode_text(texts_tokenized)  # [num_classes, D]
            text_features = F.normalize(text_features, dim=1)

        # 计算相似度 => logits: [B, num_classes]
        logits = image_features @ text_features.t()
        return logits

    def forward_for_classification_with_prototype_enhancement(self, images, text_list, use_prototype_boost=True, boost_weight=0.3):
        """
        增强版分类前向传播：集成原型回放机制
        
        Args:
            images: 输入图像 [B, C, H, W]
            text_list: 所有类别的文本描述列表
            use_prototype_boost: 是否使用原型增强
            boost_weight: 原型增强权重
            
        Returns:
            logits: 增强后的分类logits [B, num_classes]
        """
        # 1. 基础分类logits
        base_logits = self.forward_for_classification(images, text_list)
        
        if not use_prototype_boost or not hasattr(self, '_class_prototypes') or len(self._class_prototypes) == 0:
            return base_logits
            
        # 2. 计算图像特征用于原型匹配
        with torch.no_grad():
            image_features = self._network.encode_image(images)
            image_features = F.normalize(image_features, dim=1)  # [B, D]
            
            # 3. 原型增强logits
            prototype_logits = torch.zeros_like(base_logits)
            
            for class_idx, stages_dict in self._class_prototypes.items():
                if class_idx >= base_logits.size(1):  # 确保类别索引在范围内
                    continue
                    
                # 收集该类别的所有阶段原型
                class_prototypes = []
                for stage_idx, prototype in stages_dict.items():
                    if prototype is not None:
                        proto_gpu = prototype.to(self._device)
                        proto_normalized = F.normalize(proto_gpu, dim=0)
                        class_prototypes.append(proto_normalized)
                
                if len(class_prototypes) > 0:
                    # 计算与所有原型的相似度，取最大值
                    class_proto_stack = torch.stack(class_prototypes)  # [num_stages, D]
                    similarities = torch.matmul(image_features, class_proto_stack.t())  # [B, num_stages]
                    max_similarity, _ = torch.max(similarities, dim=1)  # [B]
                    prototype_logits[:, class_idx] = max_similarity
            
            # 4. 融合基础logits和原型logits
            enhanced_logits = base_logits + boost_weight * prototype_logits
            
            return enhanced_logits
    
    @torch.no_grad()
    def _compute_accuracy(self, model, loader):
        """
        测试/评估时，只使用上面定义的 forward_for_classification，
        严格保证输出维度 [batch_size, num_classes]，去除多路特征零填充。
        """
        model.eval()
        correct, total = 0, 0
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt[0]  # 只用一个模板做推理
        all_labels = class_to_label[:self._total_classes]

        for _, batch in enumerate(loader):
            if isinstance(batch[1], dict) and 'stage_id' in batch[1]:
                _, data_dict, targets = batch
                inputs = data_dict['image'].to(self._device)
            else:
                _, inputs, targets = batch
                inputs = inputs.to(self._device)

            targets = targets.long().to(self._device)

            # 准备完整类别文本
            text_list = [templates.format(lbl) for lbl in all_labels]
            logits = self.forward_for_classification(inputs, text_list)  # [B, num_classes]

            # 计算准确率
            _, preds = torch.max(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        return np.around((correct / total) * 100, decimals=2)

    @torch.no_grad()
    def _compute_accuracy_with_prototype_enhancement(self, model, loader, use_prototype_boost=True):
        """
        使用原型增强的准确率计算方法
        """
        model.eval()
        correct, total = 0, 0
        correct_base, correct_enhanced = 0, 0  # 分别统计基础方法和增强方法的准确率
        
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt[0]
        all_labels = class_to_label[:self._total_classes]

        for _, batch in enumerate(loader):
            if isinstance(batch[1], dict) and 'stage_id' in batch[1]:
                _, data_dict, targets = batch
                inputs = data_dict['image'].to(self._device)
            else:
                _, inputs, targets = batch
                inputs = inputs.to(self._device)

            targets = targets.long().to(self._device)
            text_list = [templates.format(lbl) for lbl in all_labels]
            
            # 基础分类logits
            base_logits = self.forward_for_classification(inputs, text_list)
            _, base_preds = torch.max(base_logits, dim=1)
            correct_base += (base_preds == targets).sum().item()
            
            if use_prototype_boost:
                # 原型增强logits
                enhanced_logits = self.forward_for_classification_with_prototype_enhancement(
                    inputs, text_list, use_prototype_boost=True, boost_weight=0.3
                )
                _, enhanced_preds = torch.max(enhanced_logits, dim=1)
                correct_enhanced += (enhanced_preds == targets).sum().item()
                
                # 使用增强结果作为最终结果
                correct += (enhanced_preds == targets).sum().item()
            else:
                correct += (base_preds == targets).sum().item()
                
            total += targets.size(0)

        base_acc = np.around((correct_base / total) * 100, decimals=2)
        final_acc = np.around((correct / total) * 100, decimals=2)
        
        if use_prototype_boost:
            enhanced_acc = np.around((correct_enhanced / total) * 100, decimals=2)
            logging.info(f"Stage CIL Test Accuracy - Base: {base_acc}%, Enhanced: {enhanced_acc}%, Final: {final_acc}%")
        else:
            logging.info(f"Stage CIL Test Accuracy - Base: {base_acc}%, Final: {final_acc}%")
            
        return final_acc

    def _eval_cnn(self, loader):
        """
        如果需要Top-K评估，也可使用 forward_for_classification。
        """
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt[0]
        all_labels = class_to_label[:self._total_classes]
        y_pred, y_true = [], []

        for _, batch in enumerate(loader):
            if isinstance(batch[1], dict) and 'stage_id' in batch[1]:
                _, data_dict, targets = batch
                inputs = data_dict['image'].to(self._device)
            else:
                _, inputs, targets = batch
                inputs = inputs.to(self._device)

            targets = targets.long().to(self._device)
            text_list = [templates.format(lbl) for lbl in all_labels]
            logits = self.forward_for_classification(inputs, text_list)  # [B, num_classes]

            k = min(self.topk, logits.size(1))  
            topk_preds = torch.topk(logits, k=k, dim=1)[1]
            
            # 如果k小于self.topk，需要填充结果保持一致的维度
            if k < self.topk:
                padding = torch.zeros(logits.size(0), self.topk - k, device=logits.device, dtype=torch.long)
                topk_preds = torch.cat([topk_preds, padding], dim=1)
                
            y_pred.append(topk_preds)
            y_true.append(targets)

        y_pred_tensor = torch.cat(y_pred, dim=0)
        y_true_tensor = torch.cat(y_true, dim=0)
        return y_pred_tensor.cpu().numpy(), y_true_tensor.cpu().numpy()

    
    def _create_test_loaders(self, data_manager):

        # 1. 当前任务测试集
        try:
            current_test_dataset = data_manager.get_multimodal_dataset(
                self._cur_task, source="test", mode="test")
            current_test_loader = DataLoader(
                current_test_dataset, batch_size=self.batch_size, 
                shuffle=False, num_workers=num_workers
            )
            logging.info(f"Stage CIL: Current task {self._cur_task} test dataset size: {len(current_test_dataset)}")
        except Exception as e:
            logging.warning(f"Failed to create current task test loader: {e}")
            current_test_loader = None
        
        test_datasets = []
        for task_idx in range(self._cur_task + 1):
            try:
                task_test_dataset = data_manager.get_multimodal_dataset(
                    task_idx, source="test", mode="test")
                if len(task_test_dataset) > 0:
                    test_datasets.append(task_test_dataset)
                    logging.debug(f"Added test data from task {task_idx}, size: {len(task_test_dataset)}")
            except Exception as e:
                logging.warning(f"Failed to get test data for task {task_idx}: {e}")
                continue
        
 
        if test_datasets:
            from torch.utils.data import ConcatDataset
            cumulative_test_dataset = ConcatDataset(test_datasets)
            cumulative_test_loader = DataLoader(
                cumulative_test_dataset, batch_size=self.batch_size, 
                shuffle=False, num_workers=num_workers
            )
            logging.info(f"Stage CIL: Created cumulative test dataset with {len(cumulative_test_dataset)} samples from {len(test_datasets)} tasks")
        else:

            logging.warning("No cumulative test data available, using current task only")
            cumulative_test_loader = current_test_loader
        
        return current_test_loader, cumulative_test_loader


