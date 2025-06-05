import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist

EPSILON = 1e-8
batch_size = 128


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._class_prototypes = {} 
        self.topk = 4

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    # @property
    # def exemplar_size(self):
    #     assert len(self._data_memory) == len(
    #         self._targets_memory
    #     ), "Exemplar size error."
    #     return len(self._targets_memory) # Removed

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):

        current_task_classes = data_manager.get_classes_for_cil_task(self._cur_task)
        self._update_prototypes(data_manager, current_task_classes)

    def _update_prototypes(self, data_manager, current_task_classes):
        logging.info(f"Stage CIL: Updating prototypes for classes: {current_task_classes}")
        
        # 确保双层结构初始化
        if not hasattr(self, '_class_prototypes') or self._class_prototypes is None:
            self._class_prototypes = {}
        
        for class_idx in current_task_classes:

            try:
                mapped_class_idx = data_manager._class_order.index(class_idx)
            except ValueError:
                logging.error(f"Class {class_idx} not found in class order")
                continue
            
            # Stage CIL: 初始化类别的双层字典结构
            if mapped_class_idx not in self._class_prototypes:
                self._class_prototypes[mapped_class_idx] = {}
            

            current_stage = data_manager.get_stage_for_cil_task(self._cur_task)
            
            try:
                # Get dataset for single class
                data, targets, stages, dataset = data_manager.get_dataset(
                    cil_task_idx=self._cur_task,  # Use current task to get correct stage
                    source="train", 
                    mode="test",  # Use test mode to avoid data augmentation
                    ret_data=True
                )
                
                # Filter for the specific class we want
                if len(data) > 0:
                    targets_np = np.array(targets)
                    class_mask = (targets_np == mapped_class_idx)
                    
                    if np.any(class_mask):
                        class_data = data[class_mask]
                        class_targets = targets_np[class_mask]
                        
                        # Create DataLoader for this class
                        from utils.data_manager import DummyDataset
                        class_dataset = DummyDataset(
                            class_data, class_targets, 
                            np.full(len(class_targets), current_stage, dtype=np.int32),
                            dataset.trsf, dataset.use_path
                        )
                        
                        batch_size = 32  
                        class_loader = DataLoader(
                            class_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=0  # Start with 0 for debugging
                        )
                        
                        # Extract feature vectors on GPU
                        vectors, _ = self._extract_vectors(class_loader)
                        
                        if len(vectors) > 0:
                            # Convert to tensor and move to GPU for computation
                            vectors_tensor = torch.tensor(vectors, dtype=torch.float32).to(self._device)
                            
                            # Compute prototype (mean of feature vectors)
                            prototype = torch.mean(vectors_tensor, dim=0)
                            
                            # Stage CIL: 存储双层结构原型 {class_id: {stage_id: prototype}}
                            self._class_prototypes[mapped_class_idx][current_stage] = prototype.cpu()
                            
                            logging.info(f"Stage CIL: Updated prototype for class {class_idx} (mapped: {mapped_class_idx}), "
                                       f"stage {current_stage}, prototype shape: {prototype.shape}, computed from {len(vectors)} samples")
                        else:
                            logging.warning(f"No feature vectors extracted for class {class_idx}")
                    else:
                        logging.warning(f"No samples found for class {class_idx} in current task data")
                else:
                    logging.warning(f"No data returned for current task")
                    
            except Exception as e:
                logging.error(f"Error updating prototype for class {class_idx}: {str(e)}")
                continue

    


    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        
        # 健壮的 top-k 计算
        try:
            correct = 0
            for i in range(len(y_true)):
                if y_true[i] in y_pred[i, :self.topk]:
                    correct += 1
            ret[f"top{self.topk}"] = np.around((correct * 100.0) / len(y_true), decimals=2)
        except Exception as e:
            print(f"计算top{self.topk}准确率时出错: {str(e)}")
            ret[f"top{self.topk}"] = 0.0
        
        return ret
    
    def _evaluate_zs(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._total_classes) # indx< total are old classes, >= are new unseen classes.
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around((y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),decimals=2)
        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        if self.args["convnet_type"].lower()!="clip" or self.args["model_name"].lower()=="l2p" or self.args["model_name"].lower()=="dualprompt":
            return cnn_accy, nme_accy, None, None, None, None
        else:
            y_pred, y_true = self._eval_zero_shot()
            zs_acc= self._evaluate_zs(y_pred, y_true)
            zs_seen, zs_unseen, zs_harmonic, zs_total = zs_acc["grouped"]["old"], zs_acc["grouped"]["new"], zs_acc["grouped"]["harmonic"], zs_acc["grouped"]["total"]

        return cnn_accy, nme_accy, zs_seen, zs_unseen, zs_harmonic, zs_total

    def _eval_zero_shot(self):  
        self._network.eval()
        class_to_label=self.data_manager._class_to_label
        templates=self.data_manager._data_to_prompt
        total_labels=class_to_label  # 所有类别
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).to(self._device)
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0).to(self._device)

        all_data_list = []
        all_targets_list = []
        all_stages_list = []
        
        # 遍历所有已完成的任务来收集数据
        for task_idx in range(self._cur_task + 1):
            try:
                data_task, targets_task, stages_task, _ = self.data_manager.get_dataset(
                    cil_task_idx=task_idx,
                    source="test",
                    mode="test",
                    ret_data=True
                )
                if len(data_task) > 0:
                    all_data_list.append(data_task)
                    all_targets_list.append(targets_task)
                    all_stages_list.append(stages_task)
            except Exception as e:
                logging.warning(f"Failed to get data for task {task_idx}: {e}")
                continue
        
        # 合并所有任务的数据
        if all_data_list:
            from utils.data_manager import DummyDataset
            from torchvision import transforms
            all_data = np.concatenate(all_data_list) if len(all_data_list) > 1 else all_data_list[0]
            all_targets = np.concatenate(all_targets_list) if len(all_targets_list) > 1 else all_targets_list[0]
            all_stages = np.concatenate(all_stages_list) if len(all_stages_list) > 1 else all_stages_list[0]

            test_trsf = transforms.Compose([*self.data_manager._test_trsf, *self.data_manager._common_trsf])
            
            test_dataset = DummyDataset(
                all_data, all_targets, all_stages,
                test_trsf,
                self.data_manager.use_path
            )
        else:
            # 如果没有数据，创建空数据集
            from utils.data_manager import DummyDataset
            from torchvision import transforms
            test_trsf = transforms.Compose([*self.data_manager._test_trsf, *self.data_manager._common_trsf])
            test_dataset = DummyDataset(
                np.array([]), np.array([]), np.array([]),
                test_trsf,
                self.data_manager.use_path
            )
        loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        y_pred, y_true = [], []
        for batch in loader:
            # Handle different batch formats
            if len(batch) == 3:
                # Standard format: (idx, inputs, targets)
                _, inputs, targets = batch
            elif len(batch) == 4:
                # DummyDataset format: (idx, image, label, stage)
                _, inputs, targets, stages = batch
            else:
                logging.error(f"Unexpected batch format with {len(batch)} elements")
                continue
                
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)  # 将标签也移到GPU
            with torch.no_grad():
                image_features=self._network.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                outputs= image_features @ text_features.T
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  
            y_pred.append(predicts)
            y_true.append(targets)
        
        # 在GPU上合并结果
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        
        # 最后一步转为CPU
        return y_pred.cpu().numpy(), y_true.cpu().numpy()

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        return None


    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for batch in loader:
            # Handle different batch formats
            if len(batch) == 3:
                # Standard format: (idx, inputs, targets) or InsectsMultiModalDataset format
                _, inputs, targets = batch
            elif len(batch) == 4:
                # DummyDataset format: (idx, image, label, stage)
                _, inputs, targets, stages = batch
            else:
                logging.error(f"Unexpected batch format with {len(batch)} elements")
                continue
                
            inputs = inputs.to(self._device)
            targets = targets.to(self._device) 
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts == targets).sum()  
            total += len(targets)

        return np.around((correct.item() / total) * 100, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for batch in loader:
            # Handle different batch formats
            if len(batch) == 3:
                # Standard format: (idx, inputs, targets) or InsectsMultiModalDataset format
                _, _inputs, _targets = batch
            elif len(batch) == 4:
                # DummyDataset format: (idx, image, label, stage)
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
            _preds = torch.argmax(self._network(_inputs)['logits'], dim=1)
            y_pred.append(_preds.cpu().numpy())
            y_true.append(_targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)
    

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for batch in loader:
            if len(batch) == 3:
                _, _inputs, _targets = batch
            elif len(batch) == 4:
                _, _inputs, _targets, _stages = batch
            else:
                logging.error(f"Unexpected batch format with {len(batch)} elements")
                continue
                
            # Handle various input formats
            if isinstance(_inputs, dict) and 'image' in _inputs:
                _inputs = _inputs['image'].to(self._device)
            elif isinstance(_inputs, dict) and 'stage_id' in _inputs:
                data_dict = _inputs
                _inputs = data_dict['image'].to(self._device)
            else:
                _inputs = _inputs.to(self._device)
                
            if isinstance(_targets, torch.Tensor):
                _targets_np = _targets.cpu().numpy()
            else:
                _targets_np = np.array(_targets) 

            _vectors = tensor2numpy(self._network.extract_vector(_inputs))
            vectors.append(_vectors)
            targets.append(_targets_np)

        return np.concatenate(vectors), np.concatenate(targets)
