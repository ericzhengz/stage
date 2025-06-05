import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import IIMinsects202 # Only IIMinsects202
import json

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls_count, increment_cls_count):
        if dataset_name.lower() != "iiminsects202":
            raise ValueError(f"This DataManager is configured for IIMinsects202, but got {dataset_name}")

        # load class to label name json file
        try:
            with open('./utils/label.json', 'r') as f:
                self._class_to_label_full_map = json.load(f)[dataset_name]
        except FileNotFoundError:
            logging.error("./utils/label.json not found.")
            raise
        except KeyError:
            logging.error(f"Dataset name {dataset_name} not found in label.json.")
            raise
        
        try:
            with open('./utils/templates.json', 'r', encoding="utf-8") as f:
                self._data_to_prompt_full_map = json.load(f)[dataset_name]
        except FileNotFoundError:
            logging.error("./utils/templates.json not found.")
            raise
        except KeyError:
            logging.error(f"Dataset name {dataset_name} not found in templates.json.")
            raise
        
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed) # This sets self._class_order

        # self._class_to_label and self._data_to_prompt will be ordered by self._class_order later in _setup_data

        self._class_group_configs = [] # Stores lists of actual class labels for each group
        
        current_class_idx_offset = 0
        # Initial group
        if init_cls_count > 0 and init_cls_count <= len(self._class_order):
            self._class_group_configs.append(
                list(self._class_order[current_class_idx_offset : current_class_idx_offset + init_cls_count])
            )
            current_class_idx_offset += init_cls_count
        else:
            logging.warning(f"init_cls_count ({init_cls_count}) is invalid for total classes ({len(self._class_order)}). Adjusting.")
            if len(self._class_order) > 0:
                 self._class_group_configs.append(list(self._class_order))
                 current_class_idx_offset = len(self._class_order)

        # Incremental groups
        while current_class_idx_offset < len(self._class_order):
            num_to_add = min(increment_cls_count, len(self._class_order) - current_class_idx_offset)
            if num_to_add <= 0: break # Should not happen if loop condition is correct
            self._class_group_configs.append(
                list(self._class_order[current_class_idx_offset : current_class_idx_offset + num_to_add])
            )
            current_class_idx_offset += num_to_add
            
        logging.info(f"Class groups for CIL: {self._class_group_configs}")
        logging.info(f"Total CIL tasks: {self.nb_tasks}")
        # Old _increments logic is removed

    @property
    def nb_tasks(self):
        # Each class group has two tasks (stage 0 and stage 1)
        return 2 * len(self._class_group_configs)

    def get_task_size(self, cil_task_idx):
        # Returns the number of classes in the class group for this CIL task
        class_group_idx = cil_task_idx // 2
        if class_group_idx < len(self._class_group_configs):
            return len(self._class_group_configs[class_group_idx])
        logging.warning(f"cil_task_idx {cil_task_idx} is out of bounds for class_group_configs.")
        return 0

    def get_classes_for_cil_task(self, cil_task_idx_orig):
        cil_task_idx = cil_task_idx_orig

        if isinstance(cil_task_idx, np.ndarray):
            if cil_task_idx.size == 1:
                cil_task_idx = cil_task_idx.item()  # Becomes Python scalar int
            else:
                raise ValueError(f"cil_task_idx_orig (np.ndarray) '{cil_task_idx_orig}' must have size 1, but got shape {cil_task_idx.shape}")
        
        if not isinstance(cil_task_idx, int) or isinstance(cil_task_idx, bool): # bool is subclass of int
            if isinstance(cil_task_idx, np.integer): # Catches np.int32, np.int64 etc.
                cil_task_idx = int(cil_task_idx) # Convert to Python int
            elif isinstance(cil_task_idx, bool):
                 raise TypeError(f"cil_task_idx_orig '{cil_task_idx_orig}' cannot be a boolean, type {type(cil_task_idx)}")
            else: # Not an ndarray, not np.integer, not Python int (after potential .item())
                raise TypeError(f"cil_task_idx_orig '{cil_task_idx_orig}' must be an integer type, but got {type(cil_task_idx)}")

        # Now, cil_task_idx is definitely a Python int.
        class_group_idx = cil_task_idx // 2
        
        if not (0 <= class_group_idx < len(self._class_group_configs)):
            logging.warning(
                f"Derived class_group_idx {class_group_idx} (from original cil_task_idx {cil_task_idx_orig}, processed to {cil_task_idx}) "
                f"is out of bounds for _class_group_configs with length {len(self._class_group_configs)}. "
                f"Returning empty list."
            )
            return []
            
        return self._class_group_configs[class_group_idx]

    def get_stage_for_cil_task(self, cil_task_idx_orig):
        cil_task_idx = cil_task_idx_orig

        if isinstance(cil_task_idx, np.ndarray):
            if cil_task_idx.size == 1:
                cil_task_idx = cil_task_idx.item() # Becomes Python scalar int
            else:
                raise ValueError(f"cil_task_idx_orig (np.ndarray) '{cil_task_idx_orig}' must have size 1, but got shape {cil_task_idx.shape}")

        if not isinstance(cil_task_idx, int) or isinstance(cil_task_idx, bool): # bool is subclass of int
            if isinstance(cil_task_idx, np.integer): # Catches np.int32, np.int64 etc.
                cil_task_idx = int(cil_task_idx) # Convert to Python int
            elif isinstance(cil_task_idx, bool):
                 raise TypeError(f"cil_task_idx_orig '{cil_task_idx_orig}' cannot be a boolean, type {type(cil_task_idx)}")
            else: # Not an ndarray, not np.integer, not Python int (after potential .item())
                raise TypeError(f"cil_task_idx_orig '{cil_task_idx_orig}' must be an integer type, but got {type(cil_task_idx)}")
        
        # Now, cil_task_idx is definitely a Python int.
        return cil_task_idx % 2

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
        self, cil_task_idx, source, mode, appendent=None, ret_data=True, buffer=False, balancemode='default'
    ):
        """
        获取指定CIL任务的数据集。
        cil_task_idx: 可以是单个任务索引，也可以是类别标签列表/数组。
                      在此特定调用链中，它接收的是类别标签数组。
        """
        if source == "train":
            x_all_source, y_all_source, s_all_source = self._train_data, self._train_targets, self._train_stages
        elif source == "test":
            x_all_source, y_all_source, s_all_source = self._test_data, self._test_targets, self._test_stages
        else:
            raise ValueError(f"Unknown data source {source}.")

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip": # flip is a test mode variant
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError(f"Unknown mode {mode}.")

        data_task, targets_task, stages_task = [], [], []
        
        # OLD: target_class_labels = cil_task_idx 
        target_class_labels = self.get_classes_for_cil_task(cil_task_idx) # CORRECTED: Get list of original class labels for the CIL task
        target_stage = self.get_stage_for_cil_task(cil_task_idx)

        # The existing check for has_classes should now work correctly as target_class_labels is a list.
        if hasattr(target_class_labels, '__len__'):
            has_classes = len(target_class_labels) > 0
        else:
            has_classes = target_class_labels is not None
            
        if not has_classes: # No classes for this task (e.g. task_idx out of bounds)
            logging.warning(f"No target classes for CIL task {cil_task_idx}. Returning empty dataset.")
        else:
            for cls_label in target_class_labels:
                # _select_specific filters by a single class label and a single stage label
                sel_data, sel_targets, sel_stages = self._select_specific(
                    x_all_source, y_all_source, s_all_source,
                    cls_label, 
                    target_stage
                )
                if len(sel_data) > 0:
                    data_task.append(sel_data)
                    targets_task.append(sel_targets)
                    stages_task.append(sel_stages)
        
        if len(data_task) > 0:
            data_task = np.concatenate(data_task)
            targets_task = np.concatenate(targets_task)
            stages_task = np.concatenate(stages_task)
        else: # Ensure they are empty arrays if no data was found
            data_task, targets_task, stages_task = np.array([]), np.array([]), np.array([])


        # Handle appendent (memory data)
        # Memory data should contain samples from all *classes* and *stages* learned so far.
        # It is not filtered by the current task's stage.
        if appendent is not None and len(appendent) > 0:
            # Assuming appendent is (data, targets, stages)
            appendent_data, appendent_targets, appendent_stages = appendent
            if len(appendent_data) > 0:
                if len(data_task) > 0:
                    data_task = np.concatenate([data_task, appendent_data])
                    targets_task = np.concatenate([targets_task, appendent_targets])
                    stages_task = np.concatenate([stages_task, appendent_stages])
                else:
                    data_task, targets_task, stages_task = appendent_data, appendent_targets, appendent_stages

        if ret_data:
            return data_task, targets_task, stages_task, DummyDataset(data_task, targets_task, stages_task, trsf, self.use_path)
        else:
            if len(data_task) == 0:
                logging.warning(f"No data for CIL task {cil_task_idx} (source: {source}, mode: {mode}) after processing. Returning empty DummyDataset.")
            return DummyDataset(data_task, targets_task, stages_task, trsf, self.use_path)
# ... (get_dataset_with_split would need similar overhaul if used) ...

    def _setup_data(self, dataset_name, shuffle, seed):
        self.idata = _get_idata(dataset_name)
        
        if not hasattr(self.idata, '_data_loaded') or not self.idata._data_loaded:
            logging.info(f"Initializing {dataset_name} dataset...")
            self.idata.download_data() # This loads train_data, train_targets, train_stages etc.
        else:
            logging.info(f"Using already loaded {dataset_name} dataset.")

        self._train_data, self._train_targets, self._train_stages = self.idata.train_data, self.idata.train_targets, self.idata.train_stages
        self._test_data, self._test_targets, self._test_stages = self.idata.test_data, self.idata.test_targets, self.idata.test_stages
        
        if self._train_data is None or self._test_data is None:
             raise ValueError("Data loading failed for IIMinsects202. Check paths and data.py")
        if self._train_stages is None or self._test_stages is None:
            raise ValueError("Stage information not loaded by IIMinsects202. Check data.py")


        self.use_path = self.idata.use_path
        self._train_trsf = self.idata.train_trsf
        self._test_trsf = self.idata.test_trsf
        self._common_trsf = self.idata.common_trsf

        # Determine class order (e.g. for shuffling)
        # self.idata.class_order is [0, 1, ..., 19] for IIMinsects202
        # unique_targets = np.unique(self._train_targets) # Should be 0-19
        # order = [i for i in range(len(unique_targets))]
        order = list(np.arange(len(np.unique(self.idata.train_targets))))


        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(order).tolist()
        else:
            # Use the default order from idata if it's meaningful, or just sorted unique targets
            order = self.idata.class_order if self.idata.class_order is not None else sorted(list(np.unique(self._train_targets)))
        
        self._class_order = order # This is the order of original class labels, e.g. [3,0,19,...]
        logging.info(f"Class order: {self._class_order}")

        # Map targets to new indices if class_order is a permutation of [0, ..., N-1]
        # E.g., if original class 3 is now the 0-th class in our sequence.
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)
        # Note: _train_stages and _test_stages remain aligned with _train_data and _test_data,
        # as _map_new_class_index only changes the label values, not the order of samples.

        # Update _class_to_label and _data_to_prompt to match the new class order
        # self._class_to_label_full_map and self._data_to_prompt_full_map are dictionaries keyed by original class name or index.
        # We need to create lists that are ordered according to self._class_order.
        # Assuming _class_to_label_full_map is a list/dict that can be indexed by original class indices 0-19.
        
        # If _class_to_label_full_map is a list:
        self._class_to_label = [self._class_to_label_full_map[i] for i in self._class_order]
        # If _data_to_prompt_full_map is a list of lists of prompts (one list per original class):
        # This part needs to align with how _data_to_prompt is used.
        # If it's a list of templates per class, then it also needs reordering.
        # For simplicity, assuming _data_to_prompt_full_map is a single list of templates for the dataset, not per-class.
        # If it *is* per-class, this needs: self._data_to_prompt = [self._data_to_prompt_full_map[i] for i in self._class_order]
        # The original code had: self._data_to_prompt = json.load(f)[dataset_name] which implies it's not per-class ordered.
        # Let's assume _data_to_prompt is a list of general templates for the dataset.
        self._data_to_prompt = self._data_to_prompt_full_map # No reordering if it's general templates.
        
        logging.info(f"Effective class_to_label mapping (ordered by current task sequence): {self._class_to_label}")


    def _select_specific(self, x_all, y_all_cls, s_all_stages, target_cls_label, target_stage_label):
        # Ensure inputs are numpy arrays for boolean indexing
        x_all = np.array(x_all)
        y_all_cls = np.array(y_all_cls)
        s_all_stages = np.array(s_all_stages)

        # target_cls_label is an original class label (e.g. 3, 0, 19 from _class_order)
        # y_all_cls contains mapped class indices (0, 1, 2 ... according to _class_order.indexOf(original_label))
        # So we need to find where y_all_cls corresponds to target_cls_label's position in _class_order
        try:
            mapped_target_cls_label = self._class_order.index(target_cls_label)
        except ValueError:
            logging.error(f"Target class label {target_cls_label} not found in self._class_order. This should not happen.")
            return np.array([]), np.array([]), np.array([])

        idxes = np.where((y_all_cls == mapped_target_cls_label) & (s_all_stages == target_stage_label))[0]
        
        if len(idxes) == 0:
            return np.array([]), np.array([]), np.array([])
        return x_all[idxes], y_all_cls[idxes], s_all_stages[idxes]

    # _select_rmm would need similar stage-aware modification if used.

    def get_multimodal_dataset(self, cil_task_idx, source, mode, appendent=None):
        # `get_dataset` now returns DummyDataset(data, targets, stages, ...)
        # The third element `stages_task` is not used by DummyDataset constructor yet.
        # We need to pass all three (data, targets, stages) to DummyDataset.
        # The ret_data=True path of get_dataset returns data, targets, stages, dummy_dataset
        
        # Let's call get_dataset with ret_data=False, which returns the DummyDataset directly
        dummy_dataset = self.get_dataset(cil_task_idx, source, mode, appendent, ret_data=False)
        
        # self.idata should already be set by _setup_data
        # self.idata = _get_idata(self.dataset_name) # Not needed here again
        
        return InsectsMultiModalDataset(dummy_dataset, self) # Pass self as data_manager

    def get_stage_prompt(self, class_idx_mapped, stage_id):
        # class_idx_mapped is the mapped index (0 to N-1)
        # We need the original class label to get the name from _class_to_label_full_map
        # Or, use self._class_to_label which is already ordered by self._class_order
        class_name = self._class_to_label[class_idx_mapped] # self._class_to_label is already ordered
        
        stage_name = self.idata.get_stage_description(stage_id) # Use self.idata
        
        if not self._data_to_prompt: # Check if templates are loaded
            logging.error("Prompt templates not loaded.")
            return f"Image of a {class_name} at {stage_name}."

        template = np.random.choice(self._data_to_prompt)
        prompt = template.replace("{类别}", class_name).replace("{虫态}", stage_name)
        return prompt

class DummyDataset(Dataset):
    def __init__(self, images, labels, stages, trsf, use_path=False): # Added stages
        # Assertions to check lengths if data is not empty
        if len(images) > 0:
            assert len(images) == len(labels), "Data size error (images vs labels)!"
            assert len(images) == len(stages), "Data size error (images vs stages)!"
        
        self.images = images
        self.labels = labels
        self.stages = stages # Store stages
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Add index bounds checking
        if not (0 <= idx < len(self.images)):
            logging.error(f"CRITICAL ERROR: Index {idx} is out of bounds for dataset of size {len(self.images)}!")
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.images)}")

        try:
            # Load image
            if self.use_path:
                image_path = self.images[idx]
                logging.debug(f"DummyDataset loading image path: {image_path}")
                image = self.trsf(pil_loader(image_path))
            else:
                image = self.trsf(Image.fromarray(self.images[idx]))
            
            # Get label and stage
            label = self.labels[idx]
            stage = self.stages[idx]
            
            # Debug logging
            logging.debug(f"DummyDataset __getitem__ for idx {idx}:")
            logging.debug(f"  image type: {type(image)}")
            logging.debug(f"  label type: {type(label)}, value: {label}")
            logging.debug(f"  stage type: {type(stage)}, value: {stage}")
            
            return idx, image, label, stage # Return 4 values

        except Exception as e:
            logging.error(f"Error in DummyDataset.__getitem__ for idx {idx}: {e}")
            logging.error(f"  self.use_path: {self.use_path}")
            logging.error(f"  images[{idx}]: {self.images[idx] if idx < len(self.images) else 'INDEX_OUT_OF_BOUNDS'}")
            logging.error(f"  labels[{idx}]: {self.labels[idx] if idx < len(self.labels) else 'INDEX_OUT_OF_BOUNDS'}")
            logging.error(f"  stages[{idx}]: {self.stages[idx] if idx < len(self.stages) else 'INDEX_OUT_OF_BOUNDS'}")
            raise

def _map_new_class_index(y, order):
    # y: original labels, order: list of original labels in new order
    # e.g. y=[0,1,2,0,1], order=[2,0,1] => new_y=[1,2,0,1,2] (2->0, 0->1, 1->2)
    # This means if original label 'c' is at order[i], its new label is 'i'.
    if not isinstance(y, np.ndarray): y = np.array(y)
    
    # Create a mapping from original label to new index
    map_dict = {original_label: new_idx for new_idx, original_label in enumerate(order)}
    
    new_y = np.array([map_dict[original_label] for original_label in y if original_label in map_dict])
    
    # Check if all original labels in y were in order
    if len(new_y) != len(y):
        logging.warning(f"Some labels in y were not in class_order. Original len: {len(y)}, Mapped len: {len(new_y)}")
        # Fallback for labels not in order (should not happen if order covers all unique labels in y)
        # This part might need robust handling if y can contain labels not in 'order'.
        # For now, assuming 'order' contains all unique labels present in 'y'.
        # A simple fix is to filter y first for labels present in order, but that changes data.
        # The current list comprehension already filters.
    return new_y


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "iiminsects202":
        return IIMinsects202()
    # All other elif branches for other datasets are removed
    else:
        raise NotImplementedError(f"Unknown or unsupported dataset {dataset_name}.")

# _get_idata_image_only is removed as it's not used by the refactored DataManager for IIMinsects202

def pil_loader(path):
    try:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    except Exception as e:
        logging.error(f"ERROR: Cannot open image {path}. Exception: {e}")
        # Consider returning a placeholder image or re-raising, depending on desired robustness
        raise # Re-raise the exception to halt if an image is critical

# LaionData class is kept as it might be used by other parts of the project,
# but it's not directly related to IIMinsects202 handling in DataManager.
# If it's confirmed unused, it can be removed.
# For now, keeping it.
class LaionData(Dataset):
    def __init__(self, txt_path):
        self.transform = transforms.Compose([
            transforms.Resize((224,224),transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        self.img_list = [line.split()[0] for line in lines]
        self.txt_list = [line.split()[1] for line in lines]

    def __getitem__(self, index):
        txt_path = self.txt_list[index]
        img_path = self.img_list[index]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            logging.error(f"Error loading image {img_path} in LaionData: {e}")
            # Return a placeholder or skip, depending on requirements. Here, re-raising.
            raise
        
        try:
            with open(txt_path, 'r') as f:
                txt = f.read().strip()
        except Exception as e:
            logging.error(f"Error loading text {txt_path} in LaionData: {e}")
            txt = "" # Return empty text on error

        return img, txt

    def __len__(self):
        return len(self.img_list)

class InsectsMultiModalDataset(Dataset):
    def __init__(self, dataset, data_manager): # dataset is a DummyDataset instance
        self.dataset = dataset 
        self.data_manager = data_manager
        self.insect_data = data_manager.idata # Reference to IIMinsects202 instance

        # No need for _extract_stage_info_from_paths or _build_index_mapping
        # if DummyDataset provides all necessary info including stage_id.
        # The 'orig_idx' from DummyDataset is the index within that specific subset.

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # DummyDataset now returns: local_idx_within_dummy, image, label (mapped class), stage_id
        local_idx, image, label, stage_id = self.dataset[idx]
        
        data_dict = {
            'image': image,
            'label': label,    # Mapped class label
            'stage_id': stage_id, # Stage label (0 or 1)
            'orig_idx': local_idx # Index within the current DummyDataset subset
        }
        
        # The learner expects (idx, data_dict, label_for_loss)
        return local_idx, data_dict, label