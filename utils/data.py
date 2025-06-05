import numpy as np
from torchvision import datasets, transforms
# from utils.toolkit import split_images_labels # Kept for now, can be removed if not used elsewhere
import os
# import glob # No longer needed if we target specific subfolder names
import logging # For logging

# Basic iData class
class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None
    train_data, train_targets, train_stages = None, None, None
    test_data, test_targets, test_stages = None, None, None
    use_path = False # Default for iData, IIMinsects202 will override

# Keep build_transform as IIMinsects202 uses it
def build_transform(is_train, args):
    input_size = 224
    # Simplified transform, assuming args is not used for this basic version
    if is_train:
        # Example train transform (can be adjusted)
        transform = [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ]
    else:
        # Example test transform (can be adjusted)
        transform = [
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ]
    return transform

class IIMinsects202(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [] # Normalization is included in build_transform
    
    _data_loaded = False
    class_order = np.arange(20).tolist() # 20 insect classes

    def __init__(self):
        super().__init__()
        # IMPORTANT: User must ensure this path is correct
        self.dataset_base_path = "C:/Users/ASUS/Desktop/test_0/data/IIMinsects202"
        # self.dataset_base_path = "./data/IIMinsects202" # Alternative relative path
        logging.info(f"IIMinsects202 dataset path set to: {self.dataset_base_path}")
        if not os.path.isdir(self.dataset_base_path):\
            logging.warning(f"Dataset base path does not exist: {self.dataset_base_path}. Please check the path.")

    def download_data(self):
        if self._data_loaded:
            logging.info("IIMinsects202 data already loaded. Skipping download.")
            return
            
        train_dir = os.path.join(self.dataset_base_path, "train")
        test_dir = os.path.join(self.dataset_base_path, "test")

        if not os.path.exists(train_dir):
            logging.error(f"Train directory not found: {train_dir}")
            raise FileNotFoundError(f"Train directory not found: {train_dir}")
        if not os.path.exists(test_dir):
            logging.error(f"Test directory not found: {test_dir}")
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
        logging.info("Loading IIMinsects202 train data...")
        self.train_data, self.train_targets, self.train_stages = self.load_nested_dataset(train_dir)
        if self.train_data is not None:
            logging.info(f"Loaded {len(self.train_data)} training samples with {len(np.unique(self.train_targets))} classes and {len(np.unique(self.train_stages))} stages.")
        else:
            logging.error("Failed to load training data.")

        logging.info("Loading IIMinsects202 test data...")
        self.test_data, self.test_targets, self.test_stages = self.load_nested_dataset(test_dir)
        if self.test_data is not None:
            logging.info(f"Loaded {len(self.test_data)} test samples with {len(np.unique(self.test_targets))} classes and {len(np.unique(self.test_stages))} stages.")
        else:
            logging.error("Failed to load test data.")
            
        self._data_loaded = True

    def load_nested_dataset(self, root_dir):
        images, targets, stages = [], [], []
        
        # Define the exact stage folder names and their corresponding integer IDs
        # Modified to match user's directory structure: class_name/0/ and class_name/1/
        stage_map = {"0": 0, "1": 1}

        if not os.path.isdir(root_dir):
            logging.error(f"Root directory does not exist: {root_dir}")
            return None, None, None

        # 首先需要加载标签映射
        try:
            import json
            with open('./utils/label.json', 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            # 获取 iiminsects202 的类别列表
            class_names = label_data.get("iiminsects202", [])
            if not class_names:
                logging.error("No class names found for iiminsects202 in label.json")
                return None, None, None
            
            # 创建类别名称到标签ID的映射
            name_to_label = {name: idx for idx, name in enumerate(class_names)}
            
            logging.info(f"Loaded {len(class_names)} class names from label.json")
            
        except Exception as e:
            logging.error(f"Failed to load label.json: {e}")
            return None, None, None

        for class_folder_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_folder_name)
            if not os.path.isdir(class_path):
                continue
            
            # 使用类别名称获取对应的标签ID
            if class_folder_name in name_to_label:
                class_label = name_to_label[class_folder_name]
                logging.debug(f"Found class folder: {class_folder_name} -> label {class_label}")
            else:
                logging.warning(f"Skipping unknown class folder: {class_folder_name}")
                continue
            
            for stage_folder_name, stage_id in stage_map.items():
                stage_path = os.path.join(class_path, stage_folder_name)
                if os.path.isdir(stage_path):
                    stage_images = 0
                    for img_file in sorted(os.listdir(stage_path)):
                        # Basic check for image extensions
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                            img_path = os.path.join(stage_path, img_file)
                            if os.path.isfile(img_path):
                                images.append(img_path)
                                targets.append(class_label)
                                stages.append(stage_id)
                                stage_images += 1
                    
                    if stage_images > 0:
                        logging.debug(f"Loaded {stage_images} images from {stage_path}")
                else:
                    logging.debug(f"Stage folder not found: {stage_path} for class {class_folder_name}")

        if not images:
            logging.warning(f"No images loaded from {root_dir}. Check directory structure and image files.")
            return None, None, None
            
        logging.info(f"Successfully loaded {len(images)} images with {len(set(targets))} classes and {len(set(stages))} stages")
        return np.array(images), np.array(targets, dtype=int), np.array(stages, dtype=int)

    def get_stage_description(self, stage_id):
        if stage_id == 0:
            return "initial stage" # Or "egg stage", "first stage", etc.
        elif stage_id == 1:
            return "later stage"  # Or "larva stage", "second stage", etc.
        else:
            logging.warning(f"Requested description for unknown stage_id: {stage_id}")
            return f"unknown_stage_{stage_id}"



