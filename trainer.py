import sys
import logging
import copy
import torch
import numpy as np
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import time
from datetime import datetime


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    
    # 创建日志目录
    logs_name = "logs/{}/{}/{}/{}".format(
        args["model_name"], args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    # 配置日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}_{}".format(
        args["model_name"], args["dataset"], 
        init_cls, args["increment"], args["prefix"], 
        args["seed"], args["convnet_type"], timestamp)
    
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)
    
    # 初始化数据管理器，注意昆虫数据集的特殊处理
    data_manager = DataManager(
        args["dataset"], args["shuffle"], args["seed"], 
        args["init_cls"], args["increment"]
    )
    
    # 记录数据集类别信息
    logging.info(f"数据集: {args['dataset']}")
    logging.info(f"总类别数: {data_manager.get_total_classnum()}")
    logging.info(f"初始类别数: {args['init_cls']}")
    logging.info(f"增量步长: {args['increment']}")
    logging.info(f"任务总数: {data_manager.nb_tasks}")
    
    # 创建模型
    model = factory.get_model(args["model_name"], args)
    model.save_dir = logs_name

    # 进度跟踪指标
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    
    # 遍历所有任务进行增量学习
    for task in range(data_manager.nb_tasks):
        # 打印当前任务信息
        logging.info(f"\n{'='*50}")
        logging.info(f"开始任务 {task+1}/{data_manager.nb_tasks}")
        logging.info(f"当前类别范围: {model._known_classes} - {model._known_classes + data_manager.get_task_size(task)}")
        logging.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 记录参数数量
        logging.info("模型参数总量: {}".format(count_parameters(model._network)))
        logging.info("可训练参数量: {}".format(count_parameters(model._network, True)))
        
        # 训练开始时间
        start_time = time.time()
        
        # 增量训练
        model.incremental_train(data_manager)
        
        # 评估性能
        # cnn_accy, nme_accy, zs_seen, zs_unseen, zs_harmonic, zs_total = model.eval_task()
        cnn_accy, nme_accy, *_ = model.eval_task()
        
        # 任务后处理
        model.after_task()
        
        # 移除旧的虫态距离矩阵更新逻辑
        # if hasattr(model, 'update_state_distance_matrix'):
        #     logging.info("更新虫态距离矩阵...")
        #     model.update_state_distance_matrix(data_manager)
        
        # 记录训练时间
        elapsed_time = time.time() - start_time
        logging.info(f"任务{task+1}训练耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")

        # 记录评估结果
        logging.info("CNN评估结果: {}".format(cnn_accy["grouped"]))
        logging.info("Top-1 准确率: {:.2f}%".format(cnn_accy["top1"]*100))
        # 假设评估结果字典中用 "top5" 键表示 Top-5 准确率
        if "top5" in cnn_accy:
            logging.info("Top-5 准确率: {:.2f}%".format(cnn_accy["top5"]*100))
            cnn_curve["top5"].append(cnn_accy["top5"])
        elif "top4" in cnn_accy: # Fallback or specific use of top4
            logging.info("Top-4 准确率: {:.2f}%".format(cnn_accy["top4"]*100)) # Or Top-k if it's not exactly top-4
            cnn_curve["top5"].append(cnn_accy["top4"]) # Still appending to top5 curve for consistency in logging below

        # 更新评估曲线
        cnn_curve["top1"].append(cnn_accy["top1"])
        # cnn_curve["top5"].append(cnn_accy["top4"]) # Handled above

        logging.info("CNN top1准确率曲线: {}".format([f"{acc:.4f}" for acc in cnn_curve["top1"]]))
        if cnn_curve["top5"]:
            logging.info("CNN top5准确率曲线: {}\n".format([f"{acc:.4f}" for acc in cnn_curve["top5"]]))
        else:
            logging.info("CNN top5准确率曲线: 未记录\n")


        # 计算并显示平均准确率
        avg_acc = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
        print(f'当前平均准确率: {avg_acc:.4f} ({avg_acc*100:.2f}%)')
        logging.info(f"当前平均准确率: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
        
    logging.info("\n" + "="*50)
    logging.info("训练完成!")
    logging.info("最终CNN top1准确率曲线: {}".format([f"{acc:.4f}" for acc in cnn_curve["top1"]]))
    logging.info(f"最终平均准确率: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    logging.info("="*50)


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
