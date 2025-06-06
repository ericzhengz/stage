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
    """形态演化增量学习的主训练函数"""
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train_morphology(args)


def _train_morphology(args):
    """
    形态演化训练主逻辑：
    - 每两次task为一组，表示同一批类别的不同形态（形态0 → 形态1）
    - 维护Memory Pool学习形态演化变换
    """
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    
    # 创建日志目录
    logs_name = "logs/{}/{}/{}/{}".format(
        args["model_name"], args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    # 配置日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}_morphology_{}".format(
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
    
    # 初始化数据管理器
    data_manager = DataManager(
        args["dataset"], args["shuffle"], args["seed"], 
        args["init_cls"], args["increment"]
    )
    
    # 记录数据集信息
    logging.info(f"=== 形态演化增量学习设置 ===")
    logging.info(f"数据集: {args['dataset']}")
    logging.info(f"总类别数: {data_manager.get_total_classnum()}")
    logging.info(f"初始类别数: {args['init_cls']}")
    logging.info(f"增量步长: {args['increment']}")
    logging.info(f"任务总数: {data_manager.nb_tasks}")
    
    # 计算形态对数量（每两次task为一对）
    morphology_pairs = data_manager.nb_tasks // 2
    logging.info(f"形态演化对数: {morphology_pairs} (每对包含形态0→形态1)")
    
    # 创建模型
    model = factory.get_model(args["model_name"], args)
    model.save_dir = logs_name

    # 进度跟踪指标
    cnn_curve = {"top1": [], "top5": []}
    morphology_performance = []  # 跟踪每个形态对的性能
    
    # 遍历形态对进行训练
    for pair_idx in range(morphology_pairs):
        logging.info(f"\n{'='*60}")
        logging.info(f"开始形态演化对 {pair_idx+1}/{morphology_pairs}")
        logging.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 计算对应的task索引
        task0_idx = pair_idx * 2      # 形态0的task索引
        task1_idx = pair_idx * 2 + 1  # 形态1的task索引
        
        logging.info(f"形态0任务索引: {task0_idx}, 形态1任务索引: {task1_idx}")
        
        # === 阶段1: 训练形态0 ===
        logging.info(f"\n--- 阶段1: 训练形态0 (Task {task0_idx}) ---")
        pair_start_time = time.time()
        
        # 训练形态0
        stage0_start_time = time.time()
        model.incremental_train(data_manager, task_idx=task0_idx, morphology_stage=0, pair_idx=pair_idx)
        stage0_time = time.time() - stage0_start_time
        
        # 评估形态0
        eval_results_0 = model.eval_task()
        cnn_accy_0 = eval_results_0[0]
        incremental_metrics_0 = eval_results_0[-1] if len(eval_results_0) > 6 else None
        model.after_task()
        
        logging.info(f"形态0训练耗时: {stage0_time:.2f}秒")
        logging.info(f"形态0 Top-1准确率: {cnn_accy_0['top1']:.2f}%")
        
        # 输出增量学习指标
        if incremental_metrics_0:
            logging.info(f"=== 形态0增量学习指标 ===")
            logging.info(f"  当前任务累积准确率: {incremental_metrics_0['current_task_acc']:.2f}%")
            logging.info(f"  平均增量准确率: {incremental_metrics_0['average_incremental_acc']:.2f}%")
            logging.info(f"  遗忘率: {incremental_metrics_0['forgetting_rate']:.2f}%")
            logging.info(f"  后向迁移: {incremental_metrics_0['backward_transfer']:.2f}%")
            if incremental_metrics_0['all_task_accuracies']:
                logging.info(f"  任务准确率序列: {[f'{acc:.2f}%' for acc in incremental_metrics_0['all_task_accuracies']]}")
        
        # === 阶段2: 训练形态1（利用Memory Pool） ===
        logging.info(f"\n--- 阶段2: 训练形态1 (Task {task1_idx}) ---")
        
        # 训练形态1
        logging.info(f"\n{'='*50}")
        logging.info(f"开始形态1训练 - 形态对{pair_idx}")
        logging.info(f"{'='*50}")
        
        # 在训练前报告Memory Pool状态
        model.report_memory_pool_status(pair_idx)
        
        stage1_start_time = time.time()
        model.incremental_train(
            data_manager, task_idx=task1_idx, 
            morphology_stage=1, pair_idx=pair_idx
        )
        stage1_time = time.time() - stage1_start_time
        
        # 训练后再次报告Memory Pool状态
        logging.info(f"\n形态1训练完成后的Memory Pool状态:")
        model.report_memory_pool_status(pair_idx)
        
        # 计算形态演化性能
        evolution_acc = model.eval_morphology_evolution(data_manager, pair_idx)
        logging.info(f"形态对{pair_idx}演化准确率: {evolution_acc:.2f}%")
        
        # 评估形态1
        eval_results_1 = model.eval_task()
        cnn_accy_1 = eval_results_1[0]
        incremental_metrics_1 = eval_results_1[-1] if len(eval_results_1) > 6 else None
        model.after_task()
        
        logging.info(f"形态1训练耗时: {stage1_time:.2f}秒")
        logging.info(f"形态1 Top-1准确率: {cnn_accy_1['top1']:.2f}%")
        
        # 输出增量学习指标
        if incremental_metrics_1:
            logging.info(f"=== 形态1增量学习指标 ===")
            logging.info(f"  当前任务累积准确率: {incremental_metrics_1['current_task_acc']:.2f}%")
            logging.info(f"  平均增量准确率: {incremental_metrics_1['average_incremental_acc']:.2f}%")
            logging.info(f"  遗忘率: {incremental_metrics_1['forgetting_rate']:.2f}%")
            logging.info(f"  后向迁移: {incremental_metrics_1['backward_transfer']:.2f}%")
            if incremental_metrics_1['all_task_accuracies']:
                logging.info(f"  任务准确率序列: {[f'{acc:.2f}%' for acc in incremental_metrics_1['all_task_accuracies']]}")
        
        # === 阶段3: 形态演化评估 ===
        logging.info(f"\n--- 阶段3: 形态演化评估 ---")
        
        # 记录形态对性能（包含增量学习指标）
        pair_performance = {
            'pair_idx': pair_idx,
            'morphology_0_acc': cnn_accy_0['top1'],
            'morphology_1_acc': cnn_accy_1['top1'],
            'evolution_acc': evolution_acc,
            'avg_acc': (cnn_accy_0['top1'] + cnn_accy_1['top1']) / 2,
            # 增量学习指标
            'morph0_cumulative_acc': incremental_metrics_0['current_task_acc'] if incremental_metrics_0 else 0.0,
            'morph1_cumulative_acc': incremental_metrics_1['current_task_acc'] if incremental_metrics_1 else 0.0,
            'morph0_avg_inc_acc': incremental_metrics_0['average_incremental_acc'] if incremental_metrics_0 else 0.0,
            'morph1_avg_inc_acc': incremental_metrics_1['average_incremental_acc'] if incremental_metrics_1 else 0.0,
            'morph0_forgetting': incremental_metrics_0['forgetting_rate'] if incremental_metrics_0 else 0.0,
            'morph1_forgetting': incremental_metrics_1['forgetting_rate'] if incremental_metrics_1 else 0.0
        }
        morphology_performance.append(pair_performance)
        
        # 更新整体性能曲线
        cnn_curve["top1"].extend([cnn_accy_0['top1'], cnn_accy_1['top1']])
        if "top5" in cnn_accy_0 and "top5" in cnn_accy_1:
            cnn_curve["top5"].extend([cnn_accy_0['top5'], cnn_accy_1['top5']])
        
        # 计算当前总时间
        pair_total_time = time.time() - pair_start_time
        logging.info(f"形态对{pair_idx+1}总耗时: {pair_total_time:.2f}秒 ({pair_total_time/60:.2f}分钟)")
        
        # === Memory Pool状态报告 ===
        if hasattr(model, 'report_memory_pool_status'):
            logging.info(f"\n--- Memory Pool状态报告 ---")
            model.report_memory_pool_status(pair_idx)
        
        # 阶段性总结（包含增量学习指标）
        logging.info(f"\n--- 形态对{pair_idx+1}完成总结 ---")
        logging.info(f"形态0→形态1准确率变化: {cnn_accy_0['top1']:.2f}% → {cnn_accy_1['top1']:.2f}%")
        
        if incremental_metrics_1:
            logging.info(f"=== 关键增量学习指标 ===")
            logging.info(f"  [核心] 总体累积准确率: {incremental_metrics_1['current_task_acc']:.2f}%")
            logging.info(f"  [趋势] 平均增量准确率: {incremental_metrics_1['average_incremental_acc']:.2f}%")
            logging.info(f"  [遗忘] 遗忘率: {incremental_metrics_1['forgetting_rate']:.2f}%")
            logging.info(f"  [演化] Memory Pool演化准确率: {evolution_acc*100:.2f}%")
        
        logging.info(f"形态对平均准确率: {pair_performance['avg_acc']:.2f}%")
        
        # 计算累积平均准确率
        if len(cnn_curve["top1"]) > 0:
            cumulative_avg = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
            logging.info(f"累积平均准确率: {cumulative_avg:.2f}%")
    
    # === 最终结果总结 ===
    logging.info(f"\n{'='*60}")
    logging.info("=== 形态演化增量学习完成 ===")
    
    # 总体性能统计
    final_avg_acc = sum(cnn_curve["top1"]) / len(cnn_curve["top1"]) if cnn_curve["top1"] else 0.0
    logging.info(f"最终平均准确率: {final_avg_acc:.2f}%")
    logging.info(f"完整准确率曲线: {[f'{acc:.2f}%' for acc in cnn_curve['top1']]}")
    
    # 形态演化性能总结（包含增量学习指标）
    if morphology_performance:
        avg_morph0_acc = np.mean([p['morphology_0_acc'] for p in morphology_performance])
        avg_morph1_acc = np.mean([p['morphology_1_acc'] for p in morphology_performance])
        avg_evolution_acc = np.mean([p['evolution_acc'] for p in morphology_performance])
        
        # 增量学习指标总结
        final_cumulative_acc = morphology_performance[-1]['morph1_cumulative_acc'] if morphology_performance else 0.0
        final_avg_inc_acc = morphology_performance[-1]['morph1_avg_inc_acc'] if morphology_performance else 0.0
        final_forgetting = morphology_performance[-1]['morph1_forgetting'] if morphology_performance else 0.0
        
        logging.info(f"\n=== [最终结果] 增量学习性能总结 ===")
        logging.info(f"[关键指标] 最终总体累积准确率: {final_cumulative_acc:.2f}%")
        logging.info(f"[平均性能] 最终平均增量准确率: {final_avg_inc_acc:.2f}%")
        logging.info(f"[遗忘程度] 最终遗忘率: {final_forgetting:.2f}%")
        logging.info(f"[演化能力] 平均Memory Pool演化准确率: {avg_evolution_acc*100:.2f}%")
        
        logging.info(f"\n=== [形态学习] 性能总结 ===")
        logging.info(f"平均形态0准确率: {avg_morph0_acc:.2f}%")
        logging.info(f"平均形态1准确率: {avg_morph1_acc:.2f}%")
        logging.info(f"形态演化改进: {(avg_morph1_acc - avg_morph0_acc):.2f}%")
    
    # 保存详细结果
    results_path = os.path.join(logs_name, f"morphology_results_{timestamp}.npz")
    np.savez(results_path, 
             cnn_curve=cnn_curve,
             morphology_performance=morphology_performance,
             final_avg_acc=final_avg_acc)
    logging.info(f"详细结果已保存至: {results_path}")
    
    logging.info("="*60)


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