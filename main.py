import json
import argparse
import os
import torch
from trainer import train

# 在导入其他库之前设置 OpenMP 环境变量，抑制多重初始化警告
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 设置环境变量避免潜在的多进程问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    
    # 打印使用的GPU
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"可用GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # 如果GPU内存不足，可以设置内存分配上限
        torch.cuda.empty_cache()  # 清理GPU缓存
        # torch.cuda.set_per_process_memory_fraction(0.8)
    else:
        print("警告: 未检测到GPU，将使用CPU进行训练（速度会很慢）")
    
    # 启动训练
    train(args)

def load_json(settings_path):
    print(f"加载配置文件: {settings_path}")
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='虫态时序增量学习')
    
    parser.add_argument('--config', type=str, default='./exps/iiminsects202.json',
                        help='配置文件路径')
    parser.add_argument('--name', type=str, default='',
                        help='实验名称')
    parser.add_argument('--model_name', type=str, default='proof',
                        help='模型名称')
    parser.add_argument('--dataset', type=str, default='insect',
                        help='数据集名称')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU设备ID')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--lambda_rehearsal', type=float, default=0.1,
                        help='原型回放损失权重')
    parser.add_argument('--alpha_noise', type=float, default=0.1,
                        help='原型增强高斯噪声标准差')
    
    return parser

if __name__ == '__main__':
    main()
