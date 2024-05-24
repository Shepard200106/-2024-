import logging
import os
import json
import random
import torch
import numpy as np

from models.arg import Trainer as ARGTrainer
from models.argd import Trainer as ARGDTrainer

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def frange(x, y, jump):
    while x < y:
        x = round(x, 8)
        yield x
        x += jump



class Run():
    def __init__(self, config, writer):
        self.config = config
        self.writer = writer

    def getFileLogger(self, log_file):
        # 获取文件日志记录器
        logger = logging.getLogger()
        if not logger.handlers:
            # 设置日志级别为 INFO
            logger.setLevel(logging.INFO)
            # 创建文件处理器并设置日志级别为 INFO
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            # 设置日志格式
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            # 将处理器添加到日志记录器
            logger.addHandler(handler)
        return logger

    def config2dict(self):
        # 将配置对象转换为字典
        config_dict = {}
        for k, v in self.config.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        # 获取参数日志目录，如果不存在则创建
        param_log_dir = self.config['param_log_dir']
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir)

        # 创建参数日志文件路径
        param_log_file = os.path.join(param_log_dir,
                                      self.config['model_name'] + '_' + self.config['data_name'] + '_' + 'param.txt')
        # 获取日志记录器
        logger = self.getFileLogger(param_log_file)

        # 定义训练参数
        train_param = {'lr': [self.config['lr']]}
        print(train_param)
        param = train_param
        best_param = []

        # 创建 JSON 日志目录和路径，如果不存在则创建目录
        json_dir = os.path.join('./logs/json/', self.config['model_name'] + '_' + self.config['data_name'])
        json_path = os.path.join(json_dir, 'month_' + str(self.config['month']) + '.json')
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        json_result = []
        for p, vs in param.items():
            setup_seed(self.config['seed'])
            best_metric = {'metric': 0}
            best_v = vs[0]
            best_model_path = None
            for i, v in enumerate(vs):
                self.config['lr'] = v

                # 根据模型名称选择训练器
                if self.config['model_name'] == 'ARG':
                    trainer = ARGTrainer(self.config, self.writer)
                elif self.config['model_name'] == 'ARG-D':
                    trainer = ARGDTrainer(self.config, self.writer)
                else:
                    raise ValueError('model_name is not supported')

                # 训练模型并获取指标、模型路径和训练周期数
                metrics, model_path, train_epochs = trainer.train(logger)
                json_result.append({
                    'lr': self.config['lr'],
                    'metric': metrics,
                    'train_epochs': train_epochs,
                })

                # 更新最佳指标和参数
                if metrics['metric'] > best_metric['metric']:
                    best_metric = metrics
                    best_v = v
                    best_model_path = model_path

            best_param.append({p: best_v})
            print("best model path:", best_model_path)
            print("best macro f1:", best_metric['metric'])
            print("best metric:", best_metric)
            logger.info("best model path:" + best_model_path)
            logger.info("best param " + p + ": " + str(best_v))
            logger.info("best metric:" + str(best_metric))
            logger.info('==================================================\n\n')

        # 将结果写入 JSON 文件
        with open(json_path, 'w') as file:
            json.dump(json_result, file, indent=4, ensure_ascii=False)

        return best_metric

