#!/usr/bin/env python3
"""
训练脚本 - 多阶段训练自动驾驶轨迹修正模型

使用方法:
python train.py --config src/configs/pretrain_config.yaml
python train.py --config src/configs/sft_config.yaml  
python train.py --config src/configs/full_finetune_config.yaml

支持的训练阶段:
1. pretrain - 多模态对齐预训练
2. sft - 监督微调 
3. full_finetune - 端到端全参数微调
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import (
    AutoTokenizer, 
    TrainingArguments,
    set_seed
)

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model.custom_model import TrajectoryLlamaForCausalLM, TrajectoryLlamaConfig
from src.trainer.custom_trainer import TrajectoryLlamaTrainer
from src.data_handler.data_collator import NuScenesTrajectoryDataset, MultiModalDataCollator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_model_and_tokenizer(config: Dict[str, Any]):
    """初始化模型和分词器"""
    logger.info("正在初始化模型和分词器...")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name_or_path'],
        trust_remote_code=True,
        use_fast=True
    )
    
    # 确保有pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 添加特殊token用于多模态输入
    special_tokens = ['<image>', '<bev>', '<vector>', '<trajectory>']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    # 创建模型配置
    model_config = TrajectoryLlamaConfig.from_pretrained(
        config['model_name_or_path'],
        trajectory_output_dim=config.get('trajectory_output_dim', 12),
        num_trajectory_points=config.get('num_trajectory_points', 6),
        # 损失权重
        lm_loss_weight=config.get('lm_loss_weight', 1.0),
        traj_loss_weight=config.get('traj_loss_weight', 1.5),
        align_loss_weight=config.get('align_loss_weight', 0.1),
        bev_text_loss_weight=config.get('bev_text_loss_weight', 0.1),
        bev_vector_loss_weight=config.get('bev_vector_loss_weight', 0.1),
        # 多模态设置
        vision_tower_name=config.get('vision_tower', 'openai/clip-vit-large-patch14'),
        num_camera_views=config.get('num_camera_views', 6),
        use_bev_features=config.get('use_bev_features', True),
        use_vectorized_inputs=config.get('use_vectorized_inputs', True),
        freeze_bevfusion=config.get('freeze_bevfusion', True),
        # BEV设置
        bevfusion_config_path=config.get('bevfusion_config_path'),
        bevfusion_checkpoint_path=config.get('bevfusion_checkpoint_path'),
        point_cloud_range=config.get('point_cloud_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
        voxel_size=config.get('voxel_size', [0.2, 0.2, 8.0]),
        bev_size=config.get('bev_size', [180, 180]),
        # 训练设置  
        freeze_vision_tower=config.get('freeze_vision_tower', True),
        freeze_llm=config.get('freeze_llm', False),
        enable_ablation=config.get('enable_ablation', True),
        # 特殊token索引
        image_token_index=len(tokenizer) - 4,
        bev_token_index=len(tokenizer) - 3,
        vector_token_index=len(tokenizer) - 2,
        trajectory_token_index=len(tokenizer) - 1
    )
    
    # 创建模型
    model = TrajectoryLlamaForCausalLM(model_config)
    
    # 调整词汇表大小
    model.resize_token_embeddings(len(tokenizer))
    
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    logger.info(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
    
    return model, tokenizer, model_config

def setup_datasets(config: Dict[str, Any], tokenizer):
    """设置数据集"""
    logger.info("正在加载数据集...")
    
    # 模态设置
    enable_modalities = config.get('enable_modalities', {
        'image': True, 'bev': True, 'vector': True
    })
    
    # 训练数据集
    train_dataset = NuScenesTrajectoryDataset(
        data_dir=config['dataset_dir'],
        split='train',
        tokenizer=tokenizer,
        max_length=config.get('cutoff_len', 1024),
        enable_modalities=enable_modalities,
        mode='both'  # 包含两轮对话
    )
    
    # 验证数据集
    eval_dataset = None
    eval_dir = config['dataset_dir'].replace('/train', '/val')
    if os.path.exists(eval_dir):
        eval_dataset = NuScenesTrajectoryDataset(
            data_dir=eval_dir,
            split='val',
            tokenizer=tokenizer,
            max_length=config.get('cutoff_len', 1024),
            enable_modalities=enable_modalities,
            mode='both'
        )
    
    logger.info(f"训练样本数: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"验证样本数: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset

def setup_training_args(config: Dict[str, Any]) -> TrainingArguments:
    """设置训练参数"""
    return TrainingArguments(
        output_dir=config['output_dir'],
        overwrite_output_dir=config.get('overwrite_output_dir', True),
        do_train=config.get('do_train', True),
        do_eval=config.get('do_eval', False),
        
        # 数据设置
        per_device_train_batch_size=config.get('per_device_train_batch_size', 1),
        per_device_eval_batch_size=config.get('per_device_eval_batch_size', 1),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 8),
        
        # 学习率设置
        learning_rate=config.get('learning_rate', 1e-4),
        lr_scheduler_type=config.get('lr_scheduler_type', 'cosine'),
        warmup_ratio=config.get('warmup_ratio', 0.1),
        
        # 训练步数
        num_train_epochs=config.get('num_train_epochs', 3.0),
        max_steps=config.get('max_steps', -1),
        
        # 保存和评估
        save_strategy=config.get('save_strategy', 'steps'),
        save_steps=config.get('save_steps', 500),
        evaluation_strategy=config.get('eval_strategy', 'steps'),
        eval_steps=config.get('eval_steps', 500),
        save_total_limit=config.get('save_total_limit', 3),
        
        # 日志
        logging_strategy='steps',
        logging_steps=config.get('logging_steps', 10),
        logging_first_step=config.get('logging_first_step', True),
        
        # 优化
        weight_decay=config.get('weight_decay', 0.01),
        adam_epsilon=config.get('adam_epsilon', 1e-8),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        
        # 精度
        fp16=config.get('fp16', True),
        bf16=config.get('bf16', False),
        
        # 其他
        dataloader_pin_memory=config.get('dataloader_pin_memory', False),
        remove_unused_columns=False,  # 保留多模态数据
        report_to=config.get('report_to', []),
        run_name=config.get('run_name'),
        
        # 内存优化
        gradient_checkpointing=config.get('gradient_checkpointing', False),
    )

def main():
    parser = argparse.ArgumentParser(description='训练轨迹修正模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP local rank')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    logger.info(f"加载配置: {args.config}")
    logger.info(f"训练阶段: {config.get('training_stage', 'sft')}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 初始化模型和分词器
    model, tokenizer, model_config = setup_model_and_tokenizer(config)
    
    # 设置数据集
    train_dataset, eval_dataset = setup_datasets(config, tokenizer)
    
    # 设置数据整理器
    data_collator = MultiModalDataCollator(
        tokenizer=tokenizer,
        model_config=model_config
    )
    
    # 设置训练参数
    training_args = setup_training_args(config)
    
    # 创建训练器
    trainer = TrajectoryLlamaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # 自定义参数
        training_stage=config.get('training_stage', 'sft'),
        enable_modalities=config.get('enable_modalities', {
            'image': True, 'bev': True, 'vector': True
        }),
        loss_weights={
            'lm_loss': config.get('lm_loss_weight', 1.0),
            'traj_loss': config.get('traj_loss_weight', 1.5),
            'align_loss': config.get('align_loss_weight', 0.1),
            'bev_text_loss': config.get('bev_text_loss_weight', 0.1),
            'bev_vector_loss': config.get('bev_vector_loss_weight', 0.1),
        }
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 保存最终模型
    logger.info("保存最终模型...")
    trainer.save_model()
    trainer.save_state()
    
    # 保存分词器
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("训练完成!")

if __name__ == "__main__":
    main()