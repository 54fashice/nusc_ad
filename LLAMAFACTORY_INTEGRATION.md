# LLaMA-Factory 使用指南

## 安装和环境设置

### 1. 安装 LLaMA-Factory

```bash
# 克隆 LLaMA-Factory 仓库
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 安装依赖
pip install -e .

# 或者直接安装
pip install llamafactory
```

### 2. 集成自定义模型

将我们的自定义模型集成到 LLaMA-Factory 中：

```bash
# 在 LLaMA-Factory 目录下创建符号链接
ln -s /storage/data-acc/kaile.yang/nusc_ad/src/model ./src/llamafactory/extras/trajectory_model
ln -s /storage/data-acc/kaile.yang/nusc_ad/src/trainer ./src/llamafactory/extras/trajectory_trainer
ln -s /storage/data-acc/kaile.yang/nusc_ad/src/data_handler ./src/llamafactory/extras/trajectory_data
```

## 数据集配置

### 1. 注册自定义数据集

在 LLaMA-Factory 的 `data/dataset_info.json` 中添加：

```json
{
  "nuscenes_trajectory": {
    "hf_hub_url": null,
    "ms_hub_url": null,
    "script_url": "/storage/data-acc/kaile.yang/nusc_ad/src/data_handler/data_collator.py",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversation",
      "system": "system",
      "tools": "tools"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system",
      "function_tag": "function",
      "observation_tag": "observation"
    },
    "multimodal": true
  }
}
```

### 2. 自定义数据集加载器

在 `src/llamafactory/data/loader.py` 中注册我们的数据集：

```python
# 添加到 get_dataset 函数中
if dataset_attr.dataset_name == "nuscenes_trajectory":
    from llamafactory.extras.trajectory_data.data_collator import NuScenesTrajectoryDataset
    dataset = NuScenesTrajectoryDataset(
        data_dir=dataset_attr.load_from,
        split=data_args.split,
        tokenizer=tokenizer,
        max_length=data_args.cutoff_len,
        enable_modalities=getattr(training_args, 'enable_modalities', {
            'image': True, 'bev': True, 'vector': True
        })
    )
```

## 训练配置

### 1. 预训练阶段

```yaml
# llamafactory_pretrain.yaml
### model
model_name_or_path: meta-llama/Llama-2-7b-chat-hf
use_fast_tokenizer: true
torch_dtype: float16

### method
stage: pt
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05

### custom model
use_custom_model: true
custom_model_name: trajectory_llama
custom_model_path: llamafactory.extras.trajectory_model.custom_model
custom_trainer_path: llamafactory.extras.trajectory_trainer.custom_trainer

### dataset
dataset: nuscenes_trajectory
template: llama2
cutoff_len: 1024
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/trajectory_llama2-7b/pretrain
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5.0e-05
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true

### custom training settings
training_stage: pretrain
enable_modalities:
  image: true
  bev: true
  vector: true

# 损失权重 (预训练阶段侧重对齐)
lm_loss_weight: 0.5
traj_loss_weight: 0.1
align_loss_weight: 1.0
bev_text_loss_weight: 0.5
bev_vector_loss_weight: 0.5

# 多模态设置
use_bev_features: true
use_vectorized_inputs: true
num_camera_views: 6
freeze_vision_tower: true
freeze_llm: false
freeze_bevfusion: true

# BEVFusion配置
bevfusion_config_path: /storage/data-acc/kaile.yang/nusc_ad/libs/mmdetection3d-1.4.0/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
bevfusion_checkpoint_path: null

# 点云和BEV设置
point_cloud_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size: [0.2, 0.2, 8.0]
bev_size: [180, 180]
```

### 2. 监督微调阶段

```yaml
# llamafactory_sft.yaml
### model
model_name_or_path: saves/trajectory_llama2-7b/pretrain
use_fast_tokenizer: true
torch_dtype: float16

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05

### custom model
use_custom_model: true
custom_model_name: trajectory_llama
custom_model_path: llamafactory.extras.trajectory_model.custom_model
custom_trainer_path: llamafactory.extras.trajectory_trainer.custom_trainer

### dataset
dataset: nuscenes_trajectory
template: llama2
cutoff_len: 1024
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/trajectory_llama2-7b/sft
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-04
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true

### eval
val_size: 0.1
evaluation_strategy: steps
eval_steps: 200
per_device_eval_batch_size: 1

### custom training settings
training_stage: sft
enable_modalities:
  image: true
  bev: true
  vector: true

# 损失权重 (SFT阶段平衡训练)
lm_loss_weight: 1.0
traj_loss_weight: 1.5
align_loss_weight: 0.3
bev_text_loss_weight: 0.2
bev_vector_loss_weight: 0.2

# 多模态设置
use_bev_features: true
use_vectorized_inputs: true
num_camera_views: 6
freeze_vision_tower: true
freeze_llm: false
freeze_bevfusion: true
```

### 3. 全参数微调阶段

```yaml
# llamafactory_full_finetune.yaml
### model
model_name_or_path: saves/trajectory_llama2-7b/sft
use_fast_tokenizer: true
torch_dtype: float16

### method
stage: sft
do_train: true
finetuning_type: full
# 移除LoRA设置

### custom model
use_custom_model: true
custom_model_name: trajectory_llama
custom_model_path: llamafactory.extras.trajectory_model.custom_model
custom_trainer_path: llamafactory.extras.trajectory_trainer.custom_trainer

### dataset
dataset: nuscenes_trajectory
template: llama2
cutoff_len: 1024
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/trajectory_llama2-7b/full_finetune
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
learning_rate: 1.0e-05
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
fp16: true
gradient_checkpointing: true

### custom training settings
training_stage: full_finetune
enable_modalities:
  image: true
  bev: true
  vector: true

# 损失权重 (全微调侧重轨迹质量)
lm_loss_weight: 1.0
traj_loss_weight: 2.0
align_loss_weight: 0.1
bev_text_loss_weight: 0.1
bev_vector_loss_weight: 0.1

# 多模态设置 (解冻更多组件)
use_bev_features: true
use_vectorized_inputs: true
num_camera_views: 6
freeze_vision_tower: false
freeze_llm: false
freeze_bevfusion: false
```

## 训练命令

### 1. 预训练

```bash
llamafactory-cli train llamafactory_pretrain.yaml
```

### 2. 监督微调

```bash
llamafactory-cli train llamafactory_sft.yaml
```

### 3. 全参数微调

```bash
llamafactory-cli train llamafactory_full_finetune.yaml
```

### 4. 分布式训练

```bash
# 多GPU训练
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train llamafactory_sft.yaml

# 使用 torchrun
torchrun --nproc_per_node=4 -m llamafactory.train llamafactory_sft.yaml
```

## 推理配置

### 1. API 服务器启动

```yaml
# llamafactory_api.yaml
model_name_or_path: saves/trajectory_llama2-7b/sft
template: llama2
finetuning_type: lora
use_custom_model: true
custom_model_name: trajectory_llama
custom_model_path: llamafactory.extras.trajectory_model.custom_model

# 推理参数
temperature: 0.7
top_p: 0.9
max_new_tokens: 512
```

启动API服务：

```bash
llamafactory-cli api llamafactory_api.yaml
```

### 2. Web UI 启动

```bash
llamafactory-cli webui
```

### 3. 命令行推理

```bash
llamafactory-cli chat llamafactory_api.yaml
```

## 消融实验配置

### 1. 仅视觉模态

```yaml
# llamafactory_ablation_vision_only.yaml
# ... 基础配置同 SFT ...

### custom training settings
enable_modalities:
  image: true
  bev: false
  vector: false

use_bev_features: false
use_vectorized_inputs: false

# 损失权重调整
lm_loss_weight: 1.0
traj_loss_weight: 1.5
align_loss_weight: 0.3
bev_text_loss_weight: 0.0
bev_vector_loss_weight: 0.0

output_dir: saves/trajectory_llama2-7b/ablation_vision_only
```

### 2. 仅BEV模态

```yaml
# llamafactory_ablation_bev_only.yaml
# ... 基础配置同 SFT ...

### custom training settings
enable_modalities:
  image: false
  bev: true
  vector: false

use_bev_features: true
use_vectorized_inputs: false

# 损失权重调整
lm_loss_weight: 1.0
traj_loss_weight: 1.5
align_loss_weight: 0.0
bev_text_loss_weight: 0.3
bev_vector_loss_weight: 0.0

output_dir: saves/trajectory_llama2-7b/ablation_bev_only
```

## 评估和测试

### 1. 模型评估

```bash
llamafactory-cli eval \
    --model_name_or_path saves/trajectory_llama2-7b/sft \
    --template llama2 \
    --dataset nuscenes_trajectory \
    --split test \
    --lang en \
    --batch_size 1
```

### 2. 轨迹质量评估

创建自定义评估脚本：

```python
# eval_trajectory.py
from llamafactory.api import ChatModel

# 加载模型
chat_model = ChatModel({
    "model_name_or_path": "saves/trajectory_llama2-7b/sft",
    "template": "llama2",
    "finetuning_type": "lora"
})

# 评估轨迹修正质量
def evaluate_trajectory_correction(test_samples):
    results = []
    for sample in test_samples:
        response = chat_model.chat(
            messages=sample["messages"],
            multimodal_data=sample.get("multimodal_data")
        )
        
        # 提取修正后的轨迹
        corrected_trajectory = extract_trajectory_from_response(response)
        
        # 计算评估指标
        metrics = compute_trajectory_metrics(
            predicted=corrected_trajectory,
            ground_truth=sample["gt_trajectory"]
        )
        results.append(metrics)
    
    return results
```

## 注意事项

1. **自定义模型集成**: 需要修改 LLaMA-Factory 的模型注册机制来支持我们的 `TrajectoryLlamaForCausalLM`

2. **数据集集成**: 需要在 LLaMA-Factory 中注册自定义数据集加载器

3. **多模态支持**: 确保 LLaMA-Factory 的训练循环能够处理多模态数据

4. **内存优化**: 在训练大模型时，建议使用 gradient checkpointing 和适当的批次大小

5. **监控训练**: 使用 wandb 或 tensorboard 监控训练过程中的多个损失组件

6. **模型保存**: 确保保存包含自定义组件的完整模型权重