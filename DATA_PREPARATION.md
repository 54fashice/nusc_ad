# 数据集准备说明文档

## 概述

本项目需要构建一个基于 nuScenes 数据集的多模态自动驾驶轨迹修正数据集，支持两轮对话形式的训练和推理。

## 数据集结构

```
data/
├── train/
│   ├── qa/                    # 问答对话数据
│   │   ├── {sample_token}_qa.json
│   │   └── ...
│   └── samples/               # 多模态原始数据
│       ├── surround_stitched/ # 环视拼接图像
│       │   ├── {sample_token}_surround.jpg
│       │   └── ...
│       ├── point_clouds/      # 点云数据（可选）
│       │   ├── {sample_token}.bin
│       │   └── ...
│       └── vector_maps/       # 矢量地图数据（可选）
│           ├── {sample_token}_vectors.json
│           └── ...
├── val/
│   ├── qa/
│   └── samples/
└── test/
    ├── qa/
    └── samples/
```

## 核心数据格式

### 1. 问答对话数据 (`{sample_token}_qa.json`)

每个样本的对话数据应包含两轮对话：

```json
{
  "sample_token": "00889f8a9549450aa2f32cf310a3e305",
  "scene_token": "0103c48e40f541248adb6322f1956f0c",
  "timestamp": 1531016314047991,
  
  // 第一轮：场景理解和行为评估
  "round1_conversation": [
    {
      "role": "user",
      "content": "请分析这个自动驾驶场景。当前车辆的驾驶行为是否合理？有什么潜在风险？"
    },
    {
      "role": "assistant", 
      "content": "根据场景分析，当前车辆正在进行左转操作。从多视角图像可以看到前方有行人正在过马路，车辆减速是合理的。但是从BEV特征显示车辆与右侧车道线距离较近，建议保持更大的安全距离。总体而言，驾驶行为基本合理，但需要注意行人动态。"
    }
  ],
  
  // 第二轮：轨迹修正（如果有的话）
  "round2_conversation": [
    {
      "role": "user", 
      "content": "这是其他算法生成的初始轨迹，请帮我修正并给出详细解释。"
    },
    {
      "role": "assistant",
      "content": "分析初始轨迹后，我发现以下问题：1）转弯半径过小，可能导致乘客不适；2）在行人附近没有充分减速；3）与右侧车道过于接近。修正后的轨迹采用更大的转弯半径，在行人区域增加减速点，并向左侧调整以保持安全距离。修正后的轨迹更加平滑和安全。"
    }
  ],
  
  // 轨迹数据（用于第二轮）
  "initial_trajectory": [
    [2.5, 0.0], [4.8, 0.5], [7.0, 1.2], 
    [9.0, 2.1], [10.8, 3.5], [12.2, 5.2]
  ],
  "corrected_trajectory": [
    [2.5, 0.0], [4.6, 0.3], [6.8, 0.8], 
    [8.8, 1.6], [10.6, 2.8], [12.0, 4.5]
  ],
  
  // 元数据
  "metadata": {
    "weather": "clear",
    "time_of_day": "day", 
    "location": "intersection",
    "ego_vehicle_state": {
      "position": [683.158, 1592.002],
      "heading": 2.832,
      "velocity": 4.2
    },
    "has_trajectory_correction": true
  }
}
```

### 2. 环视拼接图像

- **文件名**: `{sample_token}_surround.jpg`
- **格式**: RGB 图像，建议尺寸 1920x320 或类似比例
- **内容**: 6个环视摄像头的拼接图像（前、前左、左、后、右、前右）
- **标准化**: 像素值 [0, 255]，RGB 格式

### 3. nuScenes 集成数据（可选但推荐）

从 nuScenes 数据库提取的原始数据：

```json
{
  "sample_token": "00889f8a9549450aa2f32cf310a3e305",
  "nuScenes_data": {
    "sensor_data": {
      "CAM_FRONT": {
        "filename": "samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531016314047991.jpg",
        "calibration": {
          "camera_intrinsic": [[1142.0, 0.0, 800.0], [0.0, 1142.0, 450.0], [0.0, 0.0, 1.0]],
          "rotation": [0.0, 0.0, 0.0, 1.0],
          "translation": [1.70, 0.0, 1.54]
        }
      },
      // ... 其他 5 个摄像头
      "LIDAR_TOP": {
        "filename": "samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531016314047991.pcd.bin",
        "calibration": {
          "rotation": [0.0, 0.0, 0.0, 1.0],
          "translation": [0.0, 0.0, 1.84]
        }
      }
    },
    "annotations": [
      {
        "category_name": "vehicle.car",
        "instance_token": "ef20db0c866d4937a0de8e7bb36d6af4",
        "translation": [678.8, 1592.4, 0.3],
        "size": [1.73, 4.18, 1.56],
        "rotation": [0.0, 0.0, 0.8, 0.6],
        "velocity": [0.0, 0.0]
      }
      // ... 其他标注
    ],
    "map_data": {
      "lane_data": [
        {
          "lane_token": "lane_123",
          "centerline": [[680.1, 1590.2], [682.3, 1591.5], [684.8, 1593.1]],
          "left_boundary": [[679.6, 1592.0], [681.8, 1593.3], [684.3, 1594.9]],
          "right_boundary": [[680.6, 1588.4], [682.8, 1589.7], [685.3, 1591.3]]
        }
      ]
    }
  }
}
```

## 数据处理要求

### 多模态输入格式

#### 1. 图像数据 (`pixel_values`)
- **形状**: `[batch_size, num_views, channels, height, width]`
- **数值范围**: [0.0, 1.0] (归一化后)
- **视角数量**: 6 (前、前左、左、后、右、前右)
- **推荐尺寸**: 224x224 或 256x256

#### 2. 点云数据 (`points`) - 用于 BEV 特征提取
- **格式**: List[torch.Tensor]，每个tensor形状为 `[num_points, 5]`
- **维度含义**: [x, y, z, intensity, timestamp]
- **坐标系**: LiDAR坐标系
- **点云范围**: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

#### 3. 矢量数据

**车道线数据 (`lane_data`)**:
- **形状**: `[batch_size, max_lanes, max_points_per_lane, 4]`
- **维度含义**: [x, y, lane_id, lane_type]
- **坐标系**: 全局坐标系 (m)

**轨迹数据 (`trajectory_data`)**:
- **形状**: `[batch_size, max_agents, max_timesteps, 6]`
- **维度含义**: [x, y, vx, vy, heading, timestamp]

**智能体数据 (`agent_data`)**:
- **形状**: `[batch_size, max_agents, 5]`
- **维度含义**: [x, y, heading, velocity, agent_type]

#### 4. 轨迹标注

**初始轨迹 (`initial_trajectory`)**:
- **形状**: `[batch_size, num_waypoints, 2]`
- **维度含义**: [x, y] (相对坐标，单位：米)

**目标轨迹 (`gt_trajectory`)**:
- **形状**: `[batch_size, num_waypoints, 2]`  
- **维度含义**: [x, y] (相对坐标，单位：米)

## 数据集构建步骤

### 1. 从 nuScenes 提取原始数据

```python
# 示例代码
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-trainval', dataroot='/path/to/nuscenes', verbose=True)

def extract_sample_data(sample_token):
    sample = nusc.get('sample', sample_token)
    
    # 提取多视角图像
    cam_data = {}
    for cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
        cam_token = sample['data'][cam]
        cam_sample = nusc.get('sample_data', cam_token)
        cam_data[cam] = {
            'filename': cam_sample['filename'],
            'calibration': nusc.get('calibrated_sensor', cam_sample['calibrated_sensor_token'])
        }
    
    # 提取点云数据
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_sample = nusc.get('sample_data', lidar_token)
    
    # 提取标注数据
    annotations = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        annotations.append(ann)
    
    return {
        'camera_data': cam_data,
        'lidar_data': lidar_sample,
        'annotations': annotations
    }
```

### 2. 生成环视拼接图像

```python
import cv2
import numpy as np

def create_surround_view(cam_images, output_size=(1920, 320)):
    """
    将6个摄像头图像拼接成环视图
    """
    # 按顺序排列：前、前左、左、后、右、前右
    ordered_views = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 
                     'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
    
    resized_images = []
    view_width = output_size[0] // 6
    
    for view in ordered_views:
        img = cv2.imread(cam_images[view])
        img_resized = cv2.resize(img, (view_width, output_size[1]))
        resized_images.append(img_resized)
    
    surround_image = np.hstack(resized_images)
    return surround_image
```

### 3. 构建对话数据

对于每个样本，需要构建两轮对话：

**第一轮 - 场景理解**：
- 分析当前交通情况
- 评估自车行为合理性
- 识别潜在风险
- 基于多模态信息给出综合判断

**第二轮 - 轨迹修正**：
- 接收初始轨迹（可以是规划算法输出或添加噪声的真值轨迹）
- 分析轨迹问题
- 输出修正后的轨迹
- 提供详细的修正解释

### 4. 数据集验证

确保数据集满足以下要求：
- [ ] 所有样本都有对应的环视图像
- [ ] 对话文本格式正确，包含必要的角色标记
- [ ] 轨迹数据格式正确，坐标合理
- [ ] 多模态数据维度匹配
- [ ] 训练/验证/测试集划分合理

## 推荐的数据集大小

- **训练集**: 50,000-100,000 样本
- **验证集**: 5,000-10,000 样本  
- **测试集**: 5,000-10,000 样本

## 注意事项

1. **坐标系统一**: 确保所有空间数据使用相同的坐标系
2. **数据清洗**: 移除异常样本（如传感器故障、极端天气等）
3. **隐私保护**: 确保图像中的隐私信息得到适当处理
4. **版权合规**: 遵守nuScenes数据集的使用协议
5. **质量控制**: 建议对对话数据进行人工审核，确保质量

## 扩展选项

- **多语言支持**: 可以构建中英文双语版本
- **更多模态**: 添加雷达数据、GPS数据等
- **动态场景**: 包含更长时间序列的动态信息
- **异常情况**: 专门构建异常驾驶场景的数据集