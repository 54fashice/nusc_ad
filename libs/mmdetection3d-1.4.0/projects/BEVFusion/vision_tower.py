import torch
import torch.nn as nn
import sys
import os
from typing import Optional, Dict, List, Tuple

# MMDetection3D path setup
MMDET3D_PATH = '/storage/data-acc/kaile.yang/nusc_ad/libs/mmdetection3d-1.4.0'
if MMDET3D_PATH not in sys.path:
    sys.path.insert(0, MMDET3D_PATH)

# BEVFusion project path needs to be added for the import to work
BEVFUSION_PROJECT_PATH = os.path.join(MMDET3D_PATH, 'projects/BEVFusion')
if BEVFUSION_PROJECT_PATH not in sys.path:
    sys.path.insert(0, BEVFUSION_PROJECT_PATH)

try:
    from mmengine.config import Config
    from mmdet3d.registry import MODELS
    from mmengine.runner import load_checkpoint
    # Using register_all_modules is a robust way to ensure all components are registered.
    from mmdet3d.utils import register_all_modules

    # Initialize registries to make all modules accessible.
    # This is crucial for building models from configs.
    register_all_modules(init_default_scope=True)

    # This import is critical. It executes code in the BEVFusion project
    # that registers the custom "BEVFusion" model with the MODELS registry.
    import bevfusion  # noqa: F401 # Registers BEVFusion components
    print("Successfully imported mmdet3d and BEVFusion components.")
except ImportError as e:
    print(f"FATAL: Error importing mmdet3d components: {e}")
    print("This likely means mmdetection3d or its dependencies (e.g., mmengine, mmcv) "
          "are not properly installed or found in your Python path.")
    print(f"Please check your environment. Current MMDET3D_PATH: {MMDET3D_PATH}")
    # Re-raise the exception to halt execution, as the application cannot continue.
    raise e

from transformers.configuration_utils import PretrainedConfig
from transformers.models.clip.modeling_clip import CLIPVisionModel

class VectorFeatureEncoder(nn.Module):
    """
    矢量特征编码器，用于处理车道线、轨迹和其他矢量化数据。
    """
    # (您的原代码，未修改)
    def __init__(self, 
                 hidden_size: int = 512,
                 bev_size: Tuple[int, int] = (180, 180),
                 point_cloud_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bev_size = bev_size
        self.point_cloud_range = point_cloud_range
        
        self.lane_encoder = nn.Sequential(
            nn.Linear(2, hidden_size // 2),  # x, y
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(2, hidden_size // 2),  # x, y
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        self.agent_encoder = nn.Sequential(
            nn.Linear(5, hidden_size // 2),  # x, y, heading, vel, agent_type
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        self.bev_conv = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True)
        )
        
    def _rasterize_to_bev(self, vector_features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """将矢量特征栅格化到BEV表示"""
        B, N, _ = vector_features.shape
        H, W = self.bev_size
        
        bev_map = torch.zeros(B, self.hidden_size, H, W, 
                              device=vector_features.device, 
                              dtype=vector_features.dtype)
        
        x_min, y_min = self.point_cloud_range[0], self.point_cloud_range[1]
        bev_x = (positions[..., 0] - x_min) / (self.point_cloud_range[3] - x_min) * W
        bev_y = (positions[..., 1] - y_min) / (self.point_cloud_range[4] - y_min) * H
        
        for b in range(B):
            valid_mask = (bev_x[b] >= 0) & (bev_x[b] < W) & (bev_y[b] >= 0) & (bev_y[b] < H)
            if not valid_mask.any():
                continue
            
            x_coords, y_coords = bev_x[b][valid_mask].long(), bev_y[b][valid_mask].long()
            features = vector_features[b][valid_mask]
            
            bev_map[b].index_put_((y_coords, x_coords), features.permute(1, 0), accumulate=True)

        return self.bev_conv(bev_map)

    def forward(self, 
                lane_data: Optional[torch.Tensor] = None,
                agent_data: Optional[torch.Tensor] = None,
                initial_trajectory: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        all_features = []
        all_positions = []

        if lane_data is not None and lane_data.numel() > 0:
            B, N_lanes, N_points, _ = lane_data.shape
            lane_points = lane_data[..., :2].reshape(B, N_lanes * N_points, 2)
            encoded_lanes = self.lane_encoder(lane_points)
            all_features.append(encoded_lanes)
            all_positions.append(lane_points)

        if agent_data is not None and agent_data.numel() > 0:
            encoded_agents = self.agent_encoder(agent_data)
            all_features.append(encoded_agents)
            all_positions.append(agent_data[..., :2])

        if initial_trajectory is not None and initial_trajectory.numel() > 0:
            encoded_traj = self.trajectory_encoder(initial_trajectory)
            all_features.append(encoded_traj)
            all_positions.append(initial_trajectory)

        if not all_features:
            return None

        combined_features = torch.cat(all_features, dim=1)
        combined_positions = torch.cat(all_positions, dim=1)

        return self._rasterize_to_bev(combined_features, combined_positions)

class BEVFusionWrapper(nn.Module):
    """
    BEVFusion模型的封装器，使用MMDetection3D的注册表机制进行正确调用。
    修复了调用方式，使用registry构建模型。
    """
    
    def __init__(self, 
                 config_path: str,
                 checkpoint_path: Optional[str] = None,
                 freeze_model: bool = True):
        super().__init__()

        # The import check is now handled robustly at the top level of the file.
        # If imports fail, the program will raise an exception and stop.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._init_bevfusion(config_path, checkpoint_path)
        
        if freeze_model:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        
    def _init_bevfusion(self, config_path: str, checkpoint_path: Optional[str]) -> nn.Module:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"BEVFusion config file not found at: {config_path}")

        # Changing the current working directory is a common practice for mmdet
        # configs because they often use relative paths (_base_).
        old_cwd = os.getcwd()
        try:
            os.chdir(MMDET3D_PATH)
            
            # Now, Config, MODELS, and load_checkpoint are guaranteed to be imported correctly.
            config = Config.fromfile(config_path, import_custom_modules=False)
            
            model = MODELS.build(config.model)
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                load_checkpoint(model, checkpoint_path, map_location='cpu')
                print(f"Successfully loaded BEVFusion checkpoint from {checkpoint_path}")
            else:
                print("Warning: BEVFusion checkpoint not provided or not found. Using randomly initialized weights.")

            model = model.to(self.device)
            print("BEVFusion model initialized successfully.")
            return model
        
        except Exception as e:
            print(f"Current working directory during error: {os.getcwd()}")
            print(f"Fatal: Failed to initialize BEVFusion model: {e}")
            raise e
        finally:
            os.chdir(old_cwd)
    
    def forward(self, 
                points: List[torch.Tensor],
                images: torch.Tensor,
                data_samples: List[Dict]) -> torch.Tensor:
        is_training = self.model.training
        if not is_training:
            self.model.eval()

        inputs = {
            'points': points,
            'img': images
        }
        
        with torch.no_grad() if not is_training else torch.enable_grad():
            bev_features = self.model.extract_feat(inputs, data_samples)

        if isinstance(bev_features, (list, tuple)):
            bev_features = bev_features[0]
            
        return bev_features

class VisionTower(nn.Module):
    """
    用于自动驾驶的多模态视觉模型，集成了BEV特征和矢量编码。
    添加了消融开关（如use_bev, use_vectors）。
    """
    
    def __init__(self, vision_tower_name_or_path: str, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.is_loaded = True

        self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name_or_path)
        self.hidden_size = self.vision_tower.config.hidden_size
        self.num_views = getattr(config, "num_camera_views", 6)
        self.view_embeddings = nn.Embedding(self.num_views, self.hidden_size)

        self.use_bev = getattr(config, "use_bev_features", True)
        if self.use_bev:
            bevfusion_config_path = getattr(config, "bevfusion_config_path", "/storage/data-acc/kaile.yang/nusc_ad/libs/mmdetection3d-1.4.0/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py")
            bevfusion_checkpoint_path = getattr(config, "bevfusion_checkpoint_path", None)  # 可以设为MIT链接或None
            self.bev_fusion = BEVFusionWrapper(
                config_path=bevfusion_config_path,
                checkpoint_path=bevfusion_checkpoint_path,
                freeze_model=getattr(config, "freeze_bevfusion", True)
            )
            bev_feature_dim = getattr(config, "bev_feature_dim", 256)  # 根据配置调整
            self.bev_proj = nn.Linear(bev_feature_dim, self.hidden_size)

        self.use_vectors = getattr(config, "use_vectorized_inputs", True)
        if self.use_vectors:
            self.vector_encoder = VectorFeatureEncoder(
                hidden_size=self.hidden_size,
                bev_size=getattr(config, "bev_size", (180, 180)),
                point_cloud_range=getattr(config, "point_cloud_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
            )
            self.vector_proj = nn.Linear(self.hidden_size, self.hidden_size)


    def _process_vision_with_views(self, pixel_values: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = pixel_values.shape
        pixel_values = pixel_values.view(b * n, c, h, w)
        
        image_forward_outs = self.vision_tower(pixel_values, output_hidden_states=True)
        image_features = image_forward_outs.last_hidden_state
        
        seq_len = image_features.shape[1]
        image_features = image_features.view(b, n, seq_len, self.hidden_size)
        
        view_ids = torch.arange(n, device=self.device).view(1, n, 1, 1)
        view_embeds = self.view_embeddings(view_ids).expand(-1, -1, seq_len, -1)
        image_features_with_views = image_features + view_embeds
        
        return image_features_with_views.view(b, n * seq_len, self.hidden_size)

    def forward(self, 
                pixel_values: torch.Tensor,
                points: Optional[List[torch.Tensor]] = None,
                data_samples: Optional[List[Dict]] = None,
                lane_data: Optional[torch.Tensor] = None,
                agent_data: Optional[torch.Tensor] = None,
                initial_trajectory: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        output_features = {}
        
        image_features = self._process_vision_with_views(pixel_values)
        output_features['image_features'] = image_features
        
        if self.use_bev and hasattr(self, 'bev_fusion'):
            if points is None or data_samples is None:
                raise ValueError("`points` and `data_samples` are required for BEVFusion.")
            
            bev_features = self.bev_fusion(
                points=points,
                images=pixel_values,
                data_samples=data_samples
            )
            
            bev_flattened = bev_features.flatten(2).transpose(1, 2)
            bev_projected = self.bev_proj(bev_flattened)
            output_features['bev_features'] = bev_projected
        
        if self.use_vectors and hasattr(self, 'vector_encoder'):
            vector_bev_map = self.vector_encoder(
                lane_data=lane_data,
                agent_data=agent_data,
                initial_trajectory=initial_trajectory
            )
            
            if vector_bev_map is not None:
                vector_flattened = vector_bev_map.flatten(2).transpose(1, 2)
                vector_projected = self.vector_proj(vector_flattened)
                output_features['vector_features'] = vector_projected
        
        return output_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device