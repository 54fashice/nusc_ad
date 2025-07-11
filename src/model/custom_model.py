import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from .trajectory_head import TrajectoryHead
from .qformer import MultiModalQFormer
from .vision_tower import VisionTower

class TrajectoryLlamaConfig(LlamaConfig):
    """
    Configuration class for Trajectory LLaMA model.
    """
    model_type = "trajectory_llama"
    
    def __init__(self, 
                 trajectory_output_dim: int = 12,
                 num_trajectory_points: int = 6,
                 # Loss weights
                 lm_loss_weight: float = 1.0,
                 traj_loss_weight: float = 1.0,
                 align_loss_weight: float = 0.1,
                 bev_text_loss_weight: float = 0.1,
                 bev_vector_loss_weight: float = 0.1,
                 # Multi-modal settings
                 vision_tower_name: str = "openai/clip-vit-large-patch14",
                 num_camera_views: int = 6,
                 use_bev_features: bool = True,
                 use_vectorized_inputs: bool = True,
                 freeze_bevfusion: bool = True,
                 # BEV settings
                 bevfusion_config_path: Optional[str] = None,
                 bevfusion_checkpoint_path: Optional[str] = None,
                 bev_size: Tuple[int, int] = (180, 180),
                 point_cloud_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 voxel_size: List[float] = [0.2, 0.2, 8.0],
                 # Training settings
                 freeze_vision_tower: bool = True,
                 freeze_llm: bool = False,
                 enable_ablation: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        
        # Trajectory settings
        self.trajectory_output_dim = trajectory_output_dim
        self.num_trajectory_points = num_trajectory_points
        
        # Loss weights
        self.lm_loss_weight = lm_loss_weight
        self.traj_loss_weight = traj_loss_weight
        self.align_loss_weight = align_loss_weight
        self.bev_text_loss_weight = bev_text_loss_weight
        self.bev_vector_loss_weight = bev_vector_loss_weight
        
        # Multi-modal settings
        self.vision_tower_name = vision_tower_name
        self.num_camera_views = num_camera_views
        self.use_bev_features = use_bev_features
        self.use_vectorized_inputs = use_vectorized_inputs
        self.freeze_bevfusion = freeze_bevfusion
        
        # BEV settings
        self.bevfusion_config_path = bevfusion_config_path
        self.bevfusion_checkpoint_path = bevfusion_checkpoint_path
        self.bev_size = bev_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        
        # Training settings
        self.freeze_vision_tower = freeze_vision_tower
        self.freeze_llm = freeze_llm
        self.enable_ablation = enable_ablation


class TrajectoryLlamaForCausalLM(LlamaForCausalLM):
    """
    Multi-modal LLaMA model for autonomous driving trajectory correction.
    
    This model combines:
    1. Multi-view camera images with view encoding
    2. BEV features from BEVFusion
    3. Vector features (lanes, trajectories, agents)
    4. LLaMA for text generation and reasoning
    5. Trajectory head for trajectory prediction
    """
    
    config_class = TrajectoryLlamaConfig

    def __init__(self, config: TrajectoryLlamaConfig):
        super(LlamaForCausalLM, self).__init__(config)
        
        # Core LLaMA components
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Trajectory prediction head
        self.trajectory_head = TrajectoryHead(
            hidden_size=config.hidden_size,
            output_dim=config.trajectory_output_dim
        )
        
        # Multi-modal Q-Former for feature alignment
        self.qformer = MultiModalQFormer(
            vision_hidden_size=config.hidden_size,
            llm_hidden_size=config.hidden_size,
            num_query_tokens=32
        )
        
        # Vision tower for multi-modal feature extraction
        self.vision_tower = VisionTower(
            vision_tower_name_or_path=config.vision_tower_name,
            config=config
        )
        
        # Loss configuration
        self.lm_loss_weight = config.lm_loss_weight
        self.traj_loss_weight = config.traj_loss_weight
        self.align_loss_weight = config.align_loss_weight
        self.bev_text_loss_weight = config.bev_text_loss_weight
        self.bev_vector_loss_weight = config.bev_vector_loss_weight
        
        # Loss functions
        self.traj_loss_fn = nn.MSELoss()
        
        # Freeze components as specified
        if config.freeze_vision_tower:
            self._freeze_vision_tower()
        if config.freeze_llm:
            self._freeze_llm()
        
        self.post_init()

    def _freeze_vision_tower(self):
        """Freeze vision tower parameters."""
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        print("Vision tower frozen")

    def _freeze_llm(self):
        """Freeze LLaMA parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False
        print("LLaMA frozen")

    def get_model(self):
        """Get the underlying model."""
        return self.model

    def prepare_inputs_labels_for_multimodal(self, 
                                           input_ids: torch.LongTensor,
                                           attention_mask: torch.Tensor,
                                           labels: torch.LongTensor,
                                           multi_modal_features: Dict[str, torch.Tensor]):
        """
        Prepare inputs for multi-modal training by inserting multi-modal features.
        
        Args:
            input_ids: Text token IDs
            attention_mask: Attention mask for text
            labels: Labels for language modeling
            multi_modal_features: Dict containing aligned multi-modal features
            
        Returns:
            Modified inputs with multi-modal features inserted
        """
        # Find special tokens for inserting multi-modal features
        image_token_index = getattr(self.config, 'image_token_index', 32000)
        bev_token_index = getattr(self.config, 'bev_token_index', 32001)
        vector_token_index = getattr(self.config, 'vector_token_index', 32002)
        
        batch_size = input_ids.shape[0]
        
        # Process each sample in the batch
        new_input_embeds = []
        new_labels = []
        
        for batch_idx in range(batch_size):
            cur_input_ids = input_ids[batch_idx]
            cur_labels = labels[batch_idx] if labels is not None else None
            
            # Get text embeddings
            cur_input_embeds = self.get_input_embeddings()(cur_input_ids)
            
            # Find positions of special tokens
            image_positions = (cur_input_ids == image_token_index).nonzero(as_tuple=True)[0]
            bev_positions = (cur_input_ids == bev_token_index).nonzero(as_tuple=True)[0]
            vector_positions = (cur_input_ids == vector_token_index).nonzero(as_tuple=True)[0]
            
            # Insert multi-modal features
            if len(image_positions) > 0 and 'image' in multi_modal_features:
                image_features = multi_modal_features['image'][batch_idx]  # [num_tokens, hidden_size]
                cur_input_embeds, cur_labels = self._insert_multimodal_features(
                    cur_input_embeds, cur_labels, image_positions[0], image_features
                )
            
            if len(bev_positions) > 0 and 'bev' in multi_modal_features:
                bev_features = multi_modal_features['bev'][batch_idx]
                cur_input_embeds, cur_labels = self._insert_multimodal_features(
                    cur_input_embeds, cur_labels, bev_positions[0], bev_features
                )
            
            if len(vector_positions) > 0 and 'vector' in multi_modal_features:
                vector_features = multi_modal_features['vector'][batch_idx]
                cur_input_embeds, cur_labels = self._insert_multimodal_features(
                    cur_input_embeds, cur_labels, vector_positions[0], vector_features
                )
            
            new_input_embeds.append(cur_input_embeds)
            if cur_labels is not None:
                new_labels.append(cur_labels)
        
        # Pad sequences to same length
        max_len = max(x.shape[0] for x in new_input_embeds)
        padded_input_embeds = []
        padded_labels = []
        padded_attention_mask = []
        
        for i, input_embeds in enumerate(new_input_embeds):
            cur_len = input_embeds.shape[0]
            if cur_len < max_len:
                # Pad with zeros
                padding = torch.zeros(max_len - cur_len, input_embeds.shape[1], 
                                    device=input_embeds.device, dtype=input_embeds.dtype)
                input_embeds = torch.cat([input_embeds, padding], dim=0)
                
                # Create attention mask
                attention = torch.cat([
                    torch.ones(cur_len, device=input_embeds.device),
                    torch.zeros(max_len - cur_len, device=input_embeds.device)
                ], dim=0)
                
                if new_labels:
                    cur_labels = new_labels[i]
                    label_padding = torch.full((max_len - cur_len,), -100, 
                                             device=cur_labels.device, dtype=cur_labels.dtype)
                    cur_labels = torch.cat([cur_labels, label_padding], dim=0)
                    padded_labels.append(cur_labels)
            else:
                attention = torch.ones(cur_len, device=input_embeds.device)
                if new_labels:
                    padded_labels.append(new_labels[i])
            
            padded_input_embeds.append(input_embeds)
            padded_attention_mask.append(attention)
        
        return {
            'inputs_embeds': torch.stack(padded_input_embeds),
            'attention_mask': torch.stack(padded_attention_mask),
            'labels': torch.stack(padded_labels) if padded_labels else None
        }

    def _insert_multimodal_features(self, input_embeds, labels, position, features):
        """Insert multi-modal features at specified position."""
        # Split embeddings at position
        before = input_embeds[:position]
        after = input_embeds[position + 1:]  # Skip the special token
        
        # Concatenate
        new_input_embeds = torch.cat([before, features, after], dim=0)
        
        # Handle labels
        if labels is not None:
            before_labels = labels[:position]
            after_labels = labels[position + 1:]
            # Use -100 for multi-modal features (ignore in loss)
            mm_labels = torch.full((features.shape[0],), -100, 
                                 device=labels.device, dtype=labels.dtype)
            new_labels = torch.cat([before_labels, mm_labels, after_labels], dim=0)
        else:
            new_labels = None
            
        return new_input_embeds, new_labels

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                # Multi-modal inputs
                pixel_values: Optional[torch.FloatTensor] = None,
                points: Optional[List[torch.Tensor]] = None,
                img_metas: Optional[List[Dict]] = None,
                lane_data: Optional[torch.Tensor] = None,
                trajectory_data: Optional[torch.Tensor] = None,
                agent_data: Optional[torch.Tensor] = None,
                initial_trajectory: Optional[torch.Tensor] = None,
                gt_trajectory: Optional[torch.FloatTensor] = None,
                # Training settings
                mode: str = 'train',  # 'train', 'round1', 'round2'
                enable_modalities: Optional[Dict[str, bool]] = None,
                **kwargs) -> Tuple:
        """
        Forward pass for multi-modal trajectory correction.
        
        Args:
            mode: Training mode ('train', 'round1', 'round2')
                - 'round1': Scene understanding and behavior evaluation
                - 'round2': Trajectory correction with explanation
            enable_modalities: Dict to enable/disable modalities for ablation
        """
        
        # Default modality settings (for ablation studies)
        if enable_modalities is None:
            enable_modalities = {
                'image': True,
                'bev': True,
                'vector': True
            }
        
        total_loss = torch.tensor(0.0, device=self.device)
        loss_dict = {}
        
        # Extract multi-modal features
        multi_modal_features = {}
        
        if pixel_values is not None:
            # Extract all multi-modal features
            vision_outputs = self.vision_tower(
                pixel_values=pixel_values,
                points=points if enable_modalities.get('bev', True) else None,
                img_metas=img_metas,
                lane_data=lane_data if enable_modalities.get('vector', True) else None,
                trajectory_data=trajectory_data if enable_modalities.get('vector', True) else None,
                agent_data=agent_data if enable_modalities.get('vector', True) else None,
                initial_trajectory=initial_trajectory if enable_modalities.get('vector', True) else None
            )
            
            # Align features using Q-Former
            if enable_modalities.get('image', True) and 'image_features' in vision_outputs:
                multi_modal_features['image'] = self.qformer(
                    vision_outputs['image_features'], 'image'
                )
            
            if enable_modalities.get('bev', True) and 'bev_features' in vision_outputs:
                multi_modal_features['bev'] = self.qformer(
                    vision_outputs['bev_features'], 'bev'
                )
            
            if enable_modalities.get('vector', True) and 'vector_features' in vision_outputs:
                multi_modal_features['vector'] = self.qformer(
                    vision_outputs['vector_features'], 'vector'
                )
        
        # Prepare inputs with multi-modal features
        if inputs_embeds is None and multi_modal_features:
            prepared_inputs = self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, labels, multi_modal_features
            )
            inputs_embeds = prepared_inputs['inputs_embeds']
            attention_mask = prepared_inputs['attention_mask']
            labels = prepared_inputs['labels']
            input_ids = None  # Use embeddings instead
        
        # Forward through LLaMA
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            **kwargs
        )
        
        # Language modeling loss
        if outputs.loss is not None:
            lm_loss = outputs.loss
            total_loss += self.lm_loss_weight * lm_loss
            loss_dict['lm_loss'] = lm_loss.item()
        
        # Get last hidden state for trajectory prediction and alignment
        if outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
            # Use last token for trajectory prediction
            last_token_hidden = last_hidden_state[:, -1, :]
            
            # Trajectory regression loss (only in round2 or train mode)
            if gt_trajectory is not None and mode in ['train', 'round2']:
                predicted_trajectory = self.trajectory_head(last_token_hidden)
                traj_loss = self.traj_loss_fn(predicted_trajectory, gt_trajectory)
                total_loss += self.traj_loss_weight * traj_loss
                loss_dict['traj_loss'] = traj_loss.item()
            
            # Compute alignment losses
            if multi_modal_features and mode in ['train', 'round1']:
                alignment_losses = self._compute_alignment_losses(
                    last_token_hidden, multi_modal_features, enable_modalities
                )
                for loss_name, loss_value in alignment_losses.items():
                    total_loss += loss_value
                    loss_dict[loss_name] = loss_value.item()
        
        return (total_loss, outputs.logits, outputs.past_key_values, 
                outputs.hidden_states, outputs.attentions, loss_dict)

    def _compute_alignment_losses(self, text_features, multi_modal_features, enable_modalities):
        """Compute alignment losses between text and multi-modal features."""
        losses = {}
        
        # Normalize text features
        text_norm = F.normalize(text_features, p=2, dim=-1)
        
        # Image-text alignment
        if (enable_modalities.get('image', True) and 'image' in multi_modal_features 
            and self.align_loss_weight > 0):
            image_features = torch.mean(multi_modal_features['image'], dim=1)
            image_norm = F.normalize(image_features, p=2, dim=-1)
            similarity = torch.matmul(text_norm, image_norm.T)
            labels = torch.arange(len(similarity), device=similarity.device, dtype=torch.long)
            align_loss = F.cross_entropy(similarity, labels)
            losses['image_align_loss'] = self.align_loss_weight * align_loss
        
        # BEV-text alignment
        if (enable_modalities.get('bev', True) and 'bev' in multi_modal_features 
            and self.bev_text_loss_weight > 0):
            bev_features = torch.mean(multi_modal_features['bev'], dim=1)
            bev_norm = F.normalize(bev_features, p=2, dim=-1)
            similarity = torch.matmul(text_norm, bev_norm.T)
            labels = torch.arange(len(similarity), device=similarity.device, dtype=torch.long)
            bev_text_loss = F.cross_entropy(similarity, labels)
            losses['bev_text_loss'] = self.bev_text_loss_weight * bev_text_loss
        
        # BEV-vector alignment
        if (enable_modalities.get('bev', True) and enable_modalities.get('vector', True) 
            and 'bev' in multi_modal_features and 'vector' in multi_modal_features
            and self.bev_vector_loss_weight > 0):
            bev_features = torch.mean(multi_modal_features['bev'], dim=1)
            vector_features = torch.mean(multi_modal_features['vector'], dim=1)
            bev_norm = F.normalize(bev_features, p=2, dim=-1)
            vector_norm = F.normalize(vector_features, p=2, dim=-1)
            similarity = torch.matmul(vector_norm, bev_norm.T)
            labels = torch.arange(len(similarity), device=similarity.device, dtype=torch.long)
            bev_vector_loss = F.cross_entropy(similarity, labels)
            losses['bev_vector_loss'] = self.bev_vector_loss_weight * bev_vector_loss
        
        return losses

    def predict_trajectory(self, input_ids, attention_mask=None, **multi_modal_inputs):
        """Predict trajectory from inputs."""
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mode='round2',
                **multi_modal_inputs
            )
            
            if outputs[3] is not None:  # hidden_states
                last_hidden = outputs[3][-1][:, -1, :]
                trajectory = self.trajectory_head(last_hidden)
                return trajectory
            else:
                return None

    def generate_with_trajectory(self, input_ids, attention_mask=None, max_length=512, **multi_modal_inputs):
        """Generate text response with trajectory prediction."""
        # First generate text
        with torch.no_grad():
            generate_ids = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            # Then predict trajectory using the generated sequence
            trajectory = self.predict_trajectory(
                generate_ids, 
                attention_mask=None,  # Will be created automatically
                **multi_modal_inputs
            )
            
            return generate_ids, trajectory


# Register the model
AutoConfig.register("trajectory_llama", TrajectoryLlamaConfig)
AutoModelForCausalLM.register(TrajectoryLlamaConfig, TrajectoryLlamaForCausalLM)