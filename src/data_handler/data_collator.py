import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from transformers.data.data_collator import DataCollatorWithPadding
from torch.utils.data import Dataset
from PIL import Image
import cv2

class NuScenesTrajectoryDataset(Dataset):
    """
    Dataset for nuScenes trajectory correction with multimodal inputs.
    
    Supports two-round dialogue format:
    - Round 1: Scene understanding and behavior evaluation
    - Round 2: Trajectory correction with explanation
    """
    
    def __init__(self,
                 data_dir: str,
                 split: str = "train",
                 tokenizer=None,
                 max_length: int = 1024,
                 image_size: Tuple[int, int] = (224, 224),
                 num_views: int = 6,
                 mode: str = "both",  # "round1", "round2", "both"
                 enable_modalities: Optional[Dict[str, bool]] = None):
        
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.num_views = num_views
        self.mode = mode
        
        # Default modality settings
        self.enable_modalities = enable_modalities or {
            'image': True,
            'bev': True,
            'vector': True
        }
        
        # Load QA data
        self.qa_data = self._load_qa_data()
        
        # Load samples metadata
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_qa_data(self) -> Dict[str, Any]:
        """Load question-answer data."""
        qa_dir = os.path.join(self.data_dir, self.split, "qa")
        qa_data = {}
        
        if os.path.exists(qa_dir):
            for qa_file in os.listdir(qa_dir):
                if qa_file.endswith("_qa.json"):
                    sample_token = qa_file.replace("_qa.json", "")
                    with open(os.path.join(qa_dir, qa_file), 'r') as f:
                        qa_data[sample_token] = json.load(f)
        
        return qa_data
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load sample metadata."""
        samples = []
        
        for sample_token in self.qa_data.keys():
            qa_item = self.qa_data[sample_token]
            
            # Create samples based on mode
            if self.mode in ["round1", "both"]:
                # Round 1: Scene understanding
                sample = {
                    "sample_token": sample_token,
                    "round": 1,
                    "conversation": qa_item.get("round1_conversation", []),
                    "has_trajectory": False
                }
                samples.append(sample)
            
            if self.mode in ["round2", "both"]:
                # Round 2: Trajectory correction
                if "round2_conversation" in qa_item:
                    sample = {
                        "sample_token": sample_token,
                        "round": 2,
                        "conversation": qa_item.get("round2_conversation", []),
                        "has_trajectory": True,
                        "initial_trajectory": qa_item.get("initial_trajectory"),
                        "corrected_trajectory": qa_item.get("corrected_trajectory")
                    }
                    samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample."""
        sample = self.samples[idx]
        sample_token = sample["sample_token"]
        
        # Prepare text data
        text_data = self._prepare_text_data(sample)
        
        # Prepare multimodal data
        multimodal_data = {}
        
        if self.enable_modalities.get('image', True):
            multimodal_data.update(self._load_image_data(sample_token))
        
        if self.enable_modalities.get('bev', True) or self.enable_modalities.get('vector', True):
            multimodal_data.update(self._load_sensor_data(sample_token))
        
        # Combine all data
        result = {**text_data, **multimodal_data}
        
        # Add trajectory data if available
        if sample.get("has_trajectory", False):
            result.update(self._prepare_trajectory_data(sample))
        
        return result
    
    def _prepare_text_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare text inputs from conversation."""
        conversation = sample["conversation"]
        
        if not conversation:
            # Fallback conversation
            conversation = [
                {"role": "user", "content": "Please analyze this driving scene."},
                {"role": "assistant", "content": "I can see the driving scene and will analyze it."}
            ]
        
        # Format conversation for training
        formatted_text = self._format_conversation(conversation)
        
        # Tokenize
        encoded = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare labels (same as input_ids for causal LM)
        labels = encoded["input_ids"].clone()
        
        # Mask user tokens in labels (only train on assistant responses)
        labels = self._mask_user_tokens(labels[0], formatted_text).unsqueeze(0)
        
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "labels": labels[0]
        }
    
    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation for training."""
        if not self.tokenizer:
            return str(conversation)
        
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                conversation, 
                tokenize=False,
                add_generation_prompt=False
            )
        
        # Fallback formatting
        formatted_parts = []
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                formatted_parts.append(f"Human: {content}")
            else:
                formatted_parts.append(f"Assistant: {content}")
        
        return "\n".join(formatted_parts)
    
    def _mask_user_tokens(self, input_ids: torch.Tensor, formatted_text: str) -> torch.Tensor:
        """Mask user tokens in labels to only train on assistant responses."""
        labels = input_ids.clone()
        
        # Simple heuristic: mask everything before "Assistant:" tokens
        if hasattr(self.tokenizer, 'decode'):
            decoded = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            assistant_positions = []
            
            # Find assistant response positions
            start_pos = 0
            while True:
                pos = decoded.find("Assistant:", start_pos)
                if pos == -1:
                    break
                assistant_positions.append(pos)
                start_pos = pos + 1
            
            # Mask tokens before assistant responses
            if assistant_positions:
                for i, token_id in enumerate(input_ids):
                    token_text = self.tokenizer.decode([token_id])
                    # This is a simplified approach - you might want to use a more sophisticated method
                    if i < len(input_ids) // 2:  # Rough heuristic
                        labels[i] = -100
        
        return labels
    
    def _load_image_data(self, sample_token: str) -> Dict[str, torch.Tensor]:
        """Load multi-view camera images."""
        image_dir = os.path.join(self.data_dir, self.split, "samples", "surround_stitched")
        image_path = os.path.join(image_dir, f"{sample_token}_surround.jpg")
        
        if os.path.exists(image_path):
            # Load stitched surround view image
            image = Image.open(image_path).convert('RGB')
            image = image.resize(self.image_size)
            
            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
            
            # Simulate multi-view by splitting the stitched image
            # This is a simplification - in practice, you'd load individual camera views
            views = []
            view_width = image_tensor.shape[2] // self.num_views
            for i in range(self.num_views):
                start_col = i * view_width
                end_col = (i + 1) * view_width
                view = image_tensor[:, :, start_col:end_col]
                view = F.interpolate(view.unsqueeze(0), size=self.image_size, mode='bilinear').squeeze(0)
                views.append(view)
            
            pixel_values = torch.stack(views)  # [num_views, C, H, W]
        else:
            # Dummy images if not found
            pixel_values = torch.zeros(self.num_views, 3, *self.image_size)
        
        return {"pixel_values": pixel_values}
    
    def _load_sensor_data(self, sample_token: str) -> Dict[str, Any]:
        """Load point cloud and vector data."""
        data = {}
        
        # For now, create dummy data since we don't have the actual sensor files
        # In practice, you would load from nuScenes database
        
        if self.enable_modalities.get('bev', True):
            # Dummy point cloud data
            num_points = 1000
            points = torch.randn(num_points, 5)  # x, y, z, intensity, timestamp
            data["points"] = [points]  # List format for batch collation
            
            # Dummy image metadata for BEV transformation
            data["img_metas"] = [{"sample_token": sample_token}]
        
        if self.enable_modalities.get('vector', True):
            # Dummy vector data
            data["lane_data"] = torch.randn(10, 20, 4)  # 10 lanes, 20 points each, (x,y,lane_id,type)
            data["trajectory_data"] = torch.randn(5, 10, 6)  # 5 agents, 10 timesteps, (x,y,vx,vy,heading,time)
            data["agent_data"] = torch.randn(5, 5)  # 5 agents, (x,y,heading,vel,type)
        
        return data
    
    def _prepare_trajectory_data(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare trajectory data for round 2."""
        data = {}
        
        initial_traj = sample.get("initial_trajectory")
        corrected_traj = sample.get("corrected_trajectory")
        
        if initial_traj:
            # Convert to tensor if needed
            if isinstance(initial_traj, list):
                initial_traj = torch.tensor(initial_traj, dtype=torch.float32)
            data["initial_trajectory"] = initial_traj
        
        if corrected_traj:
            # Convert to tensor if needed
            if isinstance(corrected_traj, list):
                corrected_traj = torch.tensor(corrected_traj, dtype=torch.float32)
            data["gt_trajectory"] = corrected_traj
        else:
            # Dummy trajectory if not available
            data["gt_trajectory"] = torch.randn(6, 2)  # 6 waypoints, (x,y)
        
        return data


class MultiModalDataCollator:
    """
    Enhanced data collator for multimodal trajectory correction.
    
    Handles:
    - Text tokenization and padding
    - Multimodal feature batching
    - Trajectory data preparation
    - Dynamic masking for different training stages
    """
    
    def __init__(self, 
                 tokenizer,
                 model_config=None,
                 pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.pad_to_multiple_of = pad_to_multiple_of
        
        # Text collator for handling tokenized inputs
        self.text_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=pad_to_multiple_of
        )
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.
        
        Args:
            features: List of feature dictionaries from dataset
            
        Returns:
            Batched features ready for model input
        """
        batch = {}
        
        # Extract and collate text features
        text_features = []
        for f in features:
            text_feat = {k: v for k, v in f.items() 
                        if k in ['input_ids', 'attention_mask', 'labels']}
            text_features.append(text_feat)
        
        if text_features:
            text_batch = self.text_collator(text_features)
            batch.update(text_batch)
        
        # Handle multimodal features
        self._collate_multimodal_features(features, batch)
        
        # Handle trajectory data
        self._collate_trajectory_data(features, batch)
        
        return batch
    
    def _collate_multimodal_features(self, features: List[Dict[str, Any]], batch: Dict[str, torch.Tensor]):
        """Collate multimodal features."""
        
        # Image features
        if any('pixel_values' in f for f in features):
            pixel_values = []
            for f in features:
                if 'pixel_values' in f:
                    pixel_values.append(f['pixel_values'])
                else:
                    # Default shape: [num_views, C, H, W]
                    pixel_values.append(torch.zeros(6, 3, 224, 224))
            
            batch['pixel_values'] = torch.stack(pixel_values)
        
        # Point cloud data (keep as list for variable sizes)
        if any('points' in f for f in features):
            points_list = []
            for f in features:
                if 'points' in f:
                    points_list.extend(f['points'])
                else:
                    # Default point cloud
                    points_list.append(torch.zeros(1000, 5))
            
            batch['points'] = points_list
        
        # Image metadata (keep as list)
        if any('img_metas' in f for f in features):
            img_metas = []
            for f in features:
                if 'img_metas' in f:
                    img_metas.extend(f['img_metas'])
                else:
                    img_metas.append({})
            
            batch['img_metas'] = img_metas
        
        # Vector data
        vector_keys = ['lane_data', 'trajectory_data', 'agent_data']
        for key in vector_keys:
            if any(key in f for f in features):
                values = []
                for f in features:
                    if key in f:
                        values.append(f[key])
                    else:
                        # Default tensor based on key
                        if key == 'lane_data':
                            values.append(torch.zeros(10, 20, 4))
                        elif key == 'trajectory_data':
                            values.append(torch.zeros(5, 10, 6))
                        elif key == 'agent_data':
                            values.append(torch.zeros(5, 5))
                
                try:
                    batch[key] = torch.stack(values)
                except RuntimeError:
                    # If stacking fails, keep as list
                    batch[key] = values
    
    def _collate_trajectory_data(self, features: List[Dict[str, Any]], batch: Dict[str, torch.Tensor]):
        """Collate trajectory data."""
        
        # Initial trajectory (to be corrected)
        if any('initial_trajectory' in f for f in features):
            initial_trajs = []
            for f in features:
                if 'initial_trajectory' in f:
                    traj = f['initial_trajectory']
                    if not isinstance(traj, torch.Tensor):
                        traj = torch.tensor(traj, dtype=torch.float32)
                    initial_trajs.append(traj)
                else:
                    # Default trajectory
                    initial_trajs.append(torch.zeros(6, 2))
            
            try:
                batch['initial_trajectory'] = torch.stack(initial_trajs)
            except RuntimeError:
                # Handle variable length trajectories by padding
                max_len = max(t.shape[0] for t in initial_trajs)
                padded_trajs = []
                for traj in initial_trajs:
                    if traj.shape[0] < max_len:
                        padding = torch.zeros(max_len - traj.shape[0], traj.shape[1])
                        traj = torch.cat([traj, padding], dim=0)
                    padded_trajs.append(traj)
                batch['initial_trajectory'] = torch.stack(padded_trajs)
        
        # Ground truth trajectory
        if any('gt_trajectory' in f for f in features):
            gt_trajs = []
            for f in features:
                if 'gt_trajectory' in f:
                    traj = f['gt_trajectory']
                    if not isinstance(traj, torch.Tensor):
                        traj = torch.tensor(traj, dtype=torch.float32)
                    gt_trajs.append(traj)
                else:
                    # Default trajectory
                    gt_trajs.append(torch.zeros(6, 2))
            
            try:
                batch['gt_trajectory'] = torch.stack(gt_trajs)
            except RuntimeError:
                # Handle variable length trajectories by padding
                max_len = max(t.shape[0] for t in gt_trajs)
                padded_trajs = []
                for traj in gt_trajs:
                    if traj.shape[0] < max_len:
                        padding = torch.zeros(max_len - traj.shape[0], traj.shape[1])
                        traj = torch.cat([traj, padding], dim=0)
                    padded_trajs.append(traj)
                batch['gt_trajectory'] = torch.stack(padded_trajs)


# Alias for backward compatibility
CustomDataCollator = MultiModalDataCollator