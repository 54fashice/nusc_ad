import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.integrations import WandbCallback
import numpy as np
import logging
from llmtuner.train.llava.trainer import LLaVATrainer

logger = logging.getLogger(__name__)

class TrajectoryLlamaTrainer(LLaVATrainer):
    """
    Custom trainer for trajectory correction with multimodal LLaMA.
    
    Supports:
    - Multimodal input handling (vision, BEV, vector)
    - Composite loss (language modeling + trajectory regression + alignment)
    - Staged training (pretraining, fine-tuning)
    - Two-round dialogue training
    - Ablation experiments
    """
    
    def __init__(self, 
                 model=None,
                 args: TrainingArguments = None,
                 data_collator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 model_init=None,
                 compute_metrics=None,
                 callbacks=None,
                 optimizers=(None, None),
                 preprocess_logits_for_metrics=None,
                 # Custom arguments
                 training_stage: str = "sft",  # "pretrain", "sft", "full_finetune"
                 enable_modalities: Optional[Dict[str, bool]] = None,
                 loss_weights: Optional[Dict[str, float]] = None,
                 **kwargs):
        
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs
        )
        
        self.training_stage = training_stage
        self.enable_modalities = enable_modalities or {'image': True, 'bev': True, 'vector': True}
        self.loss_weights = loss_weights or {
            'lm_loss': 1.0,
            'traj_loss': 1.5,
            'align_loss': 0.1,
            'bev_text_loss': 0.1,
            'bev_vector_loss': 0.1
        }
        
        # Training metrics
        self.training_losses = {
            'total_loss': [],
            'lm_loss': [],
            'traj_loss': [],
            'align_loss': [],
            'bev_text_loss': [],
            'bev_vector_loss': []
        }
        
        # Add custom callbacks
        self._setup_custom_callbacks()
    
    def _setup_custom_callbacks(self):
        """Setup custom callbacks for logging and monitoring."""
        # Add custom logging callback for multimodal losses
        class MultiModalLossCallback:
            def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                if logs:
                    # Log individual loss components
                    for key, value in logs.items():
                        if '_loss' in key and key != 'train_loss':
                            logger.info(f"Step {state.global_step}: {key} = {value:.4f}")
        
        self.add_callback(MultiModalLossCallback())
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute composite loss for multimodal trajectory correction.
        
        Args:
            model: TrajectoryLlamaForCausalLM model
            inputs: Batch of inputs
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss tensor (and optionally outputs)
        """
        # Determine mode based on training stage and inputs
        mode = self._determine_training_mode(inputs)
        
        # Forward pass through model
        outputs = model(
            input_ids=inputs.get('input_ids'),
            attention_mask=inputs.get('attention_mask'),
            labels=inputs.get('labels'),
            pixel_values=inputs.get('pixel_values'),
            points=inputs.get('points'),
            img_metas=inputs.get('img_metas'),
            lane_data=inputs.get('lane_data'),
            trajectory_data=inputs.get('trajectory_data'),
            agent_data=inputs.get('agent_data'),
            initial_trajectory=inputs.get('initial_trajectory'),
            gt_trajectory=inputs.get('gt_trajectory'),
            mode=mode,
            enable_modalities=self.enable_modalities
        )
        
        # Extract losses
        if isinstance(outputs, tuple) and len(outputs) >= 6:
            total_loss, logits, past_key_values, hidden_states, attentions, loss_dict = outputs
        else:
            # Fallback for compatibility
            total_loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[1]
            loss_dict = {}
        
        # Apply loss weights
        if isinstance(loss_dict, dict) and loss_dict:
            weighted_loss = 0.0
            for loss_name, loss_value in loss_dict.items():
                weight = self.loss_weights.get(loss_name, 1.0)
                weighted_loss += weight * loss_value
                
                # Log individual losses
                self.training_losses[loss_name].append(loss_value)
            
            total_loss = weighted_loss
        
        # Log total loss
        self.training_losses['total_loss'].append(total_loss.item() if torch.is_tensor(total_loss) else total_loss)
        
        if return_outputs:
            return total_loss, (logits, past_key_values, hidden_states, attentions)
        else:
            return total_loss
    
    def _determine_training_mode(self, inputs):
        """Determine training mode based on stage and inputs."""
        if self.training_stage == "pretrain":
            return "round1"  # Focus on alignment
        elif self.training_stage == "sft":
            # Check if we have trajectory data for round2
            if inputs.get('gt_trajectory') is not None:
                return "round2"
            else:
                return "round1"
        else:  # full_finetune
            return "train"  # End-to-end training
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform a prediction step.
        
        Args:
            model: The model to evaluate
            inputs: The inputs and targets for the model
            prediction_loss_only: Whether to only compute the loss
            ignore_keys: Keys to ignore in the outputs
            
        Returns:
            Tuple of (loss, logits, labels)
        """
        with torch.no_grad():
            # Get model outputs
            outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = outputs[0]
            logits = outputs[1][0] if len(outputs) > 1 else None
            
            # Get labels
            labels = inputs.get('labels')
            
            if prediction_loss_only:
                return (loss, None, None)
            
            return (loss, logits, labels)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Run evaluation and return metrics.
        
        Args:
            eval_dataset: Dataset to evaluate on
            ignore_keys: Keys to ignore in outputs
            metric_key_prefix: Prefix for metric names
            
        Returns:
            Dictionary of evaluation metrics
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        # Run standard evaluation
        eval_results = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        # Add custom metrics
        eval_results.update(self._compute_custom_metrics())
        
        return eval_results
    
    def _compute_custom_metrics(self):
        """Compute custom metrics for multimodal training."""
        metrics = {}
        
        # Compute average losses
        for loss_name, loss_values in self.training_losses.items():
            if loss_values:
                metrics[f"avg_{loss_name}"] = np.mean(loss_values[-100:])  # Last 100 steps
        
        # Training stage specific metrics
        if self.training_stage == "pretrain":
            # Focus on alignment metrics
            align_losses = [v for k, v in self.training_losses.items() if 'align' in k and v]
            if align_losses:
                metrics["avg_alignment_loss"] = np.mean([item for sublist in align_losses for item in sublist[-50:]])
        
        elif self.training_stage == "sft":
            # Focus on trajectory and language metrics
            if self.training_losses['traj_loss']:
                metrics["avg_trajectory_loss"] = np.mean(self.training_losses['traj_loss'][-50:])
            if self.training_losses['lm_loss']:
                metrics["avg_language_loss"] = np.mean(self.training_losses['lm_loss'][-50:])
        
        return metrics
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log metrics with custom multimodal information.
        
        Args:
            logs: Dictionary of metrics to log
        """
        # Add custom metrics
        logs.update(self._compute_custom_metrics())
        
        # Add training stage info
        logs["training_stage"] = self.training_stage
        logs["enabled_modalities"] = str(self.enable_modalities)
        
        # Call parent log method
        super().log(logs)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save model with custom metadata.
        
        Args:
            output_dir: Directory to save the model
            _internal_call: Whether this is an internal call
        """
        super().save_model(output_dir, _internal_call)
        
        # Save additional metadata
        if output_dir is None:
            output_dir = self.args.output_dir
        
        metadata = {
            "training_stage": self.training_stage,
            "enable_modalities": self.enable_modalities,
            "loss_weights": self.loss_weights,
            "training_losses": self.training_losses
        }
        
        import json
        metadata_path = f"{output_dir}/training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved training metadata to {metadata_path}")
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Create optimizer and scheduler with stage-specific settings.
        
        Args:
            num_training_steps: Total number of training steps
        """
        # Stage-specific learning rate adjustments
        if self.training_stage == "pretrain":
            # Lower learning rate for alignment training
            original_lr = self.args.learning_rate
            self.args.learning_rate = original_lr * 0.1
            logger.info(f"Pretrain stage: Reduced LR to {self.args.learning_rate}")
        
        elif self.training_stage == "full_finetune":
            # Even lower learning rate for full fine-tuning
            original_lr = self.args.learning_rate
            self.args.learning_rate = original_lr * 0.01
            logger.info(f"Full finetune stage: Reduced LR to {self.args.learning_rate}")
        
        # Create optimizer and scheduler
        super().create_optimizer_and_scheduler(num_training_steps)
    
    def get_train_dataloader(self):
        """Get training dataloader with custom collation."""
        dataloader = super().get_train_dataloader()
        
        # Log modality information
        logger.info(f"Training with modalities: {self.enable_modalities}")
        logger.info(f"Training stage: {self.training_stage}")
        
        return dataloader


class CustomDataCollator:
    """
    Custom data collator for multimodal trajectory correction.
    
    Handles:
    - Text tokenization and padding
    - Multimodal feature loading and batching
    - Trajectory data preparation
    """
    
    def __init__(self, tokenizer, model_config=None):
        self.tokenizer = tokenizer
        self.model_config = model_config
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Batched features
        """
        batch = {}
        
        # Handle text features
        text_features = []
        for f in features:
            text_feat = {k: v for k, v in f.items() 
                        if k in ['input_ids', 'attention_mask', 'labels']}
            text_features.append(text_feat)
        
        # Pad text features
        from transformers.data.data_collator import DataCollatorWithPadding
        text_collator = DataCollatorWithPadding(self.tokenizer)
        text_batch = text_collator(text_features)
        batch.update(text_batch)
        
        # Handle multimodal features
        if any('pixel_values' in f for f in features):
            pixel_values = [f.get('pixel_values', torch.zeros(6, 3, 224, 224)) 
                           for f in features]
            batch['pixel_values'] = torch.stack(pixel_values)
        
        # Handle trajectory data
        if any('gt_trajectory' in f for f in features):
            gt_trajectories = [torch.tensor(f['gt_trajectory'], dtype=torch.float32) 
                             for f in features if 'gt_trajectory' in f]
            if gt_trajectories:
                batch['gt_trajectory'] = torch.stack(gt_trajectories)
        
        # Handle other optional multimodal inputs
        optional_keys = ['points', 'img_metas', 'lane_data', 'trajectory_data', 
                        'agent_data', 'initial_trajectory']
        
        for key in optional_keys:
            values = [f.get(key) for f in features if key in f]
            if values:
                if key == 'points':
                    batch[key] = values  # Keep as list for point clouds
                elif key == 'img_metas':
                    batch[key] = values  # Keep as list for metadata
                else:
                    # Try to stack other tensors
                    try:
                        batch[key] = torch.stack([torch.tensor(v, dtype=torch.float32) 
                                                for v in values])
                    except:
                        batch[key] = values  # Keep as list if stacking fails
        
        return batch