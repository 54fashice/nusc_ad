import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerModel
from transformers.models.blip_2.configuration_blip_2 import Blip2QFormerConfig

class MultiModalQFormer(nn.Module):
    """
    Multi-modal Q-Former for aligning different modalities (vision, BEV, vector) with LLM.
    
    This module processes different types of features and aligns them to the LLM's embedding space
    using a Q-Former architecture similar to BLIP-2.
    """
    
    def __init__(self, vision_hidden_size: int, llm_hidden_size: int, num_query_tokens: int = 32):
        super().__init__()
        
        # Create Q-Former configuration with proper type
        qformer_config = Blip2QFormerConfig(
            vocab_size=30522,  # Standard BERT vocab size
            hidden_size=768,   # Standard BERT hidden size
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            position_embedding_type="absolute",
            cross_attention_frequency=2,
            encoder_hidden_size=vision_hidden_size,
            num_query_tokens=num_query_tokens
        )
        
        # Initialize Q-Former model
        self.qformer = Blip2QFormerModel(qformer_config)
        
        # Projection layer to align with LLM hidden size
        self.query_proj = nn.Linear(qformer_config.hidden_size, llm_hidden_size)
        
        # Store configuration
        self.num_query_tokens = num_query_tokens
        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size
        
        # Feature adaptation layers for different modalities
        self.image_adapter = nn.Linear(vision_hidden_size, qformer_config.hidden_size)
        self.bev_adapter = nn.Linear(vision_hidden_size, qformer_config.hidden_size)  
        self.vector_adapter = nn.Linear(vision_hidden_size, qformer_config.hidden_size)
        
        # Modality-specific query tokens
        self.modality_queries = nn.ParameterDict({
            'image': nn.Parameter(torch.randn(1, num_query_tokens, qformer_config.hidden_size)),
            'bev': nn.Parameter(torch.randn(1, num_query_tokens, qformer_config.hidden_size)),
            'vector': nn.Parameter(torch.randn(1, num_query_tokens, qformer_config.hidden_size))
        })
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for new parameters."""
        for param in self.modality_queries.values():
            nn.init.normal_(param, std=0.02)
    
    def forward(self, 
                vision_features: Optional[torch.Tensor] = None,
                modality_type: str = 'image') -> Optional[torch.Tensor]:
        """
        Forward pass through Q-Former.
        
        Args:
            vision_features: Input features [B, seq_len, hidden_size]
            modality_type: Type of modality ('image', 'bev', 'vector')
            
        Returns:
            Aligned features for LLM [B, num_query_tokens, llm_hidden_size]
        """
        if vision_features is None:
            return None
            
        batch_size = vision_features.shape[0]
        
        # Adapt features based on modality type
        if modality_type == 'image':
            adapted_features = self.image_adapter(vision_features)
            query_tokens = self.modality_queries['image'].expand(batch_size, -1, -1)
        elif modality_type == 'bev':
            adapted_features = self.bev_adapter(vision_features)
            query_tokens = self.modality_queries['bev'].expand(batch_size, -1, -1)
        elif modality_type == 'vector':
            adapted_features = self.vector_adapter(vision_features)
            query_tokens = self.modality_queries['vector'].expand(batch_size, -1, -1)
        else:
            raise ValueError(f"Unknown modality type: {modality_type}")
        
        # Pass through Q-Former
        qformer_out = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=adapted_features
        )
        
        # Project to LLM hidden size
        aligned_features = self.query_proj(qformer_out.last_hidden_state)
        
        return aligned_features
    
    def forward_multi_modal(self, 
                           image_features: Optional[torch.Tensor] = None,
                           bev_features: Optional[torch.Tensor] = None,
                           vector_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multiple modalities simultaneously.
        
        Args:
            image_features: Image features [B, seq_len, hidden_size]
            bev_features: BEV features [B, seq_len, hidden_size]
            vector_features: Vector features [B, seq_len, hidden_size]
            
        Returns:
            Dictionary of aligned features for each modality
        """
        results = {}
        
        if image_features is not None:
            results['image'] = self.forward(image_features, 'image')
        
        if bev_features is not None:
            results['bev'] = self.forward(bev_features, 'bev')
        
        if vector_features is not None:
            results['vector'] = self.forward(vector_features, 'vector')
            
        return results