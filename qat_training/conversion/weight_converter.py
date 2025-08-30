"""
Weight conversion utilities for QAT to MLC-LLM format
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from safetensors.torch import save_file
import json
import os

logger = logging.getLogger(__name__)


class QATWeightConverter:
    """Convert QAT-trained weights to MLC-LLM q4f16_1 format"""
    
    def __init__(self, group_size: int = 32, num_elem_per_storage: int = 8):
        """
        Initialize weight converter
        
        Args:
            group_size: Group size for quantization (MLC q4f16_1 uses 32)
            num_elem_per_storage: Number of elements per storage unit (8 for 4bit in uint32)
        """
        self.group_size = group_size
        self.num_elem_per_storage = num_elem_per_storage
        self.max_int_value = 7  # 4-bit signed: -8 to 7
        
    def extract_qat_weights(self, qat_model) -> Dict[str, torch.Tensor]:
        """
        Extract quantized weights from QAT model
        
        Args:
            qat_model: QAT-trained model
            
        Returns:
            Dictionary of extracted weights and scales
        """
        extracted_weights = {}
        
        for name, module in qat_model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight
                
                # Check if weight is quantized
                if hasattr(weight, 'int_repr') and hasattr(weight, 'q_scale'):
                    # PyTorch quantized tensor
                    quantized_weight = weight.int_repr()
                    scales = weight.q_scale()
                    zero_points = weight.q_zero_point() if hasattr(weight, 'q_zero_point') else None
                    
                    extracted_weights[f"{name}.qweight"] = quantized_weight
                    extracted_weights[f"{name}.scales"] = scales
                    if zero_points is not None:
                        extracted_weights[f"{name}.zero_points"] = zero_points
                        
                elif hasattr(module, 'qweight') and hasattr(module, 'scales'):
                    # Already in quantized format
                    extracted_weights[f"{name}.qweight"] = module.qweight
                    extracted_weights[f"{name}.scales"] = module.scales
                    if hasattr(module, 'qzeros'):
                        extracted_weights[f"{name}.qzeros"] = module.qzeros
                        
                else:
                    # Full precision weight - need to quantize
                    logger.info(f"Quantizing full precision weight: {name}")
                    qweight, scales = self.quantize_weight_per_group(weight.data)
                    extracted_weights[f"{name}.qweight"] = qweight
                    extracted_weights[f"{name}.scales"] = scales
                
                # Handle bias if present
                if hasattr(module, 'bias') and module.bias is not None:
                    extracted_weights[f"{name}.bias"] = module.bias.data.half()
        
        logger.info(f"Extracted weights from {len(extracted_weights)} layers")
        return extracted_weights
    
    def quantize_weight_per_group(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize full precision weight using group quantization
        
        Args:
            weight: Full precision weight tensor
            
        Returns:
            Tuple of (quantized_weight, scales)
        """
        if len(weight.shape) != 2:
            # For non-2D tensors, flatten and reshape back
            original_shape = weight.shape
            weight = weight.view(weight.shape[0], -1)
        else:
            original_shape = None
        
        out_features, in_features = weight.shape
        
        # Calculate number of groups
        num_groups = (in_features + self.group_size - 1) // self.group_size
        
        # Pad weight if necessary
        padded_in_features = num_groups * self.group_size
        if padded_in_features > in_features:
            padding = torch.zeros(out_features, padded_in_features - in_features, 
                                device=weight.device, dtype=weight.dtype)
            weight_padded = torch.cat([weight, padding], dim=1)
        else:
            weight_padded = weight
        
        # Reshape for group processing
        weight_grouped = weight_padded.view(out_features, num_groups, self.group_size)
        
        # Calculate scales per group (max absolute value in each group)
        max_vals = torch.abs(weight_grouped).max(dim=2, keepdim=True)[0]
        scales = max_vals / self.max_int_value
        
        # Avoid division by zero
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        
        # Quantize
        quantized = torch.round(weight_grouped / scales).clamp(-8, 7)
        
        # Reshape back
        quantized = quantized.view(out_features, padded_in_features)
        scales = scales.squeeze(-1)  # Remove last dimension
        
        # Trim back to original size if padded
        if padded_in_features > in_features:
            quantized = quantized[:, :in_features]
        
        # Convert to int8 for storage
        quantized = quantized.to(torch.int8)
        scales = scales.half()  # Use half precision for scales
        
        return quantized, scales
    
    def convert_to_mlc_format(self, extracted_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert extracted weights to MLC-LLM format
        
        Args:
            extracted_weights: Extracted QAT weights
            
        Returns:
            Weights in MLC-LLM format
        """
        mlc_weights = {}
        
        # Process each weight
        weight_names = set()
        for key in extracted_weights.keys():
            if key.endswith('.qweight'):
                weight_names.add(key[:-8])  # Remove '.qweight'
        
        for weight_name in weight_names:
            qweight_key = f"{weight_name}.qweight"
            scales_key = f"{weight_name}.scales"
            
            if qweight_key in extracted_weights and scales_key in extracted_weights:
                qweight = extracted_weights[qweight_key]
                scales = extracted_weights[scales_key]
                
                # Convert to MLC format
                mlc_qweight, mlc_scales = self.pack_weights_for_mlc(qweight, scales)
                
                mlc_weights[f"{weight_name}.weight"] = mlc_qweight
                mlc_weights[f"{weight_name}.scales"] = mlc_scales
                
                logger.debug(f"Converted {weight_name}: {qweight.shape} -> {mlc_qweight.shape}")
            
            # Handle bias
            bias_key = f"{weight_name}.bias"
            if bias_key in extracted_weights:
                mlc_weights[bias_key] = extracted_weights[bias_key]
        
        logger.info(f"Converted to MLC format: {len(mlc_weights)} tensors")
        return mlc_weights
    
    def pack_weights_for_mlc(self, qweight: torch.Tensor, scales: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pack quantized weights into MLC-LLM storage format
        
        Args:
            qweight: Quantized weights (int8)
            scales: Scaling factors
            
        Returns:
            Tuple of (packed_weight, formatted_scales)
        """
        out_features, in_features = qweight.shape
        
        # Ensure weights are 4-bit values (-8 to 7, but we'll offset to 0-15 for packing)
        qweight_shifted = qweight + 8  # Shift to 0-15 range
        qweight_shifted = qweight_shifted.clamp(0, 15).to(torch.uint8)
        
        # Pack 8 x 4-bit values into each uint32
        assert in_features % self.num_elem_per_storage == 0, f"in_features ({in_features}) must be divisible by {self.num_elem_per_storage}"
        
        num_storage = in_features // self.num_elem_per_storage
        packed_weight = torch.zeros(out_features, num_storage, dtype=torch.uint32, device=qweight.device)
        
        # Pack weights
        qweight_reshaped = qweight_shifted.view(out_features, num_storage, self.num_elem_per_storage)
        
        for i in range(self.num_elem_per_storage):
            # Pack each 4-bit value into the appropriate position in uint32
            packed_weight += qweight_reshaped[:, :, i].to(torch.uint32) << (i * 4)
        
        # Format scales for MLC
        num_groups = (in_features + self.group_size - 1) // self.group_size
        if scales.shape[-1] != num_groups:
            # Reshape scales if needed
            scales = scales.view(out_features, num_groups)
        
        # Ensure scales are float16
        scales = scales.half()
        
        return packed_weight, scales
    
    def save_mlc_weights(self, mlc_weights: Dict[str, torch.Tensor], output_dir: str):
        """
        Save weights in MLC-LLM compatible format
        
        Args:
            mlc_weights: Converted weights
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert tensors to CPU for saving
        cpu_weights = {k: v.cpu() for k, v in mlc_weights.items()}
        
        # Save as safetensors (preferred by MLC-LLM)
        safetensors_path = os.path.join(output_dir, "model.safetensors")
        save_file(cpu_weights, safetensors_path)
        
        logger.info(f"Weights saved to: {safetensors_path}")
        
        # Create model config for MLC-LLM
        self.create_mlc_config(output_dir, len(mlc_weights))
    
    def create_mlc_config(self, output_dir: str, num_params: int):
        """
        Create MLC-LLM compatible configuration
        
        Args:
            output_dir: Output directory
            num_params: Number of parameters
        """
        # Basic config for Llama-style model
        config = {
            "model_type": "llama",
            "quantization": "q4f16_1",
            "quantization_config": {
                "group_size": self.group_size,
                "bits": 4,
                "storage_dtype": "uint32",
                "compute_dtype": "float16"
            },
            "converted_from": "qat_training",
            "conversion_timestamp": torch.datetime.now().isoformat(),
            "num_parameters": num_params
        }
        
        config_path = os.path.join(output_dir, "mlc_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"MLC config saved to: {config_path}")


def convert_qat_to_mlc(qat_model, output_dir: str, group_size: int = 32) -> None:
    """
    Convenience function to convert QAT model to MLC format
    
    Args:
        qat_model: QAT-trained model
        output_dir: Output directory
        group_size: Group size for quantization
    """
    converter = QATWeightConverter(group_size=group_size)
    
    # Extract weights from QAT model
    logger.info("Extracting weights from QAT model...")
    extracted_weights = converter.extract_qat_weights(qat_model)
    
    # Convert to MLC format
    logger.info("Converting to MLC-LLM format...")
    mlc_weights = converter.convert_to_mlc_format(extracted_weights)
    
    # Save MLC weights
    logger.info("Saving MLC-LLM compatible weights...")
    converter.save_mlc_weights(mlc_weights, output_dir)
    
    logger.info(f"Conversion completed! Weights saved to: {output_dir}")
    logger.info("Use with MLC-LLM: mlc_llm convert_weight <path> --quantization q4f16_1")