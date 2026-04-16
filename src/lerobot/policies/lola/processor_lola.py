#!/usr/bin/env python

# Copyright 2025 Lola Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Processor pipeline for LoLA (Vision-Language-Action) policy.

This module provides preprocessing and postprocessing pipelines for the LoLA policy,
which uses Qwen3.5 as the VLM backbone. The processor handles:
1. Image preprocessing for Qwen3.5 vision encoder (multi-camera support)
2. Text tokenization with Qwen3.5 chat template
3. Empty token appending for global intent control
4. Normalization of states and actions

Data flow:
1. LeRobotDataset returns data with keys:
   - observation.state: Robot state
   - observation.images.*: Image/video data (multiple cameras supported)
   - action: Actions
   - task: Task description string
   
2. Processor transforms to LoLA expected format:
   - input_ids: Tokenized text for VLM (with chat template)
   - pixel_values: Preprocessed images for vision encoder
   - image_grid_thw: Image grid information for Qwen3.5
   - attention_mask: Attention mask for text tokens
   - observation.state: Normalized state (for history actions)
   - action: Normalized target actions

Qwen3.5 Input Format:
    Qwen3.5 uses a chat template format with support for multi-image input.
    The processor uses `apply_chat_template` to format the input correctly.
    
    Example message format:
    ```python
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image1},
                {"type": "image", "image": image2},
                {"type": "text", "text": "Task description"},
            ],
        }
    ]
    ```
"""

from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.lola.configuration_lola import LoLAConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.utils.import_utils import _transformers_available

# Lazy import for type checking
if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
else:
    AutoProcessor = None


@ProcessorStepRegistry.register(name="lola_empty_token_processor")
class LolaEmptyTokenProcessor(ObservationProcessorStep):
    """
    Appends an empty token to the tokenized sequence for global intent control.
    
    According to the LoLA architecture:
    - An empty_token is appended at the end of the VLM sequence
    - This token is responsible for aggregating global task intent in self-attention
    - After extraction, it fuses with the diffusion timestep to generate modulation signals
    - These signals control DiT feature scaling and shifting through AdaLN
    """
    
    def __init__(self, empty_token_id: int, **kwargs):
        """
        Args:
            empty_token_id: The token ID to append as the empty token (default: Qwen3.5 eos_token)
        """
        super().__init__(**kwargs)
        self.empty_token_id = empty_token_id
    
    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Appends the empty token to the language tokens sequence.
        
        Args:
            observation: The observation dictionary containing language tokens.
            
        Returns:
            Updated observation with empty token appended.
        """
        if OBS_LANGUAGE_TOKENS not in observation:
            return observation
        
        new_observation = dict(observation)
        
        # Get current tokens
        language_tokens = observation[OBS_LANGUAGE_TOKENS]  # [B, seq_len]
        batch_size = language_tokens.shape[0]
        
        # Create empty token tensor
        empty_token = torch.full(
            (batch_size, 1), 
            self.empty_token_id, 
            dtype=language_tokens.dtype, 
            device=language_tokens.device
        )
        
        # Append empty token to sequence
        new_observation[OBS_LANGUAGE_TOKENS] = torch.cat([language_tokens, empty_token], dim=1)
        
        # Extend attention mask accordingly
        if OBS_LANGUAGE_ATTENTION_MASK in observation:
            attention_mask = observation[OBS_LANGUAGE_ATTENTION_MASK]  # [B, seq_len]
            new_attention = torch.ones(
                (batch_size, 1), 
                dtype=attention_mask.dtype, 
                device=attention_mask.device
            )
            new_observation[OBS_LANGUAGE_ATTENTION_MASK] = torch.cat([attention_mask, new_attention], dim=1)
        
        return new_observation
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates feature shapes to account for the additional empty token.
        
        Args:
            features: The input feature dictionary.
            
        Returns:
            Updated feature dictionary with adjusted sequence lengths.
        """
        # The sequence length increases by 1 due to the empty token
        # This is handled dynamically, so we just pass through the features
        return features


@ProcessorStepRegistry.register(name="lola_image_processor")
class LolaImageProcessor(ObservationProcessorStep):
    """
    Processes multi-camera images for Qwen3.5 vision encoder.

    This processor:
    1. Extracts images from observation.images.* keys
    2. Skips invalid cameras based on camera_valid_mask
    3. Collects PIL Images for Qwen3.5 apply_chat_template

    Supports two input formats:
    - Pretrain mode: camera values are PIL Image (valid) or None (invalid),
      passed as lists from the DataLoader collate (dynamic resolution).
    - Standard mode: camera values are tensors [C, H, W] (legacy fallback).

    Qwen3.5 supports multiple images in a single conversation turn with
    dynamic resolution, so different camera resolutions are handled natively.
    """

    def __init__(self, camera_keys: list[str] | None = None, **kwargs):
        """
        Args:
            camera_keys: List of camera keys to process (e.g., ['observation.images.left', 'observation.images.right']).
                        If None, automatically detects camera keys from observation.
        """
        super().__init__(**kwargs)
        self.camera_keys = camera_keys

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Extracts and prepares images for Qwen3.5 vision encoder.

        Args:
            observation: The observation dictionary containing image data.

        Returns:
            Updated observation with '_lola_images' key containing PIL Images list.
        """
        new_observation = dict(observation)

        # Determine camera keys
        camera_keys = self.camera_keys
        if camera_keys is None:
            # Auto-detect camera keys (keys starting with 'observation.images.')
            camera_keys = [k for k in observation.keys() if k.startswith('observation.images.')]

        if not camera_keys:
            return observation

        # Get camera validity mask
        # In pretrain mode, camera_valid_mask is routed to complementary_data by
        # batch_to_transition (it lacks the "observation." prefix). Read it from
        # self.transition, mirroring how LolaQwenProcessor reads "task".
        camera_valid_mask = observation.get('camera_valid_mask', {})
        if not camera_valid_mask:
            # Fallback: check complementary_data via transition
            if hasattr(self, 'transition') and self.transition is not None:
                from lerobot.processor.core import TransitionKey
                comp_data = self.transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
                camera_valid_mask = comp_data.get('camera_valid_mask', {})

        images = []
        for cam_key in camera_keys:
            if cam_key not in observation:
                continue

            # Skip invalid cameras
            if not camera_valid_mask.get(cam_key, True):
                continue

            img_data = observation[cam_key]

            # Handle list format (dynamic resolution from DataLoader collate)
            if isinstance(img_data, list):
                for img in img_data:
                    if img is not None:
                        if isinstance(img, Image.Image):
                            images.append(img)
                        elif isinstance(img, torch.Tensor) and img.dim() == 3:
                            # Legacy fallback: convert tensor to PIL
                            images.append(self._tensor_to_pil(img))
            elif isinstance(img_data, Image.Image):
                images.append(img_data)
            elif isinstance(img_data, torch.Tensor) and img_data.dim() == 3:
                # Legacy fallback: single tensor [C, H, W]
                images.append(self._tensor_to_pil(img_data))
            elif isinstance(img_data, dict) and 'image' in img_data:
                images.append(img_data['image'])

        # Store processed images for later use in chat template
        if images:
            new_observation['_lola_images'] = images

        return new_observation

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert [C, H, W] float32 tensor to PIL Image."""
        img = tensor.permute(1, 2, 0)  # [C, H, W] → [H, W, C]
        if img.dtype in [torch.float32, torch.float64]:
            img = (img * 255).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(img.cpu().numpy())
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="lola_qwen_processor")
class LolaQwenProcessor(ObservationProcessorStep):
    """
    Processes images and text using Qwen3.5's apply_chat_template.
    
    This is the main processor for LoLA that:
    1. Uses Qwen3.5's AutoProcessor to handle multi-modal input
    2. Formats the conversation using Qwen3.5's chat template
    3. Generates input_ids, attention_mask, pixel_values, and image_grid_thw
    
    The processor uses the apply_chat_template method which is the recommended
    way to prepare inputs for Qwen3.5 models.
    """
    
    def __init__(
        self, 
        processor_name: str = "Qwen/Qwen3.5-4B",
        max_length: int = 512,
        task_key: str = "task",
        **kwargs
    ):
        """
        Args:
            processor_name: The HuggingFace model name for the processor.
            max_length: Maximum sequence length for tokenization.
            task_key: Key in complementary_data containing the task description.
        """
        super().__init__(**kwargs)
        self.processor_name = processor_name
        self.max_length = max_length
        self.task_key = task_key
        
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install transformers`."
            )
        
        self.qwen_processor = AutoProcessor.from_pretrained(processor_name)
    
    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Processes images and text using Qwen3.5's chat template.
        
        Args:
            observation: The observation dictionary containing:
                        - '_lola_images': List of PIL Images (from LolaImageProcessor)
                        - Other observation data
                        
        Returns:
            Updated observation with:
            - 'input_ids': Token IDs from Qwen3.5 processor
            - 'attention_mask': Attention mask
            - 'pixel_values': Processed image tensors (if images present)
            - 'image_grid_thw': Image grid information (if images present)
        """
        new_observation = dict(observation)
        
        # Get task from complementary_data
        task = None
        if hasattr(self, 'transition') and self.transition is not None:
            from lerobot.processor.core import TransitionKey
            complementary_data = self.transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
            task = complementary_data.get(self.task_key, "Perform the robot task.")
        
        if task is None:
            task = "Perform the robot task."
        
        # Get images
        images = observation.get('_lola_images', [])
        
        # Build message for Qwen3.5 chat template
        # Reference from test.py: uses messages format with image and text content
        content = []
        
        # Add images first (Qwen3.5 expects images before text in content)
        for img in images:
            content.append({"type": "image", "image": img})
        
        # Add text prompt
        content.append({"type": "text", "text": task})
        
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        
        # Use apply_chat_template as in test.py
        # This handles tokenization and image processing internally
        inputs = self.qwen_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Extract outputs
        new_observation[OBS_LANGUAGE_TOKENS] = inputs["input_ids"]
        new_observation["attention_mask"] = inputs["attention_mask"]
        
        # Add visual features if present
        if "pixel_values" in inputs:
            new_observation["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            new_observation["image_grid_thw"] = inputs["image_grid_thw"]
        
        # Clean up temporary image storage
        if '_lola_images' in new_observation:
            del new_observation['_lola_images']

        # Clean up camera_valid_mask and camera key observations (not needed downstream)
        if 'camera_valid_mask' in new_observation:
            del new_observation['camera_valid_mask']
        # Remove camera key observations (PIL Image / None / tensor) — already processed into pixel_values
        for key in list(new_observation.keys()):
            if key.startswith('observation.images.'):
                del new_observation[key]

        return new_observation
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Adds feature definitions for Qwen3.5 outputs.
        
        Args:
            features: The input feature dictionary.
            
        Returns:
            Updated feature dictionary with Qwen3.5 output features.
        """
        # Add language tokens feature
        if OBS_LANGUAGE_TOKENS not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_TOKENS] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )
        
        # Add attention mask feature
        if OBS_LANGUAGE_ATTENTION_MASK not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_ATTENTION_MASK] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )
        
        return features


def make_lola_pre_post_processors(
    config: LoLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    camera_keys: list[str] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the LoLA policy.
    
    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Processing multi-camera images for Qwen3.5 vision encoder.
    3. Tokenizing text and images using Qwen3.5's apply_chat_template.
    4. Appending an empty token for global intent control.
    5. Normalizing input and output features based on dataset statistics.
    6. Moving all data to the specified device.
    
    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.
    
    Args:
        config: The configuration object for the LoLA policy.
        dataset_stats: A dictionary of statistics for normalization.
        camera_keys: List of camera keys to process (e.g., ['observation.images.left']).
                    If None, automatically detects camera keys from observation.
        
    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    
    Example:
        ```python
        from lerobot.policies.lola import LoLAConfig, make_lola_pre_post_processors
        
        config = LoLAConfig(vlm_model_name="Qwen/Qwen3.5-4B")
        preprocessor, postprocessor = make_lola_pre_post_processors(
            config,
            dataset_stats=dataset_stats,
            camera_keys=['observation.images.left', 'observation.images.right'],
        )
        ```
    """
    
    # Determine processor settings from config
    vlm_model_name = config.vlm_model_name
    max_length = getattr(config, 'tokenizer_max_length', 512)
    
    # Pre-processor steps
    # The pipeline processes data in the following order:
    # 1. Rename features (compatibility with pretrained format)
    # 2. Process images (extract and convert to PIL)
    # 3. Process with Qwen3.5 (apply_chat_template for text + images)
    # 4. Append empty token (global intent control for LoLA)
    # 5. Add batch dimension
    # 6. Move to device
    # 7. Normalize features
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To maintain compatibility with pretrained format
        LolaImageProcessor(camera_keys=camera_keys),  # Extract and prepare images for Qwen3.5
        LolaQwenProcessor(  # Process text + images with Qwen3.5's apply_chat_template
            processor_name=vlm_model_name,
            max_length=max_length,
        ),
        LolaEmptyTokenProcessor(empty_token_id=config.empty_token_id),  # Append empty token for LoLA
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    
    # Post-processor steps
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, 
            norm_map=config.normalization_mapping, 
            stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
