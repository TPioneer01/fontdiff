import math
import torch
import torch.nn as nn

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, 
                                           register_to_config)

class FontDiffuserModel(ModelMixin, ConfigMixin):
    """Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """

    @register_to_config
    def __init__(
        self, 
        unet, 
        style_encoder,
        content_encoder,
        edge_fusion_scale=0.25,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.edge_adapter_content = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.edge_adapter_style = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.edge_adapter_content.weight)
        nn.init.zeros_(self.edge_adapter_content.bias)
        nn.init.zeros_(self.edge_adapter_style.weight)
        nn.init.zeros_(self.edge_adapter_style.bias)
        
        # Use internal parameter name to avoid conflicts with @register_to_config
        # The config maintains edge_fusion_scale, but we store as _edge_fusion_scale_param
        self._edge_fusion_scale_param = nn.Parameter(torch.tensor(edge_fusion_scale, dtype=torch.float32))
    
    @property
    def edge_fusion_scale(self):
        """Public interface to access edge_fusion_scale as parameter"""
        return self._edge_fusion_scale_param
    
    @edge_fusion_scale.setter
    def edge_fusion_scale(self, value):
        """Public setter for edge_fusion_scale"""
        if isinstance(value, nn.Parameter):
            self._edge_fusion_scale_param = value
        elif isinstance(value, torch.Tensor):
            if not isinstance(self._edge_fusion_scale_param, nn.Parameter):
                self._edge_fusion_scale_param = nn.Parameter(value)
            else:
                self._edge_fusion_scale_param.data.copy_(value)
        else:
            self._edge_fusion_scale_param = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def _inject_edge(self, image, edge_map, adapter):
        if edge_map is None:
            return image
        edge_residual = torch.tanh(adapter(edge_map))
        fused = image + self._edge_fusion_scale_param * edge_residual
        return fused.clamp(-1.0, 1.0)
    
    def forward(
        self, 
        x_t, 
        timesteps, 
        style_images,
        content_images,
        content_encoder_downsample_size,
        content_edges=None,
        style_edges=None,
    ):
        content_images = self._inject_edge(content_images, content_edges, self.edge_adapter_content)
        style_images = self._inject_edge(style_images, style_edges, self.edge_adapter_style)

        style_img_feature, _, _ = self.style_encoder(style_images)
    
        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
    
        # Get the content feature
        content_img_feature, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feature)
        # Get the content feature from reference image
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [style_img_feature, content_residual_features, \
                               style_hidden_states, style_content_res_features]

        out = self.unet(
            x_t, 
            timesteps, 
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        offset_out_sum = out[1]
        
        return noise_pred, offset_out_sum


class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """DPM Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """
    @register_to_config
    def __init__(
        self, 
        unet, 
        style_encoder,
        content_encoder,
        edge_fusion_scale=0.25,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.edge_adapter_content = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.edge_adapter_style = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.edge_adapter_content.weight)
        nn.init.zeros_(self.edge_adapter_content.bias)
        nn.init.zeros_(self.edge_adapter_style.weight)
        nn.init.zeros_(self.edge_adapter_style.bias)
        
        # Use internal parameter name to avoid conflicts with @register_to_config
        # The config maintains edge_fusion_scale, but we store as _edge_fusion_scale_param
        self._edge_fusion_scale_param = nn.Parameter(torch.tensor(edge_fusion_scale, dtype=torch.float32))
    
    @property
    def edge_fusion_scale(self):
        """Public interface to access edge_fusion_scale as parameter"""
        return self._edge_fusion_scale_param
    
    @edge_fusion_scale.setter
    def edge_fusion_scale(self, value):
        """Public setter for edge_fusion_scale"""
        if isinstance(value, nn.Parameter):
            self._edge_fusion_scale_param = value
        elif isinstance(value, torch.Tensor):
            if not isinstance(self._edge_fusion_scale_param, nn.Parameter):
                self._edge_fusion_scale_param = nn.Parameter(value)
            else:
                self._edge_fusion_scale_param.data.copy_(value)
        else:
            self._edge_fusion_scale_param = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def _inject_edge(self, image, edge_map, adapter):
        if edge_map is None:
            return image
        edge_residual = torch.tanh(adapter(edge_map))
        fused = image + self._edge_fusion_scale_param * edge_residual
        return fused.clamp(-1.0, 1.0)
    
    def forward(
        self, 
        x_t, 
        timesteps, 
        cond,
        content_encoder_downsample_size,
        version,
    ):
        content_images = cond[0]
        style_images = cond[1]
        content_edges = cond[2] if len(cond) > 2 else None
        style_edges = cond[3] if len(cond) > 3 else None

        content_images = self._inject_edge(content_images, content_edges, self.edge_adapter_content)
        style_images = self._inject_edge(style_images, style_edges, self.edge_adapter_style)

        style_img_feature, _, style_residual_features = self.style_encoder(style_images)
        
        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
        
        # Get content feature
        content_img_feture, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feture)
        # Get the content feature from reference image
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [style_img_feature, content_residual_features, style_hidden_states, style_content_res_features]

        out = self.unet(
            x_t, 
            timesteps, 
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        
        return noise_pred
