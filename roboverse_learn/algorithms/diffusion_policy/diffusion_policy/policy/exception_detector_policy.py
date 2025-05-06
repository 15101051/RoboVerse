from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from torch.nn import BCEWithLogitsLoss

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.exception_detector import ExceptionDetector

class ExceptionDetectorPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TimmObsEncoder,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            input_pertub=0.1,
            inpaint_fixed_action_prefix=False,
            train_diffusion_n_samples=1,
            # parameters passed to step
            **kwargs
        ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())


        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ExceptionDetector()

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon # used for training
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: 必须包含 "obs" 键
        fixed_action_prefix: 非归一化的行动前缀
        result: 必须包含 "action" 键
        """
        assert 'past_action' not in obs_dict  # 暂未实现
        nobs = self.normalizer.normalize(obs_dict['obs'])
        B = next(iter(nobs.values())).shape[0]
        global_cond = self.obs_encoder(nobs)  
        pred_logits = self.model(global_cond)
        pred_prob = torch.sigmoid(pred_logits).squeeze()
        action = (pred_prob > 0.5).long()  # 如果概率大于 0.5，则预测为类别1，否则为类别0

        return {'pred': action}

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)  # (batch * 1568)
        
        # Extract target (binary: 0 or 1)
        target = batch['obs']['human_operation'][:,1].long()  # Assumes 'human_operation' is either 0 or 1
        
        # Ensure the model is a binary classifier MLP
        pred = self.model(global_cond)  # Output is of shape (batch_size, 1)
        
        # Use BCEWithLogitsLoss for binary classification
        loss_fn = BCEWithLogitsLoss(reduction='none')  # Raw logits output
        loss = loss_fn(pred.squeeze(), target.float())  # target needs to be float for BCEWithLogitsLoss

        # Reduce the loss
        loss = reduce(loss, 'b ... -> b (...)', 'mean')  # Mean over batch
        loss = loss.mean()  # Final average

        return loss


    def forward(self, batch):
        return self.compute_loss(batch)