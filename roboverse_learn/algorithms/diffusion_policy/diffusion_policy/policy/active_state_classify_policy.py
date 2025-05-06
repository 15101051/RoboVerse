from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from umi.real_world.state_planner import StatePlanner
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.contrastive_learning import StateContrastor

class ActiveStateClassifyPolicy(BaseImagePolicy):
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
            frequency=10,
            # parameters passed to step
            **kwargs
        ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        self.state_num = shape_meta['obs']['state']['num']
        self.state_dim = shape_meta['obs']['state']['encode_dim']
        self.state_horizon = shape_meta['action']['state_horizon']
        self.action_horizon = action_horizon
        self.frequency = frequency
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())


        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        
        state_estimator = nn.Sequential(
            nn.Linear(obs_feature_dim, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, self.state_horizon * self.state_num)
        )

        state_embedding = nn.Sequential(
            nn.Embedding(self.state_num, self.state_dim),
            nn.Flatten(start_dim=1),
            nn.Linear(2 * self.state_dim, 2 * self.state_dim),
            nn.Mish(),
        )
        self.obs_encoder = obs_encoder
        # self.contrastor = StateContrastor(obs_feature_dim, self.state_dim)
        self.model = model
        self.state_estimator = state_estimator
        self.state_embedding = state_embedding
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
        self.state = torch.tensor([[[0],[0]]], device='cuda', dtype=torch.float32) # initial state
        self.state_planner = StatePlanner(self.state_num)
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())


    def forward(self, batch):
        for param in self.obs_encoder.parameters():
            param.requires_grad = True
        nobs = self.normalizer.normalize(batch['obs'])
        label_states = batch['state']
        # 冻结obs_encoder
        global_cond = self.obs_encoder(nobs)
        state_embedding = self.state_embedding(nobs['state'].reshape(nobs['state'].shape[0], -1).long())
        global_cond = torch.cat([global_cond, state_embedding], dim=-1)
        pred_state_logit = self.state_estimator(global_cond)
        pred_state_logit = pred_state_logit.view(-1, self.state_horizon, self.state_num)
        labels = label_states.long().to(pred_state_logit.device)

        criterion = nn.CrossEntropyLoss()
        classification_loss = criterion(pred_state_logit.view(-1, self.state_num), labels.view(-1))
        return classification_loss
        
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], _=None, eval_real=None) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        global_cond = self.obs_encoder(nobs)
        state_embedding = self.state_embedding(nobs['state'].reshape(nobs['state'].shape[0], -1).long())
        global_cond = torch.cat([global_cond, state_embedding], dim=-1)
        pred_state_logit = self.state_estimator(global_cond)
        pred_state_logit = pred_state_logit.view(-1, self.state_horizon, self.state_num)
        state_pred = torch.argmax(F.softmax(pred_state_logit, dim=-1), dim=-1)
  
        result = {
            'state_pred_logit': pred_state_logit,
            'state_pred': state_pred,
        }
        return result