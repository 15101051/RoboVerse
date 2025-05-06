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

class StateDiffusionUnetTimmPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TimmObsEncoder,
            # state_encoder: TimmObsEncoder,
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
        self.state_num = shape_meta['obs']['state']['num']
        self.state_dim = shape_meta['obs']['state']['encode_dim']
        self.action_horizon = shape_meta['action']['horizon']
        self.state_horizon = shape_meta['action']['state_horizon']
        self.frequency = frequency
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())
        # state_feature_dim = np.prod(state_encoder.output_shape())


        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim + 2 * self.state_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        # MLP to predict state from obs
        state_estimator = nn.Sequential(
            nn.Linear(obs_feature_dim + 2 * self.state_dim, 512),
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
            # nn.Linear(2, 32),
            # nn.ReLU(),
        )
        # self.state_encoder = state_encoder
        self.obs_encoder = obs_encoder
        # self.contrastor = StateContrastor(obs_feature_dim, self.state_dim)
        self.model = model
        self.state_estimator = state_estimator
        self.state_embedding = state_embedding
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
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

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
        ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None, eval_real=True, obs_timestamps=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        if obs_timestamps is None and eval_real is True:
            print('warming up state planner...')
            nobs['state'] = self.state
        if eval_real and obs_timestamps is not None: # 插值器
            self.state_planner.cleanup_old_data(time.time(), 10)
            state_0 = self.state_planner.interpolate(obs_timestamps[0])
            state_1 = self.state_planner.interpolate(obs_timestamps[1])
            print(f"{state_0} -> {state_1}")
            state_pred = torch.tensor([[state_0],[state_1]], device='cuda', dtype=torch.float32)
            state_pred = state_pred.unsqueeze(0).expand(B, -1, -1)  # 形状: (B, 2, 1)
            nobs['state'] = state_pred
        # condition through global feature
        visual_features = self.obs_encoder(nobs)
        flattened_states = nobs['state'].reshape(nobs['state'].shape[0], -1)
        state_features = self.state_embedding(flattened_states.long())

        visual_features_detached = visual_features.detach()
        state_features_detached = state_features.detach()

        unet_vector = torch.cat([visual_features, state_features_detached], dim=-1)
        mlp_vector = torch.cat([visual_features_detached, state_features], dim=-1)

        output = self.state_estimator(mlp_vector)  # shape: (batch_size, state_horizon * state_num)
        state_pred_logit = output.view(-1, self.state_horizon, self.state_num)
        state_pred = torch.argmax(F.softmax(state_pred_logit, dim=-1), dim=-1)  # shape: (batch_size, action_horizon)
        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)


        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=unet_vector,
            **self.kwargs)
        
        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)
        
        result = {
            'action': action_pred,
            'action_pred': action_pred,
            'state_pred_logit': state_pred_logit,
            'state_pred': state_pred,
        }
        if eval_real and obs_timestamps is not None:
            dt = 1/self.frequency
            pred_states_prob = F.softmax(state_pred_logit, dim=-1).detach().to('cpu').numpy()[0]
            pred_states_times = (np.arange(len(pred_states_prob), dtype=np.float64)) * dt + obs_timestamps[-1]
            is_new = pred_states_times > time.time()
            if np.sum(is_new) == 0: # 表示预测的状态都是过去的状态,is_new全为False,False求和为0
                print('Over budget State')
                # TODO: 使用最新状态
            else:
                pred_states_times = pred_states_times[is_new]
                pred_states_prob = pred_states_prob[is_new] # shape : (valid_horizon, state_num)
            self.state_planner.update(pred_states_times, pred_states_prob)

        return result

    def predict_action_with_grad(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None, eval_real=True, obs_timestamps=None) -> Dict[str, torch.Tensor]:
        """
        支持梯度计算的 predict_action 版本
        obs_dict: 必须包含 "obs" 键
        fixed_action_prefix: 未归一化的动作前缀
        返回结果: 必须包含 "action" 键
        """
        assert 'past_action' not in obs_dict  # 未实现
        # 归一化输入
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        if obs_timestamps is None and eval_real is True:
            print('warming up state planner...')
            nobs['state'] = self.state
        if eval_real and obs_timestamps is not None:  # 插值器
            self.state_planner.cleanup_old_data(time.time(), 10)
            state_0 = self.state_planner.interpolate(obs_timestamps[0])
            state_1 = self.state_planner.interpolate(obs_timestamps[1])
            print(f"{state_0} -> {state_1}")
            state_pred = torch.tensor([[state_0],[state_1]], device='cuda', dtype=torch.float32, requires_grad=True)
            state_pred = state_pred.unsqueeze(0).expand(B, -1, -1)  # 形状: (B, 2, 1)
            nobs['state'] = state_pred
        # 通过全局特征进行条件化
        visual_features = self.obs_encoder(nobs)
        flattened_states = nobs['state'].reshape(nobs['state'].shape[0], -1)
        state_features = self.state_embedding(flattened_states.long())

        visual_features_detached = visual_features.detach()
        state_features_detached = state_features.detach()

        unet_vector = torch.cat([visual_features, state_features_detached], dim=-1)
        mlp_vector = torch.cat([visual_features_detached, state_features], dim=-1)

        output = self.state_estimator(mlp_vector)  # 形状: (batch_size, action_horizon * state_num)
        state_pred_logit = output.view(-1, self.state_horizon, self.state_num)
        # 移除不可微的操作（如 torch.argmax 和 F.softmax）
        # state_pred = torch.argmax(F.softmax(state_pred_logit, dim=-1), dim=-1)  # 移除
        # 为动作创建空数据
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)

        # 运行采样
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=unet_vector,
            **self.kwargs)
        
        # 反归一化预测结果
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)
        
        # 返回结果
        result = {
            'action': action_pred,
            'action_pred': action_pred,
            'state_pred_logit': state_pred_logit,  # 保留支持梯度的 logit
            # 'state_pred': state_pred,  # 移除不可微的操作
        }
        if eval_real and obs_timestamps is not None:
            dt = 1/self.frequency
            pred_states_prob = F.softmax(state_pred_logit, dim=-1).to('cpu').numpy()[0]  # 移除 detach()
            pred_states_times = (np.arange(len(pred_states_prob), dtype=np.float64)) * dt + obs_timestamps[-1]
            is_new = pred_states_times > time.time()
            if np.sum(is_new) == 0:  # 表示预测的状态都是过去的状态,is_new全为False,False求和为0
                print('Over budget State')
                # TODO: 使用最新状态
            else:
                pred_states_times = pred_states_times[is_new]
                pred_states_prob = pred_states_prob[is_new]  # 形状: (valid_horizon, state_num)
            self.state_planner.update(pred_states_times, pred_states_prob)

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        label_states = batch['state']
        assert self.obs_as_global_cond
        visual_features = self.obs_encoder(nobs)
        flattened_states = nobs['state'].reshape(nobs['state'].shape[0], -1)
        state_features = self.state_embedding(flattened_states.long())

        visual_features_detached = visual_features.detach()
        state_features_detached = state_features.detach()

        unet_vector = torch.cat([visual_features, state_features_detached], dim=-1)
        mlp_vector = torch.cat([visual_features_detached, state_features], dim=-1)
        # train on multiple diffusion samples per obs
        if self.train_diffusion_n_samples != 1:
            # repeat obs features and actions multiple times along the batch dimension
            # each sample will later have a different noise sample, effecty training 
            # more diffusion steps per each obs encoder forward pass
            unet_vector = torch.repeat_interleave(unet_vector, 
                repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(nactions, 
                repeats=self.train_diffusion_n_samples, dim=0)

        trajectory = nactions
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            local_cond=None,
            global_cond=unet_vector
        )
        pred_state_logit = self.state_estimator(mlp_vector)
        pred_state_logit = pred_state_logit.view(-1, self.state_horizon, self.state_num)

        labels = label_states.long().to(pred_state_logit.device)
        criterion = nn.CrossEntropyLoss()
        classification_loss = criterion(pred_state_logit.view(-1, self.state_num), labels.view(-1))
        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        regression_loss = F.mse_loss(pred, target, reduction='none')
        regression_loss = regression_loss.type(regression_loss.dtype)
        regression_loss = reduce(regression_loss, 'b ... -> b (...)', 'mean')
        regression_loss = regression_loss.mean()

        # 对比学习模块
        # contrastive_loss = self.contrastor(global_cond, label_states)
 
        loss = classification_loss + regression_loss
        return loss

    def forward(self, batch):
        return self.compute_loss(batch)