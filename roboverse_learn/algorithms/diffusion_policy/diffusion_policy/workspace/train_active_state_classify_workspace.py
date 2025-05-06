if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import pickle
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.dataset.umi_active_dataset import UmiActiveDataset
from diffusion_policy.policy.active_state_classify_policy import ActiveStateClassifyPolicy
from active_learning.video_manager import VideoManager
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs  # 改用这个类

import pandas as pd
OmegaConf.register_new_resolver("eval", eval, replace=True)

def log_state_loss(step_log, category, pred_state_logit, gt_state, is_warmup=False):
    B, T, _ = pred_state_logit.shape
    pred_state_logit = pred_state_logit.view(B * T, -1)
    gt_state = gt_state.view(B * T).long()
    cross_entrop = torch.nn.CrossEntropyLoss()
    category = f"warmup_{category}" if is_warmup else category
    step_log[f'{category}_state_cross_loss'] = cross_entrop(pred_state_logit, gt_state)
def log_state_succ(step_log, category, pred_state, gt_state, is_warmup=False):
    B, T = pred_state.shape  # 假设 pred_state 的形状是 (B, T)
    assert pred_state.shape == gt_state.shape  # 检查形状是否一致
    category = f"warmup_{category}" if is_warmup else category
    step_log[f'{category}_state_succ'] = torch.sum(pred_state == gt_state).float() / (B * T)  # 计算成功率

class TrainActiveStateClassifyWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: ActiveStateClassifyPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: ActiveStateClassifyPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=self.model.parameters())

        obs_encorder_lr = cfg.optimizer.lr
        if cfg.policy.obs_encoder.pretrained:
            obs_encorder_lr *= 0.1
            print('==> reduce pretrained obs_encorder\'s lr')
        obs_encorder_params = list()
        for param in self.model.obs_encoder.parameters():
            if param.requires_grad:
                obs_encorder_params.append(param)
        print(f'obs_encorder params: {len(obs_encorder_params)}')
        param_groups = [
            {'params': self.model.model.parameters()},
            {'params': obs_encorder_params, 'lr': obs_encorder_lr},
            {'params': self.model.state_estimator.parameters()}, 
            {'params': self.model.state_embedding.parameters()},
        ]
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=param_groups)
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop('_target_')
        self.optimizer = torch.optim.AdamW(
            params=param_groups,
            **optimizer_cfg
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0
        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        accelerator = Accelerator(
            log_with='wandb',
            kwargs_handlers=[ddp_kwargs]
        )
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: UmiActiveDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset) or isinstance(dataset, BaseDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        state_dataset = dataset.get_state_dataset()
        state_dataloader = DataLoader(state_dataset, **cfg.state_dataloader)

        print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))
        print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))
        print('state dataset:', len(state_dataset), 'state dataloader:', len(state_dataloader))
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(state_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # accelerator
        state_dataloader, train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            state_dataloader, train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )
        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        warmup_path = os.path.join(self.output_dir, 'warmup_logs.json.txt')
        warmup_losses = list()
        with JsonLogger(warmup_path) as warmup_logger:
            for local_epoch_idx in range(cfg.active_learning.training.num_epochs):
                self.model.train()
                step_log = dict()
                # ========= train for this epoch ==========
                with tqdm.tqdm(state_dataloader, desc=f"Warming up epoch {self.epoch}",
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        train_sampling_batch = batch

                        raw_loss = self.model(batch) # 调用model的forward
                        accelerator.backward(raw_loss)
                        
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                        if cfg.training.use_ema:
                            ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        warmup_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'epoch': self.epoch,
                            'global_step': self.global_step,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        is_last_batch = (batch_idx == (len(state_dataloader)-1))
                        if not is_last_batch:
                            accelerator.log(step_log)
                            warmup_logger.log(step_log)
                            self.global_step += 1
                warmup_loss = np.mean(warmup_losses)
                step_log['train_loss'] = warmup_loss
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()
                if accelerator.is_main_process:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        gt_state = batch['state']
                        pred = policy.predict_action(batch['obs'])

                        log_state_loss(step_log, 'train', pred['state_pred_logit'], gt_state)
                        log_state_succ(step_log, 'train', pred['state_pred'], gt_state)
                    del batch
                    del gt_state
                    with torch.no_grad():
                        gt_state_list = []
                        pred_state_list, pred_state_logit_list = [], []
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                pred = policy.predict_action(batch['obs'])

                                gt_state_list.append(batch['state'].detach().cpu())
                                pred_state_list.append(pred['state_pred'].detach().cpu())
                                pred_state_logit_list.append(pred['state_pred_logit'].detach().cpu())

                        gt_state = torch.cat(gt_state_list, dim=0)
                        pred_state = torch.cat(pred_state_list, dim=0)
                        pred_state_logit = torch.cat(pred_state_logit_list, dim=0)
                        log_state_succ(step_log, 'val', pred_state, gt_state, is_warmup=True)
                        log_state_loss(step_log, 'val', pred_state_logit, gt_state, is_warmup=True)
                    del batch
                accelerator.log(step_log)
                warmup_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
            if accelerator.is_main_process:
                    # unwrap the model to save ckpt
                model_ddp = self.model
                self.model = accelerator.unwrap_model(self.model)

                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value

                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                print(metric_dict)  # 手动保存一下

                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)

                self.model = model_ddp
        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainActiveStateClassifyWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
