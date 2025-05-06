from typing import Dict
import torch


from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.robomimic.bc_rnn import BCRNN
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class BCRNNPolicy(BaseImagePolicy):
    """
    BC-RNN policy based on robomimic BC-RNN
    """
    def __init__(self, shape_meta: dict, action_horizon: int, device='cuda:0', unnormalize_action='True'):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        self.model = BCRNN(shape_meta, action_dim, action_horizon, device)

        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.unnormalize_action = unnormalize_action

    def predict_action(self,
                       obs_dict: Dict[str, torch.Tensor],
                       fixed_action_prefix=None,
                       obs_timestamps=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        if obs_dict.get('obs', None) is None:
            nobs = self.normalizer.normalize(obs_dict)
        else:
            nobs = self.normalizer.normalize(obs_dict['obs'])
        # nactions = self.normalizer['action'].normalize(obs_dict['action'])
        b = next(iter(nobs.values())).shape[0]
        robomimic_batch = {'obs': nobs, 'actions': torch.zeros((b, self.action_horizon, self.action_dim))}
        # print('obs', nobs)

        action_pred = self.model.predict_action(robomimic_batch)['actions']

        # unnormalize prediction
        assert action_pred.shape == (b, self.action_horizon, self.action_dim), f'{action_pred.shape}'

        if self.unnormalize_action:
            # print('before unnormalize', action_pred)
            action_pred = self.normalizer['action'].unnormalize(action_pred)
            # print('after unnormalize', action_pred)

        return {'action_pred': action_pred}

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert self.unnormalize_action
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        robomimic_batch = {'obs': nobs, 'actions': nactions}
        # print(nactions.shape)
        # for ob_k, ob_v in nobs.items():
        #     print(ob_k, ob_v.shape)
        predictions = self.model.predict_action(robomimic_batch)
        # print(predictions['actions'].shape, robomimic_batch['actions'].shape)
        # print(predictions['actions'].device, robomimic_batch['actions'].device)
        losses = self.model.compute_loss(robomimic_batch, predictions)
        # exit(0)
        return losses['action_loss']

    def forward(self, batch):
        return self.compute_loss(batch)

    def get_parameters(self):
        return self.model.parameters()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        model_state_dict = self.model.nets.state_dict()
        origin_state_dict = super().state_dict(destination, prefix, keep_vars)
        return {'origin': origin_state_dict, 'model': model_state_dict}

    def load_state_dict(self, state_dict, strict = True, assign = False):
        self.model.nets.load_state_dict(state_dict['model'], strict=strict)
        super().load_state_dict(state_dict['origin'], strict=strict, assign=assign)
