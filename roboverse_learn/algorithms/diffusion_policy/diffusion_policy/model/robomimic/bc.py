from robomimic.algo import BC
from robomimic.config import BCConfig
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.common.robomimic_config_util import get_robomimic_config_bc_baseline_modified


class BehaviorCloning(BC):
    """
    wrapper for robomimic BC
    construct robomimic config and implement predict_action and compute_loss
    """
    def __init__(self, shape_meta, action_dim, action_horizon, device):
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                obs_config['rgb'].append(key)
            elif obs_type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        config: BCConfig = get_robomimic_config_bc_baseline_modified()
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            for _, modality in config.observation.encoder.items():
                if modality.obs_randomizer_class == 'CropRandomizer':
                    modality['obs_randomizer_class'] = None
        ObsUtils.initialize_obs_utils_with_config(config)
        super().__init__(algo_config=config.algo, obs_config=config.observation, global_config=config,
                        obs_key_shapes=obs_key_shapes, ac_dim=action_dim*action_horizon, device=device)
        self.action_dim = action_dim
        self.action_horizon = action_horizon

    def parameters(self):
        return self.nets.parameters()

    def predict_action(self, batch):
        batch = self.process_batch_for_training(batch)
        predictions = self._forward_training(batch)
        predictions['actions'] = predictions['actions'][:, -1, :].view(-1, self.action_horizon, self.action_dim)
        return predictions

    def compute_loss(self, batch, predictions):
        return self._compute_losses(predictions, batch)
        