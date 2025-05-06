import torch


def generate_random_data_example(shape_meta, obs_horizon, action_horizon, is_self_regression, device='cuda:0'):
    obs = {}
    for key in shape_meta['obs'].keys():
        shape = tuple([obs_horizon] + shape_meta['obs'][key]['shape'])
        obs[key] = torch.randn(shape, device=device).unsqueeze(0)
    action_length = obs_horizon + action_horizon - 1 if is_self_regression else action_horizon
    shape = (action_length, shape_meta['action']['shape'][0])
    assert len(shape_meta['action']['shape']) == 1
    return {'obs': obs, 'action': torch.randn(shape, device=device).unsqueeze(0)}
