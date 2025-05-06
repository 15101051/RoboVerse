from typing import Optional
import random
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st
from diffusion_policy.common.replay_buffer import ReplayBuffer

def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


class RNNSequenceSampler:
    def __init__(self,
        shape_meta: dict,
        replay_buffer: ReplayBuffer,
        rgb_keys: list,
        humandim_keys: list,
        lowdim_keys: list,
        obs_horizon: int,
        action_horizon: int,
        key_latency_steps: dict,
        down_sample_steps: int,
        episode_mask: Optional[np.ndarray]=None,
        action_padding: bool=False,
        repeat_frame_prob: float=0.0,
        max_duration: Optional[float]=None
    ):
        episode_ends = replay_buffer.episode_ends[:]

        # load gripper_width
        gripper_width = replay_buffer['robot0_gripper_width'][:, 0]
        gripper_width_threshold = 0.08
        self.repeat_frame_prob = repeat_frame_prob

        # create indices, including (current_idx, start_idx, end_idx)
        indices = list()
        for i, episode_end in enumerate(episode_ends):
            before_first_grasp = True # initialize for each episode
            if episode_mask is not None and not episode_mask[i]:
                # skip episode
                continue
            start_idx = 0 if i == 0 else episode_ends[i-1]
            end_idx = episode_end
            if max_duration is not None:
                end_idx = min(end_idx, max_duration * 60)
            for current_idx in range(start_idx + (obs_horizon - 1) * down_sample_steps, end_idx):
                if not action_padding and \
                    end_idx < current_idx + (action_horizon - 1) * down_sample_steps + 1:
                    continue
                if gripper_width[current_idx] < gripper_width_threshold:
                    before_first_grasp = False
                indices.append((current_idx, start_idx, end_idx, before_first_grasp))

        # load low_dim to memory and keep rgb as compressed zarr array
        self.replay_buffer = dict()
        self.num_robot = 0
        for key in lowdim_keys:
            if key.endswith('eef_pos'):
                self.num_robot += 1
            if key.endswith('pos_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                self.replay_buffer[key] = replay_buffer[key[:-4]][:, list(axis)]
            elif key.endswith('quat_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                # HACK for hybrid abs/relative proprioception
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_quat(rot_in).as_euler('XYZ')
                self.replay_buffer[key] = rot_out[:, list(axis)]
            elif key.endswith('axis_angle_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_rotvec(rot_in).as_euler('XYZ')
                self.replay_buffer[key] = rot_out[:, list(axis)]
            else:
                self.replay_buffer[key] = replay_buffer[key][:]
        for key in rgb_keys:
            self.replay_buffer[key] = replay_buffer[key]
        for key in humandim_keys:
            self.replay_buffer[key] = replay_buffer[key]

        if 'action' in replay_buffer:
            self.replay_buffer['action'] = replay_buffer['action'][:]
        else:
            # construct action (concatenation of [eef_pos, eef_rot, gripper_width])
            actions = list()
            for robot_idx in range(self.num_robot):
                for cat in ['eef_pos', 'eef_rot_axis_angle', 'gripper_width']:
                    key = f'robot{robot_idx}_{cat}'
                    if key in self.replay_buffer:
                        actions.append(self.replay_buffer[key])
            self.replay_buffer['action'] = np.concatenate(actions, axis=-1)
        self.huamndim_dict = dict()
        self.human_operations = None
        self.action_padding = action_padding
        self.indices = indices
        self.rgb_keys = rgb_keys
        self.humandim_keys = humandim_keys
        self.lowdim_keys = lowdim_keys
        self.obs_horizon, self.action_horizon = obs_horizon, action_horizon
        self.key_latency_steps = key_latency_steps
        self.down_sample_steps = down_sample_steps

        self.ignore_rgb_is_applied = False # speed up the interation when getting normalizaer

        if 'human_operation' in replay_buffer and 'human_operation' in humandim_keys:
            human_operations = self.replay_buffer['human_operation']
            self.human_operations = [0] * len(human_operations)
            current_state = "Succ"
            for idx in range(len(human_operations) - 1, -1, -1):
                item = human_operations[idx]
                if item != '':
                    current_state = item
                if current_state == "Succ":
                    self.human_operations[idx] = 1
            self.human_operations = np.array(self.human_operations)
            self.huamndim_dict['human_operation'] = self.human_operations

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        current_idx, start_idx, end_idx, before_first_grasp = self.indices[idx]

        result = dict()

        obs_keys = self.rgb_keys + self.lowdim_keys + self.humandim_keys
        if self.ignore_rgb_is_applied:
            obs_keys = self.lowdim_keys

        assert self.obs_horizon <= (current_idx - start_idx) // self.down_sample_steps + 1
        slice_start = current_idx - (self.obs_horizon - 1) * self.down_sample_steps

        # observation
        for key in obs_keys:
            input_arr = self.replay_buffer[key]
            this_latency_steps = self.key_latency_steps[key]

            if key in self.rgb_keys or key in self.humandim_keys: # humandim 和 rgb 是完全对齐的，处理方式一样
                if key in self.humandim_keys:
                    input_arr = self.huamndim_dict[key]
                assert this_latency_steps == 0

                output = input_arr[slice_start: current_idx + 1: self.down_sample_steps]
                assert output.shape[0] == self.obs_horizon

                # solve padding
                if output.shape[0] < self.obs_horizon:
                    padding = np.repeat(output[:1], self.obs_horizon - output.shape[0], axis=0)
                    output = np.concatenate([padding, output], axis=0)
                if key in self.humandim_keys: #shape从(2)变成(2,1)，增加一个维度，来和其余 low dim 形式上对齐
                    output = output.reshape(-1, 1)
            else:
                idx_with_latency = np.array([current_idx - idx * self.down_sample_steps + this_latency_steps
                                             for idx in range(self.obs_horizon)], dtype=np.float32)[::-1]
                idx_with_latency = np.clip(idx_with_latency, start_idx, end_idx - 1)
                interpolation_start = max(int(idx_with_latency[0]) - 5, start_idx)
                interpolation_end = min(int(idx_with_latency[-1]) + 2 + 5, end_idx)

                if 'rot' in key:
                    # rotation
                    rot_preprocess, rot_postprocess = None, None
                    if key.endswith('quat'):
                        rot_preprocess = st.Rotation.from_quat
                        rot_postprocess = st.Rotation.as_quat
                    elif key.endswith('axis_angle'):
                        rot_preprocess = st.Rotation.from_rotvec
                        rot_postprocess = st.Rotation.as_rotvec
                    else:
                        raise NotImplementedError
                    slerp = st.Slerp(
                        times=np.arange(interpolation_start, interpolation_end),
                        rotations=rot_preprocess(input_arr[interpolation_start: interpolation_end]))
                    output = rot_postprocess(slerp(idx_with_latency))
                else:
                    interp = si.interp1d(
                        x=np.arange(interpolation_start, interpolation_end),
                        y=input_arr[interpolation_start: interpolation_end],
                        axis=0, assume_sorted=True)
                    output = interp(idx_with_latency)

            result[key] = output

        # repeat frame before first grasp
        if self.repeat_frame_prob != 0.0:
            if before_first_grasp and random.random() < self.repeat_frame_prob:
                for key in obs_keys:
                    result[key][:-1] = result[key][-1:]

        # aciton
        assert self.key_latency_steps['action'] == 0
        slice_end = min(end_idx, current_idx + (self.action_horizon - 1) * self.down_sample_steps + 1)
        output = self.replay_buffer['action'][slice_start: slice_end: self.down_sample_steps]
        # solve padding
        action_length = self.obs_horizon + self.action_horizon - 1
        if not self.action_padding:
            assert output.shape[0] == action_length, f"{output.shape[0]} != {action_length}"
        elif output.shape[0] < action_length:
            padding = np.repeat(output[-1:], action_length - output.shape[0], axis=0)
            output = np.concatenate([output, padding], axis=0)
        result['action'] = output
        return result

    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply
