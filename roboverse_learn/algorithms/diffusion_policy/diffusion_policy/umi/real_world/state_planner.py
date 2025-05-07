import numpy as np
import scipy.interpolate as interp
from collections import deque

class StatePlanner:
    def __init__(self, state_num, max_window_size=100):
        """
        初始化动态插值器
        :param state_num: 状态数量
        :param max_window_size: 滑动窗口的最大大小（保留最近的数据量）
        """
        self.state_num = state_num
        self.init_state = 0
        self.max_window_size = max_window_size
        self.pred_states_times = deque(maxlen=max_window_size)  # 时间戳队列
        self.pred_states_prob = deque(maxlen=max_window_size)   # 概率分布队列
        self.interpolators = [None] * state_num  # 插值器列表

    def update(self, new_times, new_probs):
        """
        更新数据并重新创建插值器
        :param new_times: 新时间戳数组
        :param new_probs: 新概率分布数组（形状为 (时间点数, 状态数))
        """
        # 将新数据添加到队列中
        for t, prob in zip(new_times, new_probs):
            self.pred_states_times.append(t)
            self.pred_states_prob.append(prob)

        # 将队列转换为 NumPy 数组
        times = np.array(self.pred_states_times)
        probs = np.array(self.pred_states_prob)

        # 确保时间戳是唯一的
        unique_times, unique_indices = np.unique(times, return_index=True)
        unique_probs = probs[unique_indices]

        self.pred_states_times = deque(unique_times, maxlen=self.max_window_size)
        self.pred_states_prob = deque(unique_probs, maxlen=self.max_window_size)

        # 重新创建插值器
        for i in range(self.state_num):
            state_prob = unique_probs[:, i]
            self.interpolators[i] = interp.interp1d(
                unique_times,       # 时间戳
                state_prob,         # 当前状态的概率分布
                kind='linear',      # 使用线性插值
                fill_value="extrapolate"  # 允许外推
            )

    def get_nearby_times(self, desired_time, window=3, num_points=5):
        """
        获取插值时间点附近的时间戳
        :param desired_time: 插值时间点
        :param window: 时间窗口大小（默认 ±3)
        :param num_points: 返回的时间戳数量（默认 5)
        :return: 附近的时间戳列表
        """
        # 找到在 desired_time ± window 范围内的时间戳
        times = np.array(self.pred_states_times)
        nearby_times = times[(times >= desired_time - window) & (times <= desired_time + window)]
        
        # 如果附近的时间戳多于 num_points，只取最近的 num_points 个
        if len(nearby_times) > num_points:
            # 计算时间戳与 desired_time 的差值，取绝对值最小的 num_points 个
            time_diffs = np.abs(nearby_times - desired_time)
            nearby_times = nearby_times[np.argsort(time_diffs)[:num_points]]
        
        return nearby_times

    def interpolate(self, desired_time):
        """
        根据时间插值获取概率分布
        :param desired_time: 需要插值的时间点
        :return: 插值后的最大概率状态
        """
        # 打印插值时间点
        # print(f"插值时间点: {desired_time}")
        if len(self.pred_states_times) == 0:
            print(f"没有数据，无法进行插值。初始状态为{self.init_state}")
            return self.init_state
        # 获取插值时间点附近的时间戳
        nearby_times = self.get_nearby_times(desired_time)
        # print(f"附近的时间戳（相对时间）: {nearby_times - desired_time}")

        # 打印附近时间戳对应的各个类别的概率
        for time in nearby_times:
            probs = np.array([interp(time) for interp in self.interpolators])
            # print(f"时间 {time - desired_time} 的各个类别概率: {probs}")

        # 计算插值时间点的各个类别的概率
        interpolated_probs = np.array([interp(desired_time) for interp in self.interpolators])
        # print(f"推理使用状态类别概率: {interpolated_probs}")

        # 返回最大概率的类别
        result = np.argmax(interpolated_probs)
        return result

    def cleanup_old_data(self, current_time, max_age):
        """
        清理过时数据
        :param current_time: 当前时间
        :param max_age: 数据的最大保留时间（超过此时间的数据将被清理）
        """
        # 清理过时的时间戳和概率分布
        while self.pred_states_times and (current_time - self.pred_states_times[0]) > max_age:
            self.pred_states_times.popleft()
            self.pred_states_prob.popleft()

        # 更新插值器
        if len(self.pred_states_prob) != 0:
            self.update(np.array(self.pred_states_times), np.array(self.pred_states_prob))

if __name__ == "__main__":
    # 初始化 StatePlanner，假设有 3 个状态
    planner = StatePlanner(state_num=3, max_window_size=10)

    # 模拟一些时间戳和概率分布数据
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    probs = np.array([
        [0.8, 0.1, 0.1],  # 时间 1.0 的概率分布
        [0.7, 0.2, 0.1],  # 时间 2.0 的概率分布
        [0.6, 0.3, 0.1],  # 时间 3.0 的概率分布
        [0.5, 0.4, 0.1],  # 时间 4.0 的概率分布
        [0.4, 0.5, 0.1]   # 时间 5.0 的概率分布
    ])

    # 更新数据
    planner.update(times, probs)

    # 在时间 2.5 进行插值
    desired_time = 2.5
    result = planner.interpolate(desired_time)
    print(f"在时间 {desired_time} 插值后的最大概率状态: {result}\n")

    # 清理超过 2.0 秒的旧数据
    current_time = 3.0
    max_age = 2.0
    planner.cleanup_old_data(current_time, max_age)

    # 打印清理后的数据
    print("清理后的时间戳:", planner.pred_states_times)
    print("清理后的概率分布:", planner.pred_states_prob)

    # 在时间 3.5 进行插值
    desired_time = 3.5
    result = planner.interpolate(desired_time)
    print(f"在时间 {desired_time} 插值后的最大概率状态: {result}\n")