import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import sys
sys.path.append("/home/enco/umi_enco")
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
import can

# 定义CAN相关设置
CAN_NODE_ID = 0x001  # 根据实际情况修改
CAN_CHANNEL = 'can0'  # 根据实际硬件配置
CAN_BITRATE_ARB = 1000000  # 仲裁域1M
CAN_BITRATE_DATA = 5000000  # 数据域5M

class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2

class OmniPickerController(mp.Process):
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 can_channel=CAN_CHANNEL,
                 can_bitrate_arb=CAN_BITRATE_ARB,
                 can_bitrate_data=CAN_BITRATE_DATA,
                 can_node_id=CAN_NODE_ID,
                 frequency=30,
                 move_max_speed=255.0,  # Max speed value for OmniPicker gripper
                 command_queue_size=1024,
                 launch_timeout=3,
                 receive_latency=0.0,
                 use_meters=False,
                 verbose=False):
        super().__init__(name="OmniPickerController")
        
        # Store parameters
        self.can_channel = can_channel
        self.can_bitrate_arb = can_bitrate_arb
        self.can_bitrate_data = can_bitrate_data
        self.can_node_id = can_node_id
        self.frequency = frequency
        self.move_max_speed = move_max_speed
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.verbose = verbose
        self.scale = 1000.0 if use_meters else 1.0

        # Build input queue
        example_input = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0,
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example_input,
            buffer_size=command_queue_size
        )

        # Build ring buffer
        example_state = {
            'gripper_state': 0,
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_force': 0.0,
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        get_max_k = int(frequency * 10)
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example_state,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()

    # ========= launch methods ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[OmniPickerController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= helper methods ============
    def build_control_frame(self, pos_cmd, force_cmd=0xFF, vel_cmd=0xFF, acc_cmd=0xFF, dec_cmd=0xFF):
        """
        构建控制帧，根据协议6.3.1
        """
        data = [
            0x00,        # Reserved
            pos_cmd,     # Pos Cmd
            force_cmd,   # Force Cmd
            vel_cmd,     # Vel Cmd
            acc_cmd,     # Acc Cmd
            dec_cmd,     # Dec Cmd
            0x00,        # Reserved
            0x00         # Reserved
        ]
        return data

    def parse_status_frame(self, data):
        """
        解析状态帧，根据协议6.3.2
        """
        fault_code = data[0]
        state = data[1]
        pos = data[2]
        vel = data[3]
        force = data[4]
        # Reserved bytes 可以用于校验或忽略

        return {
            'gripper_state': state,
            'gripper_position': pos / self.scale,
            'gripper_velocity': vel / self.scale,
            'gripper_force': force,
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time() - self.receive_latency
        }

    # ========= main loop in process ============
    def run(self):
        bus = None
        try:
            # 初始化CANFD总线
            try:
                bus = can.Bus(interface='socketcan', channel=self.can_channel, bitrate=self.can_bitrate_arb)
                # 对于CANFD，需要进一步配置。python-can对CANFD的支持可能有限，以下为简化示例
                if self.verbose:
                    print(f"[OmniPickerController] CAN bus on channel {self.can_channel} initialized.")
            except Exception as e:
                print(f"[OmniPickerController] Failed to initialize CAN bus: {e}")
                return

            # 初始化目标位置和轨迹插值器
            target_pos = 0.0
            pose_interp = PoseTrajectoryInterpolator(
                times=[time.monotonic()],
                poses=[[target_pos]]  # 假设1D位置
            )
            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0

            # 设置CAN接收监听
            notifier = can.Notifier(bus, [self.AsyncCANListener(self)])

            while keep_running:
                t_now = time.monotonic()
                target_pos = pose_interp(t_now)[0]

                # 构建并发送控制帧
                control_data = self.build_control_frame(pos_cmd=int(target_pos))
                msg = can.Message(arbitration_id=self.can_node_id,
                                  data=control_data)
                try:
                    bus.send(msg)
                    if self.verbose:
                        print(f"[OmniPickerController] Sent control message: {control_data}")
                except can.CanError as e:
                    if self.verbose:
                        print(f"[OmniPickerController] Failed to send message: {e}")

                # 处理命令队列
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = {key: value[i] for key, value in commands.items()}
                    cmd = command['cmd']
                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos_new = command['target_pos'] * self.scale
                        target_time = command['target_time']
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[target_pos_new],
                            time=target_time,
                            max_pos_speed=self.move_max_speed,
                            curr_time=t_now,
                            last_waypoint_time=t_now
                        )
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = command['target_time'] - time.time() + time.monotonic()
                        iter_idx = 1
                    else:
                        keep_running = False
                        break

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx +=1

                # 调节频率
                dt = 1 / self.frequency
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)

        finally:
            self.ready_event.set()
            bus.shutdown()
            if self.verbose:
                print(f"[OmniPickerController] Disconnected from CAN bus on channel {self.can_channel}")

    class AsyncCANListener(can.Listener):
        """
        异步监听CAN消息，并处理状态帧
        """
        def __init__(self, controller):
            super().__init__()
            self.controller = controller

        def on_message_received(self, msg):
            if msg.arbitration_id != self.controller.can_node_id:
                return  # 不是目标夹爪的消息
            if self.controller.verbose:
                print(f"[OmniPickerController] Received message: {msg.data}")
            state = self.controller.parse_status_frame(msg.data)
            self.controller.ring_buffer.put(state)

def test_omni_picker_controller():
    """ 测试 OmniPickerController 的功能 """
    # 创建共享内存管理器
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    
    # 启动 OmniPickerController
    with OmniPickerController(shm_manager) as controller:

        # 测试调度目标点
        target_pos = 20.0  # 中间位置
        target_time = 1.0   # 1秒内到达

        print(f"Sending command to schedule waypoint to position {target_pos} over {target_time}s.")
        controller.schedule_waypoint(target_pos, target_time)

        # 等待一些时间以处理任务
        time.sleep(2)

        # 获取当前状态
        state = controller.get_state()
        print("Current gripper state:", state)

        # 指令结束时，停止控制器
        controller.stop(wait=True)

def main():
    """ 主函数，执行测试 """
    print("Testing OmniPickerController...")
    test_omni_picker_controller()
    print("Test completed.")

if __name__ == '__main__':
    main()
