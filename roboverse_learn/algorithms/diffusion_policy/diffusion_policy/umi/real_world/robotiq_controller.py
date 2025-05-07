import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
import serial
import binascii

# Function to calculate CRC16 checksum
def crc16(data):
    crc = 0xFFFF
    for byte in data:
        if isinstance(byte, str):
            byte = ord(byte)
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc

class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2

class RobotiqController(mp.Process):
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 port='/dev/ttyUSB0',
                 baudrate=115200,
                 timeout=1,
                 parity=serial.PARITY_NONE,
                 stopbits=serial.STOPBITS_ONE,
                 bytesize=serial.EIGHTBITS,
                 frequency=30,
                 move_max_speed=255.0,  # Max speed value for Robotiq gripper
                 command_queue_size=1024,
                 launch_timeout=3,
                 receive_latency=0.0,
                 use_meters=False,
                 verbose=False):
        super().__init__(name="RobotiqController")
        
        # Store parameters
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.parity = parity 
        self.stopbits = stopbits
        self.bytesize = bytesize
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

        # Prepare serial port settings
        self.serial_settings = {
            'port': self.port,
            'baudrate': self.baudrate,
            'timeout': self.timeout,
            'parity': self.parity,
            'stopbits': self.stopbits,
            'bytesize': self.bytesize
        }

    # ========= launch methods ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RobotiqController] Controller process spawned at {self.pid}")

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
    def activate_gripper(self, ser):
        """
        Send activation command to the gripper.
        """
        cmd = b'\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00'
        crc_value = crc16(cmd)
        crc_bytes = crc_value.to_bytes(2, 'little')  # Modbus uses little endian for CRC
        final_cmd = cmd + crc_bytes
        # print(f'activ cmd:{final_cmd}')
        ser.write(final_cmd)
        time.sleep(1)  # Wait for activation to complete
        data_raw = ser.readline()
        data = binascii.hexlify(data_raw)
        if self.verbose:
            print("Activate gripper response:", data)

    def get_current_position(self, ser):
        """
        Request and return the current position of the gripper.
        """
        # Build command to read the gripper position (Placeholder)
        cmd = b'\x09\x03\x07\xD0\x00\x01'
        crc_value = crc16(cmd)
        crc_bytes = crc_value.to_bytes(2, 'little')
        final_cmd = cmd + crc_bytes
        ser.write(final_cmd)
        data_raw = ser.read(7)  # Read expected number of bytes
        if len(data_raw) < 7:
            if self.verbose:
                print("Failed to read gripper position")
            return 0.0
        position = data_raw[3]
        if self.verbose:
            print(f"Current gripper position: {position}")
        return float(position)

    def build_move_command(self, position):
        """
        Build the command to move the gripper to the desired position.
        """
        if not 0 <= position <= 255:
            raise ValueError("Position value should be in the range of 0 - 255.")
        cmd = b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00'
        # position = 255 - position
        pos_hex = position.to_bytes(1, 'big') + b'\xFF\xFF'  # Position, speed, force
        cmd_data = cmd + pos_hex
        # print(f'cmd_data:{cmd_data}')
        crc_value = crc16(cmd_data)
        crc_bytes = crc_value.to_bytes(2, 'little')
        final_cmd = cmd_data + crc_bytes
        return final_cmd

    def move_gripper_to_position(self, ser, position):
        """
        Send command to move the gripper to the desired position.
        """
        position_value = int(position)
        final_cmd = self.build_move_command(position_value)
        # print(f'final_cmd:{final_cmd}')
        ser.write(final_cmd)
        data_raw = ser.read(8)  # Read expected number of bytes in response
        data = binascii.hexlify(data_raw)
        if self.verbose:
            print("Move gripper response:", data)

    def get_gripper_state(self, ser):
        """
        Request and return the gripper's current state.
        """
        # Build command to read the gripper status register (Placeholder)
        cmd = b'\x09\x03\x07\xD0\x00\x03'
        crc_value = crc16(cmd)
        crc_bytes = crc_value.to_bytes(2, 'little')
        final_cmd = cmd + crc_bytes
        ser.write(final_cmd)
        data_raw = ser.read(11)  # Read expected number of bytes
        if len(data_raw) < 11:
            if self.verbose:
                print("Failed to read gripper state")
            return {
                'gripper_state': 0,
                'gripper_position': 0.0,
                'gripper_velocity': 0.0,
                'gripper_force': 0.0,
                'gripper_measure_timestamp': time.time(),
                'gripper_receive_timestamp': time.time(),
                'gripper_timestamp': time.time() - self.receive_latency
            }
        # Parse data_raw to extract state information
        gripper_status = data_raw[3]
        gripper_position = data_raw[4]
        gripper_velocity = data_raw[5]  # Placeholder, adjust as per gripper data
        gripper_force = data_raw[6]     # Placeholder, adjust as per gripper data
        state = {
            'gripper_state': gripper_status,
            'gripper_position': gripper_position / self.scale,
            'gripper_velocity': gripper_velocity / self.scale,
            'gripper_force': gripper_force,
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time() - self.receive_latency
        }
        if self.verbose:
            print(f"Gripper state: {state}")
        return state

    # ========= main loop in process ============
    def run(self):
        try:
            # Open serial port
            with serial.Serial(**self.serial_settings) as ser:
                # Initialize gripper
                self.activate_gripper(ser)
                # print("gripper activated")
                
                # Initialize position and trajectory interpolator
                curr_pos = self.get_current_position(ser)
                curr_t = time.monotonic()
                last_waypoint_time = curr_t
                pose_interp = PoseTrajectoryInterpolator(
                    times=[curr_t],
                    poses=[[curr_pos]]  # Assuming 1D position
                )
                
                keep_running = True
                t_start = time.monotonic()
                iter_idx = 0
                while keep_running:
                    t_now = time.monotonic()
                    dt = 1 / self.frequency
                    t_target = t_now
                    target_pos = pose_interp(t_target)[0]
                    # Control gripper to target position
                    self.move_gripper_to_position(ser, target_pos)
                    
                    # Read state of gripper
                    gripper_state = self.get_gripper_state(ser)
                    # Put state into ring buffer
                    self.ring_buffer.put(gripper_state)

                    # Fetch commands from queue
                    try:
                        commands = self.input_queue.get_all()
                        n_cmd = len(commands['cmd'])
                    except Empty:
                        n_cmd = 0

                    # Execute commands
                    for i in range(n_cmd):
                        command = dict()
                        for key, value in commands.items():
                            command[key] = value[i]
                        cmd = command['cmd']
                        if cmd == Command.SHUTDOWN.value:
                            keep_running = False
                            break
                        elif cmd == Command.SCHEDULE_WAYPOINT.value:
                            target_pos = command['target_pos'] * self.scale
                            target_time = command['target_time']
                            # Translate global time to monotonic time
                            target_time = time.monotonic() - time.time() + target_time
                            curr_time = t_now
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=[target_pos],
                                time=target_time,
                                max_pos_speed=self.move_max_speed,
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time
                            )
                            last_waypoint_time = target_time
                        elif cmd == Command.RESTART_PUT.value:
                            t_start = command['target_time'] - time.time() + time.monotonic()
                            iter_idx = 1
                        else:
                            keep_running = False
                            break

                    if iter_idx == 0:
                        self.ready_event.set()
                    iter_idx +=1

                    # Regulate frequency
                    dt = 1 / self.frequency
                    t_end = t_start + dt * iter_idx
                    precise_wait(t_end=t_end, time_func=time.monotonic)
        finally:
            self.ready_event.set()
            if self.verbose:
                print(f"[RobotiqController] Disconnected from gripper on port {self.port}")
