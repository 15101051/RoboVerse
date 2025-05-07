import enum
import struct
import serial
import time

# =============== CRC16 for Modbus ===============
# 下面这个表可以改成适配 Modbus RTU 的 CRC 多项式 (0xA001)，
# 亦可用常规的算法循环实现，这里仅示例“Modbus 常用”查表算法之一。
# 也可直接用循环法计算 CRC16（Modbus），不再依赖原先 CCITT16 表。
MODBUS_CRC16_TABLE = []
for i in range(256):
    crc = i
    for _ in range(8):
        if crc & 0x0001:
            crc >>= 1
            crc ^= 0xA001
        else:
            crc >>= 1
    MODBUS_CRC16_TABLE.append(crc)

def modbus_crc16(data: bytes) -> int:
    """标准Modbus-RTU的CRC16校验"""
    crc = 0xFFFF
    for b in data:
        idx = (crc ^ b) & 0xFF
        crc = (crc >> 8) ^ MODBUS_CRC16_TABLE[idx]
    return crc

# =============== 枚举 & 常量保持不变（上层也在使用） ===============
class StatusCode(enum.IntEnum):
    E_SUCCESS = 0
    E_NOT_AVAILABLE = 1
    E_NO_SENSOR = 2
    E_NOT_INITIALIZED = 3
    E_ALREADY_RUNNING = 4
    E_FEATURE_NOT_SUPPORTED = 5
    E_INCONSISTENT_DATA = 6
    E_TIMEOUT = 7
    E_READ_ERROR = 8
    E_WRITE_ERROR = 9
    E_INSUFFICIENT_RESOURCES = 10
    E_CHECKSUM_ERROR = 11
    E_NO_PARAM_EXPECTED = 12
    E_NOT_ENOUGH_PARAMS = 13
    E_CMD_UNKNOWN = 14
    E_CMD_FORMAT_ERROR = 15
    E_ACCESS_DENIED = 16
    E_ALREADY_OPEN = 17
    E_CMD_FAILED = 18
    E_CMD_ABORTED = 19
    E_INVALID_HANDLE = 20
    E_NOT_FOUND = 21
    E_NOT_OPEN = 22
    E_IO_ERROR = 23
    E_INVALID_PARAMETER = 24
    E_INDEX_OUT_OF_BOUNDS = 25
    E_CMD_PENDING = 26
    E_OVERRUN = 27
    RANGE_ERROR = 28
    E_AXIS_BLOCKED = 29
    E_FILE_EXIST = 30


class CommandId(enum.IntEnum):
    Disconnect = 0x07
    Homing = 0x20
    PrePosition = 0x21
    Stop = 0x22
    FastStop = 0x23
    AckFastStop = 0x24

def args_to_bytes(*args, int_bytes=1):
    """保留原先形式, 但这里仅在部分自定义方法（custom_script等）可能用到。
       对于 Modbus 并不会直接用到这个打包逻辑，可根据需要保留或简化。"""
    buf = list()
    for arg in args:
        if isinstance(arg, float):
            # little endian 32bit float
            buf.append(struct.pack('<f', arg))
        elif isinstance(arg, int):
            buf.append(arg.to_bytes(length=int_bytes, byteorder='little', signed=False))
        elif isinstance(arg, str):
            buf.append(arg.encode('ascii'))
        else:
            raise RuntimeError(f'Unsupported type {type(arg)}')
    result = b''.join(buf)
    return result


# ================================================
# ================ Modbus Driver =================
# ================================================
class WSGBinaryDriver:
    """
    将原先的TCP + 自定义协议，改为RS485 + Modbus-RTU协议。
    方法签名、函数名基本保持不变，方便上层 wsg_driver.py 等复用。
    """
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, slave_id=1, timeout=0.1):
        """
        :param port:     串口号, 例如 'COM3' 或 '/dev/ttyUSB0'
        :param baudrate: 波特率, 默认 115200
        :param slave_id: Modbus 设备地址ID, 默认 1
        :param timeout:  读超时, 默认 0.1秒
        """
        self.port = port
        self.baudrate = baudrate
        self.slave_id = slave_id
        self.timeout = timeout

        self.ser = None  # 串口对象

    # -------------- 原先的 start/stop --------------
    def start(self):
        """原先是 TCP connect，这里改为打开串口"""
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout
        )
        if not self.ser.is_open:
            self.ser.open()
        # 也可以在这里做一些初始化动作，比如先读一下初始化状态
        # 例如：self.homing(positive_direction=True, wait=True)

    def stop(self):
        """原先是发送Stop指令+Disconnect，这里可以关串口即可"""
        time.sleep(1.0)
        if self.ser and self.ser.is_open:
            # 可以做一些“停止运动”的操作，例如将位置设置为当前位置
            self.disconnect()
            self.ser.close()

    def __enter__(self):
        self.start()
        print("Driver Init Done!")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ============= Modbus 底层读写方法 =============
    def modbus_write_register(self, reg_addr: int, value: int):
        """
        功能码 0x06, 写单个寄存器
        :param reg_addr: 寄存器地址
        :param value:    需要写入的值(0~65535)
        """
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Serial not open")

        # 1) 构造 Modbus RTU 请求帧: [ slave_id, func_code(06), reg_addr_hi, reg_addr_lo, data_hi, data_lo, crc_lo, crc_hi ]
        func_code = 0x06
        hi_addr = (reg_addr >> 8) & 0xFF
        lo_addr = reg_addr & 0xFF

        hi_val = (value >> 8) & 0xFF
        lo_val = value & 0xFF

        request = bytes([
            self.slave_id,
            func_code,
            hi_addr,
            lo_addr,
            hi_val,
            lo_val
        ])
        # print('req', request)
        # 2) 计算CRC
        crc = modbus_crc16(request)
        lo_crc = crc & 0xFF
        hi_crc = (crc >> 8) & 0xFF

        # 3) 发送帧
        frame = request + bytes([lo_crc, hi_crc])
        # print('write', value)
        self.ser.write(frame)

        # 4) 读取响应
        resp = self.ser.read(8)  # 写单寄存器的正常响应也有8字节
        if len(resp) < 8:
            # print('write_reg', resp)
            raise RuntimeError("No response or incomplete response from gripper (write reg).")

        # 5) 检查响应
        # 校验CRC
        resp_crc_calc = modbus_crc16(resp[:-2])
        resp_crc_bytes = resp[-2] | (resp[-1] << 8)
        if resp_crc_calc != resp_crc_bytes:
            raise RuntimeError("CRC error in modbus response (write reg).")

        # 判断是否与发送的功能码、地址、数据一致
        if resp[0] != self.slave_id or resp[1] != func_code:
            raise RuntimeError("Unexpected modbus response (write reg).")

        # 若无异常，则认为成功

    def modbus_read_registers(self, reg_addr: int, quantity: int) -> list:
        """
        功能码 0x03, 读保持寄存器
        :param reg_addr: 寄存器起始地址
        :param quantity: 需要读取的寄存器数量
        :return: [val1, val2, ...] 每个寄存器16位
        """
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Serial not open")

        func_code = 0x03
        hi_addr = (reg_addr >> 8) & 0xFF
        lo_addr = reg_addr & 0xFF

        hi_qty = (quantity >> 8) & 0xFF
        lo_qty = quantity & 0xFF

        request = bytes([
            self.slave_id,
            func_code,
            hi_addr,
            lo_addr,
            hi_qty,
            lo_qty
        ])
        crc = modbus_crc16(request)
        lo_crc = crc & 0xFF
        hi_crc = (crc >> 8) & 0xFF
        frame = request + bytes([lo_crc, hi_crc])
        # print('write frame:', frame)
        self.ser.write(frame)

        # 正常响应: [id, func, byteCount, dataHi1, dataLo1, dataHi2, dataLo2, ..., crcLo, crcHi]
        # dataBytes = quantity * 2
        expected_len = 5 + quantity * 2  # 前3字节 + 数据 + CRC2字节
        resp = self.ser.read(expected_len)
        # print('read_reg:', resp)
        if len(resp) < expected_len:
            raise RuntimeError("No response or incomplete response from gripper (read regs).", resp, expected_len)

        # 校验
        resp_crc_calc = modbus_crc16(resp[:-2])
        resp_crc_bytes = resp[-2] | (resp[-1] << 8)
        if resp_crc_calc != resp_crc_bytes:
            raise RuntimeError("CRC error in modbus response (read regs).")

        if resp[0] != self.slave_id or resp[1] != func_code:
            raise RuntimeError("Unexpected modbus response (read regs).")

        byte_count = resp[2]
        if byte_count != quantity * 2:
            raise RuntimeError("Read register: byte count mismatch.")

        data = resp[3:-2]
        result = []
        for i in range(quantity):
            hi = data[2*i]
            lo = data[2*i + 1]
            result.append((hi << 8) | lo)

        return result

    # =================== 兼容原先 msg_send/msg_receive/cmd_submit 的“壳” ===================
    def msg_send(self, cmd_id: int, payload: bytes):
        """
        原先自定义协议里做的是:
         1) 构造三字节 preamble (0xAA 0xAA 0xAA)
         2) 写cmd_id, size, payload, CRC
        现在用不到了，这里仅留空 or 返回0
        """
        # 因为上层现有调用可能还会调用 msg_send，
        # 我们可根据 cmd_id 分流到具体 modbus_write_register 之类
        # 但要注意：上层发送的 cmd_id 并不一定对应真正的modbus寄存器。
        # 这里建议什么也不做 or 返回0
        return 0

    def msg_receive(self) -> dict:
        """
        原先是阻塞式地读取一包自定义协议。现在用不到了。
        由于上层可能会调用，此处也做个“空实现”返回固定结构即可。
        """
        return {
            'command_id': 0,
            'status_code': StatusCode.E_SUCCESS.value,
            'payload_bytes': b''
        }

    def cmd_submit(self, cmd_id: int, payload: bytes = b'', pending: bool=True, ignore_other=False):
        """
        原先代码中是：
          1) 先 msg_send
          2) 不断 msg_receive，直到拿到与 cmd_id 对应的回复 or E_CMD_PENDING 完成
        现在已不再用这一套机制，这里做个壳子。
        实际功能都放在下方 self.act(...) => self.homing(...) / self.pre_position(...) 等。
        """
        return {
            'command_id': cmd_id,
            'status_code': StatusCode.E_SUCCESS.value,
            'payload_bytes': b''
        }

    # ============= 中层API: act(...) 保持不变 =============
    def act(self, cmd: CommandId, *args, wait=True, ignore_other=False):
        """
        原先的 act(cmd_id, payload) -> 这边内部会调用 cmd_submit 去发送。
        现在我们把真实的modbus操作放在这里。
        """
        # 根据 cmd 来做分支：
        if cmd == CommandId.Homing:
            # args[0] 表示 0,1,2 => None / True / False
            arg = args[0] if args else 1
            # 0 => 0xA5 (全程标定), 1 => 0x01(只找单向)
            if arg == 0:
                self._homing_modbus(full=True, wait=wait)
            else:
                self._homing_modbus(full=False, wait=wait)

        elif cmd == CommandId.PrePosition:
            # 传入: clamp_flag, width, speed
            # clamp_flag(0 or 1), width(float), speed(float)
            # clamp_on_block=true => clamp_flag=0
            clamp_flag = args[0]
            width_mm = float(args[1])
            speed_percent = float(args[2])
            self._pre_position_modbus(force_percent=100,  # 可根据需要扩展
                                      width_mm=width_mm,
                                      speed_percent=speed_percent,
                                      wait=wait)

        elif cmd == CommandId.Stop:
            # 没有明确的 Modbus-RTU “停止”寄存器，可在这里写目标位置=当前位置
            # 或者写一个速度=0的方式模拟停止
            self._stop_modbus()

        elif cmd == CommandId.Disconnect:
            # Modbus 没有“断开”概念，这里空一下即可
            pass

        elif cmd == CommandId.FastStop or cmd == CommandId.AckFastStop:
            # 新夹爪未必有“快速停止/清错”指令，这里空实现或自行定义
            pass

        else:
            raise RuntimeError(f"Unsupported command ID {cmd}")

        # 返回结果结构
        return {
            'command_id': cmd,
            'status_code': StatusCode.E_SUCCESS,
            'payload_bytes': b''
        }

    # ============= 高层API与旧接口保持一致 =============
    def disconnect(self):
        """原先是直接发送Disconnect命令，现在直接空实现"""
        pass

    def homing(self, positive_direction=True, wait=True):
        """
        原先: homing(positive_direction=True/False/None)
        这里我们映射到:
           True  => 只找单向 => 写0x0100=0x01
           False => 只找单向 => 写0x0100=0x01
           None  => 全行程 => 写0x0100=0xA5
        """
        if positive_direction is None:
            self._homing_modbus(full=True, wait=wait)
        else:
            self._homing_modbus(full=False, wait=wait)

    def pre_position(self, width: float, speed: float, clamp_on_block: bool=True, wait=True):
        """原先: pre_position(width, speed, clamp_on_block=True, wait=True)"""
        # 这里简单设置: force=100%, width=..., speed=...
        # clamp_on_block: 该参数在旧接口是个标记，这里可根据需要映射到具体逻辑
        # 下方仅示例无视 clamp_on_block, 强行写100% 力
        self._pre_position_modbus(force_percent=100,
                                  width_m=width,
                                  speed_percent=speed,
                                  wait=wait)
        
    def get_current_position(self):
        """原先: get_current_position()"""
        # 读取当前位置寄存器 0x0202
        current_pos = self.modbus_read_registers(0x0202, 1)[0]
        current_pos_mm = current_pos / 12.5
        current_pos_m = current_pos_mm / 1000.0
        return current_pos_m

    def ack_fault(self):
        """原先AckFastStop，这里空实现或者自定义"""
        return {
            'command_id': CommandId.AckFastStop,
            'status_code': StatusCode.E_SUCCESS,
            'payload_bytes': b''
        }

    def stop_cmd(self):
        """旧代码中stop_cmd()=act(CommandId.Stop, ...) 这里直接调用_stop_modbus"""
        return self._stop_modbus()

    def custom_script(self, cmd_id: int, *args):
        """原先自定义指令，这里新夹爪未必有对应功能，可自行修改/删除"""
        # 可以自定义某些读写寄存器做扩展，这里暂时空实现
        return {
            'state': 0,
            'position': 0.,
            'velocity': 0.,
            'force_motor': 0.,
            'measure_timestamp': 0.,
            'is_moving': False
        }

    def script_query(self):
        """空实现"""
        return self.custom_script(0xB0)

    def script_position_pd(self, 
                           position: float, velocity: float,
                           kp: float=15.0, kd: float=1e-3,
                           travel_force_limit: float=80.0, 
                           blocked_force_limit: float=None):
        """空实现"""
        return self.custom_script(0xB1, position, velocity, kp, kd, travel_force_limit, blocked_force_limit)

    # ============= 下方是真正的Modbus操作逻辑 =============
    def _homing_modbus(self, full=False, wait=True):
        """
        :param full: True => 写0xA5, False => 写0x01
        1) 写 0x0100 = 0xA5 / 0x01
        2) 若 wait=True, 轮询 0x0200 寄存器, 等待变成1(初始化成功)或超时
        """
        cmd_val = 0xA5 if full else 0x01
        self.modbus_write_register(0x0100, cmd_val)
        if wait:
            t0 = time.time()
            while (time.time() - t0) < 5.0:  # 最长等5s, 您可视机型改长一些
                state = self.modbus_read_registers(0x0200, 1)[0]
                # 0：未初始化；1：初始化成功；2：初始化中
                if state == 1:
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("Homing timeout or not successful")

    def _pre_position_modbus(self, force_percent=100, width_m=0.05, speed_percent=50, wait=True):
        """
        对应手册：
          力值寄存器: 0x0101 (20-100)
          位置寄存器: 0x0103 (0-1000 => 代表0~最大开口)
          速度寄存器: 0x0104 (1-100)
        这里演示简单写法，force=100%, position=width_m(仅做一个简单的映射),
        speed=speed_percent
        """
        # 1) force
        # 若手册定义力范围20~100，这里要做保护
        if force_percent < 20: 
            force_percent = 20
        if force_percent > 100:
            force_percent = 100
        self.modbus_write_register(0x0101, force_percent)

        # 2) position
        width_mm = width_m * 1000
        pos_val = int(width_mm) * 12.5
        if pos_val < 0:
            pos_val = 0
        if pos_val > 1000:
            pos_val = 1000
        # print('[WSG DRIVER] Send position[‰]:', pos_val)

        MAX_RETRY, succeeded = 4, False
        for i in range(MAX_RETRY):
            try:
                self.modbus_write_register(0x0103, int(pos_val))
            except RuntimeError:
                print(f'Warning ⚠️⚠️⚠️ [WSG DRIVER] Send position Failed, pos_val: {pos_val}, retrying for {i+1}/{MAX_RETRY}')
                # time.sleep(0.1)  睡眠时间太长会不会影响gripper_controller进程？
                continue
            succeeded = True
            break
        if not succeeded:
            raise RuntimeError("Pre-position not successful")

        # 3) speed
        if speed_percent < 1:
            speed_percent = 1
        if speed_percent > 100:
            speed_percent = 100
        # print('speed', int(speed_percent))
        self.modbus_write_register(0x0104, int(speed_percent))

        if wait:
            # 轮询 0x0201, 看是否停止(1=到达,2=夹到,3=掉落), or 0=运动中
            t0 = time.time()
            while (time.time() - t0) < 5.0:
                state = self.modbus_read_registers(0x0201, 1)[0]
                # 0：运动中，1：到达位置；2：夹住物体；3：物体掉落
                if state in (1, 2, 3):
                    # 到达/夹住/掉落 都算运动结束
                    break
                time.sleep(0.05)

    def _stop_modbus(self):
        """
        由于手册里没有 Stop 命令可直接写，
        这里我们用“把目标位置写成当前实时位置”的方式模拟停止。
        """
        # 1) 先读实时位置
        current_pos = self.modbus_read_registers(0x0202, 1)  # 0~1000
        current_pos = current_pos[0]
        print("Cur Pos:", current_pos)
        # 2) 将目标位置=当前位置
        self.modbus_write_register(0x0103, current_pos)
        return {
            'command_id': CommandId.Stop,
            'status_code': StatusCode.E_SUCCESS,
            'payload_bytes': b''
        }

# =============== 测试代码 ===============

import time

def test():
    """全面测试基于 Modbus-RTU 的 WSGBinaryDriver 驱动"""
    try:
        # 初始化驱动，指定串口号和参数
        driver = WSGBinaryDriver(port='/dev/ttyUSB0', baudrate=115200)

        # 打开串口连接
        driver.start()
        print("串口连接已打开。")

        # 测试夹爪初始化 (Homing)
        print("\n[TEST] 测试夹爪初始化...")
        driver.homing(positive_direction=True, wait=True)
        print("[PASS] 夹爪初始化成功！")

        # 设置力值
        print("\n[TEST] 设置夹爪力值为 50%...")
        driver.modbus_write_register(0x0101, 50)
        current_force = driver.modbus_read_registers(0x0101, 1)[0]
        if current_force == 50:
            print(f"[PASS] 力值设置成功！当前力值：{current_force}%")
        else:
            print(f"[FAIL] 力值设置失败！期望: 50, 实际: {current_force}")

        # 设置位置和速度
        print("\n[TEST] 移动夹爪到位置 300 (千分比)，速度 50%...")
        driver.pre_position(width=0.06, speed=100, clamp_on_block=True, wait=True)
        current_position = driver.get_current_position()
        driver.pre_position(width=0.07, speed=100, clamp_on_block=True, wait=True)
        current_position = driver.get_current_position()
        driver.pre_position(width=0.08, speed=100, clamp_on_block=True, wait=True)
        current_position = driver.get_current_position()
        print(f"Current Position: {current_position}")
        if abs(current_position - 0.08) <= 0.01:  # 允许一定误差范围
            print(f"[PASS] 夹爪移动成功！当前位置：{current_position} m")
        else:
            print(f"[FAIL] 夹爪移动失败！期望位置: 300, 实际位置: {current_position}")

        # 测试实时位置反馈
        print("\n[TEST] 读取实时位置反馈...")
        real_time_position = driver.modbus_read_registers(0x0202, 1)[0]
        print(f"[PASS] 当前实时位置：{real_time_position}‰")

        # 测试夹爪夹持状态反馈
        print("\n[TEST] 读取夹爪夹持状态反馈...")
        gripper_status = driver.modbus_read_registers(0x0201, 1)[0]
        status_map = {0: "运动中", 1: "到达位置", 2: "夹住物体", 3: "物体掉落"}
        print(f"[PASS] 当前夹爪状态：{status_map.get(gripper_status, '未知状态')} ({gripper_status})")

        # 测试停止夹爪
        print("\n[TEST] 测试停止夹爪...")
        driver.stop_cmd()
        print("[PASS] 夹爪停止命令发送成功！")

        # 测试写入保存配置
        print("\n[TEST] 测试写入保存配置...")
        driver.modbus_write_register(0x0300, 1)  # 写入保存指令
        print("[PASS] 配置保存成功！")


        # 测试自动初始化设置
        print("\n[TEST] 设置夹爪自动初始化...")
        # driver.modbus_write_register(0x0504, 1)  # 设置自动初始化
        # driver.modbus_write_register(0x0300, 1)  # 保存配置

    except Exception as e:
        print(f"[ERROR] 测试过程中发生异常: {e}")

    finally:
        # 关闭串口连接
        driver.stop()
        print("\n串口连接已关闭。")

# 运行测试
# test()