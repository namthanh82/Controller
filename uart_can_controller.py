import threading
import time
import math
import struct
import numpy as np
from collections import deque
import serial

from trajectory import TrapezoidalTrajectory, CubicTrajectory, QuinticTrajectory, SplineTrajectory
from kinematic import get_acc_jerk

# Define states similar to ODrive enums for GUI compatibility
AXIS_STATE_IDLE = 1
AXIS_STATE_CLOSED_LOOP_CONTROL = 8
CLOSED_LOOP_CONTROL = AXIS_STATE_CLOSED_LOOP_CONTROL
IDLE = AXIS_STATE_IDLE
DEG2RAD = math.pi / 180

gear_ratio = 100.0
g = 9.81

class ODriveThread(threading.Thread):
    """
    UART-CAN Bridge Controller
    Replaces directly connecting to ODrive via USB, uses a Serial port to an ESP32 instead.
    """
    def __init__(self, serial_port='/dev/ttyAMA0', baudrate=921600, node_id=1):
        super().__init__()

        self.serial_port = serial_port
        self.baudrate = baudrate
        self.node_id = node_id # ID của motor ODrive trên mạng CAN (1, 2, 3...)
        self.ser = None

        # running state
        self.connected = False
        self.closed_loop_control = False
        self.isOffset = False
        self.error = False
        self.data_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._estop_event = threading.Event()

        self.is_calibrating_friction = False

        # Parameters
        self.max_torque = 0.15
        self.tor_coef = 0.708282
        self.Kt = 8.27 / 270
        self.offset = 0.0

        # Physics
        self.start_pos = -90
        self.link_mass = 1.125
        self.link_length = 0.7
        self.center_distance = 0.7 / 2 - 0.04
        self.motor_inertia = 0.000643
        self.const_inertia = 1 / 12 * self.link_mass * (self.link_length ** 2) + self.link_mass * (self.center_distance ** 2) + gear_ratio ** 2 * self.motor_inertia
        
        self.hanger_mass = 0.26
        self.ext_load = 0.0
        self.hanger_distance = 0.7 - 0.04 - 0.06
        
        self.m = self.link_mass + self.hanger_mass
        self.lc = (self.center_distance * self.link_mass + self.hanger_distance * self.hanger_mass) / self.m
        self.Ic = self.const_inertia + self.hanger_mass * (self.hanger_distance ** 2)

        self.coul_friction = 0.0
        self.visc_friction = 0.00276 * gear_ratio ** 2

        # Filter settings
        self.window_size = 25
        self.poly_order = 2
        self.velFilBuf = deque(maxlen=self.window_size)
        self.timeFilBuf = deque(maxlen=self.window_size)

        # Inputs
        self.Kp = 10
        self.Kd = 5
        self.torque_set = 0.0
        self.pos_set = self.start_pos
        self.vel_set = 0.0
        self.acc_set = 0.0
        self.time_set = 5.0
        self.ctrl_bandwidth = 2000
        self.enc_bandwidth = 50

        # Trajectory (matching previous system)
        self.traj = SplineTrajectory()
        self.t_ref = -math.inf

        # State outputs
        self.pos = 0.0
        self.vel = 0.0
        self.preT = 0.0
        self.data = deque(maxlen=800)

        # ESP32 UART Buffer
        self.rx_buffer = bytearray()

    def connect(self):
        try:
            print(f"Connecting to ESP32 on {self.serial_port} at {self.baudrate}...")
            # Uncomment for windows testing
            # self.serial_port = 'COM3' 
            self.ser = serial.Serial(self.serial_port, self.baudrate, timeout=0.01)
            self.connected = True
            self.error = False
            print("Connected successfully to ESP32 bridge.")
        except Exception as e:
            self.error = True
            print(f"Serial Connection failed: {e}")

    def clear_error(self):
        self.error = False
        print("Clear error command (Not fully supported on CAN yet, but state cleared)")

    def enter_closed_loop(self):
        self.closed_loop_control = True
        print("Entered CLOSED_LOOP_CONTROL Mode")

    def return_IDLE(self):
        self.closed_loop_control = False
        self.torque_set = 0.0
        self.send_torque(0.0)

    def is_controlable(self):
        return self.connected and self.closed_loop_control and self.isOffset and not self._estop_event.is_set()

    def update_ctrlElms(self, *ctrlElms):
        with self.data_lock:
            self.t_ref = time.time()
            target = ctrlElms[0]
            max_vel = ctrlElms[1]
            self.Kp = ctrlElms[2]
            self.Kd = ctrlElms[3]
            self.ctrl_bandwidth = ctrlElms[4]
            self.enc_bandwidth = ctrlElms[5]
            
            # Setup trajectory
            self.traj.param_calc(self.pos, target, max_vel)

    def update_loadParms(self, *loadParms):
        with self.data_lock:
            self.ext_load = loadParms[0]
            self.hanger_distance = loadParms[1]
            self.coul_friction = loadParms[2]
            self.visc_friction = 0.00276 * gear_ratio ** 2 + loadParms[3]
            self.m = self.link_mass + self.hanger_mass + self.ext_load
            self.lc = (self.center_distance * self.link_mass + self.hanger_distance * (self.hanger_mass + self.ext_load)) / self.m
            self.Ic = self.const_inertia + (self.hanger_mass + self.ext_load) * (self.hanger_distance ** 2)
            self.max_torque = loadParms[4]

    def get_state(self):
        if self.connected: return CLOSED_LOOP_CONTROL if self.closed_loop_control else IDLE
        return None

    def emergency_stop(self):
        self._estop_event.set()
        self.torque_set = 0.0
        self.send_torque(0.0)

    def get_data(self):
        with self.data_lock:
            return list(self.data)

    def set_offset(self):
        # We need CAN pos_estimate at this moment. 
        self.offset = self.pos # Giả lập offset (Vì CAN trả về pos estimate)
        self.isOffset = True

    def reset(self):
        self.traj.reset()
        self.t_ref = -math.inf
        self.velFilBuf.clear()
        self.timeFilBuf.clear()
        self.return_IDLE()
        self.isOffset = False
        self._estop_event.clear()

    def stop(self):
        self._stop_event.set()
        self.send_torque(0.0)

    # --- ESP32 UART Protocol Functions ---
    def send_torque(self, torque_nm):
        if not self.ser or not self.ser.is_open: return
        # Format: [0xAA] [0xBB] [0x01] [Len] [NodeID] [Torque 4-bytes float]
        packet = bytearray([0xAA, 0xBB, 0x01, 5, self.node_id])
        packet.extend(struct.pack('<f', torque_nm)) # Little-endian float
        try:
            self.ser.write(packet)
        except Exception:
            self.connected = False

    def request_feedback(self):
        if not self.ser or not self.ser.is_open: return
        # Format: [0xAA] [0xBB] [0x02] [0]
        try:
            self.ser.write(bytearray([0xAA, 0xBB, 0x02, 0]))
        except Exception:
            self.connected = False

    def process_serial(self):
        if not self.ser or not self.ser.is_open: return
        
        try:
            while self.ser.in_waiting > 0:
                self.rx_buffer.extend(self.ser.read(self.ser.in_waiting))
                
                # Check for packet header
                while len(self.rx_buffer) >= 4:
                    if self.rx_buffer[0] == 0xAA and self.rx_buffer[1] == 0xBB:
                        cmd = self.rx_buffer[2]
                        length = self.rx_buffer[3]
                        
                        if len(self.rx_buffer) >= 4 + length:
                            payload = self.rx_buffer[4:4+length]
                            # Xóa packet này khỏi buffer
                            self.rx_buffer = self.rx_buffer[4+length:]
                            
                            # Parse Feedback Cmd
                            if cmd == 0x02:
                                # payload = [NodeID, Pos(4), Vel(4)] lặp lại
                                for i in range(0, len(payload), 9):
                                    if i+9 <= len(payload):
                                        nid = payload[i]
                                        p = struct.unpack('<f', payload[i+1:i+5])[0]
                                        v = struct.unpack('<f', payload[i+5:i+9])[0]
                                        if nid == self.node_id:
                                            with self.data_lock:
                                                # Convert từ ODrive pos/vel sang độ (deg) giống code cũ
                                                self.pos = (p - self.offset) * 360 / gear_ratio + self.start_pos
                                                self.vel = v * 360 / gear_ratio
                        else:
                            # Chưa đủ payload, chờ loop sau
                            break 
                    else:
                        # Trượt sai header -> vứt byte lỗi
                        self.rx_buffer.pop(0)

        except Exception as e:
            print("UART Read error:", e)
            self.connected = False

    def setTarget(self):
        t = time.time() - self.t_ref
        self.pos_set, self.vel_set, self.acc_set = self.traj.desired_state(t)

    def dynamic_calculation(self):
        m = self.m
        lc = self.lc
        Ic = self.Ic

        self.setTarget()
        q = self.pos * DEG2RAD
        qdot = self.vel * DEG2RAD
        q_d = self.pos_set * DEG2RAD
        qdot_d = self.vel_set * DEG2RAD
        qddot_d = self.acc_set * DEG2RAD

        Kp = self.Kp
        Kd = self.Kd
        ep = q_d - q
        ev = qdot_d - qdot
        D = self.visc_friction
        direction = math.copysign(1.0, qdot) if abs(qdot) > 0.001 else 0.0

        tor = Ic * (qddot_d + (Kp * ep + Kd * ev)) + m * g * lc * math.cos(q) + D * qdot + self.coul_friction * direction
        tor = tor / self.tor_coef / gear_ratio

        self.torque_set = max(min(tor, self.max_torque), -self.max_torque)

    def run(self):
        self.preT = time.perf_counter()

        while not self._stop_event.is_set():
            t_start = time.perf_counter()
            try:
                if not self.connected:
                    self.connect()
                    if not self.connected:
                        time.sleep(0.1)
                        continue

                if self._estop_event.is_set():
                    self.send_torque(0.0)
                    self._stop_event.wait(0.1)
                    continue

                # 1. Parse received positions/velocities via Serial
                self.process_serial()

                # 2. Ask ESP32 for the next state updates
                self.request_feedback()

                # 3. Trajectory & Dynamics
                with self.data_lock:
                    t = time.perf_counter()
                    deltaT = t - self.preT
                    if deltaT <= 0: deltaT = 0.0001
                    
                    self.velFilBuf.append(self.vel)
                    self.timeFilBuf.append(t)

                    vel_filtered = self.vel
                    acc = 0.0
                    jerk = 0.0
                    if len(self.velFilBuf) == self.window_size:
                        vel_arr = np.array(list(self.velFilBuf))
                        t_arr = np.array(list(self.timeFilBuf))
                        vel_filtered, acc, jerk = get_acc_jerk(t_arr, vel_arr, self.window_size, self.poly_order)

                    self.preT = t

                    # Record
                    self.data.append((time.time(), self.pos, vel_filtered, acc, self.pos_set, self.vel_set, self.acc_set, jerk, self.torque_set))

                # 4. Control step
                if self.is_controlable() and not self.is_calibrating_friction:
                    with self.data_lock:
                        self.dynamic_calculation()
                        self.send_torque(self.torque_set)
                else:
                    self.pos_set = self.start_pos
                    self.torque_set = 0.0
                    self.send_torque(0.0)

            except Exception as e:
                print("Loop error:", e)
                self.connected = False
                self.error = True
                time.sleep(1)

            t_end = time.perf_counter()
            # Loop delay for ~100Hz = 0.01
            t_sleep = 0.01 - (t_end - t_start)
            if t_sleep > 0:
                self._stop_event.wait(timeout=t_sleep)