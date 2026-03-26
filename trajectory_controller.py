# odrive_controller.py

import threading
import time
import odrive
import math
from odrive.enums import AXIS_STATE_CLOSED_LOOP_CONTROL, AXIS_STATE_IDLE
from kinematic_calculate import get_acc_jerk
from collections import deque
from Trajectory import TrapezoidalTrajectory, CubicTrajectory, QuinticTrajectory, SplineTrajectory
from scipy.signal import savgol_coeffs
import numpy as np

CLOSED_LOOP_CONTROL = AXIS_STATE_CLOSED_LOOP_CONTROL
IDLE = AXIS_STATE_IDLE
DEG2RAD = math.pi/180
gear_ratio = 100.0
g = 9.81

class ODriveThread(threading.Thread):
    def __init__(self):
        super().__init__()
        
        # running state
        self.connected = False
        self.closed_loop_control = False
        self.isOffset = False
        self.error = False
        self.data_lock = threading.Lock()  
        self._stop_event = threading.Event()
        self._estop_event = threading.Event() 

        # Driver components
        self.odrv = None
        self.axis = None
        self.max_torque = 0.2  # min limit
        self.max_vel = 35 # deg/s
        self.tor_coef = 0.708282
        self.Kt = 8.27/270
        self.offset = 0.0
        self.traj = QuinticTrajectory()

        # Physic components
        self.t_ref = - math.inf
        self.start_pos = -90

        self.link_mass = 1.125
        self.link_length = 0.7
        self.center_distance = 0.7/2 - 0.04 # 0.31
        self.motor_inertia = 0.000643
        self.const_inertia = 1/12 * self.link_mass * (self.link_length ** 2) + self.link_mass * (self.center_distance ** 2) + gear_ratio**2 * self.motor_inertia # 6.584

        self.hanger_mass = 0.26
        self.ext_load = 0.0
        self.hanger_distance = 0.7 - 0.04 - 0.06 # 0.6

        self.m = self.link_mass + self.hanger_mass # 1.385
        self.lc = (self.center_distance * self.link_mass + self.hanger_distance * self.hanger_mass)/self.m # 0.36444
        self.Ic = self.const_inertia + self.hanger_mass * (self.hanger_distance **2) # 6.6776

        self.coul_friction = 0.0
        self.visc_friction = 0.00276 * gear_ratio**2 

        # --- Savitzky-Golay filter settings ---
        self.window_size = 25        # number of points in sliding window
        self.poly_order = 2             # polynomial order for fitting
        self.velFilBuf = deque(maxlen=self.window_size)  # velocity buffer for filtering
        self.timeFilBuf = deque(maxlen=self.window_size)
        self.t_filter_ref = 0

        # inputs
        self.Kp = 35
        self.Kd = 6
        self.torque_set = 0.0
        self.pos_set = self.start_pos
        self.vel_set = 0.0    
        self.acc_set = 0.0   
        self.ctrl_bandwidth = 2000
        self.enc_bandwidth = 1000

        # show
        self.control_loop = 0.001
        self.pos = 0.0
        self.vel = 0.0                  # raw velocity from ODrive (used for control)
        self.preT = 0.0
        self.data = deque(maxlen=800)

    def connect(self):
        try:
            print("Connecting to Odrive...")
            self.odrv = odrive.find_any(timeout=10) # Add timeout to prevent hang
            if self.odrv:
                print(f"Connected to Odrive: {self.odrv.serial_number}")       
                self.axis = self.odrv.axis1
                self.connected = True
                self.error = False
            else:
                print("ODrive not found")
        except Exception as e:
            self.error = True
            print(f"Connection failed: {e}")

    def clear_error(self):
        self.axis.controller.error = 0
        self.axis.encoder.error = 0
        self.axis.motor.error = 0
        self.axis.error = 0
        self.error = False
        print("Errors cleared")

    def enter_closed_loop(self):
        self.clear_error()
        self.axis.requested_state = CLOSED_LOOP_CONTROL
        self.closed_loop_control = True
        print("Entered CLOSED_LOOP_CONTROL mode.")

    def return_IDLE(self):
        self.axis.controller.input_torque = 0
        self.axis.requested_state = IDLE
        self.closed_loop_control = False

    def is_controlable(self):
        #return self.connected and self.closed_loop_control and self.isOffset and not self.estop
        return self.connected and self.closed_loop_control and self.isOffset and not self._estop_event.is_set()

    def update_ctrlElms(self, *ctrlElms):
        try:
            target = 0
            with self.data_lock:
                self.t_ref = time.time()
                target = ctrlElms[0]
                self.max_vel = ctrlElms[1]
                self.Kp = ctrlElms[2]
                self.Kd = ctrlElms[3]
                self.ctrl_bandwidth = ctrlElms[4]
                self.enc_bandwidth = ctrlElms[5]
                self.axis.motor.config.current_control_bandwidth = self.ctrl_bandwidth
                self.axis.encoder.config.bandwidth = self.enc_bandwidth
            self.traj.param_calc(self.pos, target, self.max_vel)

        except Exception as e:
            print("Elements update error:", e)

    def update_loadParms(self, *loadParms):
        try:
            self.ext_load = loadParms[0]
            self.hanger_distance = loadParms[1]
            self.coul_friction = loadParms[2]
            self.visc_friction = loadParms[3]
            self.m = self.link_mass + self.hanger_mass + self.ext_load
            self.lc = (self.center_distance * self.link_mass + self.hanger_distance * (self.hanger_mass + self.ext_load))/self.m
            self.Ic = self.const_inertia + (self.hanger_mass + self.ext_load) * (self.hanger_distance **2)
            self.max_torque = loadParms[4]

        except Exception as e:
            print("Parameter update error:", e)

    def get_state(self):
        if self.connected: return self.axis.current_state
        return None
    
    def emergency_stop(self):
        """Immediate torque -> 0 and prevent further torque writes until reset"""
        self._estop_event.set()
        self.torque_set = 0.0
        try:
            if self.connected and self.axis is not None:
                self.axis.controller.input_torque = 0
        except Exception:
            pass

    def get_data(self): #cần sử dụng để kiểm tra chế độ của ODrive - Sử dụng ở GUI
        with self.data_lock:
            return list(self.data)
    
    def set_offset(self): #Sử dụng ở GUI
        try:
            self.offset = self.axis.encoder.pos_estimate
            self.isOffset = True
        except Exception:
            pass

    def reset(self):
        self.traj.reset()
        self.t_ref = - math.inf
        self.velFilBuf.clear()
        self.timeFilBuf.clear()
        self.return_IDLE()
        self.isOffset = False
        self._estop_event.clear()

    def stop(self):
        self._stop_event.set()
        try:
            if self.axis is not None:
                self.axis.controller.input_torque = 0
                time.sleep(0.05)
                self.axis.requested_state = AXIS_STATE_IDLE  
                time.sleep(0.05)
        except Exception:
            pass

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
        ep = q - q_d
        ev = qdot - qdot_d
        D = self.visc_friction

        tor = Ic * (qddot_d - (Kp * ep + Kd * ev)) + m * g * lc * math.cos(q) + D*qdot 
        tor = (tor)/self.tor_coef/gear_ratio
        self.torque_set = max(min(tor, self.max_torque), -self.max_torque)

    def run(self):
        while not self._stop_event.is_set():
            t_start = time.perf_counter()
            try:
                if not self.connected:
                    self.connect()
                    if not self.connected: 
                        time.sleep(0.1)
                        continue

                if self._estop_event.is_set():
                    self._stop_event.wait(0.1)
                    continue    

                # Data collect
                with self.data_lock:
                    t = time.time()
                    deltaT = t - self.preT
                    self.pos = (self.axis.encoder.pos_estimate - self.offset) * 360 / gear_ratio + self.start_pos
                    self.vel = self.axis.encoder.vel_estimate * 360 / gear_ratio  # raw velocity from ODrive (used for control)
                    
                    if deltaT >= self.control_loop:
                        
                        self.velFilBuf.append(self.vel)
                        self.timeFilBuf.append(t)

                        vel_filtered = 0
                        acc = 0.0
                        jerk = 0.0
                        if len(self.velFilBuf) == self.window_size:
                            vel = np.array(list(self.velFilBuf))
                            _t = np.array(list(self.timeFilBuf))
                            vel_filtered, acc, jerk = get_acc_jerk(_t, vel, self.window_size, self.poly_order)

                    tor_set = self.axis.motor.current_control.Iq_setpoint * self.Kt
                    self.data.append((t, self.pos, vel_filtered,acc, self.pos_set, self.vel_set, self.acc_set, jerk, tor_set))
                    

                    
                    if len(self.data) > 800:
                        self.data = self.data[-800:]

                if self.is_controlable():
                    with self.data_lock:
                        self.dynamic_calculation()
                        self.axis.controller.input_torque = self.torque_set
                else: 
                    self.pos_set = self.start_pos
                    self.torque_set = 0.0

            except Exception as e:
                print("ODrive error:", e)
                self.connected = False
                self.closed_loop_control = False
                self.error = True
                time.sleep(1)
            
            t_end = time.perf_counter()
            t_sleep = 0.01 - (t_end - t_start)
            if t_sleep > 0:
                self._stop_event.wait(timeout=t_sleep)
