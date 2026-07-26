"""
Low-level interface for executing motor actions on the four Dynamixels.

- Initialize motors
- Execute one RL action
- Read encoder positions
- Read motor currents
- Shutdown motors safely
"""

from __future__ import annotations
import time
import numpy as np
from dynamixel_sdk import *


ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64

ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_CURRENT = 126

EXTENDED_POSITION_MODE = 4

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

LEN_GOAL_POSITION = 4
MAX_ENCODER_DELTA = 400
MIN_POSITION = 0
MAX_POSITION = 20000 # TODO: NEED TO CHANGE THIS!!!!!!!


class DynamixelExecutor:

    def __init__(
        self,
        port_handler,
        packet_handler,
        motor_ids,
        group_sync_write,
    ):

        self.port = port_handler
        self.packet = packet_handler

        self.motor_ids = list(motor_ids)

        self.group_sync_write = group_sync_write

        # Current commanded targets
        self.targets = {}

    def initialize(self):

        print("Initializing Dynamixels...")

        for motor in self.motor_ids:

            self.write1(motor,ADDR_TORQUE_ENABLE,TORQUE_DISABLE,)
            self.write1(motor,ADDR_OPERATING_MODE,EXTENDED_POSITION_MODE,)
            self.write1(motor,ADDR_TORQUE_ENABLE,TORQUE_ENABLE,)
            self.targets[motor] = self.read_position(motor)

        print("Initialization complete.")

    # RL Action
    def execute(self,action,):
        # Values are encoder count increments.
        if len(action) != len(self.motor_ids):
            raise ValueError("Action dimension does not match number of motors.")

        self.group_sync_write.clearParam()

        for motor, delta in zip(self.motor_ids,action,):

            #clipping 
            delta = max(-1.0, min(1.0, delta))
            
            self.targets[motor] += int(delta * MAX_ENCODER_DELTA)
            self.targets[motor] = np.clip(
                self.targets[motor],
                MIN_POSITION,
                MAX_POSITION,
            )
            target = int(self.targets[motor])
            param = [

                DXL_LOBYTE(DXL_LOWORD(target)),
                DXL_HIBYTE(DXL_LOWORD(target)),
                DXL_LOBYTE(DXL_HIWORD(target)),
                DXL_HIBYTE(DXL_HIWORD(target)),

            ]

            if not self.group_sync_write.addParam(
                motor,
                param,
            ):
                raise RuntimeError(f"Could not add motor {motor}")

        result = self.group_sync_write.txPacket()
        self.group_sync_write.clearParam()

        if result != COMM_SUCCESS:

            raise RuntimeError(self.packet.getTxRxResult(result)
            )

    # Read Motor State
    def read_position(
        self,
        motor_id,
    ):

        pos, _, _ = self.packet.read4ByteTxRx(self.port,motor_id,ADDR_PRESENT_POSITION,)

        if pos > 0x7FFFFFFF:
            pos -= 0x100000000

        return pos

    def read_positions(self):
        return [self.read_position(m) for m in self.motor_ids]

    def read_current(self,motor_id,):

        current, _, _ = self.packet.read2ByteTxRx(
            self.port,
            motor_id,
            ADDR_PRESENT_CURRENT,
        )
        return current

    def read_currents(self):

        return [self.read_current(m) for m in self.motor_ids]

    # Helpers
    def write1(self,motor,address,value,):

        self.packet.write1ByteTxRx(self.port,motor,address,value,)

    # Shutdown
    def shutdown(self):

        print("Disabling motors...")

        for motor in self.motor_ids:
            self.write1(motor,ADDR_TORQUE_ENABLE,TORQUE_DISABLE,)
        print("Done.")