"""
Low-level interface for executing RL actions on four Dynamixels.

Responsibilities:
- Initialize Dynamixels
- Convert RL actions into relative encoder movements
- Execute synchronized motor commands
- Read encoder positions and currents
- Enforce relative position limits
- Shutdown safely
"""

from __future__ import annotations

import time
import numpy as np

from dynamixel_sdk import *


# =====================================================
# Dynamixel addresses
# =====================================================

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64

ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_CURRENT = 126


# =====================================================
# Constants
# =====================================================

EXTENDED_POSITION_MODE = 4

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0


# RL action scaling
# action [-1,1] -> encoder delta
MAX_ENCODER_DELTA = 400


# Safety:
# allowed movement from initial encoder position
MAX_ENCODER_TRAVEL = 10000



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


        # Current commanded encoder targets
        self.targets = {}


        # Encoder positions at initialization
        self.initial_positions = {}



    # =================================================
    # Initialization
    # =================================================

    def initialize(self):

        print("Initializing Dynamixels...")


        for motor in self.motor_ids:


            # Disable torque before changing mode
            self.write1(
                motor,
                ADDR_TORQUE_ENABLE,
                TORQUE_DISABLE,
            )


            # Extended position mode
            self.write1(
                motor,
                ADDR_OPERATING_MODE,
                EXTENDED_POSITION_MODE,
            )


            # Enable torque
            self.write1(
                motor,
                ADDR_TORQUE_ENABLE,
                TORQUE_ENABLE,
            )


            # Read current encoder position
            pos = self.read_position(motor)


            self.initial_positions[motor] = pos
            self.targets[motor] = pos


            print(
                f"Motor {motor}: "
                f"initial position = {pos}"
            )


        print("Initialization complete.")




    # =================================================
    # Execute RL action
    # =================================================

    def execute(
        self,
        action,
    ):

        """
        Execute one RL action.

        action:
            np.ndarray shape (4,)
            values in [-1,1]

        Positive:
            increase encoder position
            (wind string)

        Negative:
            decrease encoder position
            (release string)
        """


        if len(action) != len(self.motor_ids):

            raise ValueError(
                "Action dimension does not match motors"
            )


        self.group_sync_write.clearParam()


        for motor, delta in zip(
            self.motor_ids,
            action,
        ):


            # -----------------------------
            # Action clipping
            # -----------------------------

            delta = np.clip(
                delta,
                -1.0,
                1.0,
            )


            encoder_delta = int(
                delta * MAX_ENCODER_DELTA
            )


            # -----------------------------
            # Update target position
            # -----------------------------

            self.targets[motor] += encoder_delta



            # -----------------------------
            # Relative safety limits
            # -----------------------------

            lower_limit = (
                self.initial_positions[motor]
                -
                MAX_ENCODER_TRAVEL
            )


            upper_limit = (
                self.initial_positions[motor]
                +
                MAX_ENCODER_TRAVEL
            )


            self.targets[motor] = int(
                np.clip(
                    self.targets[motor],
                    lower_limit,
                    upper_limit,
                )
            )


            target = self.targets[motor]


            print(
                f"Motor {motor}: "
                f"delta={encoder_delta}, "
                f"target={target}"
            )



            # -----------------------------
            # Prepare SyncWrite packet
            # -----------------------------

            param = [

                DXL_LOBYTE(
                    DXL_LOWORD(target)
                ),

                DXL_HIBYTE(
                    DXL_LOWORD(target)
                ),

                DXL_LOBYTE(
                    DXL_HIWORD(target)
                ),

                DXL_HIBYTE(
                    DXL_HIWORD(target)
                ),

            ]


            if not self.group_sync_write.addParam(
                motor,
                param,
            ):

                raise RuntimeError(
                    f"Failed adding motor {motor}"
                )



        # Send synchronized command

        result = self.group_sync_write.txPacket()

        self.group_sync_write.clearParam()


        if result != COMM_SUCCESS:

            raise RuntimeError(
                self.packet.getTxRxResult(result)
            )



    # =================================================
    # Read motor states
    # =================================================


    def read_position(
        self,
        motor_id,
    ):

        pos, _, _ = self.packet.read4ByteTxRx(
            self.port,
            motor_id,
            ADDR_PRESENT_POSITION,
        )


        # Convert unsigned -> signed
        if pos > 0x7FFFFFFF:
            pos -= 0x100000000


        return pos



    def read_positions(self):

        return [
            self.read_position(m)
            for m in self.motor_ids
        ]




    def read_current(
        self,
        motor_id,
    ):

        current, _, _ = self.packet.read2ByteTxRx(
            self.port,
            motor_id,
            ADDR_PRESENT_CURRENT,
        )


        # Dynamixel current is signed
        if current > 0x7FFF:
            current -= 0x10000


        return current



    def read_currents(self):

        return [
            self.read_current(m)
            for m in self.motor_ids
        ]




    # =================================================
    # Helpers
    # =================================================


    def write1(
        self,
        motor,
        address,
        value,
    ):

        comm, error = self.packet.write1ByteTxRx(
            self.port,
            motor,
            address,
            value,
        )


        if comm != COMM_SUCCESS:

            raise RuntimeError(
                f"Motor {motor}: "
                f"{self.packet.getTxRxResult(comm)}"
            )


        if error != 0:

            raise RuntimeError(
                f"Motor {motor}: "
                f"{self.packet.getRxPacketError(error)}"
            )




    # =================================================
    # Shutdown
    # =================================================


    def shutdown(self):

        print("Disabling motors...")


        for motor in self.motor_ids:

            self.write1(
                motor,
                ADDR_TORQUE_ENABLE,
                TORQUE_DISABLE,
            )


        print("Shutdown complete.")

