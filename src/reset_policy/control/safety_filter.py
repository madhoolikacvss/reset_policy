# reset_policy/control/safety_filter.py

"""
Safety filter for RL actions on the cube-string system.

The safety filter intercepts PPO actions and modifies them to prevent:
1. Motors fighting against each other (high opposing currents)
2. Exceeding position limits
3. Over-current conditions
4. Voltage drops causing communication loss
5. Overloading already-stressed motors
6. Tension going too low (slack strings)
7. Single motor overload (even if pair is balanced)

The filter logs all interventions so the policy can learn to avoid them.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from enum import Enum


class SafetyReason(Enum):
    """Reasons why an action was modified."""
    SAFE = "safe"
    HIGH_HORIZONTAL_CURRENT = "high_horizontal_current"
    HIGH_VERTICAL_CURRENT = "high_vertical_current"
    SINGLE_MOTOR_OVER_CURRENT = "single_motor_over_current"
    POSITION_LIMIT_PULL = "position_limit_pull" 
    POSITION_LIMIT_RELEASE = "position_limit_release"
    OPPOSING_MOTORS = "opposing_motors"
    CURRENT_AWARE_SCALING = "current_aware_scaling"
    VOLTAGE_LOW = "voltage_low"
    VOLTAGE_CRITICAL = "voltage_critical"
    TENSION_TOO_LOW = "tension_too_low"
    TENSION_SAFETY = "tension_safety"           # <-- ADDED for single motor tension
    TEMPERATURE_HIGH = "temperature_high"     
    TEMPERATURE_CRITICAL = "temperature_critical"  
    MOTOR_STUCK = "motor_stuck"                


@dataclass
class SafetyResult:
    """Result of applying the safety filter."""
    
    safe_action: np.ndarray
    raw_action: np.ndarray
    modified: bool
    reason: Optional[SafetyReason] = None
    detail: str = ""
    affected_motors: List[int] = None
    pre_filter_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.affected_motors is None:
            self.affected_motors = []
        if self.pre_filter_metrics is None:
            self.pre_filter_metrics = {}


class SafetyFilter:
    """
    Safety filter for RL actions.
    
    Features:
        - Current-aware proactive scaling (prevents overloading)
        - Voltage-aware proactive scaling (prevents communication loss)
        - Tension safety (prevents motors fighting & single motor overload)
        - Tension constraint (prevents slack strings)
        - Pair-wise current limiting (reactive)
        - Single motor over-current protection (reactive)
        - Position limit protection
    """
    
    def __init__(
        self,
        motor_ids: List[int],
        pair_horizontal: List[int] = None,
        pair_vertical: List[int] = None,
        
        # Current thresholds (mA)
        max_pair_current: float = 1000.0,
        max_single_current: float = 800.0,
        
        # Position limits (encoder ticks relative to initial)
        max_position: float = 8000.0,
        
        # Voltage thresholds (V)
        min_voltage: float = 4.0,
        critical_voltage: float = 3.5,
        
        # Action scaling factors
        current_scale_factor: float = 0.25,
        position_scale_factor: float = 0.5,
        voltage_scale_factor: float = 0.3,
        
        # Temperature thresholds (°C)
        temp_threshold: float = 45.0,       
        temp_critical: float = 55.0, 
        temp_scale_factor: float = 0.3,
        
        # Current-aware scaling thresholds (mA)
        heavy_load_threshold: float = 400.0,
        critical_load_threshold: float = 700.0,
        heavy_load_scale: float = 0.3,
        moderate_load_scale: float = 0.6,
        
        # Tension constraint (mA)
        min_tension_threshold: float = 30.0,

        # Single motor tension thresholds (mA)
        single_motor_tension_threshold: float = 300.0,  # <-- NEW
        single_motor_tension_high: float = 500.0,       # <-- NEW
        single_motor_tension_critical: float = 700.0,   # <-- NEW

        # Enable/disable features
        enable_current_safety: bool = True,
        enable_position_safety: bool = True,
        enable_tension_safety: bool = True,
        enable_voltage_safety: bool = True,
        enable_current_aware_safety: bool = True,
        enable_tension_constraint: bool = True,  
        enable_temperature_safety: bool = True,
        enable_stuck_motor_safety: bool = True,
        enable_single_motor_tension: bool = True,  # <-- NEW
        
        # Logging
        log_interventions: bool = True,
    ):
        self.motor_ids = motor_ids
        self.num_motors = len(motor_ids)
        
        # Define motor pairs
        if pair_horizontal is None:
            pair_horizontal = [16, 17]
        if pair_vertical is None:
            pair_vertical = [18, 19]
        
        self.pair_horizontal = pair_horizontal
        self.pair_vertical = pair_vertical
        
        # Get indices for pairs
        self.h_idx = [self.motor_ids.index(m) for m in pair_horizontal]
        self.v_idx = [self.motor_ids.index(m) for m in pair_vertical]
        
        # Thresholds
        self.max_pair_current = max_pair_current
        self.max_single_current = max_single_current
        self.max_position = max_position
        
        # Voltage thresholds
        self.min_voltage = min_voltage
        self.critical_voltage = critical_voltage
        
        # Action scaling factors
        self.current_scale_factor = current_scale_factor
        self.position_scale_factor = position_scale_factor
        self.voltage_scale_factor = voltage_scale_factor
        
        # Current-aware scaling
        self.heavy_load_threshold = heavy_load_threshold
        self.critical_load_threshold = critical_load_threshold
        self.heavy_load_scale = heavy_load_scale
        self.moderate_load_scale = moderate_load_scale
        
        # Tension constraint
        self.min_tension_threshold = min_tension_threshold
        
        # Single motor tension thresholds (NEW)
        self.single_motor_tension_threshold = single_motor_tension_threshold
        self.single_motor_tension_high = single_motor_tension_high
        self.single_motor_tension_critical = single_motor_tension_critical
        
        # Feature flags
        self.enable_current_safety = enable_current_safety
        self.enable_position_safety = enable_position_safety
        self.enable_tension_safety = enable_tension_safety
        self.enable_voltage_safety = enable_voltage_safety
        self.enable_current_aware_safety = enable_current_aware_safety
        self.enable_tension_constraint = enable_tension_constraint  
        self.enable_temperature_safety = enable_temperature_safety
        self.enable_stuck_motor_safety = enable_stuck_motor_safety
        self.enable_single_motor_tension = enable_single_motor_tension  # <-- NEW
        self.log_interventions = log_interventions

        # Temperature thresholds
        self.temp_threshold = temp_threshold
        self.temp_critical = temp_critical
        self.temp_scale_factor = temp_scale_factor
            
        # Statistics
        self.intervention_count = 0
        self.intervention_reasons = {reason: 0 for reason in SafetyReason}
        self.last_result = None
        
        print(f"SafetyFilter initialized:")
        print(f"  Horizontal pair: {pair_horizontal}")
        print(f"  Vertical pair: {pair_vertical}")
        print(f"  Max pair current: {max_pair_current} mA")
        print(f"  Max single current: {max_single_current} mA")
        print(f"  Max position: {max_position}")
        print(f"  Min voltage: {min_voltage} V")
        print(f"  Critical voltage: {critical_voltage} V")
        print(f"  Heavy load threshold: {heavy_load_threshold} mA")
        print(f"  Critical load threshold: {critical_load_threshold} mA")
        print(f"  Min tension threshold: {min_tension_threshold} mA")
        print(f"  Single motor tension threshold: {single_motor_tension_threshold} mA")
        
    def filter(
        self,
        action: np.ndarray,
        currents: np.ndarray,
        positions: np.ndarray,
        initial_positions: np.ndarray,
        voltages: np.ndarray = None,
        temperatures: np.ndarray = None,
        action_count: int = 0,
    ) -> SafetyResult:
        """
        Apply safety filter to the action.
        """
        
        # Store raw action
        raw_action = action.copy()
        
        # Initialize result
        result = SafetyResult(
            safe_action=action.copy(),
            raw_action=raw_action,
            modified=False,
            reason=SafetyReason.SAFE,
            affected_motors=[],
            pre_filter_metrics={
                "max_current": float(np.max(np.abs(currents))) if currents is not None else 0,
                "horizontal_current": 0.0,
                "vertical_current": 0.0,
                "horizontal_action_sum": 0.0,
                "vertical_action_sum": 0.0,
            }
        )
        
        # ============================================================
        # 1. CURRENT-AWARE SAFETY - PROACTIVE
        # ============================================================
        
        if self.enable_current_aware_safety and currents is not None:
            action, modified, reason, detail, affected = self._apply_current_aware_safety(
                action, currents, positions
            )
            if modified:
                result.modified = True
                result.reason = reason
                result.detail = detail
                result.affected_motors = affected
                self._log_intervention(reason, detail, raw_action, action)
        
        # ============================================================
        # 2. TEMPERATURE SAFETY - PROACTIVE
        # ============================================================
        
        if self.enable_temperature_safety and temperatures is not None:
            action, modified, reason, detail, affected = self._apply_temperature_safety(
                action, temperatures
            )
            if modified:
                result.modified = True
                result.reason = reason
                result.detail = detail
                result.affected_motors = affected
                self._log_intervention(reason, detail, raw_action, action)
        
        # ============================================================
        # 3. VOLTAGE SAFETY - PROACTIVE
        # ============================================================
        
        if self.enable_voltage_safety and voltages is not None:
            action, modified, reason, detail, affected = self._apply_voltage_safety(
                action, voltages
            )
            if modified:
                result.modified = True
                result.reason = reason
                result.detail = detail
                result.affected_motors = affected
                self._log_intervention(reason, detail, raw_action, action)
        
        # ============================================================
        # 4. TENSION SAFETY - PROACTIVE (opposing motors + single motor)
        # ============================================================
        
        if self.enable_tension_safety and currents is not None:
            action, modified, reason, detail, affected = self._apply_tension_safety(
                action, currents
            )
            if modified:
                result.modified = True
                result.reason = reason
                result.detail = detail
                result.affected_motors = affected
                self._log_intervention(reason, detail, raw_action, action)
        
        # ============================================================
        # 5. TENSION CONSTRAINT - PROACTIVE (slack strings)
        # ============================================================
        
        if self.enable_tension_constraint and currents is not None:
            action, modified, reason, detail, affected = self._apply_tension_constraint(
                action, currents
            )
            if modified:
                result.modified = True
                result.reason = reason
                result.detail = detail
                result.affected_motors = affected
                self._log_intervention(reason, detail, raw_action, action)
                
        # ============================================================
        # 6. CURRENT SAFETY - REACTIVE
        # ============================================================
        
        if self.enable_current_safety and currents is not None:
            action, modified, reason, detail, affected = self._apply_current_safety(
                action, currents
            )
            if modified:
                result.modified = True
                result.reason = reason
                result.detail = detail
                result.affected_motors = affected
                self._log_intervention(reason, detail, raw_action, action)
        
        # ============================================================
        # 7. POSITION SAFETY - REACTIVE
        # ============================================================
        
        if self.enable_position_safety and positions is not None and initial_positions is not None:
            action, modified, reason, detail, affected = self._apply_position_safety(
                action, positions, initial_positions
            )
            if modified:
                result.modified = True
                result.reason = reason
                result.detail = detail
                result.affected_motors = affected
                self._log_intervention(reason, detail, raw_action, action)
        
        # --------------------------------------------------
        # Clamp to valid range
        # --------------------------------------------------
        
        action = np.clip(action, -1.0, 1.0)
        
        # Store result
        result.safe_action = action
        self.last_result = result
        
        return result
    
    # ============================================================
    # CURRENT-AWARE SAFETY - PROACTIVE
    # ============================================================
    
    def _apply_current_aware_safety(
        self,
        action: np.ndarray,
        currents: np.ndarray,
        motor_positions: np.ndarray = None,
    ) -> Tuple[np.ndarray, bool, Optional[SafetyReason], str, List[int]]:
        """
        Apply current-aware safety.
        
        Proactively scales down actions for motors that are already
        heavily loaded, preventing them from being pushed into overload.
        """
        
        modified = False
        reason = None
        detail = ""
        affected = []
        
        if currents is None or not self.enable_current_aware_safety:
            return action, modified, reason, detail, affected
        
        for i, motor_id in enumerate(self.motor_ids):
            current = abs(currents[i])
            
            # Skip if action is near zero
            if abs(action[i]) < 0.05:
                continue
            
            # Determine scaling factor based on current load
            if current < 200:
                scale = 1.0
            elif current < self.heavy_load_threshold:  # 200-400mA
                load_range = self.heavy_load_threshold - 200
                scale = 1.0 - (current - 200) / load_range * (1.0 - self.moderate_load_scale)
                scale = max(self.moderate_load_scale, min(1.0, scale))
            elif current < self.critical_load_threshold:  # 400-700mA
                load_range = self.critical_load_threshold - self.heavy_load_threshold
                scale = self.moderate_load_scale - (current - self.heavy_load_threshold) / load_range * (self.moderate_load_scale - self.heavy_load_scale)
                scale = max(self.heavy_load_scale, min(self.moderate_load_scale, scale))
            else:  # > 700mA
                scale = self.heavy_load_scale
                if current > 900:
                    scale *= 0.5
                if current > 1000:
                    scale *= 0.3
            
            # Apply scaling if action is pulling (positive)
            if action[i] > 0.05 and scale < 0.99:
                old_action = action[i]
                action[i] *= scale
                modified = True
                reason = SafetyReason.CURRENT_AWARE_SCALING
                detail = f"Motor {motor_id}: {current:.0f}mA -> scale={scale:.2f}"
                affected.append(motor_id)
                print(f"  [CURRENT-AWARE] Motor {motor_id}: {current:.0f}mA, action {old_action:.3f} -> {action[i]:.3f}")
        
        return action, modified, reason, detail, affected
    
    # ============================================================
    # VOLTAGE SAFETY - PROACTIVE
    # ============================================================
    
    def _apply_voltage_safety(
        self,
        action: np.ndarray,
        voltages: np.ndarray,
    ) -> Tuple[np.ndarray, bool, Optional[SafetyReason], str, List[int]]:
        """
        Apply voltage-based safety.
        
        When voltage drops below thresholds, scale down actions
        to prevent further voltage drop and motor disconnection.
        """
        
        modified = False
        reason = None
        detail = ""
        affected = []
        
        if voltages is None or not self.enable_voltage_safety:
            return action, modified, reason, detail, affected
        
        for i, motor_id in enumerate(self.motor_ids):
            voltage = voltages[i]
            if voltage is None:
                continue
            
            # Critical voltage - heavy scaling
            if voltage < self.critical_voltage:  # 3.5V
                scale = self.voltage_scale_factor * 0.5  # 0.15
                action[i] *= scale
                modified = True
                reason = SafetyReason.VOLTAGE_CRITICAL
                detail = f"Motor {motor_id}: {voltage:.1f}V < {self.critical_voltage}V"
                affected.append(motor_id)
                print(f"  [VOLTAGE] Motor {motor_id}: {voltage:.1f}V -> scale={scale:.2f}")
            
            # Low voltage - moderate scaling
            elif voltage < self.min_voltage:  # 4.0V
                scale_range = self.min_voltage - self.critical_voltage
                scale = self.voltage_scale_factor * (voltage - self.critical_voltage) / scale_range
                scale = max(0.1, min(1.0, scale))
                action[i] *= scale
                modified = True
                reason = SafetyReason.VOLTAGE_LOW
                detail = f"Motor {motor_id}: {voltage:.1f}V < {self.min_voltage}V"
                affected.append(motor_id)
                print(f"  [VOLTAGE] Motor {motor_id}: {voltage:.1f}V -> scale={scale:.2f}")
        
        return action, modified, reason, detail, affected
    
    # ============================================================
    # TENSION SAFETY - PROACTIVE (opposing motors + single motor)
    # ============================================================
    
    def _apply_tension_safety(
        self,
        action: np.ndarray,
        currents: np.ndarray,
    ) -> Tuple[np.ndarray, bool, Optional[SafetyReason], str, List[int]]:
        """
        Prevent motors from fighting each other and prevent single motor overload.
        
        This combines:
        1. Single-motor tension safety - scales down any motor pulling with high current
        2. Opposing motors safety - if both motors in a pair are pulling, reduce one
        """
        
        modified = False
        reason = None
        detail = ""
        affected = []
        
        if currents is None:
            return action, modified, reason, detail, affected
        
        # ============================================================
        # 1. SINGLE-MOTOR TENSION SAFETY (NEW)
        #    If a motor is pulling with high current, scale it down
        # ============================================================
        
        if self.enable_single_motor_tension:
            for i, motor_id in enumerate(self.motor_ids):
                current = abs(currents[i])
                action_val = action[i]
                
                # If motor is pulling (positive) and has high current
                if action_val > 0.1 and current > self.single_motor_tension_threshold:
                    # Scale down based on how high the current is
                    if current > self.single_motor_tension_critical:  # > 700mA
                        scale = 0.3
                    elif current > self.single_motor_tension_high:    # > 500mA
                        scale = 0.5
                    else:  # > 300mA
                        scale = 0.7
                    
                    if scale < 0.99:
                        old_action = action[i]
                        action[i] *= scale
                        modified = True
                        reason = SafetyReason.TENSION_SAFETY
                        detail = f"Motor {motor_id}: {current:.0f}mA pulling -> scale={scale:.2f}"
                        affected.append(motor_id)
                        print(f"  [TENSION] Motor {motor_id}: {current:.0f}mA -> action {old_action:.3f} → {action[i]:.3f}")
        
        # ============================================================
        # 2. OPPOSING MOTORS SAFETY (existing)
        #    If both motors in a pair are pulling with tension, reduce one
        # ============================================================
        
        # Check horizontal pair (16, 17)
        a16 = action[self.h_idx[0]]
        a17 = action[self.h_idx[1]]
        c16 = abs(currents[self.h_idx[0]])
        c17 = abs(currents[self.h_idx[1]])
        
        if a16 > 0.3 and a17 > 0.3 and c16 > 100 and c17 > 100:
            if abs(a16) > abs(a17):
                action[self.h_idx[0]] *= 0.5
                affected.append(self.pair_horizontal[0])
            else:
                action[self.h_idx[1]] *= 0.5
                affected.append(self.pair_horizontal[1])
            modified = True
            reason = SafetyReason.OPPOSING_MOTORS
            detail = f"Motors {self.pair_horizontal[0]} and {self.pair_horizontal[1]} both pulling with tension"
        
        # Check vertical pair (18, 19)
        a18 = action[self.v_idx[0]]
        a19 = action[self.v_idx[1]]
        c18 = abs(currents[self.v_idx[0]])
        c19 = abs(currents[self.v_idx[1]])
        
        if a18 > 0.3 and a19 > 0.3 and c18 > 100 and c19 > 100:
            if abs(a18) > abs(a19):
                action[self.v_idx[0]] *= 0.5
                affected.append(self.pair_vertical[0])
            else:
                action[self.v_idx[1]] *= 0.5
                affected.append(self.pair_vertical[1])
            modified = True
            reason = SafetyReason.OPPOSING_MOTORS
            detail = f"Motors {self.pair_vertical[0]} and {self.pair_vertical[1]} both pulling with tension"
        
        return action, modified, reason, detail, affected

    # ============================================================
    # TEMPERATURE SAFETY - PROACTIVE
    # ============================================================

    def _apply_temperature_safety(
        self,
        action: np.ndarray,
        temperatures: np.ndarray,
    ) -> Tuple[np.ndarray, bool, Optional[SafetyReason], str, List[int]]:
        """
        Scale down actions if motors are overheating.
        """
        modified = False
        reason = None
        detail = ""
        affected = []
        
        if temperatures is None:
            return action, modified, reason, detail, affected
        
        for i, motor_id in enumerate(self.motor_ids):
            temp = temperatures[i]
            if temp is None:
                continue
            
            # If temperature > 45°C, start scaling
            if temp > 45:
                scale = max(0.2, 1.0 - (temp - 45) / 20)
                action[i] *= scale
                modified = True
                reason = SafetyReason.TEMPERATURE_HIGH
                detail = f"Motor {motor_id}: {temp:.0f}°C -> scale={scale:.2f}"
                affected.append(motor_id)
                print(f"  [TEMPERATURE] Motor {motor_id}: {temp:.0f}°C, scaling to {scale:.2f}")
        
        return action, modified, reason, detail, affected
    
    # ============================================================
    # TENSION CONSTRAINT - PROACTIVE (slack strings)
    # ============================================================

    def _apply_tension_constraint(
        self,
        action: np.ndarray,
        currents: np.ndarray,
    ) -> Tuple[np.ndarray, bool, Optional[SafetyReason], str, List[int]]:
        """
        Ensure minimum tension is maintained.
        
        If any pair has both motors releasing (negative actions) and tension
        is too low, prevent further release to avoid slack strings.
        """
        
        modified = False
        reason = None
        detail = ""
        affected = []
        
        if currents is None:
            return action, modified, reason, detail, affected
        
        # Calculate current tension for each pair
        horizontal_tension = abs(currents[0]) + abs(currents[1])
        vertical_tension = abs(currents[2]) + abs(currents[3])
        
        # ============================================================
        # Horizontal pair (16, 17)
        # ============================================================
        
        if horizontal_tension < self.min_tension_threshold:
            # Both releasing - scale down the release
            if action[0] < 0 and action[1] < 0:
                action[0] *= 0.2
                action[1] *= 0.2
                modified = True
                reason = SafetyReason.TENSION_TOO_LOW
                detail = f"Horizontal tension {horizontal_tension:.0f}mA < {self.min_tension_threshold}mA"
                affected.extend(self.pair_horizontal)
                print(f"  [TENSION] Horizontal tension too low ({horizontal_tension:.0f}mA), scaling release")
        
        # ============================================================
        # Vertical pair (18, 19)
        # ============================================================
        
        if vertical_tension < self.min_tension_threshold:
            if action[2] < 0 and action[3] < 0:
                action[2] *= 0.2
                action[3] *= 0.2
                modified = True
                reason = SafetyReason.TENSION_TOO_LOW
                detail = f"Vertical tension {vertical_tension:.0f}mA < {self.min_tension_threshold}mA"
                affected.extend(self.pair_vertical)
                print(f"  [TENSION] Vertical tension too low ({vertical_tension:.0f}mA), scaling release")
        
        return action, modified, reason, detail, affected
    
    # ============================================================
    # CURRENT SAFETY - REACTIVE
    # ============================================================
    
    def _apply_current_safety(
        self,
        action: np.ndarray,
        currents: np.ndarray,
    ) -> Tuple[np.ndarray, bool, Optional[SafetyReason], str, List[int]]:
        """
        Apply current-based safety (reactive).
        
        Enforces hard limits on:
            1. Pair-wise current (horizontal and vertical pairs)
            2. Single motor over-current
        """
        
        modified = False
        reason = None
        detail = ""
        affected = []
        
        if currents is None:
            return action, modified, reason, detail, affected
        
        # Check pair-wise currents
        h_current = abs(currents[self.h_idx[0]]) + abs(currents[self.h_idx[1]])
        v_current = abs(currents[self.v_idx[0]]) + abs(currents[self.v_idx[1]])
        
        # Horizontal pair over-current
        if h_current > self.max_pair_current:
            scale = self.current_scale_factor
            action[self.h_idx[0]] *= scale
            action[self.h_idx[1]] *= scale
            modified = True
            reason = SafetyReason.HIGH_HORIZONTAL_CURRENT
            detail = f"Horizontal current {h_current:.1f}mA > {self.max_pair_current}mA"
            affected.extend(self.pair_horizontal)
            
        # Vertical pair over-current
        elif v_current > self.max_pair_current:
            scale = self.current_scale_factor
            action[self.v_idx[0]] *= scale
            action[self.v_idx[1]] *= scale
            modified = True
            reason = SafetyReason.HIGH_VERTICAL_CURRENT
            detail = f"Vertical current {v_current:.1f}mA > {self.max_pair_current}mA"
            affected.extend(self.pair_vertical)
        
        # Check single motor over-current (only if pair check didn't trigger)
        if not modified:
            for i, motor_id in enumerate(self.motor_ids):
                if abs(currents[i]) > self.max_single_current:
                    action[i] *= self.current_scale_factor
                    modified = True
                    reason = SafetyReason.SINGLE_MOTOR_OVER_CURRENT
                    detail = f"Motor {motor_id}: {abs(currents[i]):.1f}mA > {self.max_single_current}mA"
                    affected.append(motor_id)
        
        return action, modified, reason, detail, affected
    
    # ============================================================
    # POSITION SAFETY - REACTIVE
    # ============================================================
    
    def _apply_position_safety(
        self,
        action: np.ndarray,
        positions: np.ndarray,
        initial_positions: np.ndarray,
    ) -> Tuple[np.ndarray, bool, Optional[SafetyReason], str, List[int]]:
        """
        Prevent motors from hitting position limits.
        """
        
        modified = False
        reason = None
        detail = ""
        affected = []
        
        if positions is None or initial_positions is None:
            return action, modified, reason, detail, affected
        
        for i, motor_id in enumerate(self.motor_ids):
            pos = positions[i]
            init_pos = initial_positions[i]
            delta = pos - init_pos
            
            # Near upper limit and action pulls (positive)
            if delta > self.max_position * 0.8 and action[i] > 0:
                scale = max(0.1, 1.0 - (delta - self.max_position * 0.8) / (self.max_position * 0.2))
                action[i] *= scale
                modified = True
                reason = SafetyReason.POSITION_LIMIT_PULL
                detail = f"Motor {motor_id}: {delta:.0f} ticks near upper limit"
                affected.append(motor_id)
            
            # Near lower limit and action releases (negative)
            elif delta < -self.max_position * 0.8 and action[i] < 0:
                scale = max(0.1, 1.0 - (abs(delta) - self.max_position * 0.8) / (self.max_position * 0.2))
                action[i] *= scale
                modified = True
                reason = SafetyReason.POSITION_LIMIT_RELEASE
                detail = f"Motor {motor_id}: {delta:.0f} ticks near lower limit"
                affected.append(motor_id)
        
        return action, modified, reason, detail, affected
    
    # ============================================================
    # Logging
    # ============================================================
    
    def _log_intervention(
        self,
        reason: SafetyReason,
        detail: str,
        raw_action: np.ndarray,
        safe_action: np.ndarray,
    ):
        """Log an intervention for debugging."""
        
        if not self.log_interventions:
            return
            
        self.intervention_count += 1
        self.intervention_reasons[reason] += 1
        
        print(
            f"\n[SAFETY INTERVENTION #{self.intervention_count}]"
        )
        print(f"  Reason: {reason.value}")
        print(f"  Detail: {detail}")
        print(f"  Raw action: {raw_action}")
        print(f"  Safe action: {safe_action}")
        print(f"  Modification: {safe_action - raw_action}")
    
    def get_stats(self) -> Dict:
        """Get intervention statistics."""
        return {
            "total_interventions": self.intervention_count,
            "intervention_reasons": {
                k.value: v for k, v in self.intervention_reasons.items() if v > 0
            },
            "last_reason": self.last_result.reason.value if self.last_result else None,
            "last_modified": self.last_result.modified if self.last_result else False,
        }


# ============================================================
# Factory function for easy creation
# ============================================================

def create_safety_filter(
    motor_ids: List[int],
    config: Dict = None,
) -> SafetyFilter:
    """Create a safety filter with optional configuration."""
    
    if config is None:
        config = {}
    
    return SafetyFilter(
        motor_ids=motor_ids,
        pair_horizontal=config.get('pair_horizontal', [16, 17]),
        pair_vertical=config.get('pair_vertical', [18, 19]),
        max_pair_current=config.get('max_pair_current', 1000.0),
        max_single_current=config.get('max_single_current', 800.0),
        max_position=config.get('max_position', 8000.0),
        min_voltage=config.get('min_voltage', 4.0),
        critical_voltage=config.get('critical_voltage', 3.5),
        voltage_scale_factor=config.get('voltage_scale_factor', 0.3),
        temp_threshold=config.get('temp_threshold', 45.0),
        temp_critical=config.get('temp_critical', 55.0),
        temp_scale_factor=config.get('temp_scale_factor', 0.3),
        heavy_load_threshold=config.get('heavy_load_threshold', 400.0),
        critical_load_threshold=config.get('critical_load_threshold', 700.0),
        heavy_load_scale=config.get('heavy_load_scale', 0.3),
        moderate_load_scale=config.get('moderate_load_scale', 0.6),
        current_scale_factor=config.get('current_scale_factor', 0.25),
        position_scale_factor=config.get('position_scale_factor', 0.5),
        min_tension_threshold=config.get('min_tension_threshold', 30.0),
        single_motor_tension_threshold=config.get('single_motor_tension_threshold', 300.0),  # <-- NEW
        single_motor_tension_high=config.get('single_motor_tension_high', 500.0),           # <-- NEW
        single_motor_tension_critical=config.get('single_motor_tension_critical', 700.0),   # <-- NEW
        enable_current_safety=config.get('enable_current_safety', True),
        enable_position_safety=config.get('enable_position_safety', True),
        enable_tension_safety=config.get('enable_tension_safety', True),
        enable_voltage_safety=config.get('enable_voltage_safety', True),
        enable_current_aware_safety=config.get('enable_current_aware_safety', True),
        enable_tension_constraint=config.get('enable_tension_constraint', True),
        enable_temperature_safety=config.get('enable_temperature_safety', True),
        enable_stuck_motor_safety=config.get('enable_stuck_motor_safety', True),
        enable_single_motor_tension=config.get('enable_single_motor_tension', True),        # <-- NEW
        log_interventions=config.get('log_interventions', True),
    )