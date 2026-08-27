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

All thresholds are from config.py for centralized management.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from enum import Enum

from reset_policy.config import config


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
    TENSION_SAFETY = "tension_safety"
    TEMPERATURE_HIGH = "temperature_high"
    TEMPERATURE_CRITICAL = "temperature_critical"


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
    """Safety filter for RL actions."""
    
    def __init__(self, motor_ids: List[int] = None, **kwargs):
        """
        Initialize safety filter with config values.
        All thresholds come from config.py unless overridden via kwargs.
        """
        # Motor setup
        self.motor_ids = motor_ids if motor_ids is not None else config.motor.motor_ids
        self.num_motors = len(self.motor_ids)
        
        # Motor pairs from config
        self.pair_horizontal = kwargs.get('pair_horizontal', config.motor.horizontal_pair)
        self.pair_vertical = kwargs.get('pair_vertical', config.motor.vertical_pair)
        
        # Get indices for pairs
        self.h_idx = [self.motor_ids.index(m) for m in self.pair_horizontal]
        self.v_idx = [self.motor_ids.index(m) for m in self.pair_vertical]
        
        # Current thresholds
        self.max_pair_current = kwargs.get('max_pair_current', config.safety.max_pair_current)
        self.max_single_current = kwargs.get('max_single_current', config.safety.max_single_current)
        
        # Position limits
        self.max_position = kwargs.get('max_position', config.safety.max_position)
        
        # Voltage thresholds
        self.min_voltage = kwargs.get('min_voltage', config.safety.min_voltage)
        self.critical_voltage = kwargs.get('critical_voltage', config.safety.critical_voltage)
        
        # Action scaling factors
        self.current_scale_factor = kwargs.get('current_scale_factor', config.safety.current_scale_factor)
        self.position_scale_factor = kwargs.get('position_scale_factor', config.safety.position_scale_factor)
        self.voltage_scale_factor = kwargs.get('voltage_scale_factor', config.safety.voltage_scale_factor)
        
        # Temperature thresholds
        self.temp_threshold = kwargs.get('temp_threshold', config.safety.temp_threshold)
        self.temp_critical = kwargs.get('temp_critical', config.safety.temp_critical)
        self.temp_scale_factor = kwargs.get('temp_scale_factor', config.safety.temp_scale_factor)
        
        # Current-aware scaling
        self.heavy_load_threshold = kwargs.get('heavy_load_threshold', config.safety.heavy_load_threshold)
        self.critical_load_threshold = kwargs.get('critical_load_threshold', config.safety.critical_load_threshold)
        self.heavy_load_scale = kwargs.get('heavy_load_scale', config.safety.heavy_load_scale)
        self.moderate_load_scale = kwargs.get('moderate_load_scale', config.safety.moderate_load_scale)
        
        # Tension constraint
        self.min_tension_threshold = kwargs.get('min_tension_threshold', config.safety.min_tension_threshold)
        
        # Single motor tension
        self.single_motor_tension_threshold = kwargs.get('single_motor_tension_threshold', config.safety.single_motor_tension_threshold)
        self.single_motor_tension_high = kwargs.get('single_motor_tension_high', config.safety.single_motor_tension_high)
        self.single_motor_tension_critical = kwargs.get('single_motor_tension_critical', config.safety.single_motor_tension_critical)
        
        # Feature flags
        self.enable_current_safety = kwargs.get('enable_current_safety', config.safety.enable_current_safety)
        self.enable_position_safety = kwargs.get('enable_position_safety', config.safety.enable_position_safety)
        self.enable_tension_safety = kwargs.get('enable_tension_safety', config.safety.enable_tension_safety)
        self.enable_voltage_safety = kwargs.get('enable_voltage_safety', config.safety.enable_voltage_safety)
        self.enable_current_aware_safety = kwargs.get('enable_current_aware_safety', config.safety.enable_current_aware_safety)
        self.enable_tension_constraint = kwargs.get('enable_tension_constraint', config.safety.enable_tension_constraint)
        self.enable_temperature_safety = kwargs.get('enable_temperature_safety', config.safety.enable_temperature_safety)
        self.enable_single_motor_tension = kwargs.get('enable_single_motor_tension', config.safety.enable_single_motor_tension)
        
        # Logging
        self.log_interventions = kwargs.get('log_interventions', config.safety.log_interventions)
        
        # Statistics
        self.intervention_count = 0
        self.intervention_reasons = {reason: 0 for reason in SafetyReason}
        self.last_result = None
        
        # Print config summary
        self._print_config()
    
    def _print_config(self):
        """Print safety filter configuration."""
        print(f"SafetyFilter initialized:")
        print(f"  Motors: {self.motor_ids}")
        print(f"  Horizontal pair: {self.pair_horizontal}")
        print(f"  Vertical pair: {self.pair_vertical}")
        print(f"  Max pair current: {self.max_pair_current} mA")
        print(f"  Max single current: {self.max_single_current} mA")
        print(f"  Max position: {self.max_position}")
        print(f"  Voltage: {self.critical_voltage}-{self.min_voltage} V")
        print(f"  Temperature: {self.temp_threshold}°C (critical: {self.temp_critical}°C)")
        print(f"  Current load: {self.heavy_load_threshold}mA heavy, {self.critical_load_threshold}mA critical")
        print(f"  Min tension: {self.min_tension_threshold} mA")
        print(f"  Single motor tension: {self.single_motor_tension_threshold}mA (high: {self.single_motor_tension_high}, critical: {self.single_motor_tension_critical})")
        print(f"  Features: current={self.enable_current_safety}, position={self.enable_position_safety}, "
              f"tension={self.enable_tension_safety}, voltage={self.enable_voltage_safety}, "
              f"current_aware={self.enable_current_aware_safety}, tension_constraint={self.enable_tension_constraint}, "
              f"temperature={self.enable_temperature_safety}, single_tension={self.enable_single_motor_tension}")
    
    def filter(self, action, currents, positions, initial_positions, 
               voltages=None, temperatures=None, action_count=0):
        """Apply safety filter to action."""
        raw_action = action.copy()
        
        result = SafetyResult(
            safe_action=action.copy(),
            raw_action=raw_action,
            modified=False,
            reason=SafetyReason.SAFE,
            affected_motors=[],
            pre_filter_metrics={
                "max_current": float(np.max(np.abs(currents))) if currents is not None else 0,
            }
        )
        
        # Apply each safety layer in order
        safety_layers = [
            (self.enable_current_aware_safety and currents is not None, 
             self._apply_current_aware_safety, (action, currents)),
            (self.enable_temperature_safety and temperatures is not None,
             self._apply_temperature_safety, (action, temperatures)),
            (self.enable_voltage_safety and voltages is not None,
             self._apply_voltage_safety, (action, voltages)),
            (self.enable_tension_safety and currents is not None,
             self._apply_tension_safety, (action, currents)),
            (self.enable_tension_constraint and currents is not None,
             self._apply_tension_constraint, (action, currents)),
            (self.enable_current_safety and currents is not None,
             self._apply_current_safety, (action, currents)),
            (self.enable_position_safety and positions is not None and initial_positions is not None,
             self._apply_position_safety, (action, positions, initial_positions)),
        ]
        
        for enabled, safety_fn, args in safety_layers:
            if enabled:
                action, modified, reason, detail, affected = safety_fn(*args)
                if modified:
                    result.modified = True
                    result.reason = reason
                    result.detail = detail
                    result.affected_motors = affected
                    self._log_intervention(reason, detail, raw_action, action)
        
        # Clamp to valid range
        action = np.clip(action, -1.0, 1.0)
        result.safe_action = action
        self.last_result = result
        
        return result
    
    def _apply_current_aware_safety(self, action, currents):
        """Proactively scale down actions for heavily loaded motors."""
        modified = False
        reason = None
        detail = ""
        affected = []
        
        for i, motor_id in enumerate(self.motor_ids):
            current = abs(currents[i])
            
            if abs(action[i]) < 0.05:
                continue
            
            # Determine scaling factor
            if current < 200:
                scale = 1.0
            elif current < self.heavy_load_threshold:
                scale = 1.0 - (current - 200) / (self.heavy_load_threshold - 200) * (1.0 - self.moderate_load_scale)
                scale = max(self.moderate_load_scale, min(1.0, scale))
            elif current < self.critical_load_threshold:
                scale = self.moderate_load_scale - (current - self.heavy_load_threshold) / (self.critical_load_threshold - self.heavy_load_threshold) * (self.moderate_load_scale - self.heavy_load_scale)
                scale = max(self.heavy_load_scale, min(self.moderate_load_scale, scale))
            else:
                scale = self.heavy_load_scale
            
            # Apply scaling if pulling
            if action[i] > 0.05 and scale < 0.99:
                action[i] *= scale
                modified = True
                reason = SafetyReason.CURRENT_AWARE_SCALING
                detail = f"Motor {motor_id}: {current:.0f}mA -> scale={scale:.2f}"
                affected.append(motor_id)
        
        return action, modified, reason, detail, affected
    
    def _apply_temperature_safety(self, action, temperatures):
        """Scale down actions for overheating motors."""
        modified = False
        reason = None
        detail = ""
        affected = []
        
        for i, motor_id in enumerate(self.motor_ids):
            temp = temperatures[i]
            if temp is None:
                continue
            
            if temp > self.temp_threshold:
                # Use self.temp_threshold and self.temp_critical instead of hardcoded values
                if temp >= self.temp_critical:
                    scale = self.temp_scale_factor * 0.5
                    reason = SafetyReason.TEMPERATURE_CRITICAL
                else:
                    # Linear interpolation between threshold and critical
                    temp_range = self.temp_critical - self.temp_threshold
                    scale = 1.0 - (temp - self.temp_threshold) / temp_range * (1.0 - self.temp_scale_factor)
                    scale = max(self.temp_scale_factor, min(1.0, scale))
                    reason = SafetyReason.TEMPERATURE_HIGH
                
                action[i] *= scale
                modified = True
                detail = f"Motor {motor_id}: {temp:.0f}°C -> scale={scale:.2f}"
                affected.append(motor_id)
        
        return action, modified, reason, detail, affected
    
    def _apply_voltage_safety(self, action, voltages):
        """Scale down actions when voltage is low."""
        modified = False
        reason = None
        detail = ""
        affected = []
        
        for i, motor_id in enumerate(self.motor_ids):
            voltage = voltages[i]
            if voltage is None:
                continue
            
            if voltage < self.critical_voltage:
                scale = self.voltage_scale_factor * 0.5
                reason = SafetyReason.VOLTAGE_CRITICAL
            elif voltage < self.min_voltage:
                voltage_range = self.min_voltage - self.critical_voltage
                scale = self.voltage_scale_factor * (voltage - self.critical_voltage) / voltage_range
                scale = max(0.1, min(1.0, scale))
                reason = SafetyReason.VOLTAGE_LOW
            else:
                continue
            
            action[i] *= scale
            modified = True
            detail = f"Motor {motor_id}: {voltage:.1f}V -> scale={scale:.2f}"
            affected.append(motor_id)
        
        return action, modified, reason, detail, affected
    
    def _apply_tension_safety(self, action, currents):
        """Prevent motors from fighting and prevent single motor overload."""
        modified = False
        reason = None
        detail = ""
        affected = []
        
        # Single motor tension safety
        if self.enable_single_motor_tension:
            for i, motor_id in enumerate(self.motor_ids):
                current = abs(currents[i])
                
                if action[i] > 0.1 and current > self.single_motor_tension_threshold:
                    if current > self.single_motor_tension_critical:
                        scale = 0.3
                    elif current > self.single_motor_tension_high:
                        scale = 0.5
                    else:
                        scale = 0.7
                    
                    action[i] *= scale
                    modified = True
                    reason = SafetyReason.TENSION_SAFETY
                    detail = f"Motor {motor_id}: {current:.0f}mA -> scale={scale:.2f}"
                    affected.append(motor_id)
        
        # Opposing motors safety (check each pair)
        for pair_indices, pair_motors in [(self.h_idx, self.pair_horizontal), 
                                          (self.v_idx, self.pair_vertical)]:
            a0 = action[pair_indices[0]]
            a1 = action[pair_indices[1]]
            c0 = abs(currents[pair_indices[0]])
            c1 = abs(currents[pair_indices[1]])
            
            if a0 > 0.3 and a1 > 0.3 and c0 > 100 and c1 > 100:
                # Reduce the larger action
                if abs(a0) > abs(a1):
                    action[pair_indices[0]] *= 0.5
                    affected.append(pair_motors[0])
                else:
                    action[pair_indices[1]] *= 0.5
                    affected.append(pair_motors[1])
                
                modified = True
                reason = SafetyReason.OPPOSING_MOTORS
                detail = f"Motors {pair_motors[0]} and {pair_motors[1]} both pulling"
        
        return action, modified, reason, detail, affected
    
    def _apply_tension_constraint(self, action, currents):
        """Prevent slack strings by maintaining minimum tension."""
        modified = False
        reason = None
        detail = ""
        affected = []
        
        # Check each pair
        for pair_indices, pair_motors in [(self.h_idx, self.pair_horizontal), 
                                          (self.v_idx, self.pair_vertical)]:
            tension = abs(currents[pair_indices[0]]) + abs(currents[pair_indices[1]])
            
            if tension < self.min_tension_threshold:
                # Both releasing - prevent slack
                if action[pair_indices[0]] < 0 and action[pair_indices[1]] < 0:
                    action[pair_indices[0]] *= 0.2
                    action[pair_indices[1]] *= 0.2
                    modified = True
                    reason = SafetyReason.TENSION_TOO_LOW
                    detail = f"Pair {pair_motors} tension {tension:.0f}mA < {self.min_tension_threshold}mA"
                    affected.extend(pair_motors)
        
        return action, modified, reason, detail, affected
    
    def _apply_current_safety(self, action, currents):
        """Enforce hard limits on currents."""
        modified = False
        reason = None
        detail = ""
        affected = []
        
        # Check pair currents
        h_current = abs(currents[self.h_idx[0]]) + abs(currents[self.h_idx[1]])
        v_current = abs(currents[self.v_idx[0]]) + abs(currents[self.v_idx[1]])
        
        if h_current > self.max_pair_current:
            action[self.h_idx[0]] *= self.current_scale_factor
            action[self.h_idx[1]] *= self.current_scale_factor
            modified = True
            reason = SafetyReason.HIGH_HORIZONTAL_CURRENT
            detail = f"Horizontal current {h_current:.1f}mA > {self.max_pair_current}mA"
            affected.extend(self.pair_horizontal)
        elif v_current > self.max_pair_current:
            action[self.v_idx[0]] *= self.current_scale_factor
            action[self.v_idx[1]] *= self.current_scale_factor
            modified = True
            reason = SafetyReason.HIGH_VERTICAL_CURRENT
            detail = f"Vertical current {v_current:.1f}mA > {self.max_pair_current}mA"
            affected.extend(self.pair_vertical)
        
        # Check single motor over-current
        if not modified:
            for i, motor_id in enumerate(self.motor_ids):
                if abs(currents[i]) > self.max_single_current:
                    action[i] *= self.current_scale_factor
                    modified = True
                    reason = SafetyReason.SINGLE_MOTOR_OVER_CURRENT
                    detail = f"Motor {motor_id}: {abs(currents[i]):.1f}mA > {self.max_single_current}mA"
                    affected.append(motor_id)
        
        return action, modified, reason, detail, affected
    
    def _apply_position_safety(self, action, positions, initial_positions):
        """Prevent motors from hitting position limits."""
        modified = False
        reason = None
        detail = ""
        affected = []
        
        for i, motor_id in enumerate(self.motor_ids):
            delta = positions[i] - initial_positions[i]
            
            # Near upper limit and pulling
            if delta > self.max_position * 0.8 and action[i] > 0:
                scale = max(0.1, 1.0 - (delta - self.max_position * 0.8) / (self.max_position * 0.2))
                action[i] *= scale
                modified = True
                reason = SafetyReason.POSITION_LIMIT_PULL
                detail = f"Motor {motor_id}: {delta:.0f} ticks near upper limit"
                affected.append(motor_id)
            
            # Near lower limit and releasing
            elif delta < -self.max_position * 0.8 and action[i] < 0:
                scale = max(0.1, 1.0 - (abs(delta) - self.max_position * 0.8) / (self.max_position * 0.2))
                action[i] *= scale
                modified = True
                reason = SafetyReason.POSITION_LIMIT_RELEASE
                detail = f"Motor {motor_id}: {delta:.0f} ticks near lower limit"
                affected.append(motor_id)
        
        return action, modified, reason, detail, affected
    
    def _log_intervention(self, reason, detail, raw_action, safe_action):
        """Log intervention."""
        if not self.log_interventions:
            return
        
        self.intervention_count += 1
        self.intervention_reasons[reason] += 1
        
        print(f"\n[SAFETY INTERVENTION #{self.intervention_count}]")
        print(f"  Reason: {reason.value}")
        print(f"  Detail: {detail}")
        print(f"  Raw action: {raw_action}")
        print(f"  Safe action: {safe_action}")
        print(f"  Modification: {safe_action - raw_action}")
    
    def get_stats(self):
        """Get intervention statistics."""
        return {
            "total_interventions": self.intervention_count,
            "intervention_reasons": {
                k.value: v for k, v in self.intervention_reasons.items() if v > 0
            },
            "last_reason": self.last_result.reason.value if self.last_result else None,
            "last_modified": self.last_result.modified if self.last_result else False,
        }

