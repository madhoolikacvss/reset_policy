# reset_policy/control/safety_filter.py

"""
Safety filter for RL actions on the cube-string system.

The safety filter intercepts PPO actions and modifies them to prevent:
1. Motors fighting against each other (high opposing currents)
2. Exceeding position limits
3. Over-current conditions

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
    TENSION_SAFETY = "tension_safety"


@dataclass
class SafetyResult:
    """Result of applying the safety filter."""
    
    # The action after filtering
    safe_action: np.ndarray
    
    # The original action
    raw_action: np.ndarray
    
    # Whether the action was modified
    modified: bool
    
    # Why it was modified (if applicable)
    reason: Optional[SafetyReason] = None
    
    # Detailed reason for logging
    detail: str = ""
    
    # Which motors were affected
    affected_motors: List[int] = None
    
    # Pre-filter current/pressure metrics
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
        - Pair-wise current limiting (horizontal: 16,17 | vertical: 18,19)
        - Single motor over-current protection
        - Position limit protection
        - Logging of all interventions
        - Configurable thresholds
    """
    
    def __init__(
        self,
        motor_ids: List[int],
        pair_horizontal: List[int] = None,
        pair_vertical: List[int] = None,
        
        # Current thresholds (mA)
        max_pair_current: float = 800.0,      # Max combined current for a pair
        max_single_current: float = 600.0,    # Max current for a single motor
        
        # Position limits (encoder ticks relative to initial)
        max_position: float = 8000.0,         # Max absolute position
        
        # Action scaling factors
        current_scale_factor: float = 0.25,   # How much to scale when over-current
        position_scale_factor: float = 0.5,   # How much to scale when near limit
        
        # Tension safety
        tension_threshold: float = 0.7,       # Max allowed opposing action magnitude
        
        # Enable/disable features
        enable_current_safety: bool = True,
        enable_position_safety: bool = True,
        enable_tension_safety: bool = True,
        
        # Logging
        log_interventions: bool = True,
    ):
        self.motor_ids = motor_ids
        self.num_motors = len(motor_ids)
        
        # Define motor pairs
        if pair_horizontal is None:
            pair_horizontal = [16, 17]  # Horizontal pair (X-axis)
        if pair_vertical is None:
            pair_vertical = [18, 19]    # Vertical pair (Y-axis)
        
        self.pair_horizontal = pair_horizontal
        self.pair_vertical = pair_vertical
        
        # Get indices for pairs
        self.h_idx = [self.motor_ids.index(m) for m in pair_horizontal]
        self.v_idx = [self.motor_ids.index(m) for m in pair_vertical]
        
        # Thresholds
        self.max_pair_current = max_pair_current
        self.max_single_current = max_single_current
        self.max_position = max_position
        self.current_scale_factor = current_scale_factor
        self.position_scale_factor = position_scale_factor
        self.tension_threshold = tension_threshold
        
        # Feature flags
        self.enable_current_safety = enable_current_safety
        self.enable_position_safety = enable_position_safety
        self.enable_tension_safety = enable_tension_safety
        self.log_interventions = log_interventions
        
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
        
    def filter(
        self,
        action: np.ndarray,
        currents: np.ndarray,
        positions: np.ndarray,
        initial_positions: np.ndarray,
        action_count: int = 0,
    ) -> SafetyResult:
        """
        Apply safety filter to the action.
        
        Args:
            action: Raw action from PPO (shape: [4])
            currents: Current motor currents in mA (shape: [4])
            positions: Current motor positions (shape: [4])
            initial_positions: Initial motor positions (shape: [4])
            action_count: Current action number (for logging)
            
        Returns:
            SafetyResult with filtered action and metadata
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
        
        # --------------------------------------------------
        # 1. Tension Safety (opposing motors)
        # --------------------------------------------------
        
        if self.enable_tension_safety:
            action, modified, reason, detail, affected = self._apply_tension_safety(
                action, currents, positions
            )
            if modified:
                result.modified = True
                result.reason = reason
                result.detail = detail
                result.affected_motors = affected
                self._log_intervention(reason, detail, raw_action, action)
        
        # --------------------------------------------------
        # 2. Current Safety (pair-wise and single motor)
        # --------------------------------------------------
        
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
        
        # --------------------------------------------------
        # 3. Position Safety (near limits)
        # --------------------------------------------------
        
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
        # 4. Clamp to valid range
        # --------------------------------------------------
        
        action = np.clip(action, -1.0, 1.0)
        
        # Store result
        result.safe_action = action
        self.last_result = result
        
        return result
    
    # ============================================================
    # Current Safety
    # ============================================================
    
    def _apply_current_safety(
        self,
        action: np.ndarray,
        currents: np.ndarray,
    ) -> Tuple[np.ndarray, bool, Optional[SafetyReason], str, List[int]]:
        """
        Apply current-based safety.
        
        Checks:
            1. Pair-wise current (horizontal and vertical pairs)
            2. Single motor over-current
        """
        
        modified = False
        reason = None
        detail = ""
        affected = []
        
        # Check pair-wise currents
        h_current = abs(currents[self.h_idx[0]]) + abs(currents[self.h_idx[1]])
        v_current = abs(currents[self.v_idx[0]]) + abs(currents[self.v_idx[1]])
        
        # Horizontal pair over-current
        if h_current > self.max_pair_current:
            # Scale down both motors in the pair
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
        
        # Check single motor over-current
        if not modified:
            for i, motor_id in enumerate(self.motor_ids):
                if abs(currents[i]) > self.max_single_current:
                    # Scale down just this motor
                    action[i] *= self.current_scale_factor
                    modified = True
                    reason = SafetyReason.SINGLE_MOTOR_OVER_CURRENT
                    detail = f"Motor {motor_id} current {abs(currents[i]):.1f}mA > {self.max_single_current}mA"
                    affected.append(motor_id)
        
        return action, modified, reason, detail, affected
    
    # ============================================================
    # Tension Safety
    # ============================================================
    
    def _apply_tension_safety(
        self,
        action: np.ndarray,
        currents: np.ndarray,
        positions: np.ndarray,
    ) -> Tuple[np.ndarray, bool, Optional[SafetyReason], str, List[int]]:
        """
        Prevent motors from fighting each other.
        
        Idea: If a pair has both motors pulling (positive action), and they're
        already under tension (high current), reduce one of them.
        """
        
        modified = False
        reason = None
        detail = ""
        affected = []
        
        # Check horizontal pair (16, 17)
        a16 = action[self.h_idx[0]]
        a17 = action[self.h_idx[1]]
        c16 = abs(currents[self.h_idx[0]]) if currents is not None else 0
        c17 = abs(currents[self.h_idx[1]]) if currents is not None else 0
        
        # Both pulling AND already have tension
        if a16 > 0.3 and a17 > 0.3 and c16 > 100 and c17 > 100:
            # Scale down the larger one more
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
        c18 = abs(currents[self.v_idx[0]]) if currents is not None else 0
        c19 = abs(currents[self.v_idx[1]]) if currents is not None else 0
        
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
    # Position Safety
    # ============================================================
    
    def _apply_position_safety(
        self,
        action: np.ndarray,
        positions: np.ndarray,
        initial_positions: np.ndarray,
    ) -> Tuple[np.ndarray, bool, Optional[SafetyReason], str, List[int]]:
        """
        Prevent motors from hitting position limits.
        
        If a motor is near its limit and the action pushes it further,
        scale down the action.
        """
        
        modified = False
        reason = None
        detail = ""
        affected = []
        
        for i, motor_id in enumerate(self.motor_ids):
            pos = positions[i]
            init_pos = initial_positions[i]
            delta = pos - init_pos
            
            # Near upper limit and action pushes upward (positive)
            if delta > self.max_position * 0.8 and action[i] > 0:
                scale = max(
                    0.1,
                    1.0 - (delta - self.max_position * 0.8) / (self.max_position * 0.2)
                )
                action[i] *= scale
                modified = True
                reason = SafetyReason.POSITION_LIMIT_PULL
                detail = f"Motor {motor_id} near upper limit: {delta:.0f} ticks"
                affected.append(motor_id)
            
            # Near lower limit and action pushes downward (negative)
            elif delta < -self.max_position * 0.8 and action[i] < 0:
                scale = max(
                    0.1,
                    1.0 - (abs(delta) - self.max_position * 0.8) / (self.max_position * 0.2)
                )
                action[i] *= scale
                modified = True
                reason = SafetyReason.POSITION_LIMIT_RELEASE
                detail = f"Motor {motor_id} near lower limit: {delta:.0f} ticks"
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
    """
    Create a safety filter with optional configuration.
    
    Args:
        motor_ids: List of motor IDs
        config: Optional configuration dictionary
    
    Returns:
        SafetyFilter instance
    """
    
    if config is None:
        config = {}
    
    return SafetyFilter(
        motor_ids=motor_ids,
        pair_horizontal=config.get('pair_horizontal', [16, 17]),
        pair_vertical=config.get('pair_vertical', [18, 19]),
        max_pair_current=config.get('max_pair_current', 800.0),
        max_single_current=config.get('max_single_current', 600.0),
        max_position=config.get('max_position', 8000.0),
        current_scale_factor=config.get('current_scale_factor', 0.25),
        position_scale_factor=config.get('position_scale_factor', 0.5),
        tension_threshold=config.get('tension_threshold', 0.7),
        enable_current_safety=config.get('enable_current_safety', True),
        enable_position_safety=config.get('enable_position_safety', True),
        enable_tension_safety=config.get('enable_tension_safety', True),
        log_interventions=config.get('log_interventions', True),
    )