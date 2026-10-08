from dataclasses import dataclass
from typing import Dict, Any, Optional
import collections
import numpy as np
import numbers
import math

@dataclass(frozen=True)
class ControllerDecision:
    batch_id: int
    intensity: float

@dataclass(frozen=True)
class ControllerUpdate:
    config_id: str
    batch_id: int
    used_intensity: float
    tp: int
    fn: int
    window_start_batch_id: Optional[int]
    window_end_batch_id: Optional[int]
    configured_window_size: int
    window_batch_count: int
    window_tp_sum: int
    window_fn_sum: int
    rolling_recall: float
    state: str
    multiplier: float
    unclipped_next_intensity: float
    clipped_next_intensity: float
    hit_min_bound: bool
    hit_max_bound: bool
    zero_denominator: bool

def _validate_integral(val: Any, name: str, require_positive: bool = False) -> int:
    if isinstance(val, bool) or isinstance(val, np.bool_):
        raise TypeError(f"{name} cannot be a boolean")
    if not isinstance(val, numbers.Integral):
        raise TypeError(f"{name} must be an integral type")
    
    val_int = int(val)
    if require_positive:
        if val_int <= 0:
            raise ValueError(f"{name} must be positive")
    else:
        if val_int < 0:
            raise ValueError(f"{name} must be non-negative")
    return val_int

def _validate_real(val: Any, name: str) -> float:
    if isinstance(val, bool) or isinstance(val, np.bool_):
        raise TypeError(f"{name} cannot be a boolean")
    if not isinstance(val, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    
    val_float = float(val)
    if not math.isfinite(val_float):
        raise ValueError(f"{name} must be finite")
    return val_float

class RecallAwareController:
    @property
    def requires_feedback(self) -> bool:
        return True

    def __init__(self, config: Dict[str, Any], defense_config: Dict[str, Any], defense_name: str, zero_division_value: Any = 0.0):
        if not isinstance(config, collections.abc.Mapping):
            raise ValueError("config must be a mapping")
        if not isinstance(defense_config, collections.abc.Mapping):
            raise ValueError("defense_config must be a mapping")
            
        req_keys = ['id', 'window_size', 'Rcritical', 'Rmin', 'fast_decay', 'slow_decay', 'growth_factor']
        for k in req_keys:
            if k not in config:
                raise ValueError(f"Missing required key in config: {k}")
                
        if not isinstance(config['id'], str) or not config['id']:
            raise ValueError("config ID must be a nonempty string")
            
        req_def_keys = ['intensity_min', 'intensity_max']
        for k in req_def_keys:
            if k not in defense_config:
                raise ValueError(f"Missing required key in defense_config: {k}")

        self.config_id = config['id']
        self.window_size = _validate_integral(config['window_size'], "window_size", require_positive=True)
        self.r_critical = _validate_real(config['Rcritical'], "Rcritical")
        self.r_min = _validate_real(config['Rmin'], "Rmin")
        self.fast_decay = _validate_real(config['fast_decay'], "fast_decay")
        self.slow_decay = _validate_real(config['slow_decay'], "slow_decay")
        self.growth_factor = _validate_real(config['growth_factor'], "growth_factor")
        self.zero_division_value = _validate_real(zero_division_value, "zero_division_value")

        # Validation
        if not (0.0 < self.fast_decay < self.slow_decay < 1.0 < self.growth_factor):
            raise ValueError("Decay/growth factors must satisfy 0 < fast_decay < slow_decay < 1 < growth_factor")
        if not (0.0 <= self.r_critical < self.r_min <= 1.0):
            raise ValueError("Thresholds must satisfy 0 <= Rcritical < Rmin <= 1")
        if not (0.0 <= self.zero_division_value <= 1.0):
            raise ValueError("zero_division_value must be between 0 and 1")

        # Defense bounds extraction
        self.intensity_min = _validate_real(defense_config['intensity_min'], "intensity_min")
        self.intensity_max = _validate_real(defense_config['intensity_max'], "intensity_max")

        if self.intensity_min > self.intensity_max:
            raise ValueError("intensity_min cannot be greater than intensity_max")

        # Extract explicit frozen-base intensity
        if defense_name == "afp":
            if "epsilon_base" not in defense_config:
                raise ValueError("Missing epsilon_base in defense config")
            self.base_intensity = _validate_real(defense_config['epsilon_base'], "epsilon_base")
        elif defense_name == "randomized_smoothing":
            if "sigma" not in defense_config:
                raise ValueError("Missing sigma in defense config")
            self.base_intensity = _validate_real(defense_config['sigma'], "sigma")
        elif defense_name == "feature_squeezing":
            if "squeezing_intensity" not in defense_config:
                raise ValueError("Missing squeezing_intensity in defense config")
            self.base_intensity = _validate_real(defense_config['squeezing_intensity'], "squeezing_intensity")
        else:
            raise ValueError(f"Unrecognized defense ID: {defense_name}")

        if not (self.intensity_min <= self.base_intensity <= self.intensity_max):
            raise ValueError("Base intensity must be within bounds [intensity_min, intensity_max]")

        self.reset()

    def reset(self):
        """Deterministically isolates state and restores frozen base intensity. Aborts any pending decisions."""
        self.expected_batch_id = 0
        self.current_intensity = self.base_intensity
        self.pending_decision = None
        self.window = collections.deque()
        self.current_recall = None
        self.window_batch_count = 0

    def get_intensity(self, batch_id: Any) -> ControllerDecision:
        """Returns the intensity to use for the specified batch."""
        batch_id_int = _validate_integral(batch_id, "batch_id")
        
        if self.pending_decision is not None:
            raise RuntimeError(f"Cannot get intensity for batch {batch_id_int}; a decision is already pending for batch {self.pending_decision.batch_id}")
        
        if batch_id_int != self.expected_batch_id:
            raise ValueError(f"Expected batch_id {self.expected_batch_id}, but got {batch_id_int}")

        self.pending_decision = ControllerDecision(batch_id=batch_id_int, intensity=self.current_intensity)
        return self.pending_decision

    def submit_observations(self, batch_id: Any, tp: Any, fn: Any) -> ControllerUpdate:
        """Submits labels for the completed batch and updates intensity."""
        # 1. Validation before state check to ensure proper types
        batch_id_int = _validate_integral(batch_id, "batch_id")
        tp_int = _validate_integral(tp, "tp")
        fn_int = _validate_integral(fn, "fn")

        # 2. State enforce
        if self.pending_decision is None:
            raise RuntimeError("Cannot submit observations without first calling get_intensity()")
        if self.pending_decision.batch_id != batch_id_int:
            raise ValueError(f"Pending decision exists for batch {self.pending_decision.batch_id}, but observations submitted for {batch_id_int}")

        # Calculate values before modifying state (Atomicity)
        proposed_window = collections.deque(self.window)
        proposed_window.append((batch_id_int, tp_int, fn_int))
        if len(proposed_window) > self.window_size:
            proposed_window.popleft()

        window_tp_sum = sum(obs[1] for obs in proposed_window)
        window_fn_sum = sum(obs[2] for obs in proposed_window)
        total_obs = window_tp_sum + window_fn_sum
        
        zero_denominator = (total_obs == 0)
        rolling_recall = self.zero_division_value if zero_denominator else float(window_tp_sum) / float(total_obs)

        # State transition
        if rolling_recall < self.r_critical:
            state = "Red"
            multiplier = self.fast_decay
        elif rolling_recall < self.r_min:
            state = "Yellow"
            multiplier = self.slow_decay
        else:
            state = "Green"
            multiplier = self.growth_factor

        used_intensity = self.pending_decision.intensity
        unclipped_next_intensity = used_intensity * float(multiplier)
        clipped_next_intensity = float(np.clip(unclipped_next_intensity, self.intensity_min, self.intensity_max))

        hit_min = (clipped_next_intensity <= self.intensity_min) and (unclipped_next_intensity <= self.intensity_min)
        hit_max = (clipped_next_intensity >= self.intensity_max) and (unclipped_next_intensity >= self.intensity_max)

        window_start_batch_id = proposed_window[0][0] if proposed_window else None
        window_end_batch_id = proposed_window[-1][0] if proposed_window else None

        update_obj = ControllerUpdate(
            config_id=self.config_id,
            batch_id=batch_id_int,
            used_intensity=used_intensity,
            tp=tp_int,
            fn=fn_int,
            window_start_batch_id=window_start_batch_id,
            window_end_batch_id=window_end_batch_id,
            configured_window_size=self.window_size,
            window_batch_count=len(proposed_window),
            window_tp_sum=window_tp_sum,
            window_fn_sum=window_fn_sum,
            rolling_recall=rolling_recall,
            state=state,
            multiplier=multiplier,
            unclipped_next_intensity=unclipped_next_intensity,
            clipped_next_intensity=clipped_next_intensity,
            hit_min_bound=hit_min,
            hit_max_bound=hit_max,
            zero_denominator=zero_denominator
        )

        # Apply state changes atomically
        self.window = proposed_window
        self.current_intensity = clipped_next_intensity
        self.expected_batch_id += 1
        self.pending_decision = None
        self.current_recall = rolling_recall
        self.window_batch_count = len(proposed_window)

        return update_obj
