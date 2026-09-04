from dataclasses import dataclass
import math

@dataclass(frozen=True)
class FixedIntensityPolicy:
    """
    Immutable fixed intensity policy for the Base defense configuration.
    It returns the calibrated defense intensity for every batch and accepts no TP/FN observations.
    """
    config_id: str
    fixed_intensity: float
    intensity_min: float
    intensity_max: float
    
    @property
    def requires_feedback(self) -> bool:
        return False

    def __post_init__(self):
        if not isinstance(self.config_id, str) or not self.config_id:
            raise ValueError("config_id must be a nonempty string")
        
        if not math.isfinite(self.fixed_intensity) or self.fixed_intensity < 0:
            raise ValueError("fixed_intensity must be a finite non-negative number")
            
        if self.fixed_intensity < self.intensity_min or self.fixed_intensity > self.intensity_max:
            raise ValueError(f"fixed_intensity {self.fixed_intensity} must be within bounds [{self.intensity_min}, {self.intensity_max}]")

    def get_intensity(self, batch_id: int):
        if not isinstance(batch_id, int) or batch_id < 0:
            raise ValueError("batch_id must be a non-negative integer")
            
        from recall_aware_ids.controller import ControllerDecision
        return ControllerDecision(batch_id=batch_id, intensity=self.fixed_intensity)
        
    def reset(self):
        # Immutable and stateless
        pass

