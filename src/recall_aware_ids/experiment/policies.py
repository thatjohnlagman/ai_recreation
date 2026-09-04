class FixedIntensityPolicy:
    """
    Immutable fixed intensity policy for the Base defense configuration.
    It returns the calibrated defense intensity for every batch and accepts no TP/FN observations.
    """
    def __init__(self, config_id: str, fixed_intensity: float):
        if not isinstance(config_id, str) or not config_id:
            raise ValueError("config_id must be a nonempty string")
        
        # We store intensity as a float explicitly
        self.config_id = config_id
        self.fixed_intensity = float(fixed_intensity)
        import math
        if not math.isfinite(self.fixed_intensity):
            raise ValueError("fixed_intensity must be finite")

    def get_intensity(self, batch_id: int):
        from recall_aware_ids.controller import ControllerDecision
        # Just return the static decision immediately, no state required.
        return ControllerDecision(batch_id=batch_id, intensity=self.fixed_intensity)
        
    def submit_observations(self, batch_id: int, tp: int, fn: int):
        from recall_aware_ids.controller import ControllerUpdate
        # It logs dummy window states to adhere to the logging schema, but they are all 0/fixed.
        return ControllerUpdate(
            config_id=self.config_id,
            batch_id=batch_id,
            used_intensity=self.fixed_intensity,
            tp=int(tp),
            fn=int(fn),
            window_start_batch_id=None,
            window_end_batch_id=None,
            configured_window_size=0,
            window_batch_count=0,
            window_tp_sum=0,
            window_fn_sum=0,
            rolling_recall=0.0,
            state="Base",
            multiplier=1.0,
            unclipped_next_intensity=self.fixed_intensity,
            clipped_next_intensity=self.fixed_intensity,
            hit_min_bound=False,
            hit_max_bound=False,
            zero_denominator=False
        )

    def reset(self):
        # No state to reset, but must implement the interface
        pass
