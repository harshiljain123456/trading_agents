"""One-dimensional Kalman filter for smoothing agent signed confidence.

Model: random walk  x_t = x_{t-1} + w,  w ~ N(0, q);  z_t = x_t + v,  v ~ N(0, r).

Latency vs smoothing: the steady-state gain is K = P/(P + r) with P solving
P = (P + q) r / (P + q + r). A larger ``q/r`` ratio gives a larger gain (faster
reaction to reversals, less smoothing); a smaller ratio smooths noise but
delays response to genuine flips. Missing observations run the predict step
only, so the estimate holds while its uncertainty grows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Kalman1D:
    """Scalar Kalman filter with optional output clipping."""

    process_noise: float
    observation_noise: float
    initial_variance: float = 1.0
    lower: float = -1.0
    upper: float = 1.0

    def __post_init__(self) -> None:
        if self.process_noise <= 0 or self.observation_noise <= 0 or self.initial_variance <= 0:
            raise ValueError("Kalman noise parameters must be positive")
        if self.lower >= self.upper:
            raise ValueError("lower bound must be < upper bound")
        self.reset()

    def reset(self) -> None:
        """Forget all history (state becomes uninitialized)."""
        self.x: float | None = None
        self.p: float = self.initial_variance

    def update(self, z: float | None) -> float | None:
        """Assimilate observation ``z`` (None/NaN = missing) and return the estimate.

        The first valid observation initializes the state directly.
        Returns ``None`` until the filter has been initialized.
        """
        missing = z is None or not math.isfinite(z)
        if self.x is None:
            if missing:
                return None
            self.x = min(self.upper, max(self.lower, float(z)))  # type: ignore[arg-type]
            return self.x
        self.p += self.process_noise
        if not missing:
            k = self.p / (self.p + self.observation_noise)
            self.x = self.x + k * (float(z) - self.x)  # type: ignore[arg-type]
            self.p = (1.0 - k) * self.p
        self.x = min(self.upper, max(self.lower, self.x))
        return self.x
