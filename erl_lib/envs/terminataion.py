import numpy as np


class HealthCheck:
    def __init__(
        self,
        z_dim=None,
        healthy_state_range=(-100, 100),
        healthy_z_range=(-np.inf, np.inf),
        healthy_angle_range=(-np.inf, np.inf),
    ):
        self.z_dim = z_dim
        self.healthy_state_range = healthy_state_range
        self.healthy_z_range = healthy_z_range
        self.healthy_angle_range = healthy_angle_range

    @staticmethod
    def in_range(state: np.ndarray, min_max_range):
        """Check if state is in healthy range."""
        min_state, max_state = min_max_range
        return (min_state < state) * (state < max_state)

    def check(self, state):
        """Check if state is healthy."""
        if self.z_dim is None:
            return self.in_range(state, min_max_range=self.healthy_state_range).all(
                -1, keepdims=True
            )
        z = state[..., self.z_dim]
        angle = state[..., self.z_dim + 1]
        other = state[..., self.z_dim + 1 :]

        healthy = (
            self.in_range(z, min_max_range=self.healthy_z_range)[..., None]
            * self.in_range(angle, min_max_range=self.healthy_angle_range)[..., None]
            * self.in_range(other, min_max_range=self.healthy_state_range).all(
                -1, keepdims=True
            )
        )
        return healthy


class LargeStateTermination:
    """Hopper Termination Function."""

    def __init__(self, health_checker=None):
        self.health_checker = health_checker

    def copy(self):
        """Get copy of termination model."""
        return LargeStateTermination(self.health_checker)

    def is_healthy(self, state: np.ndarray):
        """Check if state is healthy."""
        return self.health_checker.check(state)

    def forward(self, state, action, next_state):
        """Return termination model logits."""
        if self.health_checker:
            return ~self.is_healthy(next_state)
        else:
            return False

    def __call__(self, state, action, next_state):
        return self.forward(state, action, next_state)
