from jax import vmap
import jax.numpy as jnp
from bayests.time_series_tree import TimeSeriesNode


class Spline:
    @staticmethod
    def _omega(s1, s2, t):
        return jnp.where(s1 == s2, jnp.zeros_like(t), (t - s1) / (s2 - s1))

    @staticmethod
    def _basis(t, s, order, i):
        if order == 1:
            return jnp.where(
                (t >= s[i]) * (t < s[i + 1]),
                jnp.ones_like(t),
                jnp.zeros_like(t),
            )

        # Recurse left
        w1 = Spline._omega(s[i], s[i + order - 1], t)
        B1 = Spline._basis(t, s, order - 1, i)

        # Recurse right
        w2 = Spline._omega(s[i + 1], s[i + order], t)
        B2 = Spline._basis(t, s, order - 1, i + 1)
        return w1 * B1 + (1. - w2) * B2

    @staticmethod
    def _matrix(t, s, order):
        _s = jnp.pad(s, mode="edge", pad_width=(order - 1))  # Extend knots

        def _sb(i):
            return Spline._basis(t, _s, order, i)

        X = vmap(_sb)(jnp.arange(0, len(s) + order - 2))  # Make spline basis
        return X.T


class SplineTrend(TimeSeriesNode):
    def __init__(self, k, s, order):
        self.k = k
        self.s = s
        self.order = order

    @staticmethod
    def _design(s, order, t):
        return Spline._matrix(t, s, order)

    def design(self, t):
        if self.s is None:
            self.s = jnp.linspace(jnp.min(t), jnp.max(t), num=self.k)
        self.X = self._design(self.s, self.order, t)
