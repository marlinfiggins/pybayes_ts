import jax.numpy as jnp
from bayests.time_series_tree import SeasonalityNode
from bayests.utils import ComponentType
import numpyro
import numpyro.distributions as dist


class FourierSeasonality(SeasonalityNode):
    def __init__(self, k, period, name=None):
        self.k = k
        self.period = period
        self.name = name or f"FourierSeasonality(k={k})"
        self.n_params = 2 * k

    @staticmethod
    def _design(k, period, t):
        tau = 2 * jnp.pi * jnp.arange(1, k + 1) * t[:, None] / period
        return jnp.hstack((jnp.sin(tau), jnp.cos(tau)))

    def design(self, t):
        self.features = self._design(self.k, self.period, t)
        return self.features

    def _model(self, t):
        with numpyro.plate(self.name + f"_plate_{self.k}", self.n_params):
            beta = numpyro.sample(self.name + "_beta", dist.Normal(0.0, 1.0))
        self.design(t)
        self.output = jnp.dot(self.features, beta)
