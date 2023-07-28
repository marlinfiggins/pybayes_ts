import jax.numpy as jnp
from bayests.time_series_tree import SeasonalityNode
import numpyro
import numpyro.distributions as dist


class RadialBasisSeasonality(SeasonalityNode):
    def __init__(self, k, s=None, name=None, period=365.25, alpha=1):
        assert k is not None or s is not None, (
            "Linear trend must have either given a num of change points `k`"
            + "or change points `s` directly."
        )
        super().__init__()
        self.s = s  # Centers of basis elements
        self.k = k if s is None else len(s)  # Number of basis elements
        self.period = period
        self.alpha = alpha
        self.n_params = k
        self.name = name or f"RadialBasisSeasonality(k={k})"

    @staticmethod
    def _design(s, alpha, period, t):
        t_mod = jnp.remainder(t, period)
        return jnp.exp(jnp.power((t_mod[:, None] - s) / alpha, -2))

    def design(self, t):
        if self.s is None:
            self.s = jnp.linspace(jnp.min(t), jnp.max(t), num=self.k)
        self.features = self._design(self.s, self.alpha, self.period, t)

    def _model(self, t):
        with numpyro.plate(self.name + f"_plate_{self.k}", self.n_params):
            beta = numpyro.sample(self.name + "_beta", dist.Normal(0.0, 1.0))
        self.design(t)
        self.output = jnp.dot(self.features, beta)
